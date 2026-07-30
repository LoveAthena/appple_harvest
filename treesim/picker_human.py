"""Gamepad-driven human teleop recording for imitation learning.

Mapping:
    D-pad       chassis drive (position increment per frame)
    Left stick  J1(waist) / J2(shoulder)
    Right stick J4(elbow) / J6(wrist pitch)
    LT / RT     J3(upper arm roll)
    LB / RB     J7(wrist roll)
    A           gripper toggle
    B           save + reset
    X           assisted grasp
    Y           precision mode

Usage:
    sudo python3 wsl_recv.py                          # terminal 1
    python scripts/grow_tree.py --apples --foliage \  # terminal 2
        --robot --gamepad --viewer gl
"""

from __future__ import annotations

import math
import os, glob, re
from pathlib import Path

import numpy as np

from . import robot as _robot
from .picker import ArmIK
from .gamepad import GamepadReader


class GamepadPicker:
    def __init__(self, sim, tm, wrist_cam, robot_driver, cfg,
                 dataset_dir: str = "demo_data"):
        self.sim = sim
        self.tm = tm
        self.cam = wrist_cam
        self.drv = robot_driver
        self.cfg = cfg
        self.dataset_dir = os.path.join(Path(__file__).parent.parent, dataset_dir)

        self.ik = ArmIK(list(_robot._ARM_HOME.values()))

        rb = tm.robot_data
        self._arm_tq = np.asarray(rb["arm_tq"][0], dtype=int)
        self._arm_home = np.asarray(rb["arm_home"], dtype=float)
        self._finger_tq = self._arm_tq[-2:]
        self._chassis = int(rb["chassis"][0])

        # joint index map: short joint names
        labels = [l.rsplit("/", 1)[-1] for l in sim.model.joint_label]
        jqs = sim.model.joint_q_start.numpy()
        self._ik_qstart = []
        for name in _robot._ARM_HOME:
            try:
                self._ik_qstart.append(int(jqs[labels.index(name)]))
            except ValueError:
                pass

        # seed arm at home
        self._tq_host = sim.control.joint_target_q.numpy()
        self._tq_host[self._arm_tq] = self._arm_home
        sim.control.joint_target_q.assign(self._tq_host)

        # gamepad
        self.gamepad = GamepadReader()
        self.gamepad.start()
        self._prev_btn = {}

        self.gripper_open = True
        self._precision = False
        self._j4_mode = True  # True=右摇杆X→J4, False=右摇杆X→J5
        self._buffer = []
        self._episode_id = _find_next_episode(self.dataset_dir)

        # assisted grasp
        self._g_path = None
        self._g_idx = 0
        self._assist_max = 0.20
        self._assist_steps = 15

        os.makedirs(self.dataset_dir, exist_ok=True)
        print(f"[gamepad] recording to {self.dataset_dir}")
        print(f"[gamepad] ep={self._episode_id}  "
              f"B=save  X=assist  A=gripper  Y=precision")
        if not self.gamepad.connected:
            print(f"[gamepad] ⚠ 先运行 sudo python3 wsl_recv.py")

    def update(self, viewer=None):
        if not self.gamepad.connected:
            if not self.gamepad._running:
                self.gamepad.start()
            return

        s = self.gamepad.state.snapshot()

        # edge detection
        just = {}
        for btn in ("btn_a", "btn_b", "btn_x", "btn_y", "btn_start"):
            prev = self._prev_btn.get(btn, False)
            just[btn] = s[btn] and not prev
            self._prev_btn[btn] = s[btn]

        if just["btn_a"]:
            self.gripper_open = not self.gripper_open
            print(f"  gripper {'open' if self.gripper_open else 'closed'}")
        if just["btn_b"]:
            self._save_and_reset()
            return
        if just["btn_x"]:
            self._start_assist()
        if just["btn_y"]:
            self._precision = not self._precision
            print(f"  {'precision' if self._precision else 'normal'} mode")
        if just["btn_start"]:
            self._j4_mode = not self._j4_mode
            print(f"  right stick X → {'J4 elbow' if self._j4_mode else 'J5 forearm roll'}")

        # ---- 方向键 → 底盘位置增量 ----
        dx = float(s["hat_y"]) * 0.04
        dyaw = float(s["hat_x"]) * 0.06
        if abs(dx) > 0.001 or abs(dyaw) > 0.001:
            jq = self.sim.state_0.joint_q.numpy().copy()
            for qs in self.tm.robot_data["planar_q"]:
                yaw = float(jq[qs + 3])
                jq[qs] += dx * math.cos(yaw)
                jq[qs + 1] += dx * math.sin(yaw)
                jq[qs + 3] += dyaw
            self.sim.state_0.joint_q.assign(jq)
            self.sim.model.joint_q.assign(jq)

        # ---- 关节角直接控制（SO100 风格）----
        def _curve(x):
            sign = 1.0 if x >= 0 else -1.0
            return sign * (abs(x) ** 2.0)

        speed = 0.01 if self._precision else 0.04
        dj = [0.0] * 7  # J1~J7
        dj[0] = _curve(s["left_x"]) * speed     # J1 waist
        dj[1] = _curve(s["left_y"]) * speed     # J2 shoulder
        dj[2] = (s["lt"] - s["rt"]) * speed     # J3 upper arm roll (LT/RT)
        if self._j4_mode:
            dj[3] = _curve(s["right_x"]) * speed    # J4 elbow
        else:
            dj[4] = _curve(s["right_x"]) * speed    # J5 forearm roll
        dj[5] = -_curve(s["right_y"]) * speed   # J6 wrist pitch
        dj[6] = ((1.0 if s["btn_lb"] else 0.0) - (1.0 if s["btn_rb"] else 0.0)) * speed * 2  # J7

        has_input = any(abs(v) > 1e-8 for v in dj)
        if has_input:
            jq = self.sim.state_0.joint_q.numpy()
            arm_q = jq[self._ik_qstart].copy()[:7]
            for i in range(7):
                arm_q[i] += dj[i]
            self._tq_host[self._arm_tq[:7]] = arm_q
            self.sim.control.joint_target_q.assign(self._tq_host)

        # assisted grasp
        if self._g_path is not None:
            self._step_assist()

        # gripper
        gv = 0.04 if self.gripper_open else -0.002
        self._tq_host[self._finger_tq] = gv
        self.sim.control.joint_target_q.assign(self._tq_host)

        self._record_frame()

    # -- assisted grasp (IK, only when pressing X) ---------------------------

    def _start_assist(self):
        if self.sim.apples is None:
            return
        bq = self.sim.state_0.body_q.numpy()
        ab = self.tm.apple_data["apple_body"]
        ee = bq[self._chassis + 1, :3]  # wrist approx
        best_dist = float("inf")
        best_pos = None
        for a in ab:
            if a < len(bq):
                p = bq[a, :3]
                d = np.linalg.norm(p - ee)
                if d < best_dist:
                    best_dist = d
                    best_pos = p.copy()
        if best_pos is None or best_dist > self._assist_max:
            return
        fwd = ee - best_pos
        fwd[2] = 0
        fn = np.linalg.norm(fwd)
        if fn > 1e-6: fwd /= fn
        tg = best_pos + fwd * 0.03
        tg[2] = min(tg[2] + 0.02, best_pos[2] + 0.05)
        n = self._assist_steps
        self._g_path = [ee + (tg - ee) * (i / n) for i in range(n + 1)]
        self._g_idx = 0

    def _step_assist(self):
        if self._g_idx >= len(self._g_path):
            self._g_path = None
            return
        tgt = self._g_path[self._g_idx]
        self._g_idx += 1
        ch_p = self.sim.state_0.body_q.numpy()[self._chassis, :3]
        ch_q = self.sim.state_0.body_q.numpy()[self._chassis, 3:7]
        R_ch = self._q2r(ch_q)
        t_base = R_ch.T @ (tgt - ch_p)
        jq = self.sim.state_0.joint_q.numpy()
        q_init = jq[self._ik_qstart]
        try:
            qr, err = self.ik.solve(t_base, np.array([0, 0, -1]), q_init, iters=32)
            if err < 0.03:
                self._tq_host[self._arm_tq[:len(qr)]] = qr[:len(self._arm_tq)]
                self.sim.control.joint_target_q.assign(self._tq_host)
        except Exception:
            pass

    # -- recording ------------------------------------------------------------

    def _record_frame(self):
        jq = self.sim.state_0.joint_q.numpy()
        arm = jq[self._ik_qstart][:7].copy().astype(np.float32)
        g = float(self._tq_host[self._finger_tq[0]])
        qpos = np.concatenate([arm, [g]])
        img = self._grab_wrist_image()
        self._buffer.append({
            "qpos": qpos, "action": qpos.copy(),
            "image": img, "assisted": self._g_path is not None,
        })

    def _grab_wrist_image(self):
        h, w = 144, 192
        try:
            if self.cam is not None:
                c = self.cam.color.numpy()
                if c.size > 0:
                    return c[0, 0, :, :, :3].astype(np.uint8)
        except Exception:
            pass
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _save_and_reset(self):
        if len(self._buffer) < 5:
            return
        import h5py
        T = len(self._buffer)
        qp = np.array([f["qpos"] for f in self._buffer], dtype=np.float32)
        ac = np.array([f["action"] for f in self._buffer], dtype=np.float32)
        imgs = np.array([f["image"] for f in self._buffer], dtype=np.uint8)
        as_ = np.array([f["assisted"] for f in self._buffer], dtype=bool)
        fpath = os.path.join(self.dataset_dir, f"episode_{self._episode_id}.hdf5")
        with h5py.File(fpath, "w") as f:
            f.attrs["sim"] = True
            f.attrs["success"] = False
            f.create_dataset("action", data=ac, compression="gzip", compression_opts=4)
            f.create_dataset("observations/qpos", data=qp, compression="gzip", compression_opts=4)
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = np.transpose(imgs, (0, 3, 1, 2))
            f.create_dataset("observations/images/wrist", data=imgs,
                             compression="gzip", compression_opts=4)
            f.create_dataset("meta/assisted_steps", data=as_)
        print(f"  [gamepad] saved ep{self._episode_id} ({T}f)")
        self._episode_id += 1
        self._buffer = []

    @staticmethod
    def _q2r(q):
        w, x, y, z = q[3], q[0], q[1], q[2]
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

    def close(self):
        self.gamepad.stop()
        if len(self._buffer) > 5:
            self._save_and_reset()


def _find_next_episode(dir_: str) -> int:
    os.makedirs(dir_, exist_ok=True)
    mx = -1
    for fp in glob.glob(os.path.join(dir_, "episode_*.hdf5")):
        m = re.search(r"episode_(\d+)\.hdf5$", fp)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1
