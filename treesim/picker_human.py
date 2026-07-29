"""Gamepad-driven human teleop recording for imitation learning.

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
    PHASE_DRIVE = 1
    PHASE_GRASP = 2
    PHASE_DEPOSIT = 3

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
        self._wrist = int(rb["wrist"][0])

        # IK init: use short joint names (strip "ridgeback_franka/" prefix)
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

        self.phase = self.PHASE_DRIVE
        self.gripper_open = True
        self._precision = False
        self._buffer = []
        self._episode_id = _find_next_episode(self.dataset_dir)

        # persistent IK target (world frame, updated by stick deltas)
        self._ik_target_pos = None
        self._ik_target_approach = None

        # assisted grasp
        self._g_path = None
        self._g_idx = 0
        self._assist_max = 0.20
        self._assist_steps = 15
        self._move_speed = 0.015
        self._fine_speed = 0.004
        self._drive_speed = 0.5
        self._turn_speed = 1.0

        os.makedirs(self.dataset_dir, exist_ok=True)
        print(f"[gamepad] recording to {self.dataset_dir}")
        print(f"[gamepad] ep={self._episode_id}  "
              f"B=save  X=assist  A=gripper  Y=precision")
        if not self.gamepad.connected:
            print(f"[gamepad] ⚠ 手柄未连接，先运行 sudo python3 wsl_recv.py")

    def update(self, viewer=None):
        if not self.gamepad.connected:
            if not self.gamepad._running:
                self.gamepad.start()
            return

        s = self.gamepad.state.snapshot()

        # edge detection
        just = {}
        for btn in ("btn_a", "btn_b", "btn_x", "btn_y",
                     "btn_lb", "btn_rb", "btn_start"):
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
            raise KeyboardInterrupt()

        # stick curve (square: small push = fine, full push = fast)
        def _curve(x):
            sign = 1.0 if x >= 0 else -1.0
            return sign * (abs(x) ** 2.0)

        # chassis
        fwd = _curve(-s["left_y"])
        turn = _curve(s["left_x"])
        if abs(fwd) < 0.02: fwd = 0.0
        if abs(turn) < 0.02: turn = 0.0

        # arm
        speed = self._fine_speed if self._precision else self._move_speed
        dx = _curve(s["right_x"]) * speed
        dy = _curve(s["right_y"]) * speed
        dz = (s["rt"] - s["lt"]) * speed

        dyaw = 0.0
        dpitch = 0.0
        droll = 0.0
        if s["btn_lb"]: dyaw += 0.01
        if s["btn_rb"]: dyaw -= 0.01
        dpitch = s["hat_y"] * 0.01
        droll = s["hat_x"] * 0.01

        # assisted grasp step
        if self._g_path is not None:
            self._step_assist()

        # arm IK
        has_arm = abs(dx) > 1e-8 or abs(dy) > 1e-8 or abs(dz) > 1e-8 \
                  or abs(dyaw) > 1e-8 or abs(dpitch) > 1e-8 or abs(droll) > 1e-8
        if has_arm:
            self._ik_move(dx, dy, dz, dyaw, dpitch, droll)

        # gripper
        gv = 0.04 if self.gripper_open else 0.0
        self._tq_host[self._finger_tq] = gv
        self.sim.control.joint_target_q.assign(self._tq_host)

        # chassis drive
        if abs(fwd) > 0.01 or abs(turn) > 0.01:
            self._drive_chassis(fwd, turn)

        self._record_frame()

    # -- IK -------------------------------------------------------------------

    def _ik_move(self, dx, dy, dz, dyaw, dpitch, droll):
        """Persistent world-frame target, updated by chassis-frame stick deltas."""
        bq = self.sim.state_0.body_q.numpy()
        ch_p = bq[self._chassis, :3]
        ch_q = bq[self._chassis, 3:7]
        R_ch = self._q2r(ch_q)

        # init persistent target from current pose
        ee = bq[self._wrist, :3].copy()
        if self._ik_target_pos is None:
            self._ik_target_pos = ee.copy()
            self._ik_target_approach = np.array([0.0, 0.0, -1.0])

        # world-frame delta from chassis-frame stick input
        dw = R_ch @ np.array([dx, dy, dz])
        self._ik_target_pos += dw

        # rotation (in world frame, convert approach back to chassis)
        if abs(dyaw) > 1e-8 or abs(dpitch) > 1e-8 or abs(droll) > 1e-8:
            Rz = np.array([[math.cos(dyaw), -math.sin(dyaw), 0],
                           [math.sin(dyaw), math.cos(dyaw), 0],
                           [0, 0, 1]])
            Ry = np.array([[math.cos(dpitch), 0, math.sin(dpitch)],
                           [0, 1, 0],
                           [-math.sin(dpitch), 0, math.cos(dpitch)]])
            Rx = np.array([[1, 0, 0],
                           [0, math.cos(droll), -math.sin(droll)],
                           [0, math.sin(droll), math.cos(droll)]])
            aw = R_ch @ self._ik_target_approach
            na = Rz @ Ry @ Rx @ aw
            self._ik_target_approach = R_ch.T @ na
            n = np.linalg.norm(self._ik_target_approach)
            if n > 1e-6:
                self._ik_target_approach /= n

        # safety limits (world frame)
        rel = self._ik_target_pos - ch_p
        d = np.linalg.norm(rel)
        if d > 0.65:
            rel *= 0.65 / d
            self._ik_target_pos = ch_p + rel
        self._ik_target_pos[2] = np.clip(self._ik_target_pos[2], 0.05, 1.2)

        # to chassis frame for IK
        t_base = R_ch.T @ (self._ik_target_pos - ch_p)
        app_base = self._ik_target_approach

        jq = self.sim.state_0.joint_q.numpy()
        q_init = jq[self._ik_qstart]

        try:
            qr, err = self.ik.solve(t_base, app_base, q_init, iters=32)
            if err < 0.01:
                self._tq_host[self._arm_tq] = qr[:len(self._arm_tq)]
                self.sim.control.joint_target_q.assign(self._tq_host)
        except Exception:
            pass

    def _drive_chassis(self, fwd, turn):
        try:
            q = self.sim.state_0.joint_q.numpy()
            rb = self.tm.robot_data
            for qs, ds in zip(rb["planar_q"], rb["planar_dof"]):
                yaw = float(q[qs + 3])
                self.drv._target_host[ds + 0] = fwd * self._drive_speed * math.cos(yaw)
                self.drv._target_host[ds + 1] = fwd * self._drive_speed * math.sin(yaw)
                self.drv._target_host[ds + 3] = turn * self._turn_speed
            self.sim.control.joint_target_qd.assign(self.drv._target_host)
        except Exception:
            pass

    # -- assisted grasp -------------------------------------------------------

    def _start_assist(self):
        if self.sim.apples is None:
            return
        bq = self.sim.state_0.body_q.numpy()
        ab = self.tm.apple_data["apple_body"]
        ee = bq[self._wrist, :3]
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
        bq = self.sim.state_0.joint_q.numpy()
        ch_p = bq[self._chassis, :3]
        ch_q = bq[self._chassis, 3:7]
        R_ch = self._q2r(ch_q)
        t_base = R_ch.T @ (tgt - ch_p)
        jq = self.sim.state_0.joint_q.numpy()
        q_init = jq[self._ik_qstart]
        try:
            qr, err = self.ik.solve(t_base, np.array([0, 0, -1]), q_init, iters=32)
            if err < 0.02:
                self._tq_host[self._arm_tq] = qr[:len(self._arm_tq)]
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

    # -- helpers --------------------------------------------------------------

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
