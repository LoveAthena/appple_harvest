#!/usr/bin/env python
"""Render individual scene screenshots for the OrchardBench paper.

One scene per process (a fresh GL context each time is the robust way to drive
ViewerGL headless).  Montages (foliage grid, DR gallery) are composed from the
per-scene PNGs by ``montage`` mode, which needs no GPU.

Usage::

    python scripts/render_paper_figs.py <scene> <out.png> [key=val ...]

Scenes: skeleton apples foliage break bend robot terrain depth dr
Montage: python scripts/render_paper_figs.py montage <out.png> a.png b.png ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def _kv(args):
    d = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            d[k] = v
    return d


def _save(arr, out):
    from PIL import Image
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    Image.fromarray(arr).save(out)
    print("wrote", out, arr.shape)


def _frame_camera(viewer, tm, sim, wp, zoom=2.3, az=0.15, el=0.28, up=0.15):
    import math
    lo, hi = tm.skeleton.bounds()
    C = (lo + hi) / 2.0
    dist = zoom * max(hi - lo)
    pos = np.array([C[0] + az * dist, C[1] - dist, C[2] + el * dist])
    d = C - pos
    d = d / (np.linalg.norm(d) + 1e-9)
    try:
        viewer.set_camera(pos=wp.vec3(*[float(x) for x in pos]),
                          pitch=float(math.degrees(math.asin(d[2]))),
                          yaw=float(math.degrees(math.atan2(d[1], d[0]))))
    except Exception as e:
        print("camera:", e)


def build_scene(scene, opt):
    """Return (cfg, extra) for the requested scene."""
    import warp as wp  # noqa
    from treesim.config import TreeConfig, preset, StiffnessModel
    seed = int(opt.get("seed", 7))

    cfg = TreeConfig(lsystem=preset("apple"))
    cfg.seed = seed
    cfg.device = "cuda"
    cfg.deformable = True
    cfg.physics.model = StiffnessModel.BEAM
    cfg.physics.mj_solver = "cg"

    if scene == "skeleton":
        cfg.deformable = False
        cfg.physics.model = StiffnessModel.RIGID
    elif scene == "apples":
        cfg.fruit.enabled = True
        cfg.fruit.max_count = 40
    elif scene == "foliage":
        cfg.fruit.enabled = True
        cfg.fruit.max_count = 40
        cfg.foliage.set_density(float(opt.get("density", 0.6)))
    elif scene in ("break", "bend"):
        cfg.fruit.enabled = (scene == "break")
        cfg.breaking.enabled = (scene == "break")
        if scene == "bend":
            cfg.physics.youngs_modulus = 3.0e8   # green sapling: visible bend
    elif scene in ("robot", "terrain", "depth", "dr", "robotbreak"):
        cfg.fruit.enabled = True
        cfg.fruit.max_count = 40
        cfg.foliage.set_density(float(opt.get("density", 0.5)))
        cfg.robot.enabled = scene in ("robot", "terrain", "depth", "robotbreak")
        cfg.robot.camera = scene in ("robot", "terrain", "depth")
        if scene == "terrain":
            cfg.physics.terrain = True
            cfg.physics.terrain_amplitude = float(opt.get("amp", 0.08))
        if scene == "robot":
            cfg.robot.position = (1.15, 0.25)     # close in for a hero framing
        if scene == "robotbreak":
            cfg.breaking.enabled = True
            cfg.breaking.mode = "hinge"           # droop/hang (stays visible)
            cfg.breaking.rupture_stress = 2.0e7   # softer so a bump snaps a limb
            cfg.robot.position = (0.95, 0.15)     # right at the canopy edge
    return cfg


def render(scene, out, opt):
    import warp as wp
    import newton
    from treesim import builder
    from treesim.sim import Sim

    cfg = build_scene(scene, opt)
    W = int(opt.get("w", 1100)); H = int(opt.get("h", 950))
    tm = builder.generate_and_build(cfg, num_envs=1)
    collisions = bool(cfg.robot.enabled)
    sim = Sim(tm, solver="mujoco", fps=60, substeps=3,
              enable_breaking=cfg.breaking.enabled, collisions=collisions)
    viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    sim.set_viewer(viewer)
    _frame_camera(viewer, tm, sim, wp,
                  zoom=float(opt.get("zoom", 2.3)))

    robot_driver = wrist_cam = None
    if cfg.robot.enabled:
        try:
            from treesim import robot as _robot
            robot_driver = _robot.RobotDriver(sim, tm, cfg.robot)
            if cfg.robot.camera:
                wrist_cam = _robot.WristCamera(tm.model, viewer, tm, cfg.robot)
        except Exception as e:
            print("robot setup skipped:", e)

    # settle / act
    picker = None
    if scene == "depth" and wrist_cam is not None and robot_driver is not None:
        # drive the autonomous picker so the wrist camera actually faces fruit
        try:
            from treesim.picker import AutoPicker, ArmIK
            from treesim import robot as _robot
            wrist_cam.n_detect = 1
            picker = AutoPicker(sim, tm, wrist_cam, robot_driver, cfg.robot,
                                None, env=0, ik=ArmIK(list(_robot._ARM_HOME.values())))
        except Exception as e:
            print("picker setup skipped:", e)
    settle = int(opt.get("settle", 45))
    drive = int(opt.get("drive", 320)) if picker is not None else settle
    best_dets = 0
    for i in range(drive):
        if picker is not None:
            wrist_cam.update(sim.state_0)
            picker.update()
            # stop once the camera is looking at several confirmed apples, so
            # the captured depth frame actually contains detections to overlay
            nd = len(wrist_cam.dets_env[0] or [])
            best_dets = max(best_dets, nd)
            if i > 120 and nd >= max(2, int(opt.get("min_dets", 2))):
                print("captured at frame", i, "with", nd, "detections")
                break
        sim.step()
        if robot_driver is not None and picker is None:
            robot_driver.update(viewer)
        sim.render()
    if picker is not None:
        print("best dets during drive:", best_dets)

    if scene == "robotbreak":
        # snap a canopy branch on the robot's side, as if the base/arm shoved
        # through it; it recolours dead-brown and droops/falls (breaking on)
        q = sim.state_0.body_q.numpy()
        lo, hi = tm.skeleton.bounds()
        zmid = lo[2] + 0.42 * (hi[2] - lo[2])
        cx = 0.5 * (lo[0] + hi[0])
        cand = [b for b in range(tm.n_bodies)
                if q[b, 2] > zmid and q[b, 0] > cx + 0.12]
        if cand:
            tip = min(cand, key=lambda b: abs(q[b, 2] - (zmid + 0.25)))
            for _ in range(70):
                sim.set_external_force(tip, force=(-42.0, -8.0, -22.0))
                sim.step(); sim.render()
            for b in range(tm.n_bodies):
                sim.set_external_force(b, force=(0.0, 0.0, 0.0))
        for _ in range(150):
            sim.step(); sim.render()
        n_broken = int(sim.breaker.broken_count) if sim.breaker is not None else 0
        print("robotbreak: branches snapped =", n_broken)

    if scene == "bend":
        # push the canopy sideways, capture rest|bent
        rest = viewer.get_frame().numpy()
        q = sim.state_0.body_q.numpy()
        lo, hi = tm.skeleton.bounds()
        zmid = lo[2] + 0.5 * (hi[2] - lo[2])
        for b in np.where(q[:tm.n_bodies, 2] > zmid)[0]:
            sim.set_external_force(int(b), force=(7.0, 0.0, 0.0))
        for _ in range(110):
            sim.step(); sim.render()
        bent = viewer.get_frame().numpy()
        combo = np.concatenate([rest[:, :, :3], bent[:, :, :3]], axis=1)
        _save(combo, out); viewer.close(); return

    if scene == "break":
        # yank a mid-canopy branch hard enough to snap it
        q = sim.state_0.body_q.numpy()
        lo, hi = tm.skeleton.bounds()
        zmid = lo[2] + 0.55 * (hi[2] - lo[2])
        cand = np.where(q[:tm.n_bodies, 2] > zmid)[0]
        if len(cand):
            tip = int(cand[np.argmax(np.linalg.norm(
                q[cand, :2] - (lo[:2] + hi[:2]) / 2, axis=1))])
            for _ in range(90):
                sim.set_external_force(tip, force=(60.0, 20.0, -30.0))
                sim.step(); sim.render()
            for b in range(tm.n_bodies):
                sim.set_external_force(b, force=(0.0, 0.0, 0.0))
        for _ in range(120):
            sim.step(); sim.render()

    if scene == "depth" and wrist_cam is not None:
        for _ in range(int(opt.get("cam_iters", 10))):
            wrist_cam.update(sim.state_0)
            sim.step(); sim.render()
        try:
            import matplotlib.cm as cm
            from PIL import Image
            d = wrist_cam.depth_env[0]                      # (H, W) metres
            dets = wrist_cam.dets_env[0] or []
            rng = float(cfg.robot.camera_range)
            dm = np.clip(d / rng, 0.0, 1.0)
            rgb = (cm.turbo(1.0 - dm)[:, :, :3] * 255).astype(np.uint8)  # near=warm
            depth_only = rgb.copy()
            rgba = np.dstack([rgb, np.full(d.shape, 255, np.uint8)])
            if wrist_cam.percept is not None:
                rgba = wrist_cam.percept.draw_overlay(rgba, dets, d)
            up = lambda im: np.asarray(Image.fromarray(im[:, :, :3]).resize(
                (d.shape[1] * 3, d.shape[0] * 3), Image.NEAREST))
            a, b = up(depth_only), up(rgba)
            gap = np.full((a.shape[0], 14, 3), 255, np.uint8)
            print("depth dets:", len(dets))
            _save(np.concatenate([a, gap, b], axis=1), out)
        except Exception as e:
            print("depth capture failed:", e)
            _save(viewer.get_frame().numpy(), out)
        viewer.close(); return

    frame = viewer.get_frame().numpy()
    _save(frame, out)
    viewer.close()


def montage(out, paths, cols=None, labels=None):
    from PIL import Image, ImageDraw
    imgs = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
    if not imgs:
        print("no images to montage"); return
    n = len(imgs)
    cols = cols or n
    rows = (n + cols - 1) // cols
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    w = max(im.width for im in imgs)
    pad = 8
    W = cols * w + (cols + 1) * pad
    Hh = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, Hh), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    canvas.save(out)
    print("montage ->", out, canvas.size)


if __name__ == "__main__":
    scene = sys.argv[1]
    out = sys.argv[2]
    rest = sys.argv[3:]
    if scene == "montage":
        cols = None
        paths = [a for a in rest if not a.startswith("cols=")]
        for a in rest:
            if a.startswith("cols="):
                cols = int(a.split("=")[1])
        montage(out, paths, cols=cols)
    else:
        render(scene, out, _kv(rest))
