#!/usr/bin/env python
"""Grow and simulate an L-system tree in NVIDIA Newton.

Examples
--------
    # Rigid tree, interactive OpenGL window
    python scripts/grow_tree.py --mode rigid --viewer gl

    # Deformable tree you can push around (drag with the mouse)
    python scripts/grow_tree.py --mode deformable --viewer gl

    # Deformable + breakable: yank a branch hard and it snaps off
    python scripts/grow_tree.py --mode deformable --break --viewer gl

    # Foliage + apples, recorded to a USD file for offline rendering
    python scripts/grow_tree.py --mode deformable --foliage --apples \
        --viewer usd --output output/tree.usda --frames 300

    # Headless benchmark / CI smoke test
    python scripts/grow_tree.py --viewer null --frames 60

    # RidgebackFranka you can drive around the orchard, wrist depth cam on
    python scripts/grow_tree.py --apples --break --robot --viewer gl

Interactive controls (OpenGL viewer): orbit = left-drag, pan = middle/right-drag,
zoom = scroll, apply force = grab a body and drag (see on-screen help / README),
space = pause, and the side panel exposes Newton's own options.
With --robot: W/S = drive, A/D = turn (the camera keeps arrows/Q/E/mouse); the
wrist depth camera renders into the draggable "wrist depth" image panel.
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prefer_nvidia_gl() -> None:
    """On Linux hybrid-graphics machines, render the OpenGL viewer on the NVIDIA
    GPU rather than the integrated one; otherwise the GL window falls back to slow
    CPU copies every frame. Only acts when an NVIDIA GPU is present and the user
    hasn't set these already, so it is a no-op on non-NVIDIA / non-Linux systems
    and never overrides an explicit choice. Must run before any GL context is
    created (i.e. before importing the viewer)."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("__NV_PRIME_RENDER_OFFLOAD") or \
       os.environ.get("__GLX_VENDOR_LIBRARY_NAME"):
        return
    import glob
    has_nvidia = bool(glob.glob("/dev/nvidia[0-9]*")) or \
        os.path.isdir("/proc/driver/nvidia/gpus")
    if not has_nvidia:
        import shutil, subprocess
        if shutil.which("nvidia-smi"):
            try:
                has_nvidia = subprocess.run(
                    ["nvidia-smi", "-L"], capture_output=True, timeout=5
                ).returncode == 0
            except Exception:
                has_nvidia = False
    if has_nvidia:
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        print("[grow_tree] NVIDIA GPU detected; preferring it for OpenGL "
              "rendering (PRIME offload). First run is slower while Warp compiles "
              "its CUDA kernels and the robot asset downloads; later runs are cached.",
              file=sys.stderr)


_prefer_nvidia_gl()

import warp as wp
import newton

from treesim.config import TreeConfig, preset, StiffnessModel
from treesim import builder
from treesim.sim import Sim


def make_config(args) -> TreeConfig:
    if args.auto:                            # autonomy needs a robot and fruit
        args.robot = True
        args.apples = True
    cfg = TreeConfig(lsystem=preset(args.preset))
    if args.depth > 0:                       # else keep the preset's default depth
        cfg.lsystem.n = args.depth
    cfg.lsystem.shape_jitter = args.jitter
    cfg.seed = args.seed
    cfg.device = args.device or "cuda"

    if args.mode == "rigid":
        cfg.deformable = False
        cfg.physics.model = StiffnessModel.RIGID
    else:
        cfg.deformable = True
        cfg.physics.model = StiffnessModel(args.stiffness)
    cfg.physics.dynamics_jitter = args.dyn_jitter

    cfg.breaking.enabled = args.brk
    cfg.breaking.mode = args.break_mode
    if args.rupture is not None:
        cfg.breaking.rupture_stress = args.rupture

    # foliage density dial: --foliage-density wins; else --foliage = medium; else off
    if args.foliage_density is not None:
        cfg.foliage.set_density(args.foliage_density)
    elif args.foliage:
        cfg.foliage.set_density(0.6)
    if args.leaves is not None:                 # explicit per-twig count overrides the dial
        cfg.foliage.leaves_per_terminal = args.leaves
    cfg.foliage.physics = args.foliage_physics

    cfg.fruit.enabled = args.apples
    cfg.fruit.max_count = args.apple_count

    cfg.physics.terrain = args.terrain
    cfg.physics.terrain_amplitude = args.terrain_amplitude
    cfg.physics.mj_solver = args.mj_solver

    cfg.robot.enabled = args.robot
    cfg.robot.camera = args.robot and not args.no_robot_camera

    return cfg


def make_viewer(args):
    import newton.viewer as V
    if args.viewer == "gl":
        return V.ViewerGL(headless=args.headless, paused=args.paused)
    if args.viewer == "rtx":
        return V.ViewerRTX(headless=args.headless, paused=args.paused, num_frames=args.frames)
    if args.viewer == "usd":
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        return V.ViewerUSD(output_path=args.output, num_frames=args.frames)
    if args.viewer == "null":
        return V.ViewerNull(num_frames=args.frames)
    raise ValueError(args.viewer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("tree")
    g.add_argument("--preset", default="apple", choices=["ta", "tb", "tc", "td", "apple"],
                   help="apple = stochastic apple tree (default); ta-td = ABoP ternary classes")
    g.add_argument("--depth", "-n", type=int, default=-1,
                   help="recursion depth (-1 = preset default; apple~4, ternary~5)")
    g.add_argument("--mode", default="deformable", choices=["rigid", "deformable"])
    g.add_argument("--stiffness", default="beam", choices=["beam", "rudimentary"],
                   help="compliant-joint stiffness model")
    g.add_argument("--jitter", type=float, default=0.0, help="Gaussian shape domain-randomisation sigma (paper: 0.1)")
    g.add_argument("--dyn-jitter", type=float, default=0.0, help="Gaussian dynamics randomisation sigma (paper: 1.0)")
    g.add_argument("--seed", type=int, default=-1, help="-1 = random each run (so every tree differs)")

    b = p.add_argument_group("breaking")
    b.add_argument("--break", dest="brk", action="store_true", help="enable branch snapping")
    b.add_argument("--break-mode", default="detach", choices=["detach", "hinge"])
    b.add_argument("--rupture", type=float, default=None, help="modulus of rupture [Pa] (lower => snaps easier)")

    fo = p.add_argument_group("foliage")
    fo.add_argument("--foliage", action="store_true",
                   help="add a medium canopy of leaves (rigid instanced cards on the branches; "
                        "massless, non-colliding, ~free). Shortcut for --foliage-density 0.6")
    fo.add_argument("--foliage-density", type=float, default=None,
                   help="leaf density: 0 = bare, ~1 = lush canopy, >1 = dense (leaves on inner "
                        "branches, obscures the interior). Purely visual - no physics/FPS cost.")
    fo.add_argument("--foliage-physics", action="store_true",
                   help="leaves flutter on their own bodies (EXPENSIVE: +1 body/leaf, ~5x slower)")
    fo.add_argument("--leaves", type=int, default=None,
                   help="explicit leaves-per-twig (overrides --foliage-density's leaf count)")

    ap = p.add_argument_group("fruit")
    ap.add_argument("--apples", action="store_true",
                    help="spawn apples on spurs; they detach when pulled hard enough")
    ap.add_argument("--apple-count", type=int, default=40,
                    help="how many apples (each is a free body and is the MAIN sim cost; "
                         "foliage and --break are nearly free). ~60 = lush/slow, ~20 = fast")

    ro = p.add_argument_group("robot")
    ro.add_argument("--robot", action="store_true",
                    help="add a RidgebackFranka mobile manipulator (IsaacLab's robot: "
                         "Clearpath Ridgeback base + Franka FR3 arm) to every env. "
                         "Drive it with W/S = forward/back, A/D = turn; the camera "
                         "keeps the ARROW keys, Q/E and the mouse.")
    ro.add_argument("--no-robot-camera", action="store_true",
                    help="disable the wrist depth camera (it costs a little render "
                         "time; physics is unaffected)")
    ro.add_argument("--auto", action="store_true",
                    help="autonomous fruit picking (implies --robot): the robot finds "
                         "apples with the wrist depth camera (sphere-fit detection), "
                         "drives to a stand-off, reaches with IK, grasps, pulls the "
                         "fruit off its stem and drops it in the bucket on its back. "
                         "Progress + metrics are printed and saved (see --metrics).")
    ro.add_argument("--metrics", default=None, metavar="FILE",
                    help="write run metrics JSON (fps, pick/place success, throughput, "
                         "forces, detection quality). Default with --auto: "
                         "output/metrics_<seed>.json")
    ro.add_argument("--gamepad", action="store_true",
                    help="record with Xbox controller (needs sudo wsl_recv.py). "
                         "Left stick=drive, Right stick=arm, A=gripper, "
                         "B=save, X=assist, Y=precision")

    t = p.add_argument_group("terrain")
    t.add_argument("--terrain", action="store_true",
                   help="bumpy outdoor ground (value-noise heightfield): gentle, "
                        "driveable, randomized per seed, flattened under the tree. "
                        "One global static shape, so multi-env batching is unaffected.")
    t.add_argument("--terrain-amplitude", type=float, default=0.05,
                   help="max bump height [m] (default 0.05; keep < ~0.08 or the "
                        "robot chassis visibly clips through crests)")

    r = p.add_argument_group("render/sim")
    r.add_argument("--viewer", default="gl", choices=["gl", "rtx", "usd", "null"],
                   help="gl = interactive window; null = NO rendering (physics-only)")
    r.add_argument("--num-envs", type=int, default=1,
                   help="parallel copies of the tree, batched on the GPU (IsaacLab-style). "
                        "Physics scales cheaply; rendering does not, so pair big counts with "
                        "--no-render")
    r.add_argument("--no-render", action="store_true",
                   help="disable rendering entirely (same as --viewer null); physics only")
    r.add_argument("--randomize-envs", action="store_true",
                   help="give each parallel env its own DIFFERENT apple tree (per-env "
                        "growth habit droop/lean/spread/twist, wood density, stiffness, "
                        "rupture strength, fruit size/colour/count, leaf size/colour). "
                        "Same topology AND, by default, the same branch DIMENSIONS across "
                        "the batch, so the worlds stay dimensionally homogeneous and physics "
                        "stays ~free (the shape still changes every run). Add "
                        "--distinct-geometry to also vary the dimensions per env. Only "
                        "cross-env render instancing is lost, so pair big counts with "
                        "--no-render.")
    r.add_argument("--dr-strength", type=float, default=1.0,
                   help="strength of per-env randomization (0 = identical, 1 = realistic "
                        "apple-tree spread, ~2 = exaggerated). Needs --randomize-envs.")
    r.add_argument("--distinct-geometry", action="store_true",
                   help="also randomize per-branch DIMENSIONS (segment length, thickness, "
                        "overall scale) PER ENV, not just growth habit. Visually maximal, "
                        "but distinct dimensions across the batch re-dimension every link "
                        "and drop the batched solve to ~8-13 fps (vs ~110 shared); reserve "
                        "it for the smaller batches used in evaluation. Needs "
                        "--randomize-envs.")
    r.add_argument("--headless", action="store_true")
    r.add_argument("--paused", action="store_true")
    r.add_argument("--output", default="output/tree.usda", help="USD output path (usd viewer)")
    r.add_argument("--frames", type=int, default=300, help="frames for usd/rtx/null viewers")
    r.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="step N physics frames before measuring (no metrics/render) "
                        "so the GPU reaches steady clock — for fair fps benchmarking")
    r.add_argument("--progress-every", type=int, default=0, metavar="N",
                   help="print a one-line heartbeat every N frames (frame/total, "
                        "fps, picks/places so far) — handy for long headless runs")
    r.add_argument("--solver", default="auto",
                   choices=["auto", "mujoco", "xpbd", "featherstone", "spring"],
                   help="mujoco (default, accurate); spring = experimental fast engine")
    r.add_argument("--substeps", type=int, default=3,
                   help="physics substeps/frame (lower = faster; 2-3 is plenty with MuJoCo)")
    r.add_argument("--mj-solver", default="cg", choices=["cg", "newton"],
                   help="mjwarp constraint-solver algorithm: cg (default, ~7x faster on "
                        "this workload, physics-identical on the regression matrix) or "
                        "newton (mjwarp's default algorithm, kept as a fallback)")
    r.add_argument("--collisions", action="store_true",
                   help="enable the contact pipeline (VERY slow on a full tree; the soft "
                        "ground already lands falling debris, so usually leave this OFF)")
    r.add_argument("--device", default=None)
    r.add_argument("--max-bodies", type=int, default=60000)
    return p.parse_args()


def main():
    import random
    args = parse_args()
    if args.seed < 0:
        args.seed = random.randrange(2**31)      # different tree every run
    if args.no_render:
        args.viewer = "null"
    args.num_envs = max(int(args.num_envs), 1)
    if args.device:
        wp.set_device(args.device)

    cfg = make_config(args)

    print(f"[grow_tree] building {args.mode} {args.preset} tree (seed={args.seed}, "
          f"n={cfg.lsystem.n}, num_envs={args.num_envs}"
          f"{', randomized' if (args.randomize_envs and args.num_envs > 1) else ''}) ...")
    tm = builder.generate_and_build(cfg, max_bodies=args.max_bodies, num_envs=args.num_envs,
                                    randomize_envs=args.randomize_envs,
                                    dr_strength=args.dr_strength,
                                    randomize_geometry=args.distinct_geometry)
    print(f"[grow_tree] envs={tm.num_envs} total bodies={tm.model.body_count} "
          f"joints={tm.model.joint_count} dof={tm.model.joint_dof_count} "
          f"apples={len(tm.apple_bodies)}")

    # solver / collision policy.  MuJoCo gives correct implicit torsional springs
    # and is the robust default; "spring" (XPBD + custom kernel) is experimental
    # (fast but currently unstable on large articulations).
    solver = args.solver
    if solver == "auto":
        solver = "mujoco"
    collisions = args.collisions
    if args.robot and not collisions:
        # the robot should push through the canopy, not ghost through it.
        # Collision groups keep this affordable: tree/apples are -1 (never
        # self-colliding), leaves 0 (never colliding), so the broad phase only
        # sees wood/apples vs the robot and the ground.
        collisions = True
        print("[grow_tree] --robot: enabling robot-vs-tree collisions")

    if collisions and not args.robot and tm.model.body_count > 150:
        print(f"[grow_tree] note: --collisions is rarely needed without a robot — the "
              f"soft ground already lands falling debris (collision groups keep the "
              f"cost moderate either way).")
    sim = Sim(tm, solver=solver, fps=60, substeps=args.substeps,
              enable_breaking=args.brk, collisions=collisions)
    print(f"[grow_tree] solver={solver} substeps={args.substeps} collisions={collisions}")

    viewer = make_viewer(args)
    sim.set_viewer(viewer)          # sets the model + computes per-world render offsets

    # frame the tree (or the whole grid of trees) nicely.  Parallel worlds overlap
    # in physics space; the viewer spreads them onto a grid via world_offsets, so the
    # ACTUAL rendered layout is body_q + offset[world] — frame that, not body_q alone.
    h = tm.skeleton.height()
    try:
        if tm.num_envs > 1:
            import math, numpy as np
            bp = sim.state_0.body_q.numpy()[:, :3]
            bpe = tm.model.body_count // tm.num_envs
            wo = getattr(viewer, "world_offsets", None)
            wo = wo.numpy() if wo is not None else np.zeros((tm.num_envs, 3))
            pts = np.concatenate([bp[e * bpe:(e + 1) * bpe] + wo[e]
                                  for e in range(tm.num_envs)])
            lo2, hi2 = pts.min(0), pts.max(0)
            ctr = (lo2 + hi2) / 2.0
            ext = max(float(hi2[0] - lo2[0]), float(hi2[1] - lo2[1]), 1.0)
            dist = 0.95 * ext
            pos = np.array([ctr[0] + 0.5 * dist, ctr[1] - dist, ctr[2] + 0.75 * ext])
            look = np.array([ctr[0], ctr[1], ctr[2] + 0.15 * h])
            d = look - pos; d /= (np.linalg.norm(d) + 1e-9)
            viewer.set_camera(pos=wp.vec3(*map(float, pos)),
                              pitch=float(math.degrees(math.asin(d[2]))),
                              yaw=float(math.degrees(math.atan2(d[1], d[0]))))
        else:
            viewer.set_camera(pos=wp.vec3(2.2 * h, 2.2 * h, 1.1 * h),
                              pitch=-15.0, yaw=135.0)
    except Exception:
        pass

    # RidgebackFranka teleop + wrist depth camera + autonomy + metrics
    robot_driver = None
    wrist_cam = None
    pickers = []
    metrics_envs = []
    metrics = None
    metrics_path = args.metrics or (f"output/metrics_{args.seed}.json" if args.auto else None)
    if metrics_path:
        from treesim.metrics import Metrics
        metrics = Metrics(metrics_path)
    if tm.robot_data is not None:
        from treesim import robot as _robot
        robot_driver = _robot.RobotDriver(sim, tm, cfg.robot)
        if not args.auto and _robot.take_over_wasd(viewer):
            print("[robot] W/S = drive, A/D = turn.  Camera: arrows / Q / E / mouse.")
        if cfg.robot.camera and (args.viewer == "gl" or args.auto):
            try:
                wrist_cam = _robot.WristCamera(tm.model, viewer, tm, cfg.robot)
                print("[robot] wrist depth camera on - see the 'wrist depth' and "
                      "'fruit detection' image panels.")
            except Exception as e:
                print(f"[robot] wrist camera unavailable ({e})")
        if args.auto:
            if wrist_cam is None:
                raise SystemExit("--auto needs the wrist camera (do not pass "
                                 "--no-robot-camera)")
            from treesim.picker import AutoPicker, ArmIK
            from treesim.metrics import Metrics
            # SEPARATE autonomy in EVERY env: one picker per world, each with
            # its own perception/metrics; only env 0 draws viewer overlays.
            n_auto = tm.num_envs
            wrist_cam.n_detect = n_auto
            shared_ik = ArmIK(list(_robot._ARM_HOME.values()))
            metrics_envs = ([metrics] if n_auto == 1 and metrics is not None
                            else [Metrics(None) for _ in range(n_auto)])
            pickers = [AutoPicker(sim, tm, wrist_cam, robot_driver, cfg.robot,
                                  metrics_envs[e] if metrics_envs else None,
                                  env=e, ik=shared_ik)
                       for e in range(n_auto)]
            print(f"[auto] autonomous picking on ({n_auto} env"
                  f"{'s' if n_auto > 1 else ''}): SCAN -> ALIGN -> REACH -> "
                  "GRASP -> PULL -> place in the bucket.")

    recorder = None
    if args.gamepad:
        if tm.robot_data is None:
            raise SystemExit("--gamepad needs --robot")
        from treesim.picker_human import GamepadPicker
        recorder = GamepadPicker(sim, tm, wrist_cam, robot_driver, cfg.robot)
        print("[gamepad] Left stick=drive, Right stick=arm, "
              "A=gripper, B=save, X=assist, Y=precision")

    print("[grow_tree] running. Close the window (or Ctrl-C) to stop.")
    frame = 0
    last_state = None
    # null/usd/rtx viewers stop themselves after num_frames; a HEADLESS GL
    # viewer has no window to close, so honour --frames as a hard cap there too
    # (else a headless benchmark render loops forever).
    total_frames = (args.frames if (args.viewer in ("null", "usd", "rtx")
                                     or args.headless) else 0)
    hb = max(int(args.progress_every), 0)
    import time as _time
    _t_hb = _time.time()

    def _heartbeat():
        nonlocal _t_hb
        now = _time.time()
        dt = now - _t_hb
        _t_hb = now
        fps = hb / dt if dt > 0 else 0.0
        tag = f"{frame}/{total_frames}" if total_frames else str(frame)
        extra = ""
        mets = metrics_envs if pickers else ([metrics] if metrics else [])
        if mets and pickers:
            att = sum(len(m.picks) + (1 if m._open_pick else 0) for m in mets)
            pl = sum(sum(p["placed"] for p in m.picks) for m in mets)
            extra = f" picks={att} placed={pl} state[0]={pickers[0].state.lower()}"
        pct = f" ({100.0 * frame / total_frames:.0f}%)" if total_frames else ""
        print(f"[progress] frame {tag}{pct}  {fps:.1f} fps{extra}", flush=True)

    # optional warm-up: step physics only (no metrics, no render, so the viewer's
    # frame budget is untouched) until the GPU reaches steady clock, keeping fps
    # measurements fair across cells with different build times.
    for _ in range(max(int(args.warmup), 0)):
        sim.step()

    try:
        while viewer.is_running() and not (total_frames and frame >= total_frames):
            if viewer.should_step():
                sim.step()
                frame += 1
                if hb and frame % hb == 0:
                    _heartbeat()
                if pickers:
                    for e, m in enumerate(metrics_envs):
                        m.frame(wall=(e == 0))
                    wrist_cam.update(sim.state_0)
                    for p in pickers:
                        p.update()
                    if pickers[0].state != last_state:
                        print(f"[auto] {pickers[0].state.lower()}")
                        last_state = pickers[0].state
                else:
                    if metrics is not None:
                        metrics.frame()
                    if robot_driver is not None:
                        robot_driver.update(viewer)
                if recorder is not None:
                    recorder.update(viewer)
            if not pickers and wrist_cam is not None:
                wrist_cam.update(sim.state_0)
            sim.render()
    except KeyboardInterrupt:
        pass
    viewer.close()

    if recorder is not None:
        recorder.close()

    import json as _json
    if pickers and len(metrics_envs) > 1:
        from treesim.metrics import save_combined
        combined = save_combined(metrics_path, metrics_envs, sim)
        print("[metrics] " + _json.dumps(combined, indent=2))
        if metrics_path:
            print(f"[metrics] saved to {metrics_path}")
    elif metrics is not None:
        metrics.save(sim)
        print("[metrics] " + _json.dumps(metrics.summary(sim), indent=2))
        if metrics.path:
            print(f"[metrics] saved to {metrics.path}")

    if sim.breaker is not None:
        print(f"[grow_tree] branches snapped: {sim.breaker.broken_count}")
    if sim.apples is not None:
        print(f"[grow_tree] apples detached: {sim.apples.broken_count}")

    print(f"[grow_tree] done ({frame} frames).")


if __name__ == "__main__":
    main()
