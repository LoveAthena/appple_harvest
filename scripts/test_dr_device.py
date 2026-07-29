#!/usr/bin/env python
"""Validate the on-device (GPU) domain randomization against the host path.

Checks: (1) strength=0 reproduces the base model element-wise; (2) strength=1
builds stable, diverse trees; (3) build time no longer scales with #envs;
(4) per-step cost matches homogeneous batches.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import warp as wp
import newton

from treesim.config import TreeConfig, preset, StiffnessModel
from treesim import builder
from treesim.sim import Sim


def cfg(seed):
    c = TreeConfig(lsystem=preset("apple"))
    c.seed = seed
    c.deformable = True
    c.physics.model = StiffnessModel.BEAM
    c.physics.mj_solver = "cg"
    c.fruit.enabled = False
    c.foliage.enabled = False
    c.breaking.enabled = False
    return c


def build(dr_device, n, strength, seed=7):
    if dr_device:
        os.environ["TREESIM_DR_DEVICE"] = "1"
    else:
        os.environ.pop("TREESIM_DR_DEVICE", None)
    t0 = time.time()
    # the device kernel builds DISTINCT per-world geometry, so this test runs the
    # distinct-geometry path on both sides (randomize_geometry=True); the default
    # shared-geometry path needs no per-world kernel.
    tm = builder.generate_and_build(cfg(seed), num_envs=n, randomize_envs=(strength > 0),
                                    dr_strength=strength, randomize_geometry=True)
    return tm, time.time() - t0


def test_identity():
    print("\n=== TEST 1: strength=0 identity (device vs replicated base) ===")
    tm_dev, _ = build(True, 4, 0.0)
    os.environ.pop("TREESIM_DR_DEVICE", None)
    tm_hom, _ = build(False, 4, 0.0)          # homogeneous replicate (no patch)
    ok = True
    for name in ("joint_X_p", "shape_scale", "body_mass", "body_inertia",
                 "joint_target_ke", "body_com", "shape_transform"):
        a = getattr(tm_dev.model, name).numpy().astype(np.float64)
        b = getattr(tm_hom.model, name).numpy().astype(np.float64)
        md = np.abs(a - b).max()
        flag = "OK" if md < 1e-3 else "**DIFF**"
        if md >= 1e-3:
            ok = False
        print(f"  {name:20s} max|dev-base| = {md:.2e}  {flag}")
    print("  identity:", "PASS" if ok else "FAIL")
    return ok


def test_stable_diverse():
    print("\n=== TEST 2: strength=1 stability + diversity (8 envs, 90 steps) ===")
    tm, _ = build(True, 8, 1.0)
    sim = Sim(tm, solver="mujoco", fps=60, substeps=3, collisions=False)
    v = newton.viewer.ViewerNull(num_frames=90)
    sim.set_viewer(v)
    q0 = sim.state_0.body_q.numpy()[:, :3].copy()
    nan = False
    for _ in range(90):
        sim.step()
        sim.render()
    q1 = sim.state_0.body_q.numpy()
    nan = not np.isfinite(q1).all()
    bpe = tm.model.body_count // tm.num_envs
    # rest drift: how far bodies moved over 90 steps (should be small, stable)
    drift = np.linalg.norm(q1[:, :3] - q0, axis=1)
    # cross-env diversity: variance of a mid-canopy body position across envs
    mids = np.array([q1[e * bpe + bpe // 2, :3] for e in range(tm.num_envs)])
    div = float(mids.std(axis=0).mean())
    print(f"  finite (no NaN): {not nan}")
    print(f"  rest drift over 90 steps: mean {drift.mean()*100:.2f} cm, "
          f"max {drift.max()*100:.2f} cm")
    print(f"  cross-env position spread: {div*100:.1f} cm (diverse if > ~5 cm)")
    ok = (not nan) and drift.mean() < 0.05 and div > 0.02
    print("  stability+diversity:", "PASS" if ok else "FAIL")
    return ok


def test_build_scaling():
    print("\n=== TEST 3: build time vs #envs (device vs host) ===")
    print(f"  {'envs':>5} {'host_s':>9} {'device_s':>9} {'speedup':>8}")
    for n in (16, 64, 256):
        try:
            _, th = build(False, n, 1.0)
        except Exception as e:
            th = float("nan"); print("   host fail:", e)
        _, td = build(True, n, 1.0)
        sp = th / td if td > 0 else float("nan")
        print(f"  {n:5d} {th:9.1f} {td:9.1f} {sp:7.1f}x")


def test_step_cost():
    print("\n=== TEST 4: per-step cost, device-DR vs homogeneous (64 envs) ===")
    for label, dev, strength in (("homogeneous", False, 0.0),
                                 ("device-DR  ", True, 1.0)):
        tm, _ = build(dev, 64, strength)
        sim = Sim(tm, solver="mujoco", fps=60, substeps=2, collisions=False)
        v = newton.viewer.ViewerNull(num_frames=120)
        sim.set_viewer(v)
        for _ in range(20):
            sim.step(); sim.render()          # warmup + graph capture
        t0 = time.time()
        for _ in range(100):
            sim.step(); sim.render()
        dt = time.time() - t0
        print(f"  {label}: {100/dt:.1f} fps  ({64*100/dt:.0f} env-steps/s)")


if __name__ == "__main__":
    wp.init()
    a = test_identity()
    b = test_stable_diverse()
    test_build_scaling()
    test_step_cost()
    print("\nSUMMARY:", "ALL CORE CHECKS PASS" if (a and b) else "SOME CHECKS FAILED")
