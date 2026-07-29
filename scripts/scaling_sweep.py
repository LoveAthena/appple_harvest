#!/usr/bin/env python
"""Measure physics step rate and peak GPU memory vs. #parallel environments.

Each (envs, solver) cell runs as an isolated grow_tree subprocess (physics only,
no rendering, no robot) so an out-of-memory failure at high env counts just marks
the ceiling instead of killing the sweep. Peak VRAM is polled from the driver
(one nvidia-smi thread per cell), which avoids the shell-subshell issues of an
inline sampler. Writes output/experiments/scaling_sweep.csv.

Usage: python scripts/scaling_sweep.py [--dr] [--max-envs N]
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Interpreter used to launch each run; defaults to the one running this script
# (i.e. your active env / `pixi run`). Override with NEWTON_PY if needed.
PY = os.environ.get("NEWTON_PY", sys.executable)
GROW = os.path.join(ROOT, "scripts", "grow_tree.py")
OUT = os.path.join(ROOT, "output", "experiments")


def peak_vram(stop, box):
    while not stop.is_set():
        try:
            o = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True, timeout=5)
            box[0] = max(box[0], int(o.strip().splitlines()[0]))
        except Exception:
            pass
        stop.wait(0.5)


def run_cell(envs, solver, dr, frames, timeout_s):
    mpath = os.path.join(OUT, f"_scale_{'dr' if dr else 'nd'}_{solver}_{envs}.json")
    argv = [PY, GROW, "--no-render", "--num-envs", str(envs), "--break",
            "--apples", "--apple-count", "40", "--foliage-density", "0.6",
            "--mj-solver", solver, "--frames", str(frames),
            "--progress-every", "0", "--metrics", mpath]
    if dr:
        argv.insert(3, "--randomize-envs")
    env = dict(os.environ, PYTHONPATH=ROOT)
    box = [0]; stop = threading.Event()
    t = threading.Thread(target=peak_vram, args=(stop, box), daemon=True); t.start()
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, out = -1, "TIMEOUT"
    stop.set(); wall = time.time() - t0
    fps = bodies = None
    oom = ("out of memory" in out.lower()) or ("cudaerrormemoryallocation" in out.lower())
    if rc == 0 and os.path.exists(mpath):
        try:
            fps = json.load(open(mpath))["summary"]["fps_mean"]
        except Exception:
            pass
    for ln in out.splitlines():
        if "total bodies=" in ln:
            try: bodies = int(ln.split("total bodies=")[1].split()[0])
            except Exception: pass
    if os.path.exists(mpath):
        os.remove(mpath)
    status = "ok" if (rc == 0 and fps) else ("oom" if oom else f"fail(rc={rc})")
    return dict(envs=envs, solver=solver, dr=int(dr), bodies=bodies,
                fps_mean=fps, env_fps=(round(fps * envs, 1) if fps else None),
                peak_vram_mb=box[0], wall_s=round(wall, 1), status=status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", action="store_true", help="also sweep domain-randomized (slow build)")
    ap.add_argument("--max-envs", type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "scaling_sweep.csv")
    cols = ["envs", "solver", "dr", "bodies", "fps_mean", "env_fps",
            "peak_vram_mb", "wall_s", "status"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    counts = [c for c in counts if c <= args.max_envs]
    configs = [("cg", False), ("newton", False)]
    if args.dr:
        configs.append(("cg", True))       # DR only up to where build is bearable
    for solver, dr in configs:
        for n in counts:
            if dr and n > 64:               # DR host build is O(N); stop early
                break
            frames = 120 if n <= 256 else (80 if n <= 1024 else 50)
            timeout_s = 200 if n <= 256 else (400 if n <= 1024 else 700)
            row = run_cell(n, solver, dr, frames, timeout_s)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            print(f"[{solver}{' DR' if dr else ''}] {n:5d} envs -> "
                  f"fps={row['fps_mean']} env_fps={row['env_fps']} "
                  f"vram={row['peak_vram_mb']}MiB {row['status']}", flush=True)
            if row["status"] != "ok":       # hit the ceiling for this solver
                print(f"  -> ceiling for {solver}{' DR' if dr else ''} at {n} envs")
                break
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
