#!/usr/bin/env python
"""DR-aware scaling sweep: physics step rate + peak VRAM vs #parallel envs, for
every combination of geometry-DR mode, constraint solver, and rendering.

Each cell is an isolated grow_tree subprocess so an OOM / timeout at high env
counts just marks that series' ceiling instead of killing the sweep.

Methodology notes (to keep cross-condition comparisons fair on a laptop GPU whose
clock boosts under sustained load):
  * conditions are INTERLEAVED at each env count -- at env n we run homog, shared
    and distinct (and both solvers) back-to-back, so they share thermal/clock
    state; we never run one whole condition then another.
  * a GPU warm-up run precedes the sweep, and each cell runs enough frames that
    the reported number is fps_MEDIAN over the steady-state window (the metrics
    deque keeps the last 240 frames, so early clock-ramp frames drop out).

Geometry modes: homog (no DR) | shared (--randomize-envs, dims shared across the
batch, the default) | distinct (--randomize-envs --distinct-geometry, per-env
dims -> heterogeneous geometry). Rendering: off (--no-render) or on (--viewer gl
--headless). Writes output/experiments/dr_scaling.csv.
"""
from __future__ import annotations
import csv, json, os, statistics, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def run_cell(envs, solver, geom, render, frames, warmup, timeout_s):
    tag = f"{geom}_{solver}_{'r' if render else 'n'}_{envs}"
    mpath = os.path.join(OUT, f"_dr_{tag}.json")
    argv = [PY, GROW, "--num-envs", str(envs), "--break", "--apples",
            "--apple-count", "40", "--foliage-density", "0.6",
            "--mj-solver", solver, "--frames", str(frames), "--warmup", str(warmup),
            "--progress-every", "0", "--metrics", mpath]
    if geom in ("shared", "distinct"):
        argv.append("--randomize-envs")
    if geom == "distinct":
        argv.append("--distinct-geometry")
    argv += (["--viewer", "gl", "--headless"] if render else ["--no-render"])

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
            summ = json.load(open(mpath))["summary"]
            fps = summ.get("fps_median") or summ.get("fps_mean")   # steady-state
        except Exception:
            pass
    for ln in out.splitlines():
        if "total bodies=" in ln:
            try: bodies = int(ln.split("total bodies=")[1].split()[0])
            except Exception: pass
    if os.path.exists(mpath):
        os.remove(mpath)
    status = "ok" if (rc == 0 and fps) else ("oom" if oom
                                             else ("timeout" if rc == -1 else f"fail(rc={rc})"))
    return dict(envs=envs, solver=solver, geom=geom, render=int(render), bodies=bodies,
                fps_median=fps, env_fps=(round(fps * envs, 1) if fps else None),
                peak_vram_mb=box[0], wall_s=round(wall, 1), status=status)


def measure(envs, solver, geom, render, frames, warmup, timeout_s, reps):
    """Run a cell ``reps`` times and keep the MEDIAN fps (averages out the laptop
    GPU's run-to-run clock variance). Fails fast if the first rep isn't ok."""
    rows = [run_cell(envs, solver, geom, render, frames, warmup, timeout_s)
            for _ in range(reps)]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return rows[0]
    fps = statistics.median(r["fps_median"] for r in ok)
    return dict(ok[0], fps_median=round(fps, 1), env_fps=round(fps * envs, 1),
                peak_vram_mb=max(r["peak_vram_mb"] for r in ok), status="ok")


def plan(geom, render, envs):
    """(frames, warmup, timeout_s). A per-cell warm-up at THIS config brings the
    GPU to steady clock; fps is then the median over the measured frames."""
    if geom == "distinct":
        f, w = 100, 40
    elif envs <= 64:
        f, w = 120, 60
    elif envs <= 256:
        f, w = 100, 40
    else:
        f, w = 80, 20
    to = 320 if envs <= 64 else (640 if envs <= 256 else 1000)
    if render:
        to += 150
    return f, w, to


# conditions interleaved at each env count; (geom, solver, render, [env counts])
FULL = [1, 4, 8, 16, 32, 64, 128, 256, 512]
NEWT = [1, 8, 32, 128, 256]
DIST = [1, 4, 8, 16, 32]
REND = [8, 32, 64, 128]
CONDITIONS = [
    ("homog",    "cg",     False, FULL),
    ("shared",   "cg",     False, FULL),
    ("distinct", "cg",     False, DIST),
    ("homog",    "newton", False, NEWT),
    ("shared",   "newton", False, NEWT),
    ("homog",    "cg",     True,  REND),
    ("shared",   "cg",     True,  REND),
    ("distinct", "cg",     True,  [8, 16]),
]
ENV_COUNTS = sorted({n for *_, counts in CONDITIONS for n in counts})


def main():
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "dr_scaling.csv")
    cols = ["envs", "solver", "geom", "render", "bodies", "fps_median", "env_fps",
            "peak_vram_mb", "wall_s", "status"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()

    print("[warmup] boosting GPU clocks ...", flush=True)
    run_cell(16, "cg", "homog", False, 150, 60, 200)   # discard: warm the GPU

    active = [True] * len(CONDITIONS)
    for n in ENV_COUNTS:                    # interleave conditions at each env count
        for i, cond in enumerate(CONDITIONS):
            geom, solver, render, counts = cond
            if not active[i] or n not in counts:
                continue
            frames, warmup, timeout_s = plan(geom, render, n)
            # median of 3 for the headline cg physics curves (kills clock spikes);
            # single rep for newton/render, which are read qualitatively.
            reps = 3 if (solver == "cg" and not render and n <= 256) else 1
            row = measure(n, solver, geom, render, frames, warmup, timeout_s, reps)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writerow(row)
            label = f"{geom}/{solver}/{'render' if render else 'phys'}"
            print(f"[{label:22s}] {n:4d} envs -> fps={row['fps_median']} "
                  f"env_fps={row['env_fps']} vram={row['peak_vram_mb']}MiB "
                  f"{row['status']}", flush=True)
            if row["status"] != "ok":
                active[i] = False
                print(f"  -> ceiling for {label} at {n} envs ({row['status']})")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
