#!/usr/bin/env python
"""Sequential benchmark-experiment runner for the Newton apple-orchard sim.

Runs a matrix of ``grow_tree.py`` jobs ONE AT A TIME, saving results as it
goes so a crash (or Ctrl-C) never loses more than the in-flight job:

* every job writes its own metrics JSON under ``output/experiments/<exp>/``;
* a tidy row is appended to ``output/experiments/results.csv`` after each job;
* ``output/experiments/STATUS.txt`` is a live human dashboard (progress + ETA);
* ``output/experiments/progress.log`` is an append-only, ``tail -f``-friendly log;
* GPU memory is polled per job (peak MiB) so the 8 GB-laptop story is documented.

Watch it live in another terminal with either of::

    tail -f  output/experiments/progress.log
    watch -n2 cat output/experiments/STATUS.txt

Resume after an interruption by just re-running: finished jobs (whose metrics
JSON already exists) are skipped unless ``--force`` is given.

Usage::

    python scripts/run_experiments.py                 # full suite (~3-4.5 h)
    python scripts/run_experiments.py --profile quick  # fast subset (~1-1.5 h)
    python scripts/run_experiments.py --list           # print the matrix + est.
    python scripts/run_experiments.py --dry-run        # don't actually run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Interpreter used to launch each run; defaults to the one running this script
# (i.e. your active env / `pixi run`). Override with NEWTON_PY if needed.
PY = os.environ.get("NEWTON_PY", sys.executable)
GROW = os.path.join(ROOT, "scripts", "grow_tree.py")
OUTDIR = os.path.join(ROOT, "output", "experiments")

# fixed across every picking condition so the SAME domain-randomised trees are
# reused: each sweep is then a clean paired ablation (only the swept knob
# changes, the orchard is identical).
BASE_SEED = 20250704


# --------------------------------------------------------------------------- #
# Job matrix
# --------------------------------------------------------------------------- #
def _job(exp, cond, *, auto, envs, frames, params, extra, viewer="null"):
    """One experiment cell.  ``params`` are recorded verbatim as CSV columns so
    the analyser knows which knob this row varied."""
    return dict(exp=exp, cond=cond, auto=auto, envs=envs, frames=frames,
                params=params, extra=list(extra), viewer=viewer)


def build_jobs(profile: str) -> list:
    quick = profile == "quick"
    jobs: list = []

    # ---- E1: throughput / scaling (cheap, physics only, no picking) --------
    sfr = 250 if quick else 300
    env_scale = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16, 32]
    for n in env_scale:                                    # E1a physics scaling
        jobs.append(_job(
            "E1_scaling", f"physics_envs{n:02d}", auto=False, envs=n, frames=sfr,
            params=dict(kind="physics", envs=n, render=0, dr=1, apples=40),
            extra=["--randomize-envs", "--break", "--apples", "--apple-count", "40",
                   "--foliage-density", "0.6"]))
    for n in ([1, 4] if quick else [1, 2, 4, 8]):          # E1b render cost (no DR = instanced)
        for render, vw in [(0, "null"), (1, "gl")]:
            extra = ["--break", "--apples", "--apple-count", "40",
                     "--foliage-density", "0.6"]
            if vw == "gl":
                extra += ["--headless"]
            jobs.append(_job(
                "E1_scaling", f"render{render}_envs{n:02d}", auto=False, envs=n,
                frames=sfr, viewer=vw,
                params=dict(kind="render", envs=n, render=render, dr=0, apples=40),
                extra=extra))
    if not quick:
        for algo in ["cg", "newton"]:                      # E1c solver algorithm
            jobs.append(_job(
                "E1_scaling", f"solver_{algo}_envs04", auto=False, envs=4, frames=sfr,
                params=dict(kind="solver", envs=4, solver=algo, apples=40),
                extra=["--randomize-envs", "--break", "--apples", "--apple-count",
                       "40", "--foliage-density", "0.6", "--mj-solver", algo]))
        for na in [20, 40, 60, 80]:                        # E1d fruit cost (fps vs apples)
            jobs.append(_job(
                "E1_scaling", f"applefps_{na:02d}", auto=False, envs=1, frames=400,
                params=dict(kind="applefps", envs=1, apples=na),
                extra=["--break", "--apples", "--apple-count", str(na),
                       "--foliage-density", "0.6"]))

    # ---- picking experiments (autonomous, 6 parallel DR trees per cell) -----
    P_ENVS = 4 if quick else 6
    P_FRAMES = 2400 if quick else 4500          # 40 s / 75 s of sim per env
    common = ["--randomize-envs", "--break", "--seed", str(BASE_SEED)]

    def pick_job(exp, cond, params, extra):
        return _job(exp, cond, auto=True, envs=P_ENVS, frames=P_FRAMES,
                    params=dict(params, envs=P_ENVS, seed=BASE_SEED),
                    extra=common + extra)

    # E2 foliage sweep (0.6 = the shared baseline, reused by E3/E4/E5)
    fol_levels = [0.0, 0.6, 1.2] if quick else [0.0, 0.3, 0.6, 1.0, 1.5]
    for f in fol_levels:
        cond = "baseline" if abs(f - 0.6) < 1e-9 else f"foliage_{f:.1f}"
        exp = "E5_baseline" if cond == "baseline" else "E2_foliage"
        jobs.append(pick_job(
            exp, cond,
            dict(foliage=f, apples=40, terrain=0.0),
            ["--foliage-density", str(f), "--apple-count", "40"]))

    # E3 apple-count sweep (40 = baseline)
    for na in ([20, 60] if quick else [20, 60]):
        jobs.append(pick_job(
            "E3_apples", f"apples_{na}",
            dict(foliage=0.6, apples=na, terrain=0.0),
            ["--foliage-density", "0.6", "--apple-count", str(na)]))

    # E4 terrain sweep (none = baseline).  20 cm = the realistic orchard-alley
    # roughness ceiling the user flagged; the base's z-servo tracks the ground.
    terr_levels = [0.20] if quick else [0.05, 0.10, 0.20]
    for a in terr_levels:
        jobs.append(pick_job(
            "E4_terrain", f"terrain_{int(a * 100):02d}cm",
            dict(foliage=0.6, apples=40, terrain=a),
            ["--foliage-density", "0.6", "--apple-count", "40",
             "--terrain", "--terrain-amplitude", str(a)]))

    # E5 tree/seed variation + run-to-run repeatability (extra seeds, same knobs)
    seeds = [] if quick else [BASE_SEED + 1, BASE_SEED + 2]
    for s in seeds:
        jobs.append(_job(
            "E5_variation", f"seed_{s}", auto=True, envs=P_ENVS, frames=P_FRAMES,
            params=dict(foliage=0.6, apples=40, terrain=0.0, seed=s, envs=P_ENVS),
            extra=["--randomize-envs", "--break", "--seed", str(s),
                   "--foliage-density", "0.6", "--apple-count", "40"]))
    return jobs


# --------------------------------------------------------------------------- #
# GPU memory sampler
# --------------------------------------------------------------------------- #
class VramSampler(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak = 0
        self.cur = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                self.cur = int(out.strip().splitlines()[0])
                self.peak = max(self.peak, self.cur)
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Status / logging
# --------------------------------------------------------------------------- #
CSV_COLS = ["exp", "cond", "kind", "auto", "envs", "frames", "seed", "foliage",
            "apples", "terrain", "solver", "render", "wall_s", "fps_mean",
            "fps_median", "vram_peak_mb", "picks_attempted", "grasp_success",
            "pick_success", "place_success", "total_success_rate",
            "throughput_fruit_per_min", "mean_cycle_time_s", "max_pull_force_N",
            "detection_precision", "detection_recall_visible", "branches_snapped",
            "apples_detached_total", "metrics_path", "status"]


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def log_line(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUTDIR, "progress.log"), "a") as f:
        f.write(line + "\n")


def write_status(state: dict):
    def fmt_dt(s):
        s = int(s)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

    done, total = state["done"], state["total"]
    bar_n = 30
    filled = int(bar_n * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_n - filled)
    lines = [
        "Newton apple-orchard benchmark — experiment run",
        "=" * 54,
        f"profile : {state['profile']}",
        f"progress: [{bar}] {done}/{total} jobs  ({100 * done / total:.0f}%)",
        f"elapsed : {fmt_dt(state['elapsed'])}",
        f"ETA     : {fmt_dt(state['eta']) if state['eta'] is not None else '—'}"
        f"   (est. finish {state['finish'] if state['finish'] else '—'})",
        "",
        f"current : {state['current']}",
        f"  since : {fmt_dt(state['job_elapsed'])}    GPU: {state['vram']} MiB"
        f"  (peak {state['vram_peak']})",
        f"  last  : {state['last_child']}",
        "",
        "recent jobs:",
    ]
    for r in state["recent"][-8:]:
        lines.append(f"  {r}")
    with open(os.path.join(OUTDIR, "STATUS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Run one job
# --------------------------------------------------------------------------- #
def run_job(job, state):
    exp_dir = os.path.join(OUTDIR, job["exp"])
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "logs"), exist_ok=True)
    metrics_path = os.path.join(exp_dir, job["cond"] + ".json")
    log_path = os.path.join(OUTDIR, "logs", f"{job['exp']}__{job['cond']}.log")

    argv = [PY, GROW, "--frames", str(job["frames"]),
            "--num-envs", str(job["envs"]),
            "--progress-every", "150" if job["auto"] else "100",
            "--metrics", metrics_path]
    if job["auto"]:
        argv += ["--auto", "--no-render"]
    else:
        if job["viewer"] == "null":
            argv += ["--no-render"]
        else:
            argv += ["--viewer", job["viewer"]]
    argv += job["extra"]

    env = dict(os.environ, PYTHONPATH=ROOT)
    t0 = time.time()
    vram = VramSampler()
    vram.start()
    last_child = ""
    # safety net: kill a job that overruns (a hung viewer, a solver stall).
    # Auto picking jobs get a generous budget; short physics jobs a tight one.
    timeout_s = 3000 if job["auto"] else 900
    with open(log_path, "w") as logf:
        logf.write("CMD: " + " ".join(argv) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)

        def _watchdog():
            while proc.poll() is None:
                if time.time() - t0 > timeout_s:
                    logf.write(f"\n[runner] TIMEOUT after {timeout_s}s -> killed\n")
                    logf.flush()
                    proc.kill()
                    return
                time.sleep(5)
        threading.Thread(target=_watchdog, daemon=True).start()

        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            line = line.rstrip()
            if line.startswith("[progress]") or line.startswith("[auto]") \
                    or "error" in line.lower() or "traceback" in line.lower():
                last_child = line
                state["last_child"] = line
                state["job_elapsed"] = time.time() - t0
                state["vram"] = vram.cur
                state["vram_peak"] = vram.peak
                write_status(state)
        proc.wait()
    vram.stop()
    wall = time.time() - t0
    rc = proc.returncode

    row = dict(exp=job["exp"], cond=job["cond"], auto=int(job["auto"]),
               envs=job["envs"], frames=job["frames"], wall_s=round(wall, 1),
               vram_peak_mb=vram.peak, metrics_path=os.path.relpath(metrics_path, ROOT),
               status="ok" if rc == 0 else f"exit{rc}")
    for k in ("kind", "seed", "foliage", "apples", "terrain", "solver", "render"):
        if k in job["params"]:
            row[k] = job["params"][k]

    # pull the headline numbers out of the metrics JSON the child just wrote
    if rc == 0 and os.path.exists(metrics_path):
        try:
            data = json.load(open(metrics_path))
            s = data.get("summary", {})
            for k in ("fps_mean", "fps_median", "picks_attempted", "grasp_success",
                      "pick_success", "place_success", "total_success_rate",
                      "throughput_fruit_per_min", "mean_cycle_time_s",
                      "max_pull_force_N", "detection_precision",
                      "detection_recall_visible", "branches_snapped",
                      "apples_detached_total"):
                if s.get(k) is not None:
                    row[k] = s[k]
        except Exception as e:
            row["status"] = f"ok(parse_err:{e})"
    append_csv(os.path.join(OUTDIR, "results.csv"), row)
    return row, rc


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--force", action="store_true",
                    help="re-run jobs even if their metrics JSON already exists")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="set everything up but do not launch the jobs")
    ap.add_argument("--only", default=None,
                    help="run only jobs whose exp name contains this substring")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    jobs = build_jobs(args.profile)
    if args.only:
        jobs = [j for j in jobs if args.only in j["exp"]]

    # weight ~ compute cost, for an adaptive ETA (frames x envs, x2 if rendering)
    for j in jobs:
        w = j["frames"] * max(j["envs"], 1)
        if j["viewer"] == "gl":
            w *= 2
        j["_weight"] = w
    total_weight = sum(j["_weight"] for j in jobs)

    if args.list or args.dry_run:
        print(f"profile={args.profile}  {len(jobs)} jobs")
        for j in jobs:
            print(f"  {j['exp']:14s} {j['cond']:20s} auto={int(j['auto'])} "
                  f"envs={j['envs']:2d} frames={j['frames']} {' '.join(j['extra'])}")
        # rough estimate: picking ~0.45 env-sim-s/wall-s; physics jobs measured light
        est = 0.0
        for j in jobs:
            if j["auto"]:
                est += (j["envs"] * j["frames"] / 60.0) / 0.45 + 15
            else:
                est += j["frames"] / (20.0 if j["viewer"] == "gl" else 45.0) + 15
        print(f"\nrough total estimate: {est / 3600:.1f} h "
              f"({len(jobs)} jobs, sequential)")
        if args.list:
            return

    manifest = dict(profile=args.profile, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    n_jobs=len(jobs),
                    jobs=[dict(exp=j["exp"], cond=j["cond"], params=j["params"])
                          for j in jobs])
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=2)

    log_line(f"=== starting {args.profile} suite: {len(jobs)} jobs ===")
    t_start = time.time()
    weight_done = 0.0
    state = dict(profile=args.profile, total=len(jobs), done=0, current="(starting)",
                 elapsed=0, eta=None, finish=None, job_elapsed=0, vram=0,
                 vram_peak=0, last_child="", recent=[])
    write_status(state)

    for i, j in enumerate(jobs):
        metrics_path = os.path.join(OUTDIR, j["exp"], j["cond"] + ".json")
        state["current"] = f"[{i + 1}/{len(jobs)}] {j['exp']}/{j['cond']}"
        state["elapsed"] = time.time() - t_start
        if os.path.exists(metrics_path) and not args.force:
            log_line(f"SKIP (exists) {j['exp']}/{j['cond']}")
            state["done"] += 1
            weight_done += j["_weight"]
            state["recent"].append(f"SKIP {j['exp']}/{j['cond']}")
            write_status(state)
            continue
        if args.dry_run:
            log_line(f"DRY  {j['exp']}/{j['cond']}  {' '.join(j['extra'])}")
            state["done"] += 1
            continue

        log_line(f"RUN  [{i + 1}/{len(jobs)}] {j['exp']}/{j['cond']}  "
                 f"(auto={int(j['auto'])} envs={j['envs']} frames={j['frames']})")
        row, rc = run_job(j, state)

        weight_done += j["_weight"]
        state["done"] += 1
        state["elapsed"] = time.time() - t_start
        rate = state["elapsed"] / max(weight_done, 1)      # sec per weight-unit
        remaining = total_weight - weight_done
        state["eta"] = rate * remaining
        state["finish"] = time.strftime(
            "%H:%M", time.localtime(time.time() + state["eta"]))
        # compact result note for the dashboard
        if j["auto"]:
            note = (f"OK {j['exp']}/{j['cond']}  {row.get('wall_s')}s  "
                    f"place={row.get('place_success', '?')}/"
                    f"{row.get('picks_attempted', '?')}  "
                    f"fps={row.get('fps_mean', '?')}  "
                    f"det.P={row.get('detection_precision', '?')}")
        else:
            note = (f"OK {j['exp']}/{j['cond']}  {row.get('wall_s')}s  "
                    f"fps={row.get('fps_mean', '?')}  "
                    f"vram={row.get('vram_peak_mb')}MiB")
        if rc != 0:
            note = "FAIL " + note[3:] + f"  (rc={rc}, see logs)"
        log_line(note)
        state["recent"].append(note)
        write_status(state)

    state["elapsed"] = time.time() - t_start
    state["current"] = "(all done)"
    state["eta"] = 0
    write_status(state)
    log_line(f"=== suite complete: {len(jobs)} jobs in "
             f"{state['elapsed'] / 3600:.2f} h ===")
    log_line("run:  python scripts/analyze_experiments.py   # tables + plots")


if __name__ == "__main__":
    main()
