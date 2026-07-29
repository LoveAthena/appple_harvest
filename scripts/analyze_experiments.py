#!/usr/bin/env python
"""Turn the saved experiment JSONs into paper-ready tables and figures.

Reads everything under ``output/experiments/`` (the per-job metrics JSONs and
``results.csv``) and writes, into ``output/experiments/analysis/``:

* tidy per-experiment CSVs (one row per condition) ready to plot elsewhere;
* ``canopy_zones.csv`` — pooled inner/outer x lower/middle/upper pick table;
* PNG figures for each experiment (scaling, render cost, foliage, apples,
  terrain, canopy-zone heatmap, tree-to-tree variation);
* ``report.md`` — a text summary of the headline numbers.

Safe to run at any time, including WHILE the suite is still going — it just
plots whatever has finished so far.  Re-run it after every job if you like.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "output", "experiments")
AN = os.path.join(EXP, "analysis")
ZONES_V = ["upper", "middle", "lower"]      # top row = top of tree
ZONES_R = ["inner", "outer"]


def _md(df):
    """Markdown table if ``tabulate`` is around, else a plain aligned table."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def metrics_for(exp):
    out = []
    for p in sorted(glob.glob(os.path.join(EXP, exp, "*.json"))):
        d = load_json(p)
        if d:
            out.append((os.path.splitext(os.path.basename(p))[0], d))
    return out


def savefig(fig, name):
    fig.tight_layout()
    path = os.path.join(AN, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def do_scaling(report):
    path = os.path.join(EXP, "scaling_sweep.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        return
    nd = df[df["dr"] == 0].copy()          # homogeneous envs (the RL/IL scaling story)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for solver, style in (("cg", "o-"), ("newton", "s--")):
        s = nd[nd["solver"] == solver].sort_values("envs")
        if s.empty:
            continue
        ax[0].plot(s["envs"], s["fps_mean"], style, label=f"{solver}")
        ax[1].plot(s["envs"], s["env_fps"], style, label=f"{solver}")
    for a in ax:
        a.set_xscale("log", base=2); a.set_xlabel("parallel environments")
        a.grid(alpha=.3, which="both"); a.legend(title="solver")
    ax[0].set_yscale("log"); ax[0].set_ylabel("physics step rate (fps)")
    ax[0].set_title("Physics step rate vs environments (no render)")
    ax[1].set_ylabel("environment throughput (env-steps/s)")
    ax[1].set_title("Aggregate throughput vs environments")
    savefig(fig, "fig_scaling.png")
    nd.sort_values(["solver", "envs"]).to_csv(
        os.path.join(AN, "scaling_sweep.csv"), index=False)

    cg = nd[nd["solver"] == "cg"].sort_values("envs")
    max_ok = int(cg["envs"].max()) if not cg.empty else 0
    best = cg.loc[cg["env_fps"].idxmax()] if not cg.empty else None
    report.append("## Physics scaling (homogeneous envs, no render)\n")
    report.append(_md(nd.sort_values(["solver", "envs"])[
        ["solver", "envs", "bodies", "fps_mean", "env_fps", "peak_vram_mb"]]))
    if best is not None:
        report.append(f"\nHomogeneous scaling reached **{max_ok} envs** on the "
                      f"laptop GPU; peak aggregate throughput "
                      f"$\\approx${int(best['env_fps'])} env-steps/s at "
                      f"{int(best['envs'])} envs, peak VRAM "
                      f"{int(nd['peak_vram_mb'].max())} MiB.\n")
    # DR ceiling (host build is O(N))
    dr = df[df["dr"] == 1].sort_values("envs")
    if not dr.empty:
        report.append(f"\nDomain-randomized build reached "
                      f"{int(dr['envs'].max())} envs before host build time "
                      f"dominates (per-world patching is O(N)).\n")
    # solver comparison at a fixed mid env count
    piv = nd.pivot_table(index="envs", columns="solver", values="fps_mean")
    if {"cg", "newton"}.issubset(piv.columns):
        piv["cg_speedup"] = (piv["cg"] / piv["newton"]).round(2)
        report.append("\n## CG vs Newton solver (fps)\n")
        report.append(_md(piv.reset_index()))


def _bootstrap_ci(vals, reps=2000, lo=2.5, hi=97.5):
    """Mean and (lo, hi) percentile bootstrap CI of a list of per-tree values."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return (None, None, None)
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    means = np.array([np.random.choice(v, len(v), replace=True).mean()
                      for _ in range(reps)])
    return (float(v.mean()), float(np.percentile(means, lo)),
            float(np.percentile(means, hi)))


def _picking_summary_row(cond, d):
    s = d.get("summary", {})
    census = d.get("apple_census", {}) or {}
    reachable = sum(int(v) for v in census.values())
    placed = s.get("place_success") or 0
    # harvest completeness = deposited / reachable fruit present (primary metric)
    completeness = round(placed / reachable, 3) if reachable else None
    # per-tree spread for CIs (from the per-env summaries)
    per_tree_placed = [e.get("place_success", 0) for e in d.get("envs", [])]
    pt_mean, pt_lo, pt_hi = _bootstrap_ci(per_tree_placed)
    reach_per_tree = reachable / max(len(d.get("envs", [])), 1)
    comp_lo = comp_hi = None
    if reach_per_tree and pt_lo is not None:
        comp_lo = round(pt_lo / reach_per_tree, 3)
        comp_hi = round(pt_hi / reach_per_tree, 3)
    row = dict(cond=cond, reachable=reachable, harvest_completeness=completeness,
               completeness_lo=comp_lo, completeness_hi=comp_hi,
               placed_per_tree_mean=None if pt_mean is None else round(pt_mean, 2),
               placed_per_tree_lo=None if pt_lo is None else round(pt_lo, 2),
               placed_per_tree_hi=None if pt_hi is None else round(pt_hi, 2),
               **{k: s.get(k) for k in (
                   "picks_attempted", "grasp_success", "pick_success",
                   "place_success", "total_success_rate",
                   "throughput_fruit_per_min", "mean_cycle_time_s",
                   "max_pull_force_N", "detection_precision",
                   "detection_recall_visible", "branches_snapped",
                   "apples_detached_total", "fps_mean")})
    return row


def do_failreasons(report):
    """Pool per-pick fail_reason across all autonomous runs -> a breakdown."""
    from collections import Counter
    c = Counter()
    for exp in ("E5_baseline", "E2_foliage", "E3_apples", "E4_terrain",
                "E5_variation"):
        for _, d in metrics_for(exp):
            for p in d.get("picks", []):
                if p.get("placed"):
                    c["placed"] += 1
                else:
                    c[p.get("fail_reason") or "other"] += 1
    if not c:
        return
    df = pd.DataFrame(sorted(c.items(), key=lambda kv: -kv[1]),
                      columns=["outcome", "count"])
    df.to_csv(os.path.join(AN, "failure_reasons.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["outcome"][::-1], df["count"][::-1], color="slategray")
    ax.set_xlabel("pick attempts"); ax.set_title("Outcome / failure-reason breakdown")
    ax.grid(alpha=.3, axis="x")
    savefig(fig, "fig_failure_reasons.png")
    report.append("\n## Outcome / failure-reason breakdown (pooled)\n")
    report.append(_md(df))


SIM_S = 75.0        # episode horizon (4500 frames / 60), for per-robot throughput


def _pool_group(jsons):
    """Pool run-JSONs at one swept value (possibly several seeds) into one point,
    with bootstrap CIs over the per-tree distribution."""
    placed = attempts = reach = snapped = detached = 0
    precs, cycles, per_tree = [], [], []
    n_runs = 0
    for d in jsons:
        s = d.get("summary", {})
        placed += s.get("place_success", 0) or 0
        attempts += s.get("picks_attempted", 0) or 0
        reach += sum(int(v) for v in d.get("apple_census", {}).values())
        snapped += s.get("branches_snapped", 0) or 0
        detached += s.get("apples_detached_total", 0) or 0
        if s.get("detection_precision") is not None:
            precs.append(s["detection_precision"])
        if s.get("mean_cycle_time_s") is not None:
            cycles.append(s["mean_cycle_time_s"])
        per_tree += [e.get("place_success", 0) for e in d.get("envs", [])]
        n_runs += 1
    n_trees = len(per_tree) or 1
    reach_pt = reach / n_trees if reach else 0
    pm, plo, phi = _bootstrap_ci(per_tree)
    r = dict(n_runs=n_runs, n_trees=n_trees, reachable=reach, place_success=placed,
             attempts=attempts,
             harvest_completeness=round(placed / reach, 3) if reach else None,
             place_rate=round(placed / attempts, 3) if attempts else None,
             detection_precision=round(float(np.mean(precs)), 3) if precs else None,
             mean_cycle_time_s=round(float(np.mean(cycles)), 1) if cycles else None,
             branches_snapped=round(snapped / n_runs, 1) if n_runs else None,
             fruit_dropped=round((detached - placed) / n_runs, 1) if n_runs else None)
    # per-robot throughput = 60 * placed-per-tree / episode seconds
    r["throughput_per_robot"] = round(60.0 * pm / SIM_S, 2) if pm is not None else None
    for k, v in (("throughput_lo", plo), ("throughput_hi", phi)):
        r[k] = round(60.0 * v / SIM_S, 2) if v is not None else None
    for k, v in (("completeness_lo", plo), ("completeness_hi", phi)):
        r[k] = round(v / reach_pt, 3) if (reach_pt and v is not None) else None
    return r


def do_sweep(exp, xparam, xlabel, report, xfrom_cond=None, extra_baseline=None):
    rows = metrics_for(exp)
    if extra_baseline:
        rows = rows + extra_baseline
    if not rows:
        return
    groups = {}                                       # pool seeds by swept value
    for cond, d in rows:
        x = xfrom_cond(cond, d) if xfrom_cond else cond
        groups.setdefault(x, []).append(d)
    recs = [dict(_pool_group(js), **{xparam: x}) for x, js in sorted(groups.items())]
    df = pd.DataFrame(recs).sort_values(xparam)
    df.to_csv(os.path.join(AN, f"{exp}.csv"), index=False)
    ntrees = int(df["n_trees"].max()) if "n_trees" in df else 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # left: completeness (95% bootstrap CI band) + place-rate, throughput on twin
    if df["harvest_completeness"].notna().any():
        yl = df["completeness_lo"].fillna(df["harvest_completeness"])
        yh = df["completeness_hi"].fillna(df["harvest_completeness"])
        ax[0].fill_between(df[xparam], yl, yh, alpha=.2, color="C0")
        ax[0].plot(df[xparam], df["harvest_completeness"], "o-", color="C0",
                   label="harvest completeness")
    ax[0].plot(df[xparam], df["place_rate"], "^--", color="C3",
               label="place rate / attempt")
    ax[0].set_xlabel(xlabel); ax[0].set_ylabel("success"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"Harvest success vs {xlabel}"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    if df["throughput_per_robot"].notna().any():
        ax0b = ax[0].twinx()
        ax0b.plot(df[xparam], df["throughput_per_robot"], "s:", color="green")
        ax0b.set_ylabel("fruit/min per robot", color="green")
    # centre: detection precision
    ax[1].plot(df[xparam], df["detection_precision"], "o-", color="C0")
    ax[1].set_xlabel(xlabel); ax[1].set_ylabel("detection precision")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title(f"Apple detection vs {xlabel}"); ax[1].grid(alpha=.3)
    # right: damage / loss per run
    ax[2].plot(df[xparam], df["branches_snapped"], "o-", color="firebrick",
               label="branches snapped / run")
    ax[2].plot(df[xparam], df["fruit_dropped"], "s--", color="darkorange",
               label="fruit dropped / run")
    ax[2].set_xlabel(xlabel); ax[2].set_ylabel("count"); ax[2].set_title("Damage / loss")
    ax[2].legend(); ax[2].grid(alpha=.3)
    savefig(fig, f"fig_{exp}.png")
    report.append(f"\n## {exp}: sweep over {xlabel} "
                  f"(pooled, up to {ntrees} trees/point)\n")
    cols = [xparam, "n_trees", "reachable", "place_success", "harvest_completeness",
            "place_rate", "throughput_per_robot", "detection_precision",
            "branches_snapped", "fruit_dropped"]
    report.append(_md(df[[c for c in cols if c in df.columns]]))


def do_canopy_zones(report):
    # pool every autonomous run's per-zone outcomes
    pool = {}
    n_runs = 0
    for exp in ("E5_baseline", "E2_foliage", "E3_apples", "E4_terrain",
                "E5_variation"):
        for cond, d in metrics_for(exp):
            zb = d.get("zone_breakdown")
            if not zb:
                continue
            n_runs += 1
            for z, e in zb.items():
                acc = pool.setdefault(z, dict(attempts=0, grasped=0, detached=0,
                                              placed=0, apples=0))
                for k in ("attempts", "grasped", "detached", "placed", "apples"):
                    acc[k] += int(e.get(k, 0) or 0)
    if not pool:
        return
    recs = []
    for z, e in pool.items():
        v, r = z.split("-")
        a = e["attempts"]
        recs.append(dict(zone=z, vertical=v, radial=r, apples=e["apples"],
                         attempts=a, grasped=e["grasped"], detached=e["detached"],
                         placed=e["placed"],
                         place_rate=round(e["placed"] / a, 3) if a else None,
                         grasp_rate=round(e["grasped"] / a, 3) if a else None))
    df = pd.DataFrame(recs).sort_values(["vertical", "radial"])
    df.to_csv(os.path.join(AN, "canopy_zones.csv"), index=False)

    # heatmap of place-rate over the 3x2 canopy grid
    grid = np.full((3, 2), np.nan)
    att = np.zeros((3, 2))
    for _, row in df.iterrows():
        i = ZONES_V.index(row["vertical"]); j = ZONES_R.index(row["radial"])
        if row["attempts"]:
            grid[i, j] = row["place_rate"]
        att[i, j] = row["attempts"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    im = ax.imshow(grid, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(ZONES_R)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(ZONES_V)
    ax.set_title(f"Pick place-rate by canopy zone\n(pooled over {n_runs} runs)")
    for i in range(3):
        for j in range(2):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]*100:.0f}%\n(n={int(att[i, j])})",
                        ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax, label="place success rate")
    savefig(fig, "fig_canopy_zones.png")
    report.append(f"\n## Canopy-zone success (pooled over {n_runs} autonomous runs)\n")
    report.append(_md(df[["zone", "apples", "attempts", "placed", "place_rate",
                          "grasp_rate"]]))


def do_variation(report):
    # per-tree (per-env) spread within the baseline + extra-seed runs
    recs = []
    for exp in ("E5_baseline", "E5_variation"):
        for cond, d in metrics_for(exp):
            for e, es in enumerate(d.get("envs", [])):
                recs.append(dict(run=f"{exp}/{cond}", env=e,
                                 place=es.get("place_success", 0),
                                 attempts=es.get("picks_attempted", 0),
                                 rate=es.get("total_success_rate"),
                                 det_p=es.get("detection_precision")))
    if not recs:
        return
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(AN, "tree_variation.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    mx = int(df["place"].max()) if df["place"].notna().any() else 1
    ax.hist(df["place"].dropna(), bins=range(0, mx + 2),
            align="left", rwidth=.85, color="steelblue")
    ax.set_xlabel("apples placed per tree (episode)")
    ax.set_ylabel("# trees")
    ax.set_title(f"Tree-to-tree variation ({len(df)} DR trees)")
    ax.grid(alpha=.3)
    savefig(fig, "fig_tree_variation.png")
    report.append("\n## Tree-to-tree variation & repeatability\n")
    report.append(f"{len(df)} domain-randomised trees.  Apples placed per tree: "
                  f"mean {df['place'].mean():.2f}, std {df['place'].std():.2f}, "
                  f"range {int(df['place'].min())}-{int(df['place'].max())}.\n")


def main():
    os.makedirs(AN, exist_ok=True)
    report = ["# Newton apple-orchard benchmark — results\n",
              f"_generated {pd.Timestamp.now():%Y-%m-%d %H:%M}_\n"]
    if not os.path.exists(os.path.join(EXP, "results.csv")):
        print("no results.csv yet — run scripts/run_experiments.py first")
        return
    do_scaling(report)
    # foliage 0.6 (nominal) lives in E5_baseline + E5_variation (baseline config,
    # different seeds); fold all three seeds into the foliage sweep at 0.6
    base06 = metrics_for("E5_baseline") + metrics_for("E5_variation")
    do_sweep("E2_foliage", "foliage", "foliage density", report,
             xfrom_cond=lambda c, d: float(c.split("_")[1]) if "_" in c else 0.6,
             extra_baseline=[("foliage_0.6", d) for _, d in base06])
    do_sweep("E3_apples", "apples", "apple count", report,
             xfrom_cond=lambda c, d: int(c.split("_")[1]))
    do_sweep("E4_terrain", "terrain_cm", "terrain amplitude (cm)", report,
             xfrom_cond=lambda c, d: int(c.split("_")[1].replace("cm", "")))
    do_canopy_zones(report)
    do_failreasons(report)
    do_variation(report)

    with open(os.path.join(AN, "report.md"), "w") as f:
        f.write("\n".join(str(x) for x in report) + "\n")
    print(f"wrote analysis to {AN}")
    for p in sorted(glob.glob(os.path.join(AN, "*"))):
        print("  " + os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
