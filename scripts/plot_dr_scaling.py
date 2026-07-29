#!/usr/bin/env python
"""Turn output/experiments/dr_scaling.csv into the paper's scaling figure and
print the key numbers for the results text.

  - paper/figures/fig_scaling.png : physics step rate + aggregate throughput vs
    #envs, one line per (geometry-DR mode, solver), rendering off.
  - prints a compact summary: peak env counts + throughput per condition, and a
    DR-mode x rendering fps table at a few env counts.
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "output", "experiments", "dr_scaling.csv")
FIGS = os.path.join(ROOT, "output", "experiments", "analysis")

GEOM_COLOR = {"homog": "#1f77b4", "shared": "#2ca02c", "distinct": "#d62728"}
GEOM_LABEL = {"homog": "homogeneous", "shared": "shared-dim DR", "distinct": "distinct-geom DR"}
SOLVER_STYLE = {"cg": ("o-", 1.0), "newton": ("s--", 0.85)}


def load():
    df = pd.read_csv(CSV)
    return df[df["status"] == "ok"].copy()


def fig_scaling(df):
    phys = df[df["render"] == 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    seen = []
    for geom in ("homog", "shared", "distinct"):
        for solver in ("cg", "newton"):
            s = phys[(phys["geom"] == geom) & (phys["solver"] == solver)].sort_values("envs")
            if s.empty:
                continue
            style, alpha = SOLVER_STYLE[solver]
            lab = f"{GEOM_LABEL[geom]} · {solver}"
            ax[0].plot(s["envs"], s["fps_median"], style, color=GEOM_COLOR[geom],
                       alpha=alpha, label=lab, ms=4)
            ax[1].plot(s["envs"], s["env_fps"], style, color=GEOM_COLOR[geom],
                       alpha=alpha, label=lab, ms=4)
            seen.append(lab)
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("parallel environments")
        a.grid(alpha=.3, which="both")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("physics step rate (fps)")
    ax[0].set_title("Physics step rate vs environments (no render)")
    ax[1].set_ylabel("environment throughput (env-steps/s)")
    ax[1].set_title("Aggregate throughput vs environments")
    ax[0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "fig_scaling.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def summarize(df):
    print("\n=== peak env count + best throughput per condition (no render) ===")
    phys = df[df["render"] == 0]
    for (geom, solver), s in phys.groupby(["geom", "solver"]):
        s = s.sort_values("envs")
        best = s.loc[s["env_fps"].idxmax()]
        print(f"  {GEOM_LABEL[geom]:20s} {solver:6s}: max {int(s['envs'].max()):4d} envs, "
              f"peak {int(best['env_fps']):6d} env-steps/s @ {int(best['envs'])} envs, "
              f"VRAM<= {int(s['peak_vram_mb'].max())} MiB")

    print("\n=== DR-mode x rendering: step rate (fps) at fixed env counts ===")
    for n in (8, 16, 32, 64, 128):
        row = {}
        for geom in ("homog", "shared", "distinct"):
            for r in (0, 1):
                c = df[(df["geom"] == geom) & (df["solver"] == "cg")
                       & (df["render"] == r) & (df["envs"] == n)]
                if not c.empty:
                    row[f"{geom}/{'gl' if r else 'null'}"] = round(float(c["fps_median"].iloc[0]), 1)
        if row:
            print(f"  {n:4d} envs: " + "  ".join(f"{k}={v}" for k, v in row.items()))

    # explicit shared-vs-distinct and render deltas the text can cite
    def fps(geom, r, n):
        c = df[(df["geom"] == geom) & (df["solver"] == "cg")
               & (df["render"] == r) & (df["envs"] == n)]
        return round(float(c["fps_median"].iloc[0]), 1) if not c.empty else None
    print("\n=== headline deltas (cg) ===")
    for n in (8, 16, 32):
        sh, di = fps("shared", 0, n), fps("distinct", 0, n)
        if sh and di:
            print(f"  {n} envs: shared {sh} vs distinct {di} fps  ({sh/di:.1f}x)")
    for n in (32, 64, 128):
        no, ye = fps("shared", 0, n), fps("shared", 1, n)
        if no and ye:
            print(f"  {n} envs shared: no-render {no} vs render {ye} fps  ({no/ye:.1f}x)")


if __name__ == "__main__":
    df = load()
    fig_scaling(df)
    summarize(df)
