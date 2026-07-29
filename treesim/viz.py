"""Lightweight matplotlib previews of a TreeSkeleton (no GPU needed).

Useful for quickly checking tree shape/taper before committing to a full Newton
sim + viewer.  Not a renderer for the simulation itself (use the Newton viewers
for that) — just the static generated geometry.
"""

from __future__ import annotations

import numpy as np


def render_skeleton(skel, path="output/skeleton.png", elev=12, azim=-70,
                    color_by="order", dpi=130, show_leaves=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segs = skel.segments
    lines = [[tuple(s.start), tuple(s.end)] for s in segs]
    orders = np.array([s.order for s in segs])
    radii = np.array([s.mean_radius for s in segs])
    maxo = max(orders.max(), 1)

    if color_by == "order":
        cmap = plt.get_cmap("YlOrBr_r")
        cols = cmap(0.15 + 0.7 * orders / maxo)
    else:
        cmap = plt.get_cmap("copper")
        cols = cmap(radii / radii.max())

    # line widths scaled by radius (clamped for visibility)
    lw = np.clip(radii / radii.max() * 6.0, 0.4, 6.0)

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="3d")
    lc = Line3DCollection(lines, colors=cols, linewidths=lw)
    ax.add_collection3d(lc)

    if show_leaves:
        lp = np.array([l.attach for l in show_leaves])
        if len(lp):
            ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2], s=6, c="#2e7d32", alpha=0.6)

    lo, hi = skel.bounds()
    ctr = (lo + hi) / 2
    rng = (hi - lo).max() / 2 * 1.05
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(lo[2], lo[2] + 2 * rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{len(segs)} segments, h={skel.height():.2f} m, orders 0..{maxo}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from treesim.config import preset
    from treesim import lsystem
    p = preset(sys.argv[1] if len(sys.argv) > 1 else "tb")
    p.n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    skel = lsystem.generate(p)
    print(render_skeleton(skel))
