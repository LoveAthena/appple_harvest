"""Geometric tree skeleton produced by the L-system turtle.

A :class:`TreeSkeleton` is a pure-geometry intermediate representation: an
ordered list of :class:`Segment` (one rigid cylindrical link each) plus their
parent/child topology.  It carries *no* physics or rendering state, so it can
be unit-tested without Newton and consumed by the Newton builder, the foliage
generator, the mesh/USD exporter, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    """A single internode: a capsule/cylinder from ``start`` to ``end``."""

    index: int
    parent: int                 # index of parent segment, -1 for the root
    start: np.ndarray           # (3,) world position of proximal end
    end: np.ndarray             # (3,) world position of distal end
    radius_start: float         # radius at proximal end [m]
    radius_end: float           # radius at distal end [m]
    order: int                  # branch order (0 = trunk lineage continuation)
    depth: int                  # number of segments from the root along the path
    children: list[int] = field(default_factory=list)
    # xyzw quaternion of the segment's body frame (local +Z along the branch
    # heading, +X = turtle left, +Y = turtle up).  Filled in by the turtle.
    frame: "np.ndarray | None" = None

    @property
    def axis(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        v = self.axis
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def mean_radius(self) -> float:
        return 0.5 * (self.radius_start + self.radius_end)

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0


class TreeSkeleton:
    """Container of :class:`Segment` with convenience queries and transforms."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._reindex()

    # -- topology ----------------------------------------------------------- #
    def _reindex(self) -> None:
        for s in self.segments:
            s.children.clear()
        by_index = {s.index: s for s in self.segments}
        for s in self.segments:
            if s.parent >= 0 and s.parent in by_index:
                by_index[s.parent].children.append(s.index)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, i: int) -> Segment:
        return self.segments[i]

    @property
    def roots(self) -> list[Segment]:
        return [s for s in self.segments if s.parent < 0]

    @property
    def terminals(self) -> list[Segment]:
        return [s for s in self.segments if s.is_terminal]

    def descendants(self, index: int) -> list[int]:
        """All segment indices in the subtree rooted at ``index`` (excluding it)."""
        out: list[int] = []
        stack = list(self.segments[index].children)
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.segments[i].children)
        return out

    # -- geometry ----------------------------------------------------------- #
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.array([s.start for s in self.segments] +
                       [s.end for s in self.segments])
        return pts.min(axis=0), pts.max(axis=0)

    def height(self) -> float:
        lo, hi = self.bounds()
        return float(hi[2] - lo[2])

    def rescale(self, target_height: float, base_radius: float) -> None:
        """Uniformly scale positions so the tree is ``target_height`` tall and
        set the trunk radius to ``base_radius`` (radii scale by the same factor
        as lengths so capsule aspect ratios are preserved).

        The tree is anchored so the TRUNK BASE sits exactly at the origin —
        translating by the bounds minimum instead (the old behaviour) planted
        the *lowest drooping branch tip* at z=0, which left the trunk hovering
        up to ~0.3 m in the air (and shifted it off-origin in x/y).  A branch
        that arches below the trunk base now simply rests on the ground, which
        the runtime's soft-ground penalty supports."""
        lo, hi = self.bounds()
        h = hi[2] - lo[2]
        if h < 1e-9:
            return
        s = target_height / h
        # current trunk radius (largest radius_start among roots/order-0)
        cur_base = max((seg.radius_start for seg in self.segments), default=base_radius)
        # We scale positions by ``s`` and radii by whatever makes the trunk match
        # base_radius, but keep them proportional to the geometric scale so thin
        # twigs stay thin.  Use the same factor ``s`` for radii then override base.
        radius_scale = base_radius / cur_base if cur_base > 1e-9 else s
        anchor = self.roots[0].start.copy() if self.roots else lo
        for seg in self.segments:
            seg.start = (seg.start - anchor) * s
            seg.end = (seg.end - anchor) * s
            seg.radius_start *= radius_scale
            seg.radius_end *= radius_scale

    def lift_above_ground(self, clearance: float = 0.10) -> int:
        """Pitch branches up just enough that no segment endpoint dips below
        ``clearance`` above the ground plane (z=0) — real limbs grow away from
        the soil.  Without this, a strongly gravimorphic seed can arch a twig
        BELOW the trunk base; the runtime's soft ground would then push it up
        constantly, bending it hard enough to spuriously snap with breaking on.

        A single parent-first forward-kinematic re-walk (segments are stored
        parent-before-child): each segment keeps its original rotation relative
        to its parent, and is additionally pitched toward vertical (about the
        horizontal axis perpendicular to its heading) by the smallest angle
        that keeps its distal end at ``clearance``.  Topology, lengths and
        radii are untouched, so multi-env batching is unaffected.  Returns how
        many segments were adjusted."""
        def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array([aw*bx + ax*bw + ay*bz - az*by,
                             aw*by - ax*bz + ay*bw + az*bx,
                             aw*bz + ax*by - ay*bx + az*bw,
                             aw*bw - ax*bx - ay*by - az*bz])

        def qconj(q):
            return np.array([-q[0], -q[1], -q[2], q[3]])

        def qrot(q, v):
            u, w = q[:3], q[3]
            return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

        def axis_angle(axis, ang):
            n = np.linalg.norm(axis)
            if n < 1e-12 or ang == 0.0:
                return np.array([0.0, 0.0, 0.0, 1.0])
            h = 0.5 * ang
            return np.array([*(axis / n * np.sin(h)), np.cos(h)])

        Z = np.array([0.0, 0.0, 1.0])
        old_frame = [np.asarray(s.frame, float) for s in self.segments]
        new_frame = [None] * len(self.segments)
        new_end = [None] * len(self.segments)
        lifted = 0
        for i, s in enumerate(self.segments):
            L = max(s.length, 1e-9)
            if s.parent < 0:
                start = s.start
                f = old_frame[i]
            else:
                start = new_end[s.parent]
                q_rel = qmul(qconj(old_frame[s.parent]), old_frame[i])
                f = qmul(new_frame[s.parent], q_rel)
            h = qrot(f, Z)
            need_hz = (clearance - start[2]) / L
            if h[2] < need_hz and need_hz <= 1.0:
                ang = float(np.arcsin(np.clip(need_hz, -1.0, 1.0)) -
                            np.arcsin(np.clip(h[2], -1.0, 1.0)))
                f = qmul(axis_angle(np.cross(h, Z), ang), f)
                f /= np.linalg.norm(f)
                h = qrot(f, Z)
                lifted += 1
            new_frame[i] = f
            new_end[i] = start + h * L
            s.start = start
            s.end = new_end[i]
            s.frame = f
        return lifted

    def total_length(self) -> float:
        return float(sum(s.length for s in self.segments))

    def summary(self) -> str:
        lo, hi = self.bounds()
        orders = [s.order for s in self.segments]
        radii = [s.mean_radius for s in self.segments]
        return (
            f"TreeSkeleton: {len(self.segments)} segments | "
            f"height={self.height():.3f} m | "
            f"orders 0..{max(orders)} | "
            f"radius {min(radii)*1e3:.2f}..{max(radii)*1e3:.2f} mm | "
            f"total wood length={self.total_length():.2f} m | "
            f"bbox=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}).."
            f"({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})"
        )
