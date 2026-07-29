"""Parametric ternary L-system with a 3-D turtle, producing a TreeSkeleton.

This is a faithful, self-contained implementation of the ABoP / Honda ternary
tree model used in Jacob et al. (CoRL 2024).  The grammar is

    axiom : !(w0) F(F0) /(roll0) A
    A     -> !(vr) F(F0) [ &(a) F(F0) A ] /(d1)
                          [ &(a) F(F0) A ] /(d2)
                          [ &(a) F(F0) A ]
    F(l)  -> F(l * lr)        # internodes elongate each derivation
    !(w)  -> !(w * vr)        # widths thicken each derivation

Rewriting is done on a list of ``(symbol, params)`` modules (a parametric 0L
system) rather than by string parsing, which keeps the arithmetic exact and the
code easy to extend.  The expanded string is then interpreted by a 3-D turtle
(heading/left/up frame) to emit one :class:`Segment` per ``F`` module.

Convention chosen to match Newton: a segment's **body frame** has local +Z along
the branch heading (capsules extend along local Z), local +X = turtle "left",
local +Y = turtle "up".  Each segment stores that orientation as an xyzw
quaternion so the Newton builder can place capsules and joint frames directly.
"""

from __future__ import annotations

import numpy as np

from .config import LSystemParams
from .skeleton import Segment, TreeSkeleton


# --------------------------------------------------------------------------- #
# Small rotation helpers (xyzw quaternions, right-handed)
# --------------------------------------------------------------------------- #
def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector ``v`` about unit ``axis`` by ``angle`` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _frame_to_quat(H: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """xyzw quaternion for the rotation whose columns are (X=L, Y=U, Z=H).

    i.e. local +Z maps to heading H, local +X to L, local +Y to U.
    """
    # Rotation matrix with columns [L, U, H]
    m = np.column_stack([L, U, H])
    t = np.trace(m)
    if t > 0.0:
        r = np.sqrt(1.0 + t) * 2.0
        w = 0.25 * r
        x = (m[2, 1] - m[1, 2]) / r
        y = (m[0, 2] - m[2, 0]) / r
        z = (m[1, 0] - m[0, 1]) / r
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        r = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / r
        x = 0.25 * r
        y = (m[0, 1] + m[1, 0]) / r
        z = (m[0, 2] + m[2, 0]) / r
    elif m[1, 1] > m[2, 2]:
        r = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / r
        x = (m[0, 1] + m[1, 0]) / r
        y = 0.25 * r
        z = (m[1, 2] + m[2, 1]) / r
    else:
        r = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / r
        x = (m[0, 2] + m[2, 0]) / r
        y = (m[1, 2] + m[2, 1]) / r
        z = 0.25 * r
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------- #
# Module / rewriting
# --------------------------------------------------------------------------- #
# A module is a tuple (symbol: str, params: tuple[float, ...]).
Module = tuple


def _axiom(p: LSystemParams) -> list[Module]:
    # Paper axiom: !(1) F(200) /(45) A  -> trunk internode is longer than branch
    # internodes (200 vs 50), giving a clear trunk before the first ternary fork.
    return [("!", (p.w0,)), ("F", (p.F0 * p.trunk_length_factor,)),
            ("/", (p.roll0,)), ("A", ())]


def _rewrite_once(string: list[Module], p: LSystemParams) -> list[Module]:
    out: list[Module] = []
    for sym, par in string:
        if sym == "A":
            a, d1, d2, vr, F0 = p.a, p.d1, p.d2, p.vr, p.F0
            out += [
                ("!", (vr,)),
                ("F", (F0,)),
                ("[", ()), ("&", (a,)), ("F", (F0,)), ("A", ()), ("]", ()),
                ("/", (d1,)),
                ("[", ()), ("&", (a,)), ("F", (F0,)), ("A", ()), ("]", ()),
                ("/", (d2,)),
                ("[", ()), ("&", (a,)), ("F", (F0,)), ("A", ()), ("]", ()),
            ]
        elif sym == "F":
            out.append(("F", (par[0] * p.lr,)))
        elif sym == "!":
            out.append(("!", (par[0] * p.vr,)))
        else:
            out.append((sym, par))
    return out


def expand(p: LSystemParams) -> list[Module]:
    """Apply the productions ``p.n`` times to the axiom."""
    string = _axiom(p)
    for _ in range(p.n):
        string = _rewrite_once(string, p)
    return string


# --------------------------------------------------------------------------- #
# Turtle interpretation
# --------------------------------------------------------------------------- #
class _TurtleState:
    __slots__ = ("pos", "H", "L", "U", "width", "parent", "order", "depth")

    def __init__(self, pos, H, L, U, width, parent, order, depth):
        self.pos = pos
        self.H = H        # heading (grow direction)
        self.L = L        # left
        self.U = U        # up
        self.width = width
        self.parent = parent   # parent Segment index for the next drawn segment
        self.order = order     # branch order (bracket depth)
        self.depth = depth     # number of segments from root along path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate(params: LSystemParams, seed: int = 0) -> TreeSkeleton:
    """Generate a rescaled TreeSkeleton from L-system params.

    The bracket handling above keeps a single ``_TurtleState`` per stack level;
    ``[`` pushes a *copy* and makes it current, ``]`` restores the parent.
    """
    if params.kind == "apple":
        skel = grow_apple(params, seed)
    else:
        rng = np.random.default_rng(seed) if params.shape_jitter > 0 else None
        string = expand(params)
        skel = _interpret_balanced(string, params, rng)
    if params.radius_model == "pipe":
        apply_pipe_model(skel, tip_radius=params.tip_radius, beta=params.pipe_beta)
    skel.rescale(params.target_height, params.base_radius)
    skel.lift_above_ground(0.10)     # no twig below ground: the trunk sits AT z=0 now
    return skel


def apply_pipe_model(skel: TreeSkeleton, tip_radius: float, beta: float) -> None:
    """Assign radii by the pipe model (da Vinci / Murray's law).

    Terminal twigs get ``tip_radius``; every parent's radius satisfies
    ``r_parent**beta = sum(r_child**beta)`` (beta=2 conserves cross-sectional
    area, ~2.49 is Murray's law).  Guarantees a monotonic taper with the trunk
    thickest.  Radii are computed in relative units; ``rescale`` then sets the
    absolute trunk radius.
    """
    r = {}
    # process leaves -> root: sort by depth descending guarantees children first
    for seg in sorted(skel.segments, key=lambda s: s.depth, reverse=True):
        if not seg.children:
            r[seg.index] = tip_radius
        else:
            acc = sum(r[c] ** beta for c in seg.children)
            # include the segment's own continuation (its own tip contribution)
            r[seg.index] = max(acc ** (1.0 / beta), tip_radius)
    # set start/end radii: a segment tapers from its own radius (start) to the
    # max child radius (end), giving smooth visual taper along the capsule.
    for seg in skel.segments:
        seg.radius_start = r[seg.index]
        if seg.children:
            seg.radius_end = max(r[c] for c in seg.children)
        else:
            seg.radius_end = tip_radius * 0.7


def _interpret_balanced(string, p, rng):
    """Turtle interpretation with a clean explicit stack (robust brackets)."""
    deg = np.pi / 180.0
    radius_scale = 0.5 * p.F0

    def jitter(val):
        if rng is None or p.shape_jitter <= 0.0:
            return val
        return val * (1.0 + rng.normal(0.0, p.shape_jitter))

    st = _TurtleState(np.array([0.0, 0.0, 0.0]),
                      np.array([0.0, 0.0, 1.0]),
                      np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]),
                      p.w0, -1, 0, 0)
    stack = []
    segments: list[Segment] = []

    for sym, par in string:
        if sym == "F":
            length = max(jitter(par[0]), 1e-5)
            start = st.pos.copy()
            end = start + st.H * length
            r = max(st.width * radius_scale, 1e-4)
            seg = Segment(index=len(segments), parent=st.parent,
                          start=start, end=end, radius_start=r, radius_end=r,
                          order=st.order, depth=st.depth)
            # store the turtle frame as a quaternion (local +Z = heading)
            seg.frame = _frame_to_quat(st.H / np.linalg.norm(st.H),
                                       st.L / np.linalg.norm(st.L),
                                       st.U / np.linalg.norm(st.U))
            segments.append(seg)
            st.pos = end
            st.parent = seg.index
            st.depth += 1
        elif sym == "f":
            st.pos = st.pos + st.H * par[0]
        elif sym == "!":
            st.width = par[0]
        elif sym in ("&", "^", "+", "-", "/", "\\"):
            ang = jitter(par[0]) * deg
            if sym == "&":
                st.H = _rodrigues(st.H, st.L, ang); st.U = _rodrigues(st.U, st.L, ang)
            elif sym == "^":
                st.H = _rodrigues(st.H, st.L, -ang); st.U = _rodrigues(st.U, st.L, -ang)
            elif sym == "+":
                st.H = _rodrigues(st.H, st.U, ang); st.L = _rodrigues(st.L, st.U, ang)
            elif sym == "-":
                st.H = _rodrigues(st.H, st.U, -ang); st.L = _rodrigues(st.L, st.U, -ang)
            elif sym == "/":
                st.L = _rodrigues(st.L, st.H, ang); st.U = _rodrigues(st.U, st.H, ang)
            elif sym == "\\":
                st.L = _rodrigues(st.L, st.H, -ang); st.U = _rodrigues(st.U, st.H, -ang)
        elif sym == "[":
            stack.append(st)
            st = _TurtleState(st.pos.copy(), st.H.copy(), st.L.copy(), st.U.copy(),
                              st.width, st.parent, st.order + 1, st.depth)
        elif sym == "]":
            st = stack.pop()

    return TreeSkeleton(segments)


# --------------------------------------------------------------------------- #
# Stochastic apple-tree generator (MAppleT-inspired)
# --------------------------------------------------------------------------- #
def _ortho(H, L):
    """Return an orthonormal right-handed frame (H, L, U) with H x L = U."""
    H = H / (np.linalg.norm(H) + 1e-12)
    U = np.cross(H, L)
    nU = np.linalg.norm(U)
    if nU < 1e-9:                      # L parallel to H -> pick any perpendicular
        a = np.array([1.0, 0.0, 0.0]) if abs(H[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        U = np.cross(H, a)
        nU = np.linalg.norm(U)
    U = U / nU
    L = np.cross(U, H)
    return H, L, U


def _bend_toward(H, L, U, target, angle):
    """Rotate the frame so the heading bends toward unit ``target`` by ``angle``."""
    axis = np.cross(H, target)
    n = np.linalg.norm(axis)
    if n < 1e-9 or angle == 0.0:
        return H, L, U
    axis /= n
    H = _rodrigues(H, axis, angle)
    L = _rodrigues(L, axis, angle)
    return _ortho(H, L)


def _child_frame(H, L, U, roll, pitch):
    """From a parent frame, roll about the heading by ``roll`` then pitch the
    heading away by ``pitch`` (a lateral bud)."""
    Lr = _rodrigues(L, H, roll)
    Ur = _rodrigues(U, H, roll)
    Hc = _rodrigues(H, Lr, pitch)      # pitch heading off the parent
    return _ortho(Hc, Lr)


_DOWN = np.array([0.0, 0.0, -1.0])
_UP = np.array([0.0, 0.0, 1.0])


def grow_apple(p: LSystemParams, seed: int = 0, max_segments: int = 6000) -> TreeSkeleton:
    """Grow a stochastic, gravimorphic apple tree.

    A central leader emits a phyllotactic spiral of wide-angle scaffold limbs;
    each limb recursively forks into laterals that arch downward (gravimorphism).
    Every parameter is sampled per-seed, so each tree is distinct yet plausible.
    """
    rng = np.random.default_rng(seed)
    seg: list[Segment] = []

    # --- per-instance samples (this is what makes every tree different) -------
    n_scaffold = int(rng.integers(p.ap_scaffolds[0], p.ap_scaffolds[1] + 1))
    trunk_n = int(rng.integers(p.ap_trunk_internodes[0], p.ap_trunk_internodes[1] + 1))
    form = rng.uniform(*p.ap_form_droop_bias)            # upright(<1) .. weeping(>1)
    droop = rng.uniform(*p.ap_droop) * form
    len_decay = rng.uniform(*p.ap_len_decay)
    div_mean = rng.uniform(*p.ap_divergence)
    scaffold_pitch = rng.uniform(*p.ap_scaffold_pitch)
    wobble = rng.uniform(*p.ap_wobble)
    max_order = p.n
    base_int = 0.32                                       # relative internode (rescaled later)
    scaffold_len = base_int * rng.uniform(3.4, 5.2)
    upturn = rng.uniform(0.0, 0.5)                        # tips curve back up a little

    def add(start, end, H, L, U, parent, order, depth):
        H, L, U = _ortho(H, L)
        s = Segment(index=len(seg), parent=parent, start=start.copy(), end=end.copy(),
                    radius_start=1.0, radius_end=1.0, order=order, depth=depth)
        s.frame = _frame_to_quat(H, L, U)
        seg.append(s)
        return s.index, H, L, U

    def grow(pos, H, L, U, length, order, parent, depth):
        if order > max_order or length < 0.05 or len(seg) >= max_segments:
            return
        n_int = int(rng.integers(p.ap_internodes_per_axis[0], p.ap_internodes_per_axis[1] + 1))
        ilen = length / n_int
        roll = rng.uniform(0, 2 * np.pi)
        cur = parent
        ppos = pos.copy()
        Hc, Lc, Uc = _ortho(H, L)
        # NOTE: a bounded number of laterals PER AXIS (not per internode) keeps the
        # body count tractable; they are distributed over the axis's internodes.
        n_child = 0
        if order < max_order:
            base = int(rng.integers(p.ap_children[0], p.ap_children[1] + 1))
            n_child = max(0, base - (1 if rng.random() < rng.uniform(*p.ap_prune) else 0))
        child_at = sorted(rng.integers(0, n_int, size=n_child)) if n_child else []
        ci = 0
        for i in range(n_int):
            # gravimorphism: outer/older wood arches down more; tips can upturn
            frac = (i + 1) / n_int
            d = np.deg2rad(droop) * (0.35 + 0.65 * order / max(max_order, 1))
            tgt = _UP if (order >= max_order - 1 and frac > 0.6 and rng.random() < upturn) else _DOWN
            Hc, Lc, Uc = _bend_toward(Hc, Lc, Uc, tgt, d)
            # crookedness
            Hc = _rodrigues(Hc, Lc, np.deg2rad(rng.normal(0, wobble)))
            Hc = _rodrigues(Hc, Uc, np.deg2rad(rng.normal(0, wobble)))
            Hc, Lc, Uc = _ortho(Hc, Lc)
            end = ppos + Hc * ilen
            cur, Hc, Lc, Uc = add(ppos, end, Hc, Lc, Uc, cur, order, depth + i)
            ppos = end
            # spawn the laterals assigned to this internode
            while ci < len(child_at) and child_at[ci] == i:
                ci += 1
                roll += np.deg2rad(div_mean + rng.normal(0, 14))
                pitch = np.deg2rad(rng.uniform(*p.ap_lateral_pitch))
                cH, cL, cU = _child_frame(Hc, Lc, Uc, roll, pitch)
                grow(end, cH, cL, cU, length * len_decay * rng.uniform(0.8, 1.08),
                     order + 1, cur, depth + i + 1)

    # --- central leader (trunk), emitting scaffolds in a spiral --------------
    pos = np.array([0.0, 0.0, 0.0])
    H, L, U = _ortho(_UP.copy(), np.array([1.0, 0.0, 0.0]))
    cur = -1
    roll = rng.uniform(0, 2 * np.pi)
    ilen = base_int * rng.uniform(1.0, 1.3)
    remaining = n_scaffold
    for t in range(trunk_n + n_scaffold):
        H = _rodrigues(H, L, np.deg2rad(rng.normal(0, wobble * 0.4)))
        H = _rodrigues(H, U, np.deg2rad(rng.normal(0, wobble * 0.4)))
        H, L, U = _ortho(H, L)
        end = pos + H * ilen
        cur, H, L, U = add(pos, end, H, L, U, cur, 0, t)
        pos = end
        if t >= trunk_n - 1 and remaining > 0:
            roll += np.deg2rad(div_mean + rng.normal(0, 18))
            pitch = np.deg2rad(scaffold_pitch + rng.normal(0, 5))
            sH, sL, sU = _child_frame(H, L, U, roll, pitch)
            grow(end, sH, sL, sU, scaffold_len * rng.uniform(0.85, 1.15),
                 order=1, parent=cur, depth=t + 1)
            remaining -= 1
        ilen *= p.ap_leader_decay
    # weak central-leader tip continuing upward
    if rng.random() < 0.7:
        grow(pos, H, L, U, scaffold_len * 0.6, order=1, parent=cur, depth=trunk_n + n_scaffold)

    return TreeSkeleton(seg)
