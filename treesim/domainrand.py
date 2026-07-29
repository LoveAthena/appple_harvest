"""Per-environment domain randomization (IsaacLab-style).

The GPU-batched MuJoCo solver only batches **homogeneous** worlds: every world
must have the *same* body/joint/shape counts and types, in the same order.  It
does **not** require the same *values* — masses, initial poses, joint rest
frames and shape sizes may all differ per world (this is exactly how IsaacLab
does domain randomization).  So to get a whole *stand* of visibly different
apple trees that still batches "for free", we keep one shared discrete
structure (the L-system expansion, segment count, branch topology, apple/leaf
counts) and randomize only continuous quantities per env:

* geometry  — per-segment bend and whole-tree growth habit (droop/lean/spread/
  twist) vary per env; the branch DIMENSIONS (scale, length, thickness) are by
  default SHARED across a batch so the worlds stay dimensionally homogeneous and
  step at full speed, varying instead across resets/reloads (see
  ``DRParams.randomize_geometry`` to vary them per env too). All of it is a
  forward-kinematic re-walk that never changes the parent/child graph;
* material  — wood density, Young's modulus (stiffness) and rupture strength;
* fruit     — apple size, colour, mass and how many are visibly "grown";
* foliage   — leaf-card size.

Everything here is pure geometry/parameters (no Newton), so it is cheap and
testable in isolation.  :func:`treesim.builder.generate_and_build` calls it once
per env and stitches the resulting sub-models into parallel worlds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .skeleton import TreeSkeleton
from .config import TreeConfig
from .fruit import ApplePlacement
from .foliage import LeafPlacement


# --------------------------------------------------------------------------- #
# xyzw quaternion helpers (numpy)
# --------------------------------------------------------------------------- #
def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q, v):
    u = np.array([q[0], q[1], q[2]])
    w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _axis_angle_quat(axis, angle):
    n = np.linalg.norm(axis)
    if n < 1e-12 or angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    h = 0.5 * angle
    s = np.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(h)])


_Z = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
@dataclass
class DRParams:
    """Continuous per-env randomization ranges, tuned to a realistic apple tree.

    ``strength`` uniformly scales every deviation from the nominal (1.0): 0 = no
    randomization (identical envs), 1 = the ranges below, ~2 = exaggerated.
    """
    strength: float = 1.0

    # Per-branch DIMENSIONS (segment length, radius, whole-tree scale) are the
    # ONE per-step-expensive DR axis: distinct dimensions across the worlds of a
    # batched step re-dimension every link and gate the batched solve (~8-13 fps
    # vs ~110 homogeneous).  When False (the default for a batch) all envs SHARE
    # one dimension draw, so the worlds stay dimensionally homogeneous and step
    # at full speed; the shape still changes across resets/reloads (a new base
    # seed).  Growth-habit ANGLES (droop/lean/spread/twist/bend) and all cheap
    # axes (mass, stiffness, damping, fruit, foliage, colour) stay per-env either
    # way, since they perturb frames/values, not link dimensions, and are free.
    # True gives every env its own dimensions too (visually maximal, but slow).
    randomize_geometry: bool = False

    # whole-tree ---------------------------------------------------------------
    scale_lo: float = 0.86    # overall tree scale (height & girth together)
    scale_hi: float = 1.16
    rad_bias_lo: float = 0.90  # overall stockiness (thickness) bias
    rad_bias_hi: float = 1.16

    # per-segment --------------------------------------------------------------
    len_sigma: float = 0.10   # branch-length jitter (std, multiplicative)
    len_lo: float = 0.78
    len_hi: float = 1.26
    ang_sigma_deg: float = 5.0  # relative bend jitter per segment (compounds to tips)
    rad_sigma: float = 0.09   # thickness jitter (std, multiplicative)
    rad_lo: float = 0.76
    rad_hi: float = 1.30

    # growth habit (whole-tree, continuous -> keeps batching) --------------------
    droop_lo: float = 0.55    # gravimorphic arch multiplier: <1 upright, >1 weeping
    droop_hi: float = 1.60
    droop_base_deg: float = 1.3   # per-internode extra droop at multiplier 2.0
    lean_max_deg: float = 5.0     # whole-tree trunk lean (wind-formed), random azimuth
    spread_lo: float = 0.85   # crotch-angle scale at forks: <1 columnar, >1 spreading
    spread_hi: float = 1.18
    twist_max_deg: float = 22.0   # env-wide phyllotaxy roll offset at forks

    # coloration (render-only values, no physics) --------------------------------
    wood_bright_lo: float = 0.80  # bark brightness
    wood_bright_hi: float = 1.22
    wood_warm_lo: float = 0.90    # bark warmth (red/blue balance)
    wood_warm_hi: float = 1.14
    leaf_bright_lo: float = 0.78  # foliage brightness
    leaf_bright_hi: float = 1.28
    leaf_yellow_lo: float = 0.85  # foliage yellowness (fresh -> late-summer)
    leaf_yellow_hi: float = 1.40
    apple_hue_sigma: float = 0.06  # env-wide shift of the apple colour palette

    # material -----------------------------------------------------------------
    dens_lo: float = 0.86     # x wood_density  (green apple wood ~600-900 kg/m^3)
    dens_hi: float = 1.16
    mod_lo: float = 0.78      # x youngs_modulus (green wood ~5-12 GPa)
    mod_hi: float = 1.30
    rupt_lo: float = 0.82     # x rupture_stress (modulus of rupture ~30-80 MPa)
    rupt_hi: float = 1.25

    # apples -------------------------------------------------------------------
    apple_size_lo: float = 0.82
    apple_size_hi: float = 1.22
    apple_mass_lo: float = 0.80
    apple_mass_hi: float = 1.25
    apple_visible_lo: float = 0.55   # min fraction of apples "grown" (rest tiny)
    # "not grown" apples must be truly INVISIBLE, not pea-sized: 1.2 mm is
    # sub-pixel at any working distance (5 mm read as tiny fruit on the tree)
    apple_hidden_radius: float = 0.0012

    # foliage ------------------------------------------------------------------
    leaf_lo: float = 0.82
    leaf_hi: float = 1.22

    # ---- range helpers (apply ``strength`` by widening/narrowing about 1) ----
    def _u(self, rng, lo, hi):
        s = self.strength
        lo = 1.0 - (1.0 - lo) * s
        hi = 1.0 + (hi - 1.0) * s
        return float(rng.uniform(lo, hi))

    def _nclip(self, rng, sigma, lo, hi):
        s = self.strength
        v = 1.0 + rng.normal(0.0, sigma * s)
        lo = 1.0 - (1.0 - lo) * s
        hi = 1.0 + (hi - 1.0) * s
        return float(np.clip(v, lo, hi))


# --------------------------------------------------------------------------- #
def env_rng(base_seed: int, env_index: int) -> np.random.Generator:
    """A stable, independent RNG per (base seed, env)."""
    return np.random.default_rng([int(base_seed) & 0x7FFFFFFF, int(env_index)])


def dim_rng(base_seed: int) -> np.random.Generator:
    """A stable RNG for the SHARED branch-dimension draw (one per batch).

    Seeded off the base seed but decorrelated from every env's habit stream, so
    when geometry is shared each env draws identical dimensions (length, radius,
    scale) while keeping its own growth habit.  A fresh instance per env yields
    the same sequence, so all envs share the same dimensions; changing the base
    seed (a reset/reload) resamples the shared shape."""
    return np.random.default_rng([(int(base_seed) ^ 0x9E3779B1) & 0x7FFFFFFF, 0])


# --------------------------------------------------------------------------- #
def perturb_skeleton(base: TreeSkeleton, rng, dr: DRParams, dim_rng=None) -> TreeSkeleton:
    """Return a geometry-perturbed **copy** of ``base`` with identical topology.

    A single forward-kinematic re-walk (segments are stored parent-before-child,
    so a plain index-order pass suffices): each segment inherits its parent's
    *new* frame composed with the original relative rotation plus a small random
    bend, and hangs off the parent's *new* distal end at a jittered length.  The
    parent/child graph, orders, depths and terminal set are untouched, so the
    built model stays homogeneous with every other env.
    """
    skel = copy.deepcopy(base)
    segs = skel.segments
    n = len(segs)

    b_start = [np.asarray(s.start, float) for s in base.segments]
    b_frame = [_qnorm(np.asarray(s.frame, float)) for s in base.segments]
    b_len = [max(float(s.length), 1e-6) for s in base.segments]
    b_r0 = [float(s.radius_start) for s in base.segments]
    b_r1 = [float(s.radius_end) for s in base.segments]

    # DIMENSIONS (scale, stockiness, and per-segment length & radius in the loop
    # below) are drawn from ``drd``; when a shared dim_rng is passed these are
    # identical across the batch, so the worlds stay dimensionally homogeneous
    # and step at full speed.  Growth-habit angles and the per-segment bend stay
    # on the per-env ``rng`` (they perturb frames, not link dimensions, and are
    # free at scale).  With dim_rng=None (default) drd IS rng, so a single call
    # is byte-for-byte identical to the old fully-per-env behaviour.
    drd = rng if dim_rng is None else dim_rng
    scale = dr._u(drd, dr.scale_lo, dr.scale_hi)          # whole-tree size
    rad_bias = dr._u(drd, dr.rad_bias_lo, dr.rad_bias_hi)  # whole-tree stockiness
    ang_sigma = np.deg2rad(dr.ang_sigma_deg) * dr.strength

    # growth-habit draws (one per env; continuous, so the worlds still batch) --
    droop = dr._u(rng, dr.droop_lo, dr.droop_hi)          # arch: upright..weeping
    droop_gain = (droop - 1.0) * np.deg2rad(dr.droop_base_deg)
    spread = dr._u(rng, dr.spread_lo, dr.spread_hi)       # crotch angles at forks
    twist = np.deg2rad(rng.uniform(-dr.twist_max_deg, dr.twist_max_deg)) * dr.strength
    lean_ang = np.deg2rad(rng.uniform(0.0, dr.lean_max_deg)) * dr.strength
    lean_az = rng.uniform(0.0, 2.0 * np.pi)
    q_lean = _axis_angle_quat(np.array([np.cos(lean_az), np.sin(lean_az), 0.0]), lean_ang)

    new_start = [None] * n
    new_end = [None] * n
    new_frame = [None] * n

    for i, s in enumerate(segs):
        f_len = dr._nclip(drd, dr.len_sigma, dr.len_lo, dr.len_hi)
        f_rad = dr._nclip(drd, dr.rad_sigma, dr.rad_lo, dr.rad_hi)
        length = b_len[i] * f_len * scale

        # small random relative bend (about a random axis)
        if ang_sigma > 0.0:
            ang = float(rng.normal(0.0, ang_sigma))
            jq = _axis_angle_quat(rng.normal(size=3), ang)
        else:
            jq = np.array([0.0, 0.0, 0.0, 1.0])

        p = s.parent
        if p < 0:
            # trunk stays rooted; a small wind-formed LEAN tilts the whole tree
            new_start[i] = b_start[i] * scale
            new_frame[i] = _qnorm(_qmul(q_lean, b_frame[i]))
        else:
            q_rel = _qmul(_qconj(b_frame[p]), b_frame[i])   # original child-in-parent
            fork = segs[p].order != s.order                  # branching point
            if fork:
                # SPREAD: scale how far the child tilts off the parent's axis
                # (crotch angle) without touching its roll, by correcting the
                # child heading in the parent frame.  TWIST: an env-constant
                # extra phyllotaxy roll about the parent's axis.
                if spread != 1.0:
                    d = _qrot(q_rel, _Z)
                    tilt = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
                    axis = np.cross(d, _Z)
                    q_rel = _qmul(_axis_angle_quat(axis, (1.0 - spread) * tilt), q_rel)
                if twist != 0.0:
                    q_rel = _qmul(_axis_angle_quat(_Z, twist), q_rel)
            q_rel = _qmul(jq, q_rel)                        # + random bend
            new_frame[i] = _qnorm(_qmul(new_frame[p], q_rel))
            # DROOP: gravimorphic arch — bend each internode a little toward
            # (droop>1) or away from (droop<1) the ground, in proportion to how
            # horizontal it is; compounds along an axis exactly like real
            # branch arching.  World-frame correction after composition.
            if droop_gain != 0.0 and s.order > 0:
                h = _qrot(new_frame[i], _Z)
                horiz = float(np.hypot(h[0], h[1]))
                if horiz > 1e-6:
                    axis = np.cross(h, np.array([0.0, 0.0, -1.0]))
                    new_frame[i] = _qnorm(_qmul(
                        _axis_angle_quat(axis, droop_gain * horiz), new_frame[i]))
            new_start[i] = new_end[p]

        heading = _qrot(new_frame[i], _Z)
        new_end[i] = new_start[i] + heading * length

        s.start = new_start[i]
        s.end = new_end[i]
        s.frame = new_frame[i]
        s.radius_start = b_r0[i] * f_rad * rad_bias * scale
        s.radius_end = b_r1[i] * f_rad * rad_bias * scale

    skel._reindex()          # topology is unchanged; harmless & keeps invariants
    # extra per-env droop can arch a twig below ground -> pitch it back up
    # (values-only FK pass, so the worlds still batch)
    skel.lift_above_ground(0.10)
    return skel


# --------------------------------------------------------------------------- #
def perturb_config(base_cfg: TreeConfig, rng, dr: DRParams, env_index: int) -> TreeConfig:
    """Return a **copy** of the config with per-env continuous material/size
    randomization.  Discrete knobs (counts, enums, recursion depth, max apple
    count) are left untouched so every env stays structurally identical."""
    c = copy.deepcopy(base_cfg)
    # A distinct seed only drives *continuous* per-joint / per-apple jitter here
    # (placements are supplied externally, so counts can't drift).
    c.seed = int(base_cfg.seed) + 1 + int(env_index)

    c.physics.wood_density = float(np.clip(
        base_cfg.physics.wood_density * dr._u(rng, dr.dens_lo, dr.dens_hi), 500.0, 1000.0))
    c.physics.youngs_modulus = float(
        base_cfg.physics.youngs_modulus * dr._u(rng, dr.mod_lo, dr.mod_hi))
    c.breaking.rupture_stress = float(
        base_cfg.breaking.rupture_stress * dr._u(rng, dr.rupt_lo, dr.rupt_hi))
    c.fruit.mass = float(np.clip(
        base_cfg.fruit.mass * dr._u(rng, dr.apple_mass_lo, dr.apple_mass_hi), 0.09, 0.26))

    leaf = dr._u(rng, dr.leaf_lo, dr.leaf_hi)
    c.foliage.leaf_length = base_cfg.foliage.leaf_length * leaf
    c.foliage.leaf_width = base_cfg.foliage.leaf_width * leaf

    # coloration (render-only): bark tint and foliage colour vary tree-to-tree
    wb = dr._u(rng, dr.wood_bright_lo, dr.wood_bright_hi)
    ww = dr._u(rng, dr.wood_warm_lo, dr.wood_warm_hi)
    c.render.wood_tint = (wb * ww, wb, wb / max(ww, 1e-3))
    lb = dr._u(rng, dr.leaf_bright_lo, dr.leaf_bright_hi)
    ly = dr._u(rng, dr.leaf_yellow_lo, dr.leaf_yellow_hi)
    r0, g0, b0 = base_cfg.foliage.leaf_color
    c.foliage.leaf_color = (float(np.clip(r0 * lb * ly, 0.0, 1.0)),
                            float(np.clip(g0 * lb, 0.0, 1.0)),
                            float(np.clip(b0 * lb / max(ly, 1e-3), 0.0, 1.0)))
    return c


# --------------------------------------------------------------------------- #
def _fraction_along(seg, attach) -> float:
    """Where ``attach`` sits along ``seg`` (0=proximal .. 1=distal)."""
    t = float(np.dot(np.asarray(attach) - seg.start, seg.direction) / max(seg.length, 1e-6))
    return float(np.clip(t, 0.0, 1.0))


def remap_apples(canonical, base: TreeSkeleton, skel_e: TreeSkeleton,
                 rng, dr: DRParams):
    """Re-home the *fixed* apple set onto this env's perturbed geometry.

    The apple COUNT and which spurs bear fruit are frozen (shared ``canonical``
    placement), so every env has the same number of apple bodies.  Per env we
    move each apple to the matching spot on the moved spur and jitter its size &
    colour; a random subset is shrunk to a tiny "not grown" radius so the *number
    of visible* apples varies between trees without changing the body count.
    """
    m = len(canonical)
    if m == 0:
        return []
    n_vis = int(round(dr._u(rng, dr.apple_visible_lo, 1.0) * m))
    n_vis = int(np.clip(n_vis, 1, m))
    visible = set(rng.choice(m, size=n_vis, replace=False).tolist())
    # env-wide palette shift: some trees bear redder fruit, some greener
    hue = rng.normal(0.0, dr.apple_hue_sigma * dr.strength, size=3)

    out = []
    for j, ap in enumerate(canonical):
        ps = ap.parent_seg
        t = _fraction_along(base[ps], ap.attach)
        eseg = skel_e[ps]
        attach_e = eseg.start + eseg.direction * (t * eseg.length)
        if j in visible:
            # realistic dessert-apple band: 4.8-6.8 cm diameter.  The floor
            # keeps DR from producing plum-sized fruit; the ceiling keeps the
            # fruit graspable (Franka opens 8 cm, and finger clearance on
            # approach shrinks fast past ~7 cm)
            r = float(np.clip(ap.radius * dr._u(rng, dr.apple_size_lo, dr.apple_size_hi),
                              0.024, 0.034))
            # 2 mm size classes: one GL instancer per unique sphere size (see
            # fruit.place_apples) — imperceptible, keeps the draw batched
            r = float(np.clip(round(r / 0.002) * 0.002, 0.024, 0.034))
        else:
            r = float(dr.apple_hidden_radius)
        col = tuple(float(np.clip(c + h + rng.normal(0.0, 0.03), 0.0, 1.0))
                    for c, h in zip(ap.color, hue))
        out.append(ApplePlacement(parent_seg=ps, attach=attach_e, radius=r, color=col))
    return out


def remap_leaves(canonical, base: TreeSkeleton, skel_e: TreeSkeleton):
    """Re-home the fixed leaf set onto this env's perturbed geometry (same count,
    same relative position/orientation on each twig).  Leaf-card *size* comes from
    the per-env config, so nothing here needs to change size."""
    out = []
    for lp in canonical:
        ps = lp.parent_seg
        bseg = base[ps]
        eseg = skel_e[ps]
        t = _fraction_along(bseg, lp.attach)
        attach_e = eseg.start + eseg.direction * (t * eseg.length)
        local_q = _qmul(_qconj(_qnorm(np.asarray(bseg.frame, float))),
                        _qnorm(np.asarray(lp.frame, float)))
        frame_e = _qnorm(_qmul(_qnorm(np.asarray(eseg.frame, float)), local_q))
        out.append(LeafPlacement(parent_seg=ps, attach=attach_e, frame=frame_e,
                                 length=lp.length, width=lp.width))
    return out
