"""Convert a TreeSkeleton into a Newton articulated model.

Each L-system segment becomes one rigid body with a capsule collider (axis =
local +Z, matching the skeleton frame).  Segments are connected to their parent
by a joint whose rest pose reproduces the skeleton:

* ``deformable=False`` -> ``add_joint_fixed`` (welded; the whole tree is rigid).
* ``deformable=True``  -> a torsional spring-damper:
    - ``joint_type="d6"``      : 2 bending DOFs (local X & Y), twist+linear locked.
    - ``joint_type="revolute"``: 1 bending DOF (local X).
    - ``joint_type="spherical"``: 3 angular DOFs (X, Y, twist).

The trunk base is welded to the world with a fixed joint, so the tree is rooted.

The returned :class:`TreeModel` keeps every mapping needed downstream (segment
<-> body, joint <-> segment, per-DOF stiffness, rupture thresholds, DOF offsets)
so deformation, breaking and foliage can all be layered on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton

from .config import TreeConfig, StiffnessModel
from .skeleton import TreeSkeleton
from . import physics


# --------------------------------------------------------------------------- #
# xyzw quaternion helpers (numpy)
# --------------------------------------------------------------------------- #
def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qrot(q, v):
    """Rotate vec3 ``v`` by xyzw quaternion ``q``."""
    u = np.array([q[0], q[1], q[2]])
    w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _wq(q):
    """numpy xyzw -> wp.quat (xyzw)."""
    return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _wv(v):
    return wp.vec3(float(v[0]), float(v[1]), float(v[2]))


# --------------------------------------------------------------------------- #
@dataclass
class TreeModel:
    model: "newton.Model"
    config: TreeConfig
    skeleton: TreeSkeleton
    seg_to_body: np.ndarray              # segment index -> body index
    joint_ids: list[int]                 # branch joint ids (excludes fixed root)
    joint_seg: np.ndarray                # branch joint -> child segment index
    joint_kp: np.ndarray                 # (Jbranch, ndof) per-dof stiffness
    joint_kd: np.ndarray                 # (Jbranch, ndof) per-dof damping
    joint_dof_start: np.ndarray          # branch joint -> first global dof index
    joint_ndof: int                      # dofs per branch joint
    bend_mask: np.ndarray                # (ndof,) bool: which local dofs bend
    rupture: np.ndarray                  # branch joint -> M_max [N*m]
    base_color: np.ndarray = field(default=None)   # (Nbody,3) for recolour
    n_bodies: int = 0
    num_envs: int = 1                               # parallel copies of the tree (worlds)
    leaf_bodies: list = field(default_factory=list)   # body indices of leaves
    apple_bodies: list = field(default_factory=list)  # body indices of apples
    apple_data: dict = field(default=None)            # AppleField arrays (free-body apples)

    robot_data: dict = field(default=None)             # RidgebackFranka maps (per env, tiled)
    terrain_height: object = None                       # callable (x, y) -> ground z, or None
    env_pitch: tuple = None                             # (px, py) display-grid pitch [m]
    env_cols: int = 1                                   # display-grid columns

    # Unified list of *breakable* joints (branch joints + apple stems) for the
    # runtime breaker.  Parallel arrays, one entry per breakable joint.
    brk_joint_id: np.ndarray = field(default=None)    # model joint id (for joint_enabled)
    brk_parent_body: np.ndarray = field(default=None)
    brk_child_body: np.ndarray = field(default=None)
    brk_kp: np.ndarray = field(default=None)          # bending stiffness for moment calc
    brk_mmax: np.ndarray = field(default=None)        # rupture moment [N*m]
    brk_dof_start: np.ndarray = field(default=None)   # first global dof (for hinge zeroing)
    brk_dof_count: np.ndarray = field(default=None)
    brk_descend: list = field(default_factory=list)   # bodies to recolour per break

    def descendants_bodies(self, joint_idx: int) -> list[int]:
        """Body indices of the subtree below branch joint ``joint_idx`` (used by
        the spring-mode breaker, which indexes branch joints in segment order)."""
        seg = int(self.joint_seg[joint_idx])
        idxs = [seg] + self.skeleton.descendants(seg)
        return [int(self.seg_to_body[i]) for i in idxs]

    def state_pair(self):
        s0 = self.model.state()
        s1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s1)
        return s0, s1


# --------------------------------------------------------------------------- #
def _add_terrain(b: "newton.ModelBuilder", ph, seed: int,
                 pitch: tuple | None = None, grid: tuple = (1, 1)):
    """Gentle value-noise heightfield terrain (one global static shape, like
    the ground plane — all envs share it, so batching is unaffected).

    Multi-env: every world sits at the physics ORIGIN and the viewer spreads
    them on a (cols x rows) grid with per-dimension ``pitch``, so the noise is
    generated PERIODIC with exactly that period and tiled across the whole
    displayed grid (plus one apron tile all round).  The ground drawn under
    every displayed env is then identical to the terrain the physics (and the
    base z-servo) samples at the origin, and every trunk sits on a flattened
    disc.  Amplitude is kept small enough to drive over."""
    rng = np.random.default_rng((int(seed) * 2654435761) & 0x7FFFFFFF)
    if pitch is None:
        ext = float(ph.terrain_extent)
        px, py = 2.0 * ext, 2.0 * ext
        cols = rows = 1
        apron = 0
    else:
        px, py = float(pitch[0]), float(pitch[1])
        cols, rows = max(int(grid[0]), 1), max(int(grid[1]), 1)
        apron = 1

    # ONE periodic tile of value noise (wrapped lattice, bilinear upsample);
    # sample j sits at x = -px/2 + j*px/res so tile copies join seamlessly
    res_x = int(np.clip(round(px / 0.16), 48, 160))
    res_y = int(np.clip(round(py / 0.16), 48, 160))
    tile = np.zeros((res_y, res_x))
    for octave, w in enumerate([1.0, 0.5, 0.25]):
        wl = ph.terrain_wavelength / (octave + 1)
        cx, cy = max(int(round(px / wl)), 2), max(int(round(py / wl)), 2)
        g = rng.standard_normal((cy, cx))
        u = np.arange(res_x) * cx / res_x
        v = np.arange(res_y) * cy / res_y
        j0 = u.astype(int) % cx
        i0 = v.astype(int) % cy
        j1, i1 = (j0 + 1) % cx, (i0 + 1) % cy
        fu, fv = u - u.astype(int), (v - v.astype(int))[:, None]
        top = g[np.ix_(i0, j0)] * (1 - fu) + g[np.ix_(i0, j1)] * fu
        bot = g[np.ix_(i1, j0)] * (1 - fu) + g[np.ix_(i1, j1)] * fu
        tile += w * (top + (bot - top) * fv)
    tile -= tile.min()
    tile /= max(tile.max(), 1e-9)
    # flatten under the trunk (tile centre == env origin == trunk)
    xx = (np.arange(res_x) / res_x - 0.5) * px
    yy = (np.arange(res_y) / res_y - 0.5) * py
    r = np.hypot(xx[None, :], yy[:, None])
    tile *= np.clip((r - 0.9) / 1.4, 0.0, 1.0)

    # tile the displayed grid (+ apron); duplicate the far row/col so the
    # heightfield's corner-inclusive sampling keeps the exact tile period
    fld = np.tile(tile, (rows + 2 * apron, cols + 2 * apron))
    fld = np.concatenate([fld, fld[:1, :]], axis=0)
    fld = np.concatenate([fld, fld[:, :1]], axis=1)
    hx = 0.5 * (cols + 2 * apron) * px
    hy = 0.5 * (rows + 2 * apron) * py
    centre = (0.5 * (cols - 1) * px, 0.5 * (rows - 1) * py)
    # float the field a few mm above the global ground plane: the flattened
    # discs are at height 0 and would z-fight the checkerboard otherwise
    lift = 0.004
    hf = newton.Heightfield(fld.astype(np.float32),
                            nrow=fld.shape[0], ncol=fld.shape[1],
                            hx=hx, hy=hy,
                            min_z=lift, max_z=float(ph.terrain_amplitude) + lift)
    b.add_shape_heightfield(
        heightfield=hf,
        xform=wp.transform(wp.vec3(centre[0], centre[1], 0.0), wp.quat_identity()),
        cfg=b.ShapeConfig(mu=0.9, restitution=0.0, collision_group=1),
        color=(0.30, 0.36, 0.22), label="terrain")

    amp = float(ph.terrain_amplitude)
    zg = tile * amp + lift
    wrap = pitch is not None

    def sample(x, y, _g=zg, _px=px, _py=py, _rx=res_x, _ry=res_y, _w=wrap):
        u = (float(x) / _px + 0.5) * _rx
        v = (float(y) / _py + 0.5) * _ry
        if _w:
            u, v = u % _rx, v % _ry
        elif not (0.0 <= u <= _rx - 1 and 0.0 <= v <= _ry - 1):
            return 0.0
        j0, i0 = int(u) % _rx, int(v) % _ry
        j1, i1 = (j0 + 1) % _rx, (i0 + 1) % _ry
        fu, fv = u - int(u), v - int(v)
        return float((_g[i0, j0] * (1 - fu) + _g[i0, j1] * fu) * (1 - fv)
                     + (_g[i1, j0] * (1 - fu) + _g[i1, j1] * fu) * fv)

    return sample


def _env_grid(aabbs, num_envs: int) -> tuple:
    """Display-grid geometry for multi-env rendering: per-dimension pitch from
    the union AABB of the env contents (robot pads x only — a single square
    pitch left the x gaps tight and the y gaps wide), plus an even visual gap.
    ``aabbs`` is a list of (lo_xy, hi_xy) pairs."""
    lo = np.min([a[0] for a in aabbs], axis=0)
    hi = np.max([a[1] for a in aabbs], axis=0)
    gap = 2.2                       # even gap both axes; covers shape overhang
    pitch = (float(hi[0] - lo[0]) + gap, float(hi[1] - lo[1]) + gap)
    cols = int(np.ceil(np.sqrt(max(num_envs, 1))))
    rows = int(np.ceil(num_envs / max(cols, 1)))
    return pitch, cols, rows


def _builder_aabb(b: "newton.ModelBuilder") -> tuple:
    q = np.asarray(b.body_q, dtype=np.float64).reshape(-1, 7)[:, :2]
    return q.min(axis=0), q.max(axis=0)


def _wood_color(order: int, max_order: int, tint=(1.0, 1.0, 1.0)) -> tuple:
    """Brown -> tan gradient from trunk to twigs (placeholder vertex colour).
    ``tint`` is a per-env multiplier used by domain randomization."""
    t = order / max(max_order, 1)
    base = np.array([0.36, 0.24, 0.14])   # dark bark brown
    tip = np.array([0.55, 0.41, 0.24])    # lighter outer wood
    c = np.clip((base * (1 - t) + tip * t) * np.asarray(tint), 0.0, 1.0)
    return float(c[0]), float(c[1]), float(c[2])


def build(config: TreeConfig, skeleton: TreeSkeleton,
          max_bodies: int = 4000, num_envs: int = 1,
          apple_placements=None, leaf_placements=None, _sub: bool = False,
          env_aabb=None):
    """Build a Newton model from a skeleton according to ``config``.

    ``num_envs`` > 1 replicates the whole tree into that many parallel worlds
    (mjwarp batches them on the GPU), for IsaacLab-style throughput.

    ``apple_placements`` / ``leaf_placements`` override the internal
    ``place_apples`` / ``place_leaves`` calls — used by the domain-randomized
    multi-env path, where the apple/leaf set is fixed once on a base skeleton
    (so every env has identical counts) and re-homed onto each env's perturbed
    geometry.  With ``_sub=True`` the un-finalized builder and its env-relative
    index maps are returned (``(builder, maps)``) instead of a finished
    :class:`TreeModel`, so several differently-randomized sub-builders can be
    stitched into parallel worlds by :func:`_assemble_dr`.
    """
    n = len(skeleton)
    if n > max_bodies:
        raise ValueError(
            f"skeleton has {n} bodies (> max_bodies={max_bodies}); reduce "
            f"L-system depth n or raise max_bodies. Each body costs VRAM.")

    rng = np.random.default_rng(config.seed)
    builder = newton.ModelBuilder()
    builder.gravity = config.physics.gravity   # along -up (Z)

    # collision_group = -1: branches collide with the ground and with *external*
    # positive-group objects (e.g. a robot arm) but NOT with each other.  Tree
    # self-collision among hundreds of capsules is an O(N^2) broad-phase blowup
    # and almost never wanted, so it is disabled by default (this is what makes
    # --collisions usable on a full tree).  Contact gains are SOFT (green wood
    # gives): stiff contacts against a 130 kg robot explode gram-scale bodies.
    wood = builder.ShapeConfig(
        density=config.physics.wood_density,
        mu=0.7, restitution=0.0, collision_group=-1,
        ke=600.0, kd=40.0,
    )
    # Thin twigs DO take rigid contacts against the robot (group -1, like the
    # thicker wood), so the manipulator cannot pass through them.  The naive
    # danger is the mass ratio: a gram-scale body squeezed against a 130 kg robot
    # is a ~10^4:1 contact that explodes the solver.  We condition it two ways:
    # (i) extra JOINT ARMATURE on the twig's joint (``twig_armature`` below),
    # invisible inertia in DOF space that regularises the contact impulse without
    # changing the static bend; and (ii) a small contact ``margin`` so the contact
    # is caught before the fast arm has moved through the thin capsule (a
    # speculative buffer against discrete-step tunnelling).  A branch the robot
    # cannot push aside then either stalls the base or ruptures (breaking) rather
    # than being penetrated.
    twig = builder.ShapeConfig(
        density=config.physics.wood_density,
        mu=0.7, restitution=0.0, collision_group=-1,
        ke=800.0, kd=60.0, margin=float(os.environ.get("TWIG_MARGIN", 0.012)),
    )
    TWIG_R = 0.010
    twig_armature = float(os.environ.get("TWIG_ARMATURE", 0.02))

    max_order = max(s.order for s in skeleton.segments)
    seg_to_body = np.full(n, -1, dtype=int)
    colors: list[tuple] = []   # one per body in add order (branches then leaves)
    wood_tint = tuple(getattr(config.render, "wood_tint", (1.0, 1.0, 1.0)))

    # --- bodies + capsule shapes --------------------------------------------
    for seg in skeleton.segments:
        # add_link (not add_body) so Newton does NOT auto-attach a free joint;
        # we connect every segment explicitly below.
        body = builder.add_link(label=f"seg{seg.index}")
        seg_to_body[seg.index] = body
        L = max(seg.length, 1e-3)
        r = max(seg.mean_radius, 5e-4)
        col = _wood_color(seg.order, max_order, wood_tint)
        colors.append(col)
        # capsule axis = local +Z; centre at L/2 so body origin is the proximal end
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, L * 0.5), q=wp.quat_identity()),
            radius=r, half_height=L * 0.5,
            cfg=(wood if r >= TWIG_R else twig), color=col,
        )

    deformable = config.deformable and config.physics.model != StiffnessModel.RIGID
    jt = config.physics.joint_type
    # All compliant joints are 2-bending-DOF D6 (stable in MuJoCo across long runs;
    # 6-DOF stems destabilise on the compliant spurs).  On rupture the runtime
    # zeroes the bending stiffness so the branch/apple goes limp and swings about
    # the break; aerodynamic drag + the soft ground then decelerate and settle it.
    breaking = False

    # Decide the DOF layout for branch joints.  D6 orders dofs as
    # [linear..., angular...].  ``bend_mask`` flags the angular bending dofs
    # used for the rupture-moment calculation.
    X = wp.vec3(1.0, 0.0, 0.0)
    Y = wp.vec3(0.0, 1.0, 0.0)
    Z = wp.vec3(0.0, 0.0, 1.0)
    if not deformable:
        ndof, bend_mask = 0, np.zeros(0, dtype=bool)
    elif breaking:
        # 3 linear (near-rigid) + 3 angular (2 bending + twist); zeroing all 6
        # on rupture frees the child -> the subtree detaches and falls.
        ndof, bend_mask = 6, np.array([False, False, False, True, True, False])
    elif jt == "revolute":
        ndof, bend_mask = 1, np.array([True])
    elif jt == "spherical":
        ndof, bend_mask = 3, np.array([True, True, False])
    else:  # d6, 2 bending dofs
        ndof, bend_mask = 2, np.array([True, True])

    joint_ids: list[int] = []
    joint_seg: list[int] = []
    joint_kp: list[list[float]] = []
    joint_kd: list[list[float]] = []
    rupture: list[float] = []
    all_joint_ids: list[int] = []

    # unified breakable-joint records (branch joints + apple stems)
    brk_jid: list[int] = []
    brk_parent: list[int] = []
    brk_child: list[int] = []
    brk_kp: list[float] = []
    brk_mmax: list[float] = []
    brk_ndof: list[int] = []
    brk_descend: list[list[int]] = []

    lim = config.physics.joint_limit
    arm = config.physics.armature
    cur_arm = arm                    # per-joint armature, raised for thin twigs
    klin = config.breaking.linear_stiffness

    # NOTE: hard joint limits become MuJoCo *constraints* (one per DOF), which
    # both overflows njmax and dominates solve time on a large tree.  The
    # torsional spring (target_ke, a passive force, not a constraint) already
    # keeps branches near their rest pose, so we leave joints unlimited by
    # default (limit_ke=0).  Set ``physics.joint_limit > 0`` and
    # ``use_limits=True`` to re-enable soft limits (and bump njmax in the solver).
    use_limits = config.physics.joint_limit > 0 and getattr(config.physics, "use_limits", False)

    def dof(axis, kp, kd, limit=None, armature=None):
        lo = -(limit if limit is not None else lim)
        hi = (limit if limit is not None else lim)
        lke = 1.0e4 if use_limits else 0.0
        lkd = 1.0e1 if use_limits else 0.0
        if not use_limits:
            lo, hi = -1.0e9, 1.0e9
        return builder.JointDofConfig(
            axis=axis, target_pos=0.0, target_ke=kp, target_kd=kd,
            limit_lower=lo, limit_upper=hi, limit_ke=lke, limit_kd=lkd,
            armature=(cur_arm if armature is None else armature),
            actuator_mode=newton.JointTargetMode.POSITION,
        )

    # --- root: weld trunk base to the world ---------------------------------
    root = skeleton.roots[0]
    rj = builder.add_joint_fixed(
        parent=-1, child=int(seg_to_body[root.index]),
        parent_xform=wp.transform(p=_wv(root.start), q=_wq(root.frame)),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    all_joint_ids.append(rj)

    # --- inter-branch joints -------------------------------------------------
    for seg in skeleton.segments:
        if seg.parent < 0:
            continue
        parent = skeleton[seg.parent]
        pbody = int(seg_to_body[seg.parent])
        cbody = int(seg_to_body[seg.index])
        q_rel = _qmul(_qconj(parent.frame), seg.frame)
        pxform = wp.transform(p=wp.vec3(0.0, 0.0, max(parent.length, 1e-3)), q=_wq(q_rel))
        cxform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())

        if not deformable:
            jid = builder.add_joint_fixed(parent=pbody, child=cbody,
                                          parent_xform=pxform, child_xform=cxform)
            all_joint_ids.append(jid)
            continue

        # thin twigs get extra joint armature so their rigid contact with the
        # 130 kg robot is well-conditioned despite the mass ratio; armature is
        # inertia in DOF space, so it does not change the static bend under load.
        cur_arm = twig_armature if seg.mean_radius < TWIG_R else arm
        kp, kd = physics.stiffness_for(seg, config.physics, rng)
        if config.breaking.enabled and not getattr(config.breaking, "free_fall", False):
            # FALLBACK breaking (non-MuJoCo): softer damping so a snapped branch
            # droops/flops instead of creeping.  With free-fall breaking the
            # actuator (incl. damping) is removed at rupture, so the tree keeps
            # the normal damping ratio and pre-break dynamics are identical to a
            # non-breakable tree.
            kd = config.breaking.limp_damping_ratio * kp
        kpt = min(kp * config.breaking.twist_stiffness_scale, config.physics.max_stiffness)
        if breaking:
            linear = [dof(X, klin, klin * 0.1, limit=1.0e6),
                      dof(Y, klin, klin * 0.1, limit=1.0e6),
                      dof(Z, klin, klin * 0.1, limit=1.0e6)]
            angular = [dof(X, kp, kd), dof(Y, kp, kd), dof(Z, kpt, kd)]
            jid = builder.add_joint_d6(parent=pbody, child=cbody,
                                       linear_axes=linear, angular_axes=angular,
                                       parent_xform=pxform, child_xform=cxform)
            kps = [klin, klin, klin, kp, kp, kpt]
            kds = [klin * 0.1, klin * 0.1, klin * 0.1, kd, kd, kd]
        elif jt == "revolute":
            jid = builder.add_joint_d6(parent=pbody, child=cbody,
                                       angular_axes=[dof(X, kp, kd)],
                                       parent_xform=pxform, child_xform=cxform)
            kps, kds = [kp], [kd]
        elif jt == "spherical":
            jid = builder.add_joint_d6(parent=pbody, child=cbody,
                                       angular_axes=[dof(X, kp, kd), dof(Y, kp, kd),
                                                     dof(Z, kpt, kd)],
                                       parent_xform=pxform, child_xform=cxform)
            kps, kds = [kp, kp, kpt], [kd, kd, kd]
        else:  # d6 -> 2 bending dofs
            jid = builder.add_joint_d6(parent=pbody, child=cbody,
                                       angular_axes=[dof(X, kp, kd), dof(Y, kp, kd)],
                                       parent_xform=pxform, child_xform=cxform)
            kps, kds = [kp, kp], [kd, kd]

        all_joint_ids.append(jid)
        joint_ids.append(jid)
        joint_seg.append(seg.index)
        joint_kp.append(kps)
        joint_kd.append(kds)
        mmax = physics.rupture_moment(seg, config.breaking)
        rupture.append(mmax)
        # register as a breakable joint (branches only rupture if breaking is on;
        # otherwise an infinite threshold so apples can detach without snapping
        # the woody branches)
        sub = [seg.index] + skeleton.descendants(seg.index)
        brk_jid.append(jid)
        brk_parent.append(pbody)
        brk_child.append(cbody)
        brk_kp.append(kp)                       # bending stiffness
        brk_mmax.append(mmax if config.breaking.enabled else float("inf"))
        brk_ndof.append(len(kps))
        brk_descend.append([int(seg_to_body[i]) for i in sub])

    # --- optional foliage: leaf cards on the outer twigs --------------------
    # By default leaves are added as extra *shapes on the parent branch body* —
    # they render as a full canopy and move rigidly with the branch at ZERO extra
    # physics cost (no new bodies/DOFs).  --foliage-physics instead makes each
    # leaf its own body on a compliant petiole so it flutters (much heavier:
    # hundreds of extra bodies).
    leaf_bodies: list[int] = []
    if config.foliage.enabled:
        from . import foliage as _foliage
        fp = config.foliage
        leaf_col = tuple(getattr(fp, "leaf_color", (0.18, 0.42, 0.12)))
        placements = (leaf_placements if leaf_placements is not None
                      else _foliage.place_leaves(skeleton, fp, seed=config.seed))
        if fp.physics:
            # each leaf = its own body on a compliant petiole (flutters; EXPENSIVE:
            # +1 body & joint per leaf, so keep leaf counts small in this mode)
            leaf_cfg = builder.ShapeConfig(density=max(fp.leaf_mass / max(
                fp.leaf_length * fp.leaf_width * 0.004, 1e-7), 50.0), mu=0.6,
                collision_group=0)
            for lp in placements:
                pseg = skeleton[lp.parent_seg]
                pbody = int(seg_to_body[lp.parent_seg])
                half_w = max(lp.width * 0.5, 1e-3); half_l = max(lp.length * 0.5, 1e-3)
                lbody = builder.add_link(label=f"leaf{len(leaf_bodies)}")
                leaf_bodies.append(lbody)
                colors.append(leaf_col)
                builder.add_shape_box(
                    lbody, xform=wp.transform(p=wp.vec3(0.0, 0.0, half_l), q=wp.quat_identity()),
                    hx=half_w, hy=0.0008, hz=half_l, cfg=leaf_cfg, color=leaf_col)
                off = _qrot(_qconj(pseg.frame), lp.attach - pseg.start)
                pxf = wp.transform(p=_wv(off), q=_wq(_qmul(_qconj(pseg.frame), lp.frame)))
                cxf = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
                ke, kd = fp.petiole_stiffness, fp.petiole_damping
                all_joint_ids.append(builder.add_joint_d6(
                    parent=pbody, child=lbody,
                    angular_axes=[dof(wp.vec3(1.0, 0, 0), ke, kd), dof(wp.vec3(0, 1.0, 0), ke, kd)],
                    parent_xform=pxf, child_xform=cxf))
        else:
            # DEFAULT: render-only leaf BLADES (folded/curled elliptical meshes).
            # All leaves of one size class share ONE Mesh object, so the GL
            # viewer INSTANCES them exactly like the old identical boxes — a few
            # discrete size classes = a few draw batches, independent of leaf
            # count (a continuous per-leaf size would make every leaf a unique
            # geometry and tank the fps).  Colour is a per-shape attribute (no
            # instancing cost), so each leaf gets its own green.  The blades are
            # massless & non-colliding and move rigidly with the branch, so
            # foliage stays PURELY VISUAL: zero physics cost.
            leaf_cfg = builder.ShapeConfig(density=0.0, mu=0.5, collision_group=0)
            meshes = _foliage.leaf_meshes(fp)
            leaf_rng = np.random.default_rng(config.seed + 313)
            ncls = len(meshes)
            for lp in placements:
                pseg = skeleton[lp.parent_seg]
                pbody = int(seg_to_body[lp.parent_seg])
                cpos = _qrot(_qconj(pseg.frame), lp.attach - pseg.start)
                cquat = _qmul(_qconj(pseg.frame), lp.frame)
                # per-leaf tone: brightness + a touch of yellow, around leaf_col
                b = float(np.clip(leaf_rng.normal(1.0, 0.16), 0.6, 1.5))
                y = float(np.clip(leaf_rng.normal(1.0, 0.14), 0.7, 1.5))
                col = (float(np.clip(leaf_col[0] * b * y, 0.0, 1.0)),
                       float(np.clip(leaf_col[1] * b, 0.0, 1.0)),
                       float(np.clip(leaf_col[2] * b / y, 0.0, 1.0)))
                builder.add_shape_mesh(
                    pbody, xform=wp.transform(p=_wv(cpos), q=_wq(cquat)),
                    mesh=meshes[int(leaf_rng.integers(0, ncls))],
                    cfg=leaf_cfg, color=col)

    builder.add_articulation(all_joint_ids, label="tree")

    # --- optional apples: each is an INDEPENDENT free-floating rigid body (its
    #     own 1-body articulation), deliberately NOT jointed into the tree.  A
    #     runtime one-sided spring-damper tether (treesim.fruit.AppleField) holds
    #     it at its hang point under the spur and a hard pull cuts that tether so
    #     the apple free-falls.  Because the apple is never connected to the
    #     compliant branches by a joint, its (heavy) mass can't destabilise the
    #     thin spur, and "detaching" is a flag — no model edit, notify or
    #     graph-recapture.  Built AFTER the tree articulation so joint indices
    #     stay contiguous (add_articulation requires that). ------------------- #
    apple_bodies: list[int] = []
    apple_jids: list[int] = []
    ap_parent: list[int] = []
    ap_offset: list[list[float]] = []
    ap_drop: list[float] = []
    ap_detach: list[float] = []
    if config.fruit.enabled and deformable:
        from . import fruit as _fruit
        fr = config.fruit
        apple_rng = np.random.default_rng(config.seed + 99)
        _aplace = (apple_placements if apple_placements is not None
                   else _fruit.place_apples(skeleton, fr, seed=config.seed))
        slide = getattr(fr, "joint", "slide") == "slide"
        for ap in _aplace:
            acfg = builder.ShapeConfig(
                density=max(fr.mass / ((4.0 / 3.0) * np.pi * ap.radius ** 3), 50.0), mu=1.0,
                collision_group=-1)   # apples don't self-collide / hit branches; grippy skin
            pseg = skeleton[ap.parent_seg]
            pbody = int(seg_to_body[ap.parent_seg])
            drop = fr.stem_length + ap.radius            # apple centre hangs this far below attach
            hang = ap.attach + np.array([0.0, 0.0, -drop])
            # body whose ORIGIN is the apple centre (sphere at local origin),
            # spawned already at its hang position so it starts attached-looking
            abody = builder.add_link(
                xform=wp.transform(p=_wv(hang), q=wp.quat_identity()),
                label=f"apple{len(apple_bodies)}")
            builder.add_shape_sphere(
                abody, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                radius=ap.radius, cfg=acfg, color=ap.color)
            if slide:
                # 3 translational DOFs to the WORLD (an apple never needs to
                # spin): HALF the dofs of a free joint, and the free-body dof
                # count is the dominant sim cost.  No actuators (mode NONE), no
                # limits; the runtime tether + gravity do everything.  Parented
                # to the world (fixed base), NOT to the compliant spur — a
                # heavy child on a thin spur is what used to blow up.
                fdof = lambda ax: builder.JointDofConfig(
                    axis=ax, limit_lower=-1.0e9, limit_upper=1.0e9,
                    limit_ke=0.0, limit_kd=0.0, target_ke=0.0, target_kd=0.0,
                    actuator_mode=newton.JointTargetMode.NONE)
                ajid = builder.add_joint_d6(
                    parent=-1, child=abody,
                    linear_axes=[fdof(X), fdof(Y), fdof(Z)],
                    parent_xform=wp.transform(p=_wv(hang), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
            else:
                ajid = builder.add_joint_free(child=abody)
            # each apple is its OWN articulation so the viewer pick clamp uses
            # the apple's own mass (keeps detach_force thresholds meaningful)
            builder.add_articulation([ajid], label=f"apple{len(apple_bodies)}")
            apple_jids.append(ajid)
            apple_bodies.append(abody)
            colors.append(ap.color)
            # attach point in the PARENT body's local frame, so the runtime can
            # track it as the spur sways:  attach_world = X_parent * offset
            off = _qrot(_qconj(pseg.frame), ap.attach - pseg.start)
            ap_parent.append(pbody)
            ap_offset.append([float(off[0]), float(off[1]), float(off[2])])
            ap_drop.append(float(drop))
            ap_detach.append(float(apple_rng.uniform(*fr.detach_force)))

    # --- optional RidgebackFranka mobile manipulator (one per env; identical
    #     in every env so the worlds stay homogeneous and keep batching) ------- #
    robot_maps = None
    if getattr(config, "robot", None) is not None and config.robot.enabled:
        from . import robot as _robot
        robot_maps = _robot.build_robot(builder, config.robot)
        # keep the per-body colour table aligned (chassis + arm links)
        for _ in range(robot_maps["nbody"]):
            colors.append((0.45, 0.45, 0.47))

    # ---- domain-randomized multi-env: hand the un-finalized builder + its
    #      env-relative index maps back so _assemble_dr can stitch several
    #      differently-randomized (but structurally identical) sub-builders into
    #      parallel worlds.  No ground plane / finalize here (the main builder
    #      owns those). ---------------------------------------------------------
    if _sub:
        maps = dict(
            nbody=builder.body_count, njoint=builder.joint_count, n=n,
            ndof=ndof, bend_mask=bend_mask,
            seg_to_body=seg_to_body,
            joint_ids=joint_ids, joint_seg=joint_seg,
            joint_kp=joint_kp, joint_kd=joint_kd, rupture=rupture,
            colors=colors, leaf_bodies=leaf_bodies,
            apple_bodies=apple_bodies, ap_parent=ap_parent, ap_offset=ap_offset,
            ap_drop=ap_drop, ap_detach=ap_detach,
            brk_jid=brk_jid, brk_parent=brk_parent, brk_child=brk_child,
            brk_kp=brk_kp, brk_mmax=brk_mmax, brk_ndof=brk_ndof,
            brk_descend=brk_descend,
            robot=robot_maps,
        )
        return builder, maps

    # ---- finalize, optionally REPLICATED into `num_envs` parallel worlds -------
    # (IsaacLab-style batching: mjwarp steps every world in one GPU call.  The
    #  tree is a deep serial chain so one tree barely uses the GPU; replicas run
    #  almost for free until it saturates.)  The ground plane is global (world -1).
    num_envs = max(int(num_envs), 1)
    aabbs = [_builder_aabb(builder)]
    if env_aabb is not None:
        aabbs.append(env_aabb)      # DR headroom: union over the perturbed skeletons
    env_pitch, env_cols, env_rows = _env_grid(aabbs, num_envs)
    terrain_fn = None
    if num_envs == 1:
        builder.add_ground_plane()
        if getattr(config.physics, "terrain", False):
            terrain_fn = _add_terrain(builder, config.physics, config.seed)
        model = builder.finalize(device=config.device)
    else:
        main = newton.ModelBuilder()
        main.gravity = config.physics.gravity
        # Every world sits at the SAME physics origin (newton's own advice:
        # better conditioning, and the one global terrain/ground is then valid
        # for every env).  The viewer spreads them on the display grid.
        main.replicate(builder, world_count=num_envs)
        main.add_ground_plane()
        if getattr(config.physics, "terrain", False):
            terrain_fn = _add_terrain(main, config.physics, config.seed,
                                      pitch=env_pitch, grid=(env_cols, env_rows))
        model = main.finalize(device=config.device)

    # per-env body/joint counts (ground is global, adds neither), for offsetting
    nbpe = model.body_count // num_envs
    njpe = model.joint_count // num_envs

    def _tile_idx(arr, per_env):
        a = np.asarray(arr, dtype=np.int64)
        if a.size == 0:
            return a.astype(int)
        return np.concatenate([a + w * per_env for w in range(num_envs)]).astype(int)

    def _tile_rep(arr):
        a = np.asarray(arr)
        return np.tile(a, (num_envs,) + (1,) * (a.ndim - 1)) if a.size else a

    # runtime (MuJoCo) mappings replicated across every world
    dof_start_all = model.joint_target_q_start.numpy()
    brk_jid_all = _tile_idx(brk_jid, njpe)
    brk_dof_start = dof_start_all[brk_jid_all].astype(int) if brk_jid_all.size else np.zeros(0, dtype=int)

    apple_data = None
    apple_bodies_all = _tile_idx(apple_bodies, nbpe).tolist()
    if apple_bodies:
        apple_data = dict(
            apple_body=np.asarray(apple_bodies_all, dtype=np.int32),
            parent_body=_tile_idx(ap_parent, nbpe).astype(np.int32),
            offset=_tile_rep(np.asarray(ap_offset, dtype=np.float32)),
            hang_drop=_tile_rep(np.asarray(ap_drop, dtype=np.float32)),
            detach_force=_tile_rep(np.asarray(ap_detach, dtype=np.float32)),
        )

    robot_data = None
    if robot_maps is not None:
        # translate joint ids -> final-model coordinate/dof/target indices
        # (builder-time starts do NOT survive finalize in general)
        jq_s = model.joint_q_start.numpy()
        jqd_s = model.joint_qd_start.numpy()
        jtq_s = model.joint_target_q_start.numpy()
        pj = robot_maps["planar_joint"]
        ajs = robot_maps["arm_jids"]
        robot_data = dict(
            chassis=_tile_idx([robot_maps["chassis"]], nbpe).tolist(),
            wrist=_tile_idx([robot_maps["wrist"]], nbpe).tolist(),
            planar_q=[int(jq_s[pj + w * njpe]) for w in range(num_envs)],
            planar_dof=[int(jqd_s[pj + w * njpe]) for w in range(num_envs)],
            planar_tq=[int(jtq_s[pj + w * njpe]) for w in range(num_envs)],
            arm_tq=[[int(jtq_s[j + w * njpe]) for j in ajs] for w in range(num_envs)],
            arm_dofs=[[int(jqd_s[j + w * njpe]) for j in ajs] for w in range(num_envs)],
            arm_home=list(robot_maps["arm_home"]),
        )

    brk_descend_all = [[b + w * nbpe for b in d] for w in range(num_envs) for d in brk_descend]

    tm = TreeModel(
        model=model, config=config, skeleton=skeleton,
        num_envs=num_envs,
        seg_to_body=seg_to_body,                     # env-relative (diagnostics/spring only)
        joint_ids=joint_ids,
        joint_seg=np.array(joint_seg, dtype=int),
        joint_kp=np.array(joint_kp, dtype=float) if joint_kp else np.zeros((0, ndof)),
        joint_kd=np.array(joint_kd, dtype=float) if joint_kd else np.zeros((0, ndof)),
        joint_dof_start=(dof_start_all[_tile_idx(joint_ids, njpe)].astype(int) if joint_ids else np.zeros(0, dtype=int)),
        joint_ndof=ndof,
        bend_mask=bend_mask,
        rupture=np.array(rupture, dtype=float) if rupture else np.zeros(0),
        base_color=_tile_rep(np.array(colors, dtype=float)),
        n_bodies=n,
        leaf_bodies=_tile_idx(leaf_bodies, nbpe).tolist(),
        apple_bodies=apple_bodies_all,
        apple_data=apple_data,
        robot_data=robot_data,
        terrain_height=terrain_fn,
        env_pitch=env_pitch, env_cols=env_cols,
        brk_joint_id=_tile_idx(brk_jid, njpe),
        brk_parent_body=_tile_idx(brk_parent, nbpe),
        brk_child_body=_tile_idx(brk_child, nbpe),
        brk_kp=_tile_rep(np.array(brk_kp, dtype=float)),
        brk_mmax=_tile_rep(np.array(brk_mmax, dtype=float)),
        brk_dof_start=brk_dof_start,
        brk_dof_count=_tile_rep(np.array(brk_ndof, dtype=int)),
        brk_descend=brk_descend_all,
    )
    return tm


def _assemble_dr(config: TreeConfig, base_skeleton: TreeSkeleton, subs,
                 num_envs: int) -> TreeModel:
    """Stitch ``num_envs`` structurally-identical but continuously-randomized
    sub-builders into ONE batched model (IsaacLab-style separate worlds) and
    concatenate their env-relative index maps with the right per-world offsets.

    Every world has the same body/joint/shape counts and types (only the numeric
    values differ), so ``SolverMuJoCo`` batches them on the GPU exactly like the
    identical-replica path — the per-env variety is "free" for physics.  Index
    maps (body/joint ids) are the same env-relative values shifted by each
    world's offset (``cat_i``); value maps (stiffness, rupture, colours, apple
    tether params) are the genuinely-different per-env values, concatenated
    verbatim (``cat_r``).
    """
    main = newton.ModelBuilder()
    main.gravity = config.physics.gravity
    for e, (b, _m) in enumerate(subs):
        # Every world sits at the SAME local origin (worlds are independent, batched
        # copies — they never interact), exactly like ModelBuilder.replicate for the
        # identical path.  The GL viewer spreads them onto a grid visually via its
        # own per-world offsets, so we must NOT also bake a grid into body_q here
        # (that would double-offset and fling the trees out of frame).  A distinct
        # label_prefix keeps each world's shapes uniquely named.
        main.add_world(b, label_prefix=f"env{e}_")
    main.add_ground_plane()
    env_pitch, env_cols, env_rows = _env_grid(
        [_builder_aabb(b) for (b, _m) in subs], num_envs)
    terrain_fn = None
    if getattr(config.physics, "terrain", False):
        terrain_fn = _add_terrain(main, config.physics, config.seed,
                                  pitch=env_pitch, grid=(env_cols, env_rows))
    model = main.finalize(device=config.device)

    env_maps = [m for (_, m) in subs]
    nb = [int(m["nbody"]) for m in env_maps]
    nj = [int(m["njoint"]) for m in env_maps]
    body_off = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
    joint_off = np.concatenate([[0], np.cumsum(nj)[:-1]]).astype(np.int64)

    def cat_i(key, off):
        parts = [np.asarray(env_maps[e][key], dtype=np.int64) + off[e]
                 for e in range(num_envs) if len(env_maps[e][key])]
        return np.concatenate(parts).astype(int) if parts else np.zeros(0, dtype=int)

    def cat_r(key):
        parts = [np.asarray(env_maps[e][key]) for e in range(num_envs)
                 if len(env_maps[e][key])]
        return np.concatenate(parts) if parts else np.asarray(env_maps[0][key])

    m0 = env_maps[0]
    ndof0 = m0["ndof"]

    brk_joint_id = cat_i("brk_jid", joint_off)
    dof_start_all = model.joint_target_q_start.numpy()
    brk_dof_start = (dof_start_all[brk_joint_id].astype(int) if brk_joint_id.size
                     else np.zeros(0, dtype=int))

    apple_bodies_all = cat_i("apple_bodies", body_off).tolist()
    apple_data = None
    if apple_bodies_all:
        ap_off = np.concatenate([
            np.asarray(env_maps[e]["ap_offset"], dtype=np.float32).reshape(-1, 3)
            for e in range(num_envs) if len(env_maps[e]["ap_offset"])])
        apple_data = dict(
            apple_body=np.asarray(apple_bodies_all, dtype=np.int32),
            parent_body=cat_i("ap_parent", body_off).astype(np.int32),
            offset=ap_off,
            hang_drop=cat_r("ap_drop").astype(np.float32),
            detach_force=cat_r("ap_detach").astype(np.float32),
        )

    brk_descend_all = [[b + int(body_off[e]) for b in d]
                       for e in range(num_envs) for d in env_maps[e]["brk_descend"]]

    robot_data = None
    if env_maps[0].get("robot") is not None:
        jq_s = model.joint_q_start.numpy()
        jqd_s = model.joint_qd_start.numpy()
        jtq_s = model.joint_target_q_start.numpy()
        rms = [m["robot"] for m in env_maps]
        robot_data = dict(
            chassis=[rms[e]["chassis"] + int(body_off[e]) for e in range(num_envs)],
            wrist=[rms[e]["wrist"] + int(body_off[e]) for e in range(num_envs)],
            planar_q=[int(jq_s[rms[e]["planar_joint"] + int(joint_off[e])])
                      for e in range(num_envs)],
            planar_dof=[int(jqd_s[rms[e]["planar_joint"] + int(joint_off[e])])
                        for e in range(num_envs)],
            planar_tq=[int(jtq_s[rms[e]["planar_joint"] + int(joint_off[e])])
                       for e in range(num_envs)],
            arm_tq=[[int(jtq_s[j + int(joint_off[e])]) for j in rms[e]["arm_jids"]]
                    for e in range(num_envs)],
            arm_dofs=[[int(jqd_s[j + int(joint_off[e])]) for j in rms[e]["arm_jids"]]
                      for e in range(num_envs)],
            arm_home=list(rms[0]["arm_home"]),
        )

    joint_ids_all = cat_i("joint_ids", joint_off)
    return TreeModel(
        model=model, config=config, skeleton=base_skeleton, num_envs=num_envs,
        seg_to_body=np.asarray(m0["seg_to_body"]),   # env-0 (diagnostics/spring only)
        joint_ids=list(m0["joint_ids"]),
        joint_seg=np.asarray(m0["joint_seg"], dtype=int),
        joint_kp=np.asarray(m0["joint_kp"], dtype=float) if m0["joint_kp"] else np.zeros((0, ndof0)),
        joint_kd=np.asarray(m0["joint_kd"], dtype=float) if m0["joint_kd"] else np.zeros((0, ndof0)),
        joint_dof_start=(dof_start_all[joint_ids_all].astype(int)
                         if joint_ids_all.size else np.zeros(0, dtype=int)),
        joint_ndof=ndof0,
        bend_mask=m0["bend_mask"],
        rupture=np.asarray(m0["rupture"], dtype=float) if len(m0["rupture"]) else np.zeros(0),
        base_color=cat_r("colors"),
        n_bodies=m0["n"],
        leaf_bodies=cat_i("leaf_bodies", body_off).tolist(),
        apple_bodies=apple_bodies_all,
        apple_data=apple_data,
        robot_data=robot_data,
        terrain_height=terrain_fn,
        env_pitch=env_pitch, env_cols=env_cols,
        brk_joint_id=brk_joint_id,
        brk_parent_body=cat_i("brk_parent", body_off),
        brk_child_body=cat_i("brk_child", body_off),
        brk_kp=cat_r("brk_kp").astype(float),
        brk_mmax=cat_r("brk_mmax").astype(float),
        brk_dof_start=brk_dof_start,
        brk_dof_count=cat_r("brk_ndof").astype(int),
        brk_descend=brk_descend_all,
    )


def _capsule_mass_inertia(r, L, rho):
    """Vectorized capsule (axis z) mass/inertia about its COM — exactly
    newton's convention (validated to 6 digits against ModelBuilder)."""
    r = np.asarray(r, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    m_cyl = rho * np.pi * r * r * L
    m_cap = rho * (4.0 / 3.0) * np.pi * r ** 3          # both hemispherical caps
    m = m_cyl + m_cap
    izz = 0.5 * m_cyl * r * r + 0.4 * m_cap * r * r
    ixx = (m_cyl * (L * L / 12.0 + 0.25 * r * r)
           + m_cap * (0.4 * r * r + 0.25 * L * L + 0.375 * L * r))
    return m, ixx, izz


def _dr_fast(config: TreeConfig, base, num_envs: int, drp,
             max_bodies: int) -> TreeModel:
    """Fast domain randomization: build ONE base env, ``replicate`` it (array
    tiling, seconds instead of a minute at 50 envs), then PATCH the per-env
    continuous values straight into the finalized model arrays — joint rest
    transforms, capsule/leaf/apple shape scales & transforms, masses/inertias,
    joint gains, colours, tether/rupture tables.  Exactly the values the slow
    per-env-builder path produces (same RNG streams, validated element-wise),
    at ~5x the startup speed, and cross-env GL instancing survives because all
    envs share the base geometry assets."""
    from . import domainrand as _dr
    from . import fruit as _fruit
    from . import foliage as _foliage
    from . import physics as _physics

    canon_apples = (_fruit.place_apples(base, config.fruit, seed=config.seed)
                    if (config.fruit.enabled and config.deformable) else None)
    canon_leaves = (_foliage.place_leaves(base, config.foliage, seed=config.seed)
                    if config.foliage.enabled else None)

    # the display pitch (viewer grid + terrain tiling) must fit the LARGEST
    # perturbed env, not the base tree — union the perturbed skeleton AABBs
    # (fresh env_rng instances, so the patch loop's streams are untouched)
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for e in range(num_envs):
        d_rng = None if drp.randomize_geometry else _dr.dim_rng(config.seed)
        b0_, b1_ = _dr.perturb_skeleton(base, _dr.env_rng(config.seed, e), drp,
                                        dim_rng=d_rng).bounds()
        lo = np.minimum(lo, np.asarray(b0_, dtype=float)[:2])
        hi = np.maximum(hi, np.asarray(b1_, dtype=float)[:2])

    tm = build(config, base, max_bodies=max_bodies, num_envs=num_envs,
               apple_placements=canon_apples, leaf_placements=canon_leaves,
               env_aabb=(lo, hi))
    model = tm.model
    n_seg = len(base)
    n_leaf = len(canon_leaves) if canon_leaves else 0
    n_ap = len(canon_apples) if canon_apples else 0
    jpe = model.joint_count // num_envs
    spe = model.shape_count // num_envs
    bpe = model.body_count // num_envs
    qd_start = model.joint_qd_start.numpy()

    jxp = model.joint_X_p.numpy()
    sxf = model.shape_transform.numpy()
    ssc = model.shape_scale.numpy()
    scol = model.shape_color.numpy()
    bm = model.body_mass.numpy()
    bi = model.body_inertia.numpy()
    bcom = model.body_com.numpy()
    tke = model.joint_target_ke.numpy()
    tkd = model.joint_target_kd.numpy()

    limp = (config.breaking.enabled
            and not getattr(config.breaking, "free_fall", False))
    brk_kp_all, brk_mmax_all = [], []
    ap_off_all, ap_drop_all, ap_det_all = [], [], []

    for e in range(num_envs):
        rng = _dr.env_rng(config.seed, e)
        d_rng = None if drp.randomize_geometry else _dr.dim_rng(config.seed)
        skel = _dr.perturb_skeleton(base, rng, drp, dim_rng=d_rng)
        cfg_e = _dr.perturb_config(config, rng, drp, e)
        ap_e = (_dr.remap_apples(canon_apples, base, skel, rng, drp)
                if canon_apples is not None else [])
        lf_e = (_dr.remap_leaves(canon_leaves, base, skel)
                if canon_leaves is not None else [])
        rho = cfg_e.physics.wood_density
        tint = cfg_e.render.wood_tint
        max_order = max(s.order for s in skel.segments)

        j0, s0, b0 = e * jpe, e * spe, e * bpe
        # --- branch capsules: geometry, mass, colour -----------------------
        L = np.maximum([s.length for s in skel.segments], 1e-3)
        R = np.maximum([s.mean_radius for s in skel.segments], 5e-4)
        m, ixx, izz = _capsule_mass_inertia(R, L, rho)
        idx = np.arange(n_seg)
        ssc[s0 + idx] = np.stack([R, 0.5 * L, np.zeros(n_seg)], axis=1)
        sxf[s0 + idx, :3] = np.stack([np.zeros(n_seg), np.zeros(n_seg), 0.5 * L], axis=1)
        bm[b0 + idx] = m
        bcom[b0 + idx] = np.stack([np.zeros(n_seg), np.zeros(n_seg), 0.5 * L], axis=1)
        bi[b0 + idx] = 0.0
        bi[b0 + idx, 0, 0] = ixx
        bi[b0 + idx, 1, 1] = ixx
        bi[b0 + idx, 2, 2] = izz
        scol[s0 + idx] = [_wood_color(s.order, max_order, tint) for s in skel.segments]

        # --- joints: rest transforms + stiffness ---------------------------
        root = skel.roots[0]
        jxp[j0 + 0] = (*root.start, *root.frame)
        srng = np.random.default_rng(cfg_e.seed)     # matches builder's rng use
        jid = 1
        for seg in skel.segments:
            if seg.parent < 0:
                continue
            parent = skel[seg.parent]
            q_rel = _qmul(_qconj(parent.frame), seg.frame)
            jxp[j0 + jid] = (0.0, 0.0, max(parent.length, 1e-3), *q_rel)
            kp, kd = _physics.stiffness_for(seg, cfg_e.physics, srng)
            if limp:
                kd = config.breaking.limp_damping_ratio * kp
            d0 = qd_start[j0 + jid]
            tke[d0:d0 + 2] = kp
            tkd[d0:d0 + 2] = kd
            brk_kp_all.append(kp)
            brk_mmax_all.append(_physics.rupture_moment(seg, cfg_e.breaking)
                                if config.breaking.enabled else float("inf"))
            jid += 1

        # --- leaves: re-homed transforms, per-env size, per-leaf colour ----
        if n_leaf:
            leaf_rng = np.random.default_rng(cfg_e.seed + 313)
            lc = cfg_e.foliage.leaf_color
            srel = cfg_e.foliage.leaf_length / max(config.foliage.leaf_length, 1e-9)
            for k, lp in enumerate(lf_e):
                pseg = skel[lp.parent_seg]
                cpos = _qrot(_qconj(pseg.frame), lp.attach - pseg.start)
                cquat = _qmul(_qconj(pseg.frame), lp.frame)
                si = s0 + n_seg + k
                sxf[si] = (*cpos, *cquat)
                ssc[si] = (srel, srel, srel)
                b = float(np.clip(leaf_rng.normal(1.0, 0.16), 0.6, 1.5))
                y = float(np.clip(leaf_rng.normal(1.0, 0.14), 0.7, 1.5))
                leaf_rng.integers(0, 3)              # keep the class draw in sync
                scol[si] = (np.clip(lc[0] * b * y, 0, 1), np.clip(lc[1] * b, 0, 1),
                            np.clip(lc[2] * b / y, 0, 1))

        # --- apples: hang joints, size, mass, colour, tether table ---------
        if n_ap:
            apple_rng = np.random.default_rng(cfg_e.seed + 99)
            for k, ap in enumerate(ap_e):
                pseg = skel[ap.parent_seg]
                drop = config.fruit.stem_length + ap.radius
                hang = ap.attach + np.array([0.0, 0.0, -drop])
                jxp[j0 + jid + k] = (*hang, 0.0, 0.0, 0.0, 1.0)
                si = s0 + n_seg + n_leaf + k
                base_r = canon_apples[k].radius
                ssc[si] = ssc[si] * (ap.radius / max(base_r, 1e-9))
                scol[si] = ap.color
                ab = b0 + (tm.apple_bodies[k] % bpe)
                bm[ab] = cfg_e.fruit.mass
                bi[ab] = 0.0
                iap = 0.4 * cfg_e.fruit.mass * ap.radius ** 2
                bi[ab, 0, 0] = bi[ab, 1, 1] = bi[ab, 2, 2] = iap
                bcom[ab] = 0.0
                off = _qrot(_qconj(pseg.frame), ap.attach - pseg.start)
                ap_off_all.append(off.astype(np.float32))
                ap_drop_all.append(drop)
                ap_det_all.append(float(apple_rng.uniform(*config.fruit.detach_force)))

    # push everything to the device (contents only; solver not created yet)
    model.joint_X_p.assign(jxp)
    model.shape_transform.assign(sxf)
    model.shape_scale.assign(ssc)
    model.shape_color.assign(scol)
    model.body_mass.assign(bm)
    model.body_com.assign(bcom)
    model.body_inv_mass.assign(np.where(bm > 0, 1.0 / np.maximum(bm, 1e-12), 0.0))
    model.body_inertia.assign(bi)
    binv = np.zeros_like(bi)
    for a in range(3):
        d = bi[:, a, a]
        binv[:, a, a] = np.where(d > 0, 1.0 / np.maximum(d, 1e-15), 0.0)
    model.body_inv_inertia.assign(binv)
    model.joint_target_ke.assign(tke)
    model.joint_target_kd.assign(tkd)

    # per-env value tables on the TreeModel
    tm.brk_kp = np.asarray(brk_kp_all, dtype=float)
    tm.brk_mmax = np.asarray(brk_mmax_all, dtype=float)
    tm.base_color = scol.copy()
    if n_ap and tm.apple_data is not None:
        tm.apple_data["offset"] = np.asarray(ap_off_all, dtype=np.float32)
        tm.apple_data["hang_drop"] = np.asarray(ap_drop_all, dtype=np.float32)
        tm.apple_data["detach_force"] = np.asarray(ap_det_all, dtype=np.float32)
    return tm


def generate_and_build(config: TreeConfig, max_bodies: int = 4000,
                       num_envs: int = 1, randomize_envs: bool = False,
                       dr_strength: float = 1.0,
                       randomize_geometry: bool = False) -> TreeModel:
    from . import lsystem
    import os
    num_envs = max(int(num_envs), 1)
    base = lsystem.generate(config.lsystem, seed=config.seed)
    if num_envs == 1 or not randomize_envs:
        return build(config, base, max_bodies=max_bodies, num_envs=num_envs)

    # Per-env DOMAIN RANDOMIZATION: every env is the SAME tree topology (so the
    # worlds stay homogeneous and batch on the GPU) but with continuously
    # randomized geometry, material, fruit and foliage — a whole stand of
    # different-looking apple trees for "free" parallel physics.
    from . import domainrand as _dr
    from . import fruit as _fruit
    from . import foliage as _foliage
    drp = _dr.DRParams(strength=max(float(dr_strength), 0.0),
                       randomize_geometry=bool(randomize_geometry))

    # DEVICE path (opt-in, TREESIM_DR_DEVICE=1): perturb every world's wood
    # geometry in ONE Warp kernel on the GPU instead of an O(N) host re-walk, so
    # the build no longer scales with the env count.  Wood only (fruit/foliage
    # re-homing stays on the host path); see treesim/domainrand_gpu.py.  It only
    # makes sense for DISTINCT per-world geometry; with shared dimensions (the
    # default) there is nothing per-world to perturb, so fall through to the fast
    # replicate-and-patch path, which builds one shape and tiles it.
    if randomize_geometry and os.environ.get("TREESIM_DR_DEVICE"):
        from . import domainrand_gpu as _drg
        tm = build(config, base, max_bodies=max_bodies, num_envs=num_envs)
        _drg.patch_device(tm.model, base, num_envs, drp, config)
        return tm

    # FAST path (default): replicate once + patch per-env values into the
    # finalized arrays (~5x faster startup at 50 envs, instancing preserved).
    # TREESIM_SLOW_DR=1 forces the original per-env-builder path (validation).
    if not os.environ.get("TREESIM_SLOW_DR"):
        return _dr_fast(config, base, num_envs, drp, max_bodies)

    # Freeze the apple / leaf SET once on the base skeleton (fixes counts + which
    # spurs bear fruit) so every env is structurally identical; each env just
    # re-homes them onto its own perturbed geometry and jitters their size/colour.
    canon_apples = (_fruit.place_apples(base, config.fruit, seed=config.seed)
                    if (config.fruit.enabled and config.deformable) else None)
    canon_leaves = (_foliage.place_leaves(base, config.foliage, seed=config.seed)
                    if config.foliage.enabled else None)

    subs = []
    for e in range(num_envs):
        rng = _dr.env_rng(config.seed, e)
        d_rng = None if drp.randomize_geometry else _dr.dim_rng(config.seed)
        skel_e = _dr.perturb_skeleton(base, rng, drp, dim_rng=d_rng)
        cfg_e = _dr.perturb_config(config, rng, drp, e)
        ap_e = (_dr.remap_apples(canon_apples, base, skel_e, rng, drp)
                if canon_apples is not None else None)
        lf_e = (_dr.remap_leaves(canon_leaves, base, skel_e)
                if canon_leaves is not None else None)
        subs.append(build(cfg_e, skel_e, max_bodies=max_bodies, num_envs=1,
                          apple_placements=ap_e, leaf_placements=lf_e, _sub=True))
    return _assemble_dr(config, base, subs, num_envs)
