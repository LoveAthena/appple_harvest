"""On-device (GPU) domain randomization for the tree geometry.

The host-side DR path (`builder._dr_fast`) runs a Python forward-kinematic
re-walk of the skeleton once per environment, so its build cost is ``O(N)`` in
the number of worlds and dominates startup past a few tens of envs.  This module
does the same per-world perturbation in a single Warp kernel, parallel across
environments: one thread per world walks the (parent-before-child) segment list,
applies that world's random geometry perturbation, and writes the finalized
model arrays (joint rest transforms, capsule scales/transforms, masses/inertias,
joint stiffness, colours) directly on the device.  Build then costs one kernel
launch instead of ``N`` Python re-walks, and, because the worlds stay
structurally identical, the per-step simulation cost is unchanged.

Scope: this randomizes the WOOD (branch geometry + articulation), which is the
part whose FK re-walk is the ``O(N)`` bottleneck.  Fruit/foliage re-homing would
follow the same pattern and is left to the host path when enabled.

Validated by the ``strength=0`` identity: with no perturbation the kernel must
reproduce the replicated base model element-wise (see ``scripts`` test).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .config import TreeConfig, StiffnessModel
from . import domainrand as _dr

_ZUP = wp.constant(wp.vec3(0.0, 0.0, 1.0))
_ZDN = wp.constant(wp.vec3(0.0, 0.0, -1.0))
_BARK0 = wp.constant(wp.vec3(0.36, 0.24, 0.14))     # dark bark brown (trunk)
_BARK1 = wp.constant(wp.vec3(0.55, 0.41, 0.24))     # lighter outer wood (tips)


@wp.struct
class DRGpu:
    strength: float
    scale_lo: float; scale_hi: float
    rad_bias_lo: float; rad_bias_hi: float
    len_sigma: float; len_lo: float; len_hi: float
    rad_sigma: float; rad_lo: float; rad_hi: float
    ang_sigma: float                      # radians (already x strength on host)
    droop_lo: float; droop_hi: float; droop_base: float   # droop_base in radians
    spread_lo: float; spread_hi: float
    twist_max: float; lean_max: float     # radians (pre-strength)
    dens_lo: float; dens_hi: float
    mod_lo: float; mod_hi: float
    wood_bright_lo: float; wood_bright_hi: float
    wood_warm_lo: float; wood_warm_hi: float
    # material / model
    density: float; youngs: float; beam_factor: float
    damping_ratio: float; min_stiff: float; max_stiff: float
    clearance: float


@wp.func
def _u(r: float, lo: float, hi: float, s: float):
    """Map a uniform draw ``r`` in [0,1) onto [lo,hi] widened by ``strength``."""
    lo2 = 1.0 - (1.0 - lo) * s
    hi2 = 1.0 + (hi - 1.0) * s
    return lo2 + (hi2 - lo2) * r


@wp.func
def _nclip(z: float, sigma: float, lo: float, hi: float, s: float):
    """Map a standard-normal draw ``z`` onto ``1 + z*sigma*s`` clipped to [lo,hi]."""
    v = 1.0 + z * sigma * s
    lo2 = 1.0 - (1.0 - lo) * s
    hi2 = 1.0 + (hi - 1.0) * s
    return wp.clamp(v, lo2, hi2)


@wp.kernel
def _dr_walk(
    # --- base per-segment data (shared by all envs) ---
    seg_parent: wp.array(dtype=wp.int32),
    seg_order: wp.array(dtype=wp.int32),
    seg_fork: wp.array(dtype=wp.int32),
    seg_joint: wp.array(dtype=wp.int32),
    base_qrel: wp.array(dtype=wp.quat),      # root: base world frame; child: rel rot
    base_start: wp.array(dtype=wp.vec3),     # only the root's is used
    base_len: wp.array(dtype=wp.float32),
    base_r0: wp.array(dtype=wp.float32),
    base_r1: wp.array(dtype=wp.float32),
    n_seg: wp.int32, max_order: wp.int32,
    spe: wp.int32, bpe: wp.int32, jpe: wp.int32, seed: wp.int32,
    dr: DRGpu,
    # --- scratch (per env x per seg) ---
    nf: wp.array(dtype=wp.quat, ndim=2),
    ne: wp.array(dtype=wp.vec3, ndim=2),
    nl: wp.array(dtype=wp.float32, ndim=2),
    # --- model outputs ---
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_X_p: wp.array(dtype=wp.transform),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    shape_color: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=wp.float32),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    joint_target_ke: wp.array(dtype=wp.float32),
    joint_target_kd: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    s = dr.strength
    st = wp.rand_init(seed, e)

    # whole-tree draws (all randf/randn called in the kernel body so the RNG
    # state advances; helpers take the drawn value)
    scale = _u(wp.randf(st), dr.scale_lo, dr.scale_hi, s)
    rad_bias = _u(wp.randf(st), dr.rad_bias_lo, dr.rad_bias_hi, s)
    droop = _u(wp.randf(st), dr.droop_lo, dr.droop_hi, s)
    droop_gain = (droop - 1.0) * dr.droop_base
    spread = _u(wp.randf(st), dr.spread_lo, dr.spread_hi, s)
    twist = (2.0 * wp.randf(st) - 1.0) * dr.twist_max * s
    lean_ang = wp.randf(st) * dr.lean_max * s
    lean_az = wp.randf(st) * 6.2831853
    q_lean = wp.quat_from_axis_angle(
        wp.vec3(wp.cos(lean_az), wp.sin(lean_az), 0.0), lean_ang)
    E_e = dr.youngs * _u(wp.randf(st), dr.mod_lo, dr.mod_hi, s)
    rho_e = dr.density * _u(wp.randf(st), dr.dens_lo, dr.dens_hi, s)
    bright = _u(wp.randf(st), dr.wood_bright_lo, dr.wood_bright_hi, s)
    warm = _u(wp.randf(st), dr.wood_warm_lo, dr.wood_warm_hi, s)

    root_jnt = int(0)
    root_pos = wp.vec3(0.0, 0.0, 0.0)
    root_rot = wp.quat_identity()
    min_z = float(1.0e9)

    for i in range(n_seg):
        p = seg_parent[i]
        f_len = _nclip(wp.randn(st), dr.len_sigma, dr.len_lo, dr.len_hi, s)
        f_rad = _nclip(wp.randn(st), dr.rad_sigma, dr.rad_lo, dr.rad_hi, s)
        length = base_len[i] * f_len * scale
        # small random relative bend about a random axis
        jq = wp.quat_identity()
        if dr.ang_sigma > 0.0:
            ang = wp.randn(st) * dr.ang_sigma
            ax = wp.vec3(wp.randn(st), wp.randn(st), wp.randn(st))
            if wp.length(ax) > 1.0e-9:
                jq = wp.quat_from_axis_angle(wp.normalize(ax), ang)

        frame = wp.quat_identity()
        start = wp.vec3(0.0, 0.0, 0.0)
        if p < 0:
            start = base_start[i] * scale
            frame = wp.normalize(q_lean * base_qrel[i])
        else:
            qrel = base_qrel[i]
            if seg_fork[i] != 0:
                if spread != 1.0:
                    d = wp.quat_rotate(qrel, _ZUP)
                    tilt = wp.acos(wp.clamp(d[2], -1.0, 1.0))
                    axis = wp.cross(d, _ZUP)
                    if wp.length(axis) > 1.0e-9:
                        qrel = wp.quat_from_axis_angle(
                            wp.normalize(axis), (1.0 - spread) * tilt) * qrel
                if twist != 0.0:
                    qrel = wp.quat_from_axis_angle(_ZUP, twist) * qrel
            qrel = jq * qrel
            pf = nf[e, p]
            frame = wp.normalize(pf * qrel)
            if droop_gain != 0.0 and seg_order[i] > 0:
                h = wp.quat_rotate(frame, _ZUP)
                horiz = wp.sqrt(h[0] * h[0] + h[1] * h[1])
                if horiz > 1.0e-6:
                    axis = wp.cross(h, _ZDN)
                    if wp.length(axis) > 1.0e-9:
                        frame = wp.normalize(wp.quat_from_axis_angle(
                            wp.normalize(axis), droop_gain * horiz) * frame)
            start = ne[e, p]

        heading = wp.quat_rotate(frame, _ZUP)
        end = start + heading * length
        nf[e, i] = frame
        ne[e, i] = end
        nl[e, i] = length
        min_z = wp.min(min_z, wp.min(start[2], end[2]))

        # ---- capsule geometry / mass / inertia / colour ----
        R = 0.5 * (base_r0[i] + base_r1[i]) * f_rad * rad_bias * scale
        r0 = base_r0[i] * f_rad * rad_bias * scale
        L = length
        shp = e * spe + i
        bod = e * bpe + i
        shape_scale[shp] = wp.vec3(R, 0.5 * L, 0.0)
        shape_transform[shp] = wp.transform(
            wp.vec3(0.0, 0.0, 0.5 * L), wp.quat_identity())
        body_com[bod] = wp.vec3(0.0, 0.0, 0.5 * L)
        m_cyl = rho_e * 3.14159265 * R * R * L
        m_cap = rho_e * (4.0 / 3.0) * 3.14159265 * R * R * R
        body_mass[bod] = m_cyl + m_cap
        izz = 0.5 * m_cyl * R * R + 0.4 * m_cap * R * R
        ixx = (m_cyl * (L * L / 12.0 + 0.25 * R * R)
               + m_cap * (0.4 * R * R + 0.25 * L * L + 0.375 * L * R))
        body_inertia[bod] = wp.mat33(ixx, 0.0, 0.0, 0.0, ixx, 0.0, 0.0, 0.0, izz)
        t = float(seg_order[i]) / float(wp.max(max_order, 1))
        col = _BARK0 * (1.0 - t) + _BARK1 * t
        shape_color[shp] = wp.vec3(
            wp.clamp(col[0] * bright * warm, 0.0, 1.0),
            wp.clamp(col[1] * bright, 0.0, 1.0),
            wp.clamp(col[2] * bright / warm, 0.0, 1.0))

        # ---- joint rest transform + stiffness ----
        jnt = seg_joint[i]
        if p < 0:
            root_jnt = e * jpe + jnt
            root_pos = start
            root_rot = frame
        else:
            qrel_j = wp.quat_inverse(nf[e, p]) * frame
            joint_X_p[e * jpe + jnt] = wp.transform(
                wp.vec3(0.0, 0.0, nl[e, p]), qrel_j)
            kp = dr.beam_factor * E_e * (r0 * r0 * r0 * r0) / L
            kp = wp.clamp(kp, dr.min_stiff, dr.max_stiff)
            kd = dr.damping_ratio * kp
            d0 = joint_qd_start[e * jpe + jnt]
            joint_target_ke[d0] = kp
            joint_target_ke[d0 + 1] = kp
            joint_target_kd[d0] = kd
            joint_target_kd[d0 + 1] = kd

    # lift the whole tree so no segment starts below the ground clearance
    dz = wp.max(0.0, dr.clearance - min_z)
    joint_X_p[root_jnt] = wp.transform(
        root_pos + wp.vec3(0.0, 0.0, dz), root_rot)


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def patch_device(model, base, num_envs: int, drp, config: TreeConfig):
    """Fill the replicated ``model`` with per-env randomized wood geometry on the
    GPU.  ``model`` must already be the base tree replicated ``num_envs`` times.
    Returns nothing; ``model`` is modified in place."""
    dev = model.device
    segs = base.segments
    n_seg = len(segs)
    frames = [np.asarray(s.frame, float) for s in segs]

    parent = np.array([s.parent for s in segs], np.int32)
    order = np.array([s.order for s in segs], np.int32)
    fork = np.zeros(n_seg, np.int32)
    qrel = np.zeros((n_seg, 4), np.float32)
    for i, s in enumerate(segs):
        if s.parent < 0:
            qrel[i] = frames[i]
        else:
            qrel[i] = _qmul(_qconj(frames[s.parent]), frames[i])
            fork[i] = int(order[s.parent] != order[i])
    # seg -> joint index: root weld is joint 0, then non-root segments in order
    seg_joint = np.zeros(n_seg, np.int32)
    jid = 1
    n_root = 0
    for i, s in enumerate(segs):
        if s.parent < 0:
            seg_joint[i] = 0; n_root += 1
        else:
            seg_joint[i] = jid; jid += 1
    if n_root != 1:
        raise ValueError(f"device DR expects a single trunk root, got {n_root}")

    spe = model.shape_count // num_envs
    bpe = model.body_count // num_envs
    jpe = model.joint_count // num_envs

    def arr(a, dt):
        return wp.array(a, dtype=dt, device=dev)

    p = config.physics
    d = DRGpu()
    d.strength = float(drp.strength)
    d.scale_lo, d.scale_hi = drp.scale_lo, drp.scale_hi
    d.rad_bias_lo, d.rad_bias_hi = drp.rad_bias_lo, drp.rad_bias_hi
    d.len_sigma, d.len_lo, d.len_hi = drp.len_sigma, drp.len_lo, drp.len_hi
    d.rad_sigma, d.rad_lo, d.rad_hi = drp.rad_sigma, drp.rad_lo, drp.rad_hi
    d.ang_sigma = float(np.deg2rad(drp.ang_sigma_deg) * drp.strength)
    d.droop_lo, d.droop_hi = drp.droop_lo, drp.droop_hi
    d.droop_base = float(np.deg2rad(drp.droop_base_deg))
    d.spread_lo, d.spread_hi = drp.spread_lo, drp.spread_hi
    d.twist_max = float(np.deg2rad(drp.twist_max_deg))
    d.lean_max = float(np.deg2rad(drp.lean_max_deg))
    d.dens_lo, d.dens_hi = drp.dens_lo, drp.dens_hi
    d.mod_lo, d.mod_hi = drp.mod_lo, drp.mod_hi
    d.wood_bright_lo, d.wood_bright_hi = drp.wood_bright_lo, drp.wood_bright_hi
    d.wood_warm_lo, d.wood_warm_hi = drp.wood_warm_lo, drp.wood_warm_hi
    d.density = float(p.wood_density)
    d.youngs = float(p.youngs_modulus)
    d.beam_factor = float(p.beam_factor)
    d.damping_ratio = float(p.damping_ratio)
    d.min_stiff = float(p.min_stiffness)
    d.max_stiff = float(p.max_stiffness)
    d.clearance = 0.10

    nf = wp.zeros((num_envs, n_seg), dtype=wp.quat, device=dev)
    ne = wp.zeros((num_envs, n_seg), dtype=wp.vec3, device=dev)
    nl = wp.zeros((num_envs, n_seg), dtype=wp.float32, device=dev)

    wp.launch(_dr_walk, dim=num_envs, device=dev, inputs=[
        arr(parent, wp.int32), arr(order, wp.int32), arr(fork, wp.int32),
        arr(seg_joint, wp.int32), arr(qrel, wp.quat),
        arr(np.array([s.start for s in segs], np.float32), wp.vec3),
        arr(np.array([max(s.length, 1e-6) for s in segs], np.float32), wp.float32),
        arr(np.array([s.radius_start for s in segs], np.float32), wp.float32),
        arr(np.array([s.radius_end for s in segs], np.float32), wp.float32),
        int(n_seg), int(max(order.max(), 1)), int(spe), int(bpe), int(jpe),
        int(config.seed), d,
        nf, ne, nl,
        model.joint_qd_start, model.joint_X_p, model.shape_scale,
        model.shape_transform, model.shape_color, model.body_mass,
        model.body_com, model.body_inertia,
        model.joint_target_ke, model.joint_target_kd,
    ])
    wp.synchronize_device(dev)
    # keep inverse-mass/inertia consistent with the new masses
    bm = model.body_mass.numpy()
    model.body_inv_mass.assign(np.where(bm > 0, 1.0 / np.maximum(bm, 1e-12), 0.0))
    bi = model.body_inertia.numpy()
    binv = np.zeros_like(bi)
    for a in range(3):
        dd = bi[:, a, a]
        binv[:, a, a] = np.where(dd > 0, 1.0 / np.maximum(dd, 1e-15), 0.0)
    model.body_inv_inertia.assign(binv)
