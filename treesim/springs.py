"""Custom torsional spring-damper field (the paper's "crude spring abstraction").

This is the *fast* deformable engine.  A maximal-coordinate base solver (XPBD)
keeps the branch segments rigidly connected at their pivots (cheap, implicit,
stable), and this module adds, every substep, a torsional restoring torque at
each inter-branch joint:

    tau = -Kp * rotvec(q_child_rel * q_rest^-1)  -  Kd * (w_child - w_parent)

applied as an equal-and-opposite wrench on the child and parent bodies via
``state.body_f``.  Because the stiffness lives in a Warp array we own, branch
*breaking* is instantaneous and solver-agnostic: set ``Kp[j]=0`` (limp hinge)
and optionally ``joint_enabled[j]=0`` (full detach) — no model recompile.

Unlike the MuJoCo solver this scales to thousands of bodies in real time, at the
cost of an explicit stiffness cap for stability (very stiff trunk joints are
clamped; they barely move anyway).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .builder import TreeModel
from . import physics


@wp.kernel
def _spring_torque(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    parent_body: wp.array(dtype=wp.int32),
    child_body: wp.array(dtype=wp.int32),
    q_rest: wp.array(dtype=wp.quat),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    j = wp.tid()
    k = kp[j]
    if k <= 0.0:
        return
    pb = parent_body[j]
    cb = child_body[j]
    qp = wp.transform_get_rotation(body_q[pb])
    qc = wp.transform_get_rotation(body_q[cb])

    q_rel = wp.mul(wp.quat_inverse(qp), qc)          # child in parent frame
    dq = wp.mul(q_rel, wp.quat_inverse(q_rest[j]))   # deviation from rest
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])  # shortest arc

    sh = wp.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2])
    angle = 2.0 * wp.atan2(sh, dq[3])
    axis = wp.vec3(0.0, 0.0, 0.0)
    if sh > 1.0e-7:
        axis = wp.vec3(dq[0], dq[1], dq[2]) / sh
    rotvec = axis * angle                            # in parent frame

    tau = wp.quat_rotate(qp, -k * rotvec)            # restoring torque, world
    rel_w = wp.spatial_top(body_qd[cb]) - wp.spatial_top(body_qd[pb])
    tau = tau - kd[j] * rel_w                         # damping

    w = wp.spatial_vector(tau, wp.vec3(0.0, 0.0, 0.0))
    wp.atomic_add(body_f, cb, w)
    wp.atomic_sub(body_f, pb, w)


@wp.kernel
def _measure_moment(
    body_q: wp.array(dtype=wp.transform),
    parent_body: wp.array(dtype=wp.int32),
    child_body: wp.array(dtype=wp.int32),
    q_rest: wp.array(dtype=wp.quat),
    kp: wp.array(dtype=wp.float32),
    moment: wp.array(dtype=wp.float32),
):
    j = wp.tid()
    pb = parent_body[j]
    cb = child_body[j]
    qp = wp.transform_get_rotation(body_q[pb])
    qc = wp.transform_get_rotation(body_q[cb])
    q_rel = wp.mul(wp.quat_inverse(qp), qc)
    dq = wp.mul(q_rel, wp.quat_inverse(q_rest[j]))
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])
    sh = wp.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2])
    angle = 2.0 * wp.atan2(sh, dq[3])
    moment[j] = kp[j] * angle


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


class MomentMeter:
    """Measures the per-joint bending moment ``|M| = Kp*theta`` from the current
    body poses (always valid, unlike ``joint_q`` under the MuJoCo solver).

    Used by the breakers to decide rupture for *any* solver.  ``kp`` is the
    physical stiffness, ``q_rest`` the rest relative orientation taken from the
    finalized bodies, and ``m_max`` the per-joint rupture moment.
    """

    def __init__(self, parent_body, child_body, kp, rest_body_q, device):
        self.device = device
        bq = np.asarray(rest_body_q)
        parent_body = np.asarray(parent_body, dtype=np.int32)
        child_body = np.asarray(child_body, dtype=np.int32)
        self.n = len(parent_body)
        qrest = np.array([_qmul(_qconj(bq[p, 3:7]), bq[c, 3:7])
                          for p, c in zip(parent_body, child_body)],
                         dtype=np.float32).reshape(-1, 4)
        self.parent_body = wp.array(parent_body, device=device)
        self.child_body = wp.array(child_body, device=device)
        self.q_rest = wp.array(qrest, dtype=wp.quat, device=device)
        self.kp = wp.array(np.asarray(kp, dtype=np.float32), device=device)
        self._moment = wp.zeros(max(self.n, 1), dtype=wp.float32, device=device)

    def moments(self, state) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        wp.launch(_measure_moment, dim=self.n,
                  inputs=[state.body_q, self.parent_body, self.child_body,
                          self.q_rest, self.kp, self._moment],
                  device=self.device)
        return self._moment.numpy()[:self.n]


class SpringField:
    """Torsional springs for all inter-branch joints of a TreeModel.

    Built from the *skeleton* (rest frames) and the per-branch beam stiffness,
    so it is independent of the base solver.  ``max_kp`` caps stiffness for
    explicit-integration stability.
    """

    def __init__(self, tree: TreeModel, rest_body_q=None):
        self.tree = tree
        self.model = tree.model
        dev = self.model.device
        skel = tree.skeleton
        phys = tree.config.physics

        # Rest relative orientations are taken from the *actual* finalized body
        # poses (so the spring sees zero deviation at t=0).  Falls back to the
        # skeleton frames if no rest state is provided.
        bq = None if rest_body_q is None else np.asarray(rest_body_q)

        # subtree mass per segment (for the inertia budget of the stability cap)
        seg_mass = {s.index: physics.segment_mass(s, phys.wood_density) for s in skel.segments}
        subtree_mass = {}
        for s in sorted(skel.segments, key=lambda s: s.depth, reverse=True):
            subtree_mass[s.index] = seg_mass[s.index] + sum(
                subtree_mass[c] for c in s.children)

        parents, children, qrest, kps, kds, mmax, inertia = [], [], [], [], [], [], []
        for seg in skel.segments:
            if seg.parent < 0:
                continue
            par = skel[seg.parent]
            pb = int(tree.seg_to_body[seg.parent])
            cb = int(tree.seg_to_body[seg.index])
            parents.append(pb)
            children.append(cb)
            if bq is not None:
                qrest.append(_qmul(_qconj(bq[pb, 3:7]), bq[cb, 3:7]))
            else:
                qrest.append(_qmul(_qconj(par.frame), seg.frame))
            kp, kd = physics.stiffness_for(seg, phys)
            kps.append(kp)
            kds.append(kd)
            mmax.append(physics.rupture_moment(seg, tree.config.breaking))
            # rotational inertia of the subtree about this joint (rod-about-end
            # for the segment + lumped subtree mass a segment-length away)
            L = max(seg.length, 1e-3)
            I = seg_mass[seg.index] * L * L / 3.0 + \
                (subtree_mass[seg.index] - seg_mass[seg.index]) * L * L
            inertia.append(max(I, 1e-9))

        self.n = len(parents)
        self.parent_body = wp.array(np.array(parents, dtype=np.int32), device=dev)
        self.child_body = wp.array(np.array(children, dtype=np.int32), device=dev)
        self.q_rest = wp.array(np.array(qrest, dtype=np.float32), dtype=wp.quat, device=dev)
        self.kp_phys = np.array(kps, dtype=np.float64)
        self.kd_phys = np.array(kds, dtype=np.float64)
        self.inertia = np.array(inertia, dtype=np.float64)
        self.kp = wp.array(self.kp_phys.astype(np.float32), device=dev)
        self.kd = wp.array(self.kd_phys.astype(np.float32), device=dev)
        self.m_max = np.array(mmax, dtype=np.float64)
        self._moment = wp.zeros(self.n, dtype=wp.float32, device=dev)

    def cap_for_dt(self, dt: float, safety: float = 0.5):
        """Clamp per-joint stiffness/damping for explicit stability at substep
        ``dt``:  Kp <= I*(2*safety/dt)^2,  Kd <= safety*I/dt.  Thin twigs (tiny
        inertia) are softened automatically; the trunk keeps near-physical Kp."""
        kp_max = self.inertia * (2.0 * safety / dt) ** 2
        kp = np.minimum(self.kp_phys, kp_max)
        kd_max = safety * self.inertia / dt
        kd = np.minimum(np.maximum(self.kd_phys, 0.2 * np.sqrt(kp * self.inertia)), kd_max)
        self.kp.assign(kp.astype(np.float32))
        self.kd.assign(kd.astype(np.float32))
        return kp

    def apply(self, state):
        wp.launch(_spring_torque, dim=self.n,
                  inputs=[state.body_q, state.body_qd, self.parent_body,
                          self.child_body, self.q_rest, self.kp, self.kd,
                          state.body_f], device=self.model.device)

    def moments(self, state) -> np.ndarray:
        wp.launch(_measure_moment, dim=self.n,
                  inputs=[state.body_q, self.parent_body, self.child_body,
                          self.q_rest, self.kp, self._moment],
                  device=self.model.device)
        return self._moment.numpy()
