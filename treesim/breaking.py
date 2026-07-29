"""Branch snapping.

A branch ruptures when the bending moment transmitted by its base joint exceeds
the modulus-of-rupture limit ``M_max = sigma_r * pi * r**3 / 4`` (see
``physics.rupture_moment``).  The transmitted moment is estimated from the
torsional spring law of the joint's bending DOFs::

    tau_i = Kp_i * q_i + Kd_i * qd_i        (per bending DOF)
    |M|   = sqrt(sum_i tau_i**2)

When ``|M| > M_max`` the joint is "snapped": all of its DOF stiffnesses are
zeroed and its limits opened.  In ``detach`` mode the joint has 6 DOFs, so this
frees the child completely and the whole subtree falls under gravity; in
``hinge`` mode only the bending DOFs exist, so the branch goes limp and droops.
The change is pushed to the solver via ``notify_model_changed``.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from .builder import TreeModel


@wp.kernel
def _cancel_stiffness(joint_q: wp.array(dtype=wp.float32),
                      coord_start: wp.array(dtype=wp.int32),
                      dof_start: wp.array(dtype=wp.int32),
                      dof_count: wp.array(dtype=wp.int32),
                      kp: wp.array(dtype=wp.float32),
                      broken: wp.array(dtype=wp.int32),
                      joint_f: wp.array(dtype=wp.float32)):
    """For each *broken* joint add ``+Ke*q`` in JOINT space, exactly cancelling
    that joint's position-actuator restoring term ``-Ke*(q-0)`` so the joint goes
    limp — a free, still-attached hinge that droops under gravity.  The MuJoCo
    solver maps ``control.joint_f`` straight into ``qfrc_applied`` (no frame
    attenuation, unlike a Cartesian body_f couple, which a position servo just
    overrides), so this actually limps the joint.  Only the STIFFNESS is
    cancelled — the actuator's damping is integrated implicitly by MuJoCo and
    cancelling it explicitly would anti-damp and blow up; the residual damping
    plus aerodynamic angular drag stop the drooped branch from spinning forever.
    Pure forces + a flag: no model edit, no notify, runs inside the CUDA graph."""
    j = wp.tid()
    if broken[j] == 0:
        return
    qa = coord_start[j]      # joint_q is indexed by COORD start
    fa = dof_start[j]        # control.joint_f is indexed by VELOCITY-dof start
    for d in range(dof_count[j]):
        joint_f[fa + d] = kp[j] * joint_q[qa + d]


class Breaker:
    """Bending rupture for the breakable branch joints.

    When the transmitted bending moment ``|M| = Kp*theta`` (measured from the
    body poses, always valid under MuJoCo) exceeds a joint's rupture moment for a
    few consecutive frames, the joint is "snapped".  Two mechanisms:

    * **free-fall (default, MuJoCo)**: the joint's position-actuator gain AND
      bias rows are zeroed *in place* in the mjwarp model arrays (they are laid
      out ``(num_worlds, nu)`` and read fresh every step, so this needs no
      ``notify``, no recompile and no CUDA-graph recapture, and it is per-env).
      That removes the spring *and its implicit damping*, so the branch becomes
      a genuinely free hinge and swings down at gravity rate — the old
      stiffness-only cancellation left the actuator damping in, which is why
      snapped branches used to creep down in slow motion.  The subtree's aero
      drag is also reduced (via ``Sim.set_drag_scale``) so air resistance does
      not fake a slow fall; the soft ground still settles it.
    * **fallback (non-MuJoCo solvers)**: the original in-graph joint-space
      stiffness cancellation (``control.joint_f = +Ke*q``); damping cannot be
      cancelled explicitly (anti-damping blows up), so the tree is then built
      with ``limp_damping_ratio`` soft damping instead.

    The MuJoCo model structure is never edited; breaking stays a per-frame flag
    plus small array writes.  A short rupture-hysteresis avoids transient
    spikes chain-breaking the whole tree, and snapped wood is recoloured so the
    break is visible.
    """

    def __init__(self, tree: TreeModel, solver, rest_body_q, hysteresis: int = 3):
        from . import springs as _springs
        self.tree = tree
        self.solver = solver
        self.model = tree.model
        dev = self.model.device

        self.n = len(tree.brk_joint_id)
        self.broken = np.zeros(self.n, dtype=bool)
        self._over = np.zeros(self.n, dtype=np.int32)        # consecutive over-threshold frames
        self._hyst = max(int(hysteresis), 1)
        self.broken_count = 0
        self.meter = _springs.MomentMeter(
            tree.brk_parent_body, tree.brk_child_body, tree.brk_kp,
            rest_body_q, dev)
        self.m_max = tree.brk_mmax
        self.broken_flag = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._bf_host = np.zeros(max(self.n, 1), dtype=np.int32)

        # joint-space cancellation: control.joint_f[dof] = +Ke*joint_q[coord].
        # joint_f is indexed by the velocity DOF (joint_qd_start) while joint_q is
        # indexed by the COORD (joint_q_start).  For a single tree these coincide
        # for the branch joints, but with num_envs>1 each world's branches sit
        # after the previous worlds' apple FREE joints (7 coords vs 6 dofs each),
        # so the two indices diverge and MUST be read separately.
        jids = np.asarray(tree.brk_joint_id)
        qd_start = self.model.joint_qd_start.numpy()
        q_start = self.model.joint_q_start.numpy()
        ds = (qd_start[jids] if self.n else np.zeros(0))
        cs = (q_start[jids] if self.n else np.zeros(0))
        self.dof_start = wp.array(np.asarray(ds, dtype=np.int32), device=dev)
        self.coord_start = wp.array(np.asarray(cs, dtype=np.int32), device=dev)
        self.dof_count = wp.array(np.asarray(tree.brk_dof_count, dtype=np.int32), device=dev)
        self.kp_dof = wp.array(np.asarray(tree.brk_kp, dtype=np.float32), device=dev)
        self._ds_host = np.asarray(ds, dtype=np.int64)
        self._dc_host = np.asarray(tree.brk_dof_count, dtype=np.int64)

        self._pending_recolor: list[int] = []
        self._drag_hook = None            # set by Sim: called with newly-snapped body ids
        self._rest_q = np.asarray(rest_body_q)
        self.dev = dev
        # body -> its FIRST shape (the wood capsule; leaf cards etc. come later),
        # for recolouring snapped wood.  update_shape_colors wants shape indices.
        self._wood_shape = {}
        for s, b in enumerate(self.model.shape_body.numpy()):
            self._wood_shape.setdefault(int(b), int(s))
        # broken subtrees stop taking rigid contacts (group 0, live-read by the
        # broad phase every collide()): a limp zero-stiffness branch pinned by
        # the 130 kg robot is unresolvable for the contact solver (explodes);
        # physically the dangling debris just sweeps aside.
        self._colgroup_host = (self.model.shape_collision_group.numpy()
                               if self.model.shape_collision_group is not None else None)
        # apples riding a snapped subtree go inert WITH it: their sphere stays
        # group -1 otherwise, and the robot rolling over a dead limb's fruit
        # was the last remaining robot-vs-debris contact (mass ratio ~10^4)
        self._ap_parent = self._ap_body = None
        ad = getattr(tree, "apple_data", None)
        if ad is not None and len(ad.get("apple_body", [])):
            self._ap_parent = np.asarray(ad["parent_body"], dtype=np.int64)
            self._ap_body = np.asarray(ad["apple_body"], dtype=np.int64)

        # ---- free-fall: zero the broken dofs' actuator gain+bias in place ----
        # (see class docstring).  Must be set up BEFORE the CUDA graph is
        # captured only because the arrays are bound by pointer at capture —
        # we never swap them, just rewrite their contents, so any time works.
        self._direct = False
        if getattr(tree.config.breaking, "free_fall", False):
            try:
                self._setup_direct(solver)
            except Exception as e:              # non-MuJoCo solver etc.
                print(f"[breaking] free-fall unavailable ({e}); using limp-hinge fallback")

    def _setup_direct(self, solver):
        mjw = solver.mjw_model
        a2n = solver.mjc_actuator_to_newton_idx.numpy()
        num_envs = max(int(self.tree.num_envs), 1)
        ndof_env = self.model.joint_dof_count // num_envs
        # invert actuator -> template(world-0) newton dof; positive entries are
        # position actuators (the solver encodes velocity ones as -(dof+2))
        act_of = np.full(ndof_env, -1, dtype=np.int64)
        pos = np.where((a2n >= 0) & (a2n < ndof_env))[0]
        act_of[a2n[pos]] = pos
        # host mirrors of the (num_worlds, nu, 10) gain/bias tables
        self._gain_wp = mjw.actuator_gainprm
        self._bias_wp = mjw.actuator_biasprm
        self._gain_host = self._gain_wp.numpy()
        self._bias_host = self._bias_wp.numpy()
        if self._gain_host.shape[0] != num_envs and num_envs > 1:
            raise RuntimeError(f"gainprm worlds={self._gain_host.shape[0]} != envs={num_envs}")
        # Coulomb friction at the break (dof_frictionloss, (num_worlds, nv)):
        # a fraction of each subtree's rest gravity moment about its pivot —
        # barely resists the fall, pins the branch once it hangs.
        self._fric_wp = mjw.dof_frictionloss
        self._fric_host = self._fric_wp.numpy()
        frac = float(getattr(self.tree.config.breaking, "break_friction", 0.0))
        g = abs(float(self.tree.config.physics.gravity))
        masses = self.model.body_mass.numpy()
        rest = self._rest_q[:, :3]
        self._fric_val = np.zeros(self.n)
        for j in range(self.n):
            sub = self.tree.brk_descend[j]
            msub = masses[sub]
            if msub.sum() <= 0.0:
                continue
            com = (rest[sub] * msub[:, None]).sum(0) / msub.sum()
            pivot = rest[int(self.tree.brk_child_body[j])]
            d = float(np.linalg.norm(com - pivot))
            self._fric_val[j] = frac * float(msub.sum()) * g * max(d, 0.01)
        self._act_of = act_of
        self._ndof_env = ndof_env
        self._n_env = max(self.n // num_envs, 1)
        # post-break friction ramp (fibers seize): age counter per joint
        brk = self.tree.config.breaking
        self._ramp_max = float(getattr(brk, "break_friction_ramp_max", 1.0))
        ramp_t = max(float(getattr(brk, "break_friction_ramp_time", 1.0)), 1e-3)
        self._ramp_frames = max(int(ramp_t * 60), 1)     # update() runs per frame
        self._age = np.zeros(self.n, dtype=np.int64)
        # freeze-at-bottom state machine (see BreakParams.freeze_at_bottom)
        self._freeze = bool(getattr(brk, "freeze_at_bottom", False))
        self._freeze_fric = float(getattr(brk, "freeze_friction", 60.0))
        self._fall_speed = float(getattr(brk, "freeze_fall_speed", 0.25))
        self._freeze_frames = max(int(float(getattr(brk, "freeze_timeout", 3.0)) * 60), 1)
        self._fell = np.zeros(self.n, dtype=bool)
        self._frozen = np.zeros(self.n, dtype=bool)
        self._sub_mass = [None] * self.n                 # lazy per-joint mass vectors
        self._direct = True

    def _free_joints(self, joints: np.ndarray):
        """Zero gain+bias of every actuator of the given breakable joints (per
        that joint's world) and set the break friction, then push the small
        tables to the device."""
        for j in joints:
            env = int(j) // self._n_env
            w = env if self._gain_host.shape[0] > 1 else 0
            wf = env if self._fric_host.shape[0] > 1 else 0
            base = int(self._ds_host[j]) - env * self._ndof_env
            for d in range(int(self._dc_host[j])):
                a = self._act_of[base + d]
                if a >= 0:
                    self._gain_host[w, a] = 0.0
                    self._bias_host[w, a] = 0.0
                self._fric_host[wf, base + d] = self._fric_val[int(j)]
        self._gain_wp.assign(self._gain_host)
        self._bias_wp.assign(self._bias_host)
        self._fric_wp.assign(self._fric_host)

    def apply_cancel(self, state, control):
        """Limp the broken joints in joint space (call each substep, in-graph).
        FALLBACK path only — with free-fall the actuator itself is gone."""
        if self.n == 0 or self._direct or control.joint_f is None:
            return
        control.joint_f.zero_()
        wp.launch(_cancel_stiffness, dim=self.n,
                  inputs=[state.joint_q, self.coord_start, self.dof_start, self.dof_count,
                          self.kp_dof, self.broken_flag, control.joint_f],
                  device=self.dev)

    def _set_fric(self, j: int, val: float):
        env = int(j) // self._n_env
        wf = env if self._fric_host.shape[0] > 1 else 0
        base = int(self._ds_host[j]) - env * self._ndof_env
        for d in range(int(self._dc_host[j])):
            self._fric_host[wf, base + d] = val

    def _tick_friction(self, state):
        """Per-frame post-break bookkeeping: ramp the break friction (fibers
        seizing) while the branch falls, and the moment its subtree's vertical
        motion flips from falling to rising — the bottom of the drop — FREEZE
        the joint (friction >> gravity moment).  The branch just stays where it
        landed; the pick force on snapped subtrees is already dropped, so it
        cannot be disturbed afterwards."""
        if not self._direct or not self.broken.any():
            return
        live = self.broken & ~self._frozen
        self._age[self.broken] += 1
        if not live.any():
            return
        changed = False
        vz = None
        if self._freeze:
            qd = state.body_qd.numpy()          # [linear (top), angular]
            vz = qd[:, 2]
        for j in np.where(live)[0]:
            jj = int(j)
            if self._freeze:
                if self._sub_mass[jj] is None:
                    sub = np.asarray(self.tree.brk_descend[jj], dtype=int)
                    m = self.model.body_mass.numpy()[sub]
                    self._sub_mass[jj] = (sub, m / max(m.sum(), 1e-9))
                sub, w = self._sub_mass[jj]
                v = float((vz[sub] * w).sum())   # subtree COM vertical velocity
                if not self._fell[jj]:
                    if v < -self._fall_speed:
                        self._fell[jj] = True
                if ((self._fell[jj] and v >= 0.0)
                        or self._age[jj] >= self._freeze_frames):
                    self._frozen[jj] = True
                    self._set_fric(jj, self._fric_val[jj] * self._freeze_fric)
                    changed = True
                    continue
            if self._age[jj] <= self._ramp_frames:
                t = min(float(self._age[jj]) / self._ramp_frames, 1.0)
                self._set_fric(jj, self._fric_val[jj] * (1.0 + (self._ramp_max - 1.0) * t))
                changed = True
        if changed:
            self._fric_wp.assign(self._fric_host)

    def update(self, state) -> int:
        """Detect new ruptures (read-only) and free/limp them.  No model edit."""
        if self.n == 0:
            return 0
        self._tick_friction(state)
        if self.broken.all():
            return 0
        moment = self.meter.moments(state)
        over = (moment > self.m_max) & (~self.broken)
        self._over[over] += 1
        self._over[~over] = 0
        newly = (self._over >= self._hyst) & (~self.broken)
        if not newly.any():
            return 0
        self.broken[newly] = True
        self._bf_host[:self.n][newly] = 1
        self.broken_flag.assign(self._bf_host)
        self.broken_count = int(self.broken.sum())
        new_idx = np.where(newly)[0]
        if self._direct:
            self._free_joints(new_idx)
        new_bodies: list[int] = []
        for j in new_idx:
            new_bodies.extend(self.tree.brk_descend[int(j)])
        self._pending_recolor.extend(new_bodies)          # wood only: apples stay red
        dead = list(new_bodies)
        if self._ap_parent is not None and new_bodies:
            on_dead = np.isin(self._ap_parent, np.asarray(new_bodies, dtype=np.int64))
            dead.extend(self._ap_body[on_dead].tolist())
        if self._drag_hook is not None and dead:
            self._drag_hook(dead)
        if self._colgroup_host is not None and dead:
            for b in dead:
                s = self._wood_shape.get(int(b))
                if s is not None:
                    self._colgroup_host[s] = 0
            self.model.shape_collision_group.assign(self._colgroup_host)
        return int(newly.sum())

    def render(self, viewer):
        """Recolour newly-snapped wood to dead-wood brown so breaks are visible."""
        if not self._pending_recolor:
            return
        bodies, self._pending_recolor = self._pending_recolor, []
        col = self.tree.config.breaking.broken_color
        try:
            shape_cols = {}
            for b in bodies:
                if self.tree.base_color is not None:
                    self.tree.base_color[b] = col
                s = self._wood_shape.get(int(b))
                if s is not None:
                    shape_cols[s] = col
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                viewer.update_shape_colors(shape_cols)
        except Exception:
            pass


class SpringBreaker:
    """Breaking for the custom-spring engine (solver-agnostic, instant).

    Reads the restoring moment each frame from the :class:`SpringField`; when it
    exceeds the branch's rupture moment, zeroes that joint's spring stiffness
    (limp hinge) and, in ``detach`` mode, disables the underlying joint so the
    subtree separates and falls.  No model recompile / notify is needed for the
    hinge part, so this is real-time even with many simultaneous breaks.
    """

    def __init__(self, tree: TreeModel, spring, solver):
        self.tree = tree
        self.spring = spring
        self.solver = solver
        self.model = tree.model
        self.mode = tree.config.breaking.mode
        n = spring.n
        self.broken = np.zeros(n, dtype=bool)
        self.broken_count = 0
        self.kp_host = spring.kp.numpy().copy()
        self.kd_host = spring.kd.numpy().copy()
        self.joint_enabled = self.model.joint_enabled.numpy().copy()
        self.joint_ids = np.array(tree.joint_ids, dtype=int)
        self._pending_recolor: list[int] = []

    def update(self, state) -> bool:
        if self.broken.all():
            return False
        m = self.spring.moments(state)
        newly = (m > self.spring.m_max) & (~self.broken)
        if not newly.any():
            return False
        idx = np.where(newly)[0]
        self.kp_host[idx] = 0.0
        self.kd_host[idx] = 0.0
        self.broken[idx] = True
        self.spring.kp.assign(self.kp_host)
        self.spring.kd.assign(self.kd_host)
        self.broken_count = int(self.broken.sum())

        detached = False
        for j in idx:
            self._pending_recolor.extend(
                self.tree.descendants_bodies(int(j)))
            if self.mode == "detach":
                self.joint_enabled[self.joint_ids[j]] = 0
                detached = True
        if detached:
            self.model.joint_enabled.assign(self.joint_enabled)
            try:
                self.solver.notify_model_changed(
                    newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)
            except Exception:
                pass
        return True

    def render(self, viewer):
        if not self._pending_recolor:
            return
        bodies, self._pending_recolor = self._pending_recolor, []
        col = self.tree.config.breaking.broken_color
        try:
            for b in bodies:
                self.tree.base_color[b] = col
            viewer.update_shape_colors(self.tree.base_color)
        except Exception:
            pass
