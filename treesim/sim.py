"""Simulation runtime: wraps a TreeModel with states, solver, viewer and the
step loop.  Handles interactive forces (via the viewer), programmatic external
forces, optional branch breaking, and CUDA-graph acceleration.

The step loop mirrors Newton's canonical idiom (see the bundled
``example_basic_pendulum``):  for each substep -> clear forces, let the viewer
inject mouse-drag forces, collide, solver.step, swap states.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp
import newton

from .builder import TreeModel
from . import breaking as _breaking
from . import springs as _springs
from . import fruit as _fruit


def _make_solver(name, model, *, collisions, iterations, ls_iterations,
                 algo: str = "cg"):
    if name == "mujoco":
        # Use MuJoCo's default (implicit) integration/iterations — they resolve
        # the stiff torsional springs stably.  Only drop contacts when unused.
        # njmax (constraints/world) is otherwise sized from the calm rest state; a
        # hard yank spawns many transient constraints, so give a generous margin
        # to avoid "nefc overflow" (esp. with several breaking branches / envs).
        # With contacts on, use NEWTON's collision pipeline (model.collide each
        # substep) instead of MuJoCo's internal one: it honours newton's
        # collision-group semantics exactly (tree = -1 self-filtered, leaves 0,
        # robot chassis 3 / arm 1), which is what makes robot-vs-tree contact
        # affordable — only ~400 group-enabled shapes ever reach the broad phase.
        # nconmax/njmax must survive the worst case, not the calm rest state: a
        # robot shoved deep into the canopy produced ~666 newton contacts (the
        # defaults sized 256 -> "nconmax exceeded / nefc overflow" and the sim
        # exploded).  Each contact contributes several constraint rows, so
        # njmax gets a matching margin.  Memory cost is a few MB; step cost is
        # driven by ACTIVE contacts, not capacity.
        # Constraint-solver algorithm: mjwarp's default ("newton") spends
        # ~85% of the step in ONE blocked-Cholesky kernel whose parallelism is
        # per-WORLD — a single ~700-dof tree leaves the GPU almost idle
        # (measured 12.7 ms per launch on this model; 33.7 -> 4.5 ms/frame by
        # switching).  "cg" solves the same optimization matrix-free and
        # validated bit-for-bit on the full physics regression matrix (rest
        # drift, pull-bend, detach, break fall rate, rams, e2e).  Iteration
        # caps don't matter for either (early termination on tolerance).
        algo_i = {"cg": 1, "newton": 2}.get(str(algo).lower(), 1)
        return newton.solvers.SolverMuJoCo(model, disable_contacts=not collisions,
                                           use_mujoco_contacts=False if collisions else True,
                                           solver=algo_i,
                                           nconmax=2048 if collisions else None,
                                           njmax=8192 if collisions else 1024)
    if name == "featherstone":
        return newton.solvers.SolverFeatherstone(model)
    if name == "semiimplicit":
        return newton.solvers.SolverSemiImplicit(model)
    if name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=max(iterations, 10))
    raise ValueError(f"unknown solver {name!r}")


class Sim:
    def __init__(self, tree: TreeModel, *, solver: str = "mujoco",
                 fps: int = 60, substeps: int = 8, enable_breaking: bool = False,
                 collisions: bool = False, iterations: int = 4, ls_iterations: int = 4,
                 spring_base: str = "xpbd", spring_safety: float = 0.25):
        self.tree = tree
        self.model = tree.model
        self.fps = fps
        self.frame_dt = 1.0 / fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.solver_name = solver
        # Branch-vs-branch self-collision is O(N^2) and usually unwanted; leave
        # it off for fast interactive bending (forces apply directly).  Turn on
        # for falling snapped debris / external-object interaction.
        self.collisions = collisions

        # aerodynamic drag + soft ground (see PhysicsParams); applied every substep
        ph = tree.config.physics
        self._drag = (float(ph.linear_drag), float(ph.angular_drag), float(ph.ground_z),
                      float(ph.ground_stiffness), float(ph.ground_damping),
                      float(ph.ground_friction))
        self._vcap = (float(getattr(ph, "max_speed", 25.0)),
                      float(getattr(ph, "max_omega", 60.0)),
                      float(getattr(ph, "brake", 120.0)),
                      float(getattr(ph, "fade_speed", 8.0)),
                      float(getattr(ph, "fade_omega", 30.0)))
        self._ang_broken = float(getattr(tree.config.breaking, "broken_ang_drag_scale", 1.0))
        self.body_mass = self.model.body_mass
        # Per-body scalar rotational inertia (mean of the inertia-tensor diagonal).
        # Angular drag is applied as tau = -angular_drag * I * w so the angular
        # decay rate is uniform (= angular_drag) and EXPLICITLY STABLE for every
        # body — using mass there instead would over-damp tiny-inertia twigs and
        # apples (m/I huge) and blow them up.
        _bi = self.model.body_inertia.numpy()
        _iscalar = np.maximum((_bi[:, 0, 0] + _bi[:, 1, 1] + _bi[:, 2, 2]) / 3.0, 1e-9)
        self._body_inertia = wp.array(_iscalar.astype(np.float32), device=self.model.device)
        # Per-body aero-drag multiplier (default 1 = exactly the old behaviour).
        # Snapped subtrees get scaled down (breaking.broken_drag_scale) so a
        # falling branch isn't slowed to a fake terminal velocity by the tree's
        # sway-damping drag.  Contents are rewritten in place, so this works
        # inside the captured CUDA graph.
        self._drag_scale_host = np.ones(self.model.body_count, dtype=np.float32)
        self._drag_scale = wp.array(self._drag_scale_host, device=self.model.device)

        # "spring" mode = fast maximal-coordinate base + our torsional-spring kernel.
        base_solver = spring_base if solver == "spring" else solver
        self.solver = _make_solver(base_solver, self.model, collisions=collisions,
                                   iterations=iterations, ls_iterations=ls_iterations,
                                   algo=getattr(ph, "mj_solver", "cg"))
        self.state_0, self.state_1 = tree.state_pair()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        # host-side pose cache shared across all callers (e.g. every env's
        # picker): body_q/joint_q are copied to the host at most ONCE per step,
        # so N pickers cost one device->host sync per frame, not N.
        self._host_step = 0
        self._bq_np = None
        self._bq_np_step = -1
        self._jq_np = None
        self._jq_np_step = -1

        self.spring = None
        if solver == "spring" and len(tree.joint_ids):
            # The base solver must NOT also drive the joints (it would fight our
            # kernel); zero the model's PD targets so only our springs act.
            self.model.joint_target_ke.zero_()
            self.model.joint_target_kd.zero_()
            # rest pose from the actual finalized bodies -> zero initial torque
            self.spring = _springs.SpringField(tree, rest_body_q=self.state_0.body_q.numpy())
            self.spring.cap_for_dt(self.sim_dt, safety=spring_safety)

        self.viewer = None
        self._graph = None

        # external programmatic forces (persistent buffer, added each substep)
        self._ext_force = wp.zeros(self.model.body_count, dtype=wp.spatial_vector,
                                   device=self.model.device)

        self.breaker = None
        if enable_breaking and len(tree.brk_joint_id):
            if self.spring is not None:
                self.breaker = _breaking.SpringBreaker(tree, self.spring, self.solver)
            else:
                self.breaker = _breaking.Breaker(
                    tree, self.solver, self.state_0.body_q.numpy(),
                    hysteresis=getattr(tree.config.breaking, "hysteresis_frames", 3))
                # snapped subtrees fall with much less aero drag (see _drag_scale)
                bds = float(getattr(tree.config.breaking, "broken_drag_scale", 1.0))
                self.breaker._drag_hook = lambda bodies: self.set_drag_scale(bodies, bds)

        # free-body apples held by a one-sided tether; pulled past their stem
        # tension they detach and fall.  Independent of branch breaking and of
        # the solver — pure forces + a flag, so it never hitches or destabilises.
        self.apples = None
        if tree.apple_data is not None and len(tree.apple_data["apple_body"]):
            self.apples = _fruit.AppleField(tree, tree.config)

        # thin twigs don't take rigid contacts (mass-ratio explosion); the
        # brush kernel is what lets the robot push small branches aside
        self.brush = None
        # thin twigs now take real (armature-conditioned) rigid contacts, so the
        # penalty brush is retired; keep it available for A/B via TREESIM_TWIG_BRUSH
        if tree.robot_data is not None and collisions and os.environ.get("TREESIM_TWIG_BRUSH"):
            from . import robot as _robot
            self.brush = _robot.TwigBrush(tree)
        if self.brush is not None and self.breaker is not None \
                and getattr(self.breaker, "_drag_hook", None) is not None:
            # dead subtrees are fully INERT: the breaker drops their rigid
            # contacts and pick force at break; also stop brushing their twigs
            # (a frozen limb must not shove the robot, nor cost kernel time)
            _drag = self.breaker._drag_hook

            def _on_snapped(bodies, _drag=_drag):
                _drag(bodies)
                self.brush.deactivate_bodies(bodies)
            self.breaker._drag_hook = _on_snapped

    # -- viewer wiring -------------------------------------------------------
    def set_viewer(self, viewer):
        self.viewer = viewer
        viewer.set_model(self.model)
        # UNIFORM grid for multi-env rendering, with a per-dimension pitch
        # computed at build time from the env content's AABB (the robot pads x
        # only — a single square pitch left x gaps tight and y gaps wide).
        # The SAME pitch drives the terrain tiling, so the ground drawn under
        # every displayed env matches the physics terrain at the origin.
        try:
            wo = getattr(viewer, "world_offsets", None)
            if wo is not None and self.tree.num_envs > 1 and self.tree.env_pitch:
                px, py = self.tree.env_pitch
                cols = max(int(self.tree.env_cols), 1)
                grid = np.zeros((self.tree.num_envs, 3), dtype=np.float32)
                for e in range(self.tree.num_envs):
                    grid[e, 0] = (e % cols) * px
                    grid[e, 1] = (e // cols) * py
                wo.assign(grid)
        except Exception:
            pass
        # raise the mouse-pick strength ceiling (needed to beat the apple stems;
        # the impulse cap + force fade keep even the hardest yank integrable)
        try:
            pk = getattr(viewer, "picking", None)
            if pk is not None and hasattr(pk, "pick_state"):
                st = pk.pick_state.numpy()
                st[0]["pick_max_acceleration"] = float(
                    getattr(self.tree.config.physics, "pick_max_acceleration", 5.0))
                pk.pick_state.assign(st)
        except Exception:
            pass
        self._capture()
        return self

    def _capture(self):
        """(Re)capture the substep loop into a CUDA graph for fast replay.  Cheap
        to call again after a break changes the model."""
        self._graph = None
        if wp.get_device(self.model.device).is_cuda:
            try:
                with wp.ScopedCapture() as cap:
                    self._simulate()
                self._graph = cap.graph
            except Exception:
                self._graph = None

    # -- external force API --------------------------------------------------
    def set_external_force(self, body: int, force=(0, 0, 0), torque=(0, 0, 0)):
        """Set a persistent world-frame wrench on ``body``.  Newton's body wrench
        convention is spatial_vector = [linear force (top, 3), angular torque
        (bottom, 3)] — confirmed against the solver/viewer kernels."""
        host = self._ext_force.numpy()
        host[body] = (*force, *torque)
        self._ext_force.assign(host)

    def clear_external_forces(self):
        self._ext_force.zero_()

    def set_drag_scale(self, bodies, scale: float):
        """Scale the aero drag of the given bodies (in-place array write; safe
        while the CUDA graph is captured)."""
        self._drag_scale_host[list(bodies)] = float(scale)
        self._drag_scale.assign(self._drag_scale_host)

    # -- core loop -----------------------------------------------------------
    def _simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)        # mouse-drag forces
            wp.launch(_add_wrench, dim=self.model.body_count,
                      inputs=[self._ext_force, self.state_0.body_f],
                      device=self.model.device)
            if self.spring is not None:
                self.spring.apply(self.state_0)
            # Apple tether runs HERE — after the pick/external pull is in body_f
            # but before drag/ground — so each apple's recorded pull force is the
            # user's direct tug on the fruit (used to decide detachment), and the
            # one-sided hold force is added without ever loading the spur.
            if self.apples is not None:
                self.apples.apply(self.state_0)
            ld, ad, gz, gk, gd, gf = self._drag
            vmax, wmax, brake, vfade, wfade = self._vcap
            wp.launch(_drag_ground, dim=self.model.body_count,
                      inputs=[self.state_0.body_q, self.state_0.body_qd, self.body_mass,
                              self._body_inertia, self._drag_scale,
                              ld, ad, gz, gk, gd, gf, vmax, wmax, brake, vfade, wfade,
                              self.sim_dt, self._ang_broken, self.state_0.body_f],
                      device=self.model.device)
            # twig brush runs AFTER the drag/impulse cap: an attached twig is
            # joint-constrained (it cannot teleport like a freed body), so it is
            # exempt from the per-body impulse cap and can be pushed with enough
            # authority to stay out of the robot; the reaction on the heavy robot
            # is far within its own cap, so this stays stable.
            if self.brush is not None:
                self.brush.apply(self.state_0)
            if self.breaker is not None:
                self.breaker.apply_cancel(self.state_0, self.control)   # limp snapped joints (in-graph)
            if self.collisions:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        if self.breaker is not None:
            self.breaker.update(self.state_0)   # flag new breaks; cancel kernel limps them
        if self.apples is not None:
            self.apples.update(self.state_0)    # flag over-pulled apples; tether kernel drops them
        self.sim_time += self.frame_dt
        self._host_step += 1        # invalidates the shared host pose cache below

    def body_q_np(self):
        """Host copy of ``state_0.body_q``, cached once per :meth:`step` and
        shared across callers, so every env's picker reads poses without each
        triggering its own device->host copy (one sync per frame, not N)."""
        if self._bq_np_step != self._host_step:
            self._bq_np = self.state_0.body_q.numpy()
            self._bq_np_step = self._host_step
        return self._bq_np

    def joint_q_np(self):
        """Host copy of ``state_0.joint_q``, cached once per step (see
        :meth:`body_q_np`)."""
        if self._jq_np_step != self._host_step:
            self._jq_np = self.state_0.joint_q.numpy()
            self._jq_np_step = self._host_step
        return self._jq_np

    def render(self):
        if self.viewer is None:
            return
        if self.breaker is not None and hasattr(self.breaker, "render"):
            self.breaker.render(self.viewer)   # recolour newly-snapped branches
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # -- diagnostics ---------------------------------------------------------
    def tip_positions(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()[:, :3]
        return q[[self.tree.seg_to_body[t.index] for t in self.tree.skeleton.terminals]]

    def max_bend(self) -> float:
        if not len(self.tree.joint_ids):
            return 0.0
        return float(np.abs(self.state_0.joint_q.numpy()).max())


@wp.kernel
def _add_wrench(ext: wp.array(dtype=wp.spatial_vector),
                body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    body_f[i] = body_f[i] + ext[i]


@wp.kernel
def _drag_ground(body_q: wp.array(dtype=wp.transform),
                 body_qd: wp.array(dtype=wp.spatial_vector),
                 body_mass: wp.array(dtype=wp.float32),
                 body_inertia: wp.array(dtype=wp.float32),
                 drag_scale: wp.array(dtype=wp.float32),
                 lin_drag: float, ang_drag: float,
                 ground_z: float, ground_k: float, ground_d: float, ground_fric: float,
                 vmax: float, wmax: float, brake: float, vfade: float, wfade: float,
                 dt: float, ang_broken: float,
                 body_f: wp.array(dtype=wp.spatial_vector)):
    """Air drag + a soft ground plane.  Decelerates detached branches/fruit so
    they settle instead of spinning forever, and gently damps the whole tree.

    Newton convention: body_qd = [linear vel (top), angular vel (bottom)] and
    body_f = [force (top), torque (bottom)].  Getting this right is what makes the
    soft ground actually push a fallen apple *up* (an upward force) instead of
    spinning it (a stray torque)."""
    i = wp.tid()
    m = body_mass[i]
    if m <= 0.0:
        return
    qd = body_qd[i]
    v = wp.spatial_top(qd)      # linear velocity
    w = wp.spatial_bottom(qd)   # angular velocity
    vn = wp.length(v)
    wn = wp.length(w)

    # ---- power limit on the APPLIED forces already in body_f (pick/external/
    # tether).  Gravity and the joint springs are computed inside the solver
    # and are unaffected.  The force component pushing the body ALONG its own
    # velocity fades linearly between vfade and 2*vfade — static loading (the
    # slow bend that snaps a branch) sees the full force, but a pick-clamp
    # force (~1000 N on a 5 g twig!) can no longer accelerate it without
    # bound: that runaway used to NaN the articulation and "collapse" the tree.
    F0 = wp.spatial_top(body_f[i])
    T0 = wp.spatial_bottom(body_f[i])
    s = drag_scale[i]                   # 1 normally; <1 marks a snapped subtree
    if s < 0.999:
        # SNAPPED subtree: drop the pick/external force entirely (exactly what
        # detached apples do).  A sustained pick on a limp free hinge pumps the
        # pendulum a little more every swing — bounded for a while, then the
        # amplitude ratchets past the integrable range and the sim "spazzes".
        # The break already released the wood; the hand comes away.
        F0 = wp.vec3(0.0, 0.0, 0.0)
        T0 = wp.vec3(0.0, 0.0, 0.0)
    # impulse cap: an applied force may not accelerate this body by more than
    # vmax within one substep (a STATIC load — bending a branch until it snaps —
    # is held by the structure and never actually produces that acceleration,
    # so it passes through untouched).  This is what stops the snap instant
    # from teleporting a freed 20 g twig to 100 m/s in the substeps before the
    # host notices the break and drops the force.
    fcap = m * vmax / dt
    fn0 = wp.length(F0)
    if fn0 > fcap:
        F0 = F0 * (fcap / fn0)
    tcap = body_inertia[i] * wmax / dt
    tn0 = wp.length(T0)
    if tn0 > tcap:
        T0 = T0 * (tcap / tn0)
    if vn > vfade:
        fpar = wp.dot(F0, v) / vn
        if fpar > 0.0:
            keep = wp.max(0.0, 1.0 - (vn - vfade) / vfade)
            F0 = F0 - (1.0 - keep) * fpar * (v / vn)
    if wn > wfade:
        tpar = wp.dot(T0, w) / wn
        if tpar > 0.0:
            keep = wp.max(0.0, 1.0 - (wn - wfade) / wfade)
            T0 = T0 - (1.0 - keep) * tpar * (w / wn)

    f = -lin_drag * s * m * v           # aerodynamic linear drag (mass-proportional: stable)
    # angular drag keeps most of its strength on snapped subtrees (ang_broken):
    # it barely resists the slow early fall but bleeds off the fast swing
    # through the bottom, so a dropped branch doesn't pendulum back up.
    sa = s
    if s < 0.999 and ang_broken > s:
        sa = ang_broken
    tau = -ang_drag * sa * body_inertia[i] * w  # angular drag, inertia-proportional (stable for all)

    # soft velocity limiter: a strong braking force above (vmax, wmax) — never
    # active in normal motion, a hard backstop against blow-ups.  brake*dt < 1
    # keeps the explicit brake itself stable.
    if vn > vmax:
        f = f - brake * m * (vn - vmax) * (v / vn)
    if wn > wmax:
        tau = tau - brake * body_inertia[i] * (wn - wmax) * (w / wn)

    z = wp.transform_get_translation(body_q[i])[2]
    if z < ground_z:
        pen = ground_z - z
        f = f + wp.vec3(0.0, 0.0, ground_k * m * pen - ground_d * m * v[2])
        f = f - ground_fric * m * wp.vec3(v[0], v[1], 0.0)   # ground friction

    body_f[i] = wp.spatial_vector(F0 + f, T0 + tau)
