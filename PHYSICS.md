# Modelling & physics

This documents how the tree is generated and simulated, and the equations behind
each part.  The approach follows Jacob et al., *Gentle Manipulation of
Tree-Branches* (CoRL 2024), grounded in standard botanical L-systems and beam
theory.

**Parameter provenance.**  Most default values and ranges in `config.py` are
grounded in the literature, with the source cited inline at the parameter.
Highlights: green apple wood — density ~850 kg/m³, MOE ~7 GPa, MOR ~50 MPa
[The Wood Database; USDA FPL Wood Handbook]; apple stem detachment ~14–23 N pull
[Bu et al. 2020; Magni/Hussain PSU; drops with ripeness, Parameswarakumar &
Gupta 1991]; fruit ~52–70 mm and ~0.16 kg [Stemilt grades; USDA], capped small
for the 80 mm Franka Hand; phyllotaxis 137.5° golden angle [Okabe 2015]; pipe
exponent 2.0–2.5 [Shinozaki 1964; Lehnebach 2018]; Ridgeback 1.1 m/s / ~900 N
traction [Clearpath datasheet]; RealSense D435i FOV/range [Intel]; leaf 8.5×5 cm
[CFIA]; orchard-alley roughness ~3 cm [soil random-roughness lit., a proxy].
Where a value is a deliberate sim compromise (tree height, apple size, camera
FOV) the inline comment says so and gives the real figure.

## 0. Apple-tree generator (`lsystem.py: grow_apple`, default)
A stochastic recursive grower in the spirit of **MAppleT** (Costes et al.,
mixed stochastic/deterministic L-systems): a central leader emits a phyllotactic
spiral of 3–6 **scaffold limbs at wide crotch angles (≈45–65° from vertical)**;
each limb recursively forks into a bounded number of laterals that **arch
downward (gravimorphism)** with random wobble.  Every parameter (scaffold count,
angles, droop, length decay, prune probability, "form" from upright to weeping)
is **sampled per seed**, so each tree is distinct but plausible.  Radii come from
the same pipe model as below.  `treesim/config.py` exposes all the ranges
(`ap_*`).  Apples and leaves are placed on the outer canopy spurs.

## 1. L-system morphology (`lsystem.py`, `skeleton.py`)
The `ta`–`td` presets use the classic ternary model below; the default `apple`
preset uses the grower above.
A **parametric ternary 0-L-system** (Prusinkiewicz & Lindenmayer, *The
Algorithmic Beauty of Plants*; Honda's model — the four classes Ta–Td, with **Tb
the paper's class**):

```
axiom : !(w0) F(F0·trunk) /(roll0) A
A     -> !(vr) F(F0) [ &(a) F(F0) A ] /(d1) [ &(a) F(F0) A ] /(d2) [ &(a) F(F0) A ]
F(l)  -> F(l·lr)      # internodes elongate each derivation
!(w)  -> !(w·vr)      # widths thicken each derivation
```

Symbols are interpreted by a 3-D turtle carrying a heading/left/up frame
(`F`=draw internode, `&`=pitch, `/`=roll, `[ ]`=branch).  Tb parameters
(matching the paper's `Σ_b`): `a=18.95°, d1=d2=137.5°, lr=1.109, vr=√3, n=8`.
Body count `= 1 + 2·(3ⁿ−1)` (n=5 → 485).

Each drawn `F` becomes a `Segment` with world endpoints, branch order, and an
**xyzw frame quaternion** (local +Z = branch heading) so capsules and joint
frames map directly into Newton.

**Taper (pipe model).**  Radii are assigned by the da-Vinci / Murray rule
`r_parent^β = Σ r_child^β` (β≈2.3) from a fixed tip radius up to the trunk,
guaranteeing a monotonic taper with the trunk thickest.

**Grounding.**  The finished skeleton is anchored so the **trunk base sits at
the origin** (translating by the bounds minimum — the old behaviour — planted
the lowest drooping twig at z=0 and left the trunk hovering in mid-air), and a
ground-clearance FK pass (`skeleton.lift_above_ground`) pitches any branch
that would dip below ~10 cm back up, the way real limbs grow away from the
soil.  Both are values-only, so multi-env batching is unaffected.

## 2. Newton articulation (`builder.py`)
- One rigid **link** per segment (`add_link`, *not* `add_body`, to avoid an
  auto free joint); a **capsule** collider along local +Z; mass & inertia come
  from `ShapeConfig(density=wood_density)`.
- The trunk base is **welded to the world** (fixed joint).
- Each branch connects to its parent with a joint whose rest pose reproduces the
  skeleton: `parent_xform = (0,0,L_parent)·q_rel`, `child_xform = identity`,
  `q_rel = conj(q_parent)·q_child`.
- **Rigid** mode → fixed joints.  **Deformable** → `D6` with 2 free bending DOFs
  (X,Y), the twist/linear DOFs locked (`revolute`/`spherical` also available).

## 3. Branch stiffness (`physics.py`)
Each compliant joint is a torsional spring-damper.  Two models:

- **Beam (default, Euler-Bernoulli):** `Kp = k·E·r⁴/l` with `k=π/4` (= `E·I/l`,
  `I=πr⁴/4`); the paper uses `k=π/2`.  `Kd = 0.1·Kp`.  Because `Kp ∝ r⁴`, thin
  outer twigs are very compliant and the trunk is stiff — exactly the observed
  behaviour.
- **Rudimentary:** `Kp = φ_u·decay^order`, floored at `φ_l` (the paper's per-level
  exponential decay).

Domain randomisation: Gaussian σ on shape params (paper 0.1) and dynamics
(paper 1.0).

## 4. Deformability engines (`sim.py`, `springs.py`)
Newton's MuJoCo solver gives exact implicit springs but is built for *many small*
articulations, so one large tree is slow.  Two engines:

- **`mujoco`** — exact PD joint springs (`target_ke/kd`).  Real-time for small
  trees (n≤4 ≈ 160 bodies).  Best accuracy.
- **`spring`** *(experimental — fast but currently unstable on the XPBD base)* —
  a Warp **torsional-spring kernel** over a fast maximal-coordinate
  base (XPBD): each substep applies, per joint,
  `τ = −Kp·rotvec(q_rel·q_rest⁻¹) − Kd·(ω_c−ω_p)` as an equal/opposite wrench on
  child/parent (`state.body_f`).  This is the paper's "crude spring abstraction"
  made explicit.  Scales to thousands of branches in real time.  For explicit
  stability, stiffness is capped per joint by the sub-tree inertia,
  `Kp ≤ I·(2·safety/dt)²`, so thin twigs soften automatically.

Interactivity: the OpenGL viewer's mouse-drag forces are injected via
`viewer.apply_forces(state)`; programmatic forces via `Sim.set_external_force`.

## 5. Branch snapping (`breaking.py`)
A branch ruptures when the transmitted bending moment exceeds the modulus of
rupture times the section modulus of a solid circular cross-section:

```
M_max = σ_r · π · r³ / 4
```

The moment is read from the spring law (`|M| = Kp·θ`).

**Snapping = a genuinely free hinge, via in-place actuator-gain zeroing
(`breaking.py`, default `breaking.free_fall`).**  MuJoCo can't cheaply free a
constrained body at runtime (`joint_enabled=0` *welds* it), and editing the
Newton model mid-sim needs a recompile (`notify`, ~1 s) which on a hard pull
cascades into a blow-up.  But the joint's position actuator lives in the
**mjwarp model arrays** `actuator_gainprm`/`actuator_biasprm`, laid out
`(num_worlds, nu)` and *read fresh every step* — so on rupture the broken
DOFs' gain and bias rows are simply **zeroed in place** (per env).  That removes
the spring **and its implicit damping** with no `notify`, no recompile and no
CUDA-graph recapture.  The branch becomes a real free hinge and **swings down
at gravity rate** (the older stiffness-only force-cancellation left the
actuator damping in place, which made snapped branches creep down in slow
motion).  The subtree's aerodynamic drag is also scaled down
(`breaking.broken_drag_scale`, a per-body multiplier array read in-graph) so
air resistance doesn't fake a slow fall.  Rupture-hysteresis (N consecutive
over-threshold frames) prevents transient spikes from chain-snapping the tree,
and snapped wood is recoloured dead-brown.  On non-MuJoCo solvers the old
joint-space stiffness-cancellation kernel (`+Kp·q` into `control.joint_f`)
remains as a fallback, paired with `limp_damping_ratio` soft damping at build.

**Break friction (ramped).**  Snapped joints also get Coulomb friction at the
break (mjwarp `dof_frictionloss`, written in place like the gains, sized as
`break_friction ×` the subtree's rest gravity moment) which **ramps up
~5× over ~1.5 s** as the torn fibers seize.  The branch therefore falls at
near-gravity rate (low friction early), swings through the bottom once, and is
then pinned — it does not pendulum back up past its release height or sway
forever.  Angular drag on snapped subtrees keeps most of its strength
(`broken_ang_drag_scale`) for the same reason; only their *linear* drag is
reduced.

**Settling & safety rails (`sim.py: _drag_ground`).**  Every substep a Warp
kernel processes `state.body_f`:

* aerodynamic drag (`F=-c·m·v`, `τ=-c·I·ω`, scaled per body) and a soft
  ground penalty (`z<0`) land falling branches/apples and gently damp the tree;
* **applied-force safety rails** — the viewer's pick clamp scales with the
  *whole articulation's* mass, so grabbing a 5 g twig can legally apply
  ~1000 N; when such a load is suddenly freed by a rupture it used to teleport
  the twig to ~100 m/s in one substep, NaN the articulation and "collapse" the
  whole tree.  Three rails make that impossible while leaving static loading
  (bending a branch until it snaps) untouched: an **impulse cap** (an applied
  force may not accelerate a body past `max_speed` in one substep), a **power
  limit** (the force component pushing a body along its own velocity fades out
  above `fade_speed` — a hand can't keep pushing something flying away), and a
  **soft speed limiter** (strong braking above `max_speed`/`max_omega`).
  Additionally, once a subtree is snapped the pick/external force on it is
  dropped entirely (exactly as detached apples already did) — sustained
  pick force on a limp free hinge pumps the pendulum without bound.

## 5b. Fruit (`fruit.py`, `builder.py`)
Each apple is an **independent body** held at its hang point under the spur by a
one-sided **spring-damper tether** (a Warp kernel, not a joint — a heavy fruit
jointed to a thin compliant spur is numerically unstable).  By default an apple
is a **3-DOF translational ("slide") body**: an apple never needs to spin, and
halving the per-apple DOF count (vs a 6-DOF free body) speeds the whole sim up
~45 % — apples are the dominant cost.  (`fruit.joint="free"` restores the
legacy free bodies.)

**Pulling the fruit loads the branch** (`fruit.branch_reaction`): the reaction
of the *elastic* part of the tether force, **minus the apple's static weight**
(`f_react = −(f_spring − m·g·ẑ)`), is applied to the spur at the attach point
(wrench about the parent's COM).  At rest `f_spring = m·g·ẑ`, so the reaction
vanishes — the rest pose and rupture margins are exactly as without apples —
but a tug on the fruit visibly bends the branch, in proportion to the pull, and
more on thin outer spurs than on stiff scaffold wood.  Only the spring term is
reacted (the damper acts on the apple's absolute velocity; its reaction is not
passive) and the reaction is clamped (`fruit.reaction_max`) so a detach-strength
yank can never chain-snap the tree.

**Detachment** has two rupture paths: the *direct pull force on the fruit*
(read from `body_f`; mouse pick, hold spring) sustained for
`detach_hysteresis` frames, and the *stem tension itself* (the tether spring's
stretch force) sustained for the longer `tension_hysteresis` with a 1.2x
margin — the second is what lets a gripper holding the fruit purely by contact
friction snap the stem, since contact forces never appear in `body_f`.
Stems are strong (`detach_force` 13–20 N; the pick ceiling is 16 g so a
determined grab tops out ~25 N on a 160 g apple).  A light tug just bends the
branch; a whipping branch still never sheds fruit (transient tension doesn't
survive the hysteresis).  A detached apple is a plain ballistic body — normal
drag and the ground land it, and the mouse pick keeps working so you can carry
it around.  Apples are 4.8–6.4 cm across, sized for the Franka's 8 cm gripper
opening, with grippy skin (mu 1.0) for contact grasps.

## 6. Photoreal bark (`materials.py`, `usd_export.py`)
Procedural, tileable albedo / normal / roughness maps from fractal value noise
plus vertical furrows, bound as a `UsdPreviewSurface` on the recorded USD for
offline path tracing (Omniverse / Blender / Newton ViewerRTX).

## 7. Foliage (`foliage.py`)
Leaves are placed on the outer twigs with phyllotactic spacing.  Each leaf is a
**folded, curled elliptical blade mesh** (~40 triangles: midrib fold, tip curl,
overall droop, double-sided) instead of a flat card, so it catches light like a
real leaf.  Efficiency comes from instancing: all leaves of one **size class**
share ONE `newton.Mesh` object, and there are only 3 discrete size classes, so
the whole canopy renders in 3 draw batches regardless of leaf count (a
continuous per-leaf size would make every leaf a unique geometry and tank the
frame rate).  Per-leaf **colour** variation (brightness/yellowness) is a
per-shape attribute and costs nothing.  Blades stay massless and non-colliding
(collision group 0): physics cost is exactly zero.  `--foliage-physics`
instead puts each leaf on its own compliant petiole joint (expensive).

## 8. Per-env domain randomization (`domainrand.py`, `--randomize-envs`)
mjwarp batches only **structurally homogeneous** worlds, but the *values* may
differ per world.  So every env shares one topology and randomizes continuous
quantities: global scale/stockiness, per-segment length/bend/thickness, wood
density, Young's modulus, rupture stress, apple size/mass/visible count, leaf
size — plus **growth habit** (gravimorphic droop multiplier from upright to
weeping, whole-tree lean, fork spread — crotch-angle scale — and an env-wide
phyllotaxy twist) and **coloration** (bark brightness/warmth tint, foliage
brightness/yellowness, an env-wide apple-palette hue shift).  Habit terms are
applied inside the forward-kinematic re-walk (droop compounds per internode
exactly like real branch arching), so the parent/child graph never changes and
physics stays batched.

## 9. RidgebackFranka (`robot.py`, `--robot`)
IsaacLab's mobile manipulator, modeled IsaacLab's way: the Clearpath Ridgeback
omni base is driven through three dummy planar DOFs (x, y, yaw) with
**velocity actuators** (`JointTargetMode.VELOCITY`; targets written into
`control.joint_target_qd`, read in-graph — no recapture), and the Franka FR3
(real URDF from newton-assets) is welded on top with position servos holding a
camera-forward ready pose.  One robot per env keeps the worlds homogeneous.
**W/S/A/D** drive the base (the viewer camera keeps arrows/Q/E/mouse); the
**wrist depth camera** is Newton's tiled-camera sensor (GPU raycast) mounted on
`fr3_hand`, displayed live in a viewer image panel, one camera per env.  Its
display range is deliberately short (3 m): the tree only "appears" as the robot
closes in, and at manipulation distance apples read as bright round blobs
clearly distinct from the speckled foliage.

**Terrain (`--terrain`).**  Optional bumpy outdoor ground: a value-noise
heightfield (`newton.Heightfield`, 3 octaves, randomized per seed, flattened
under the tree so the trunk stays planted), one global static shape shared by
all envs — batching unaffected, one collision shape total.  Multi-env, the
noise is generated **periodic with exactly the display-grid pitch** and tiled
across the whole displayed stand (plus an apron ring): every world physically
sits at the origin, so the ground drawn under each displayed env is identical
to the terrain the physics samples — every trunk gets its flattened disc and
every robot rides the bumps it appears to stand on.  Gentle by default (5 cm
bumps, ~1.8 m wavelength) so the base stays driveable.

**Solver algorithm.**  The mjwarp constraint solver runs **CG** by default
(`physics.mj_solver`, `--mj-solver`): mjwarp's default "newton" algorithm
spends ~85 % of every substep in a single blocked-Cholesky kernel whose
parallelism is per *world* — one ~700-dof tree keeps a laptop GPU almost idle
(measured 12.7 ms per launch; 33.7 → 4.5 ms/frame by switching, and 10 DR
envs 109 → 12.5 ms).  Iteration caps are irrelevant for both (early
termination on tolerance).  CG reproduced the entire regression matrix
bit-for-bit (rest drift, 17 N pull-bend 0.64 cm, detach, break-fall ratio
0.70, 800 N hard yank, 3 m/s canopy rams, robot stop position).

**Collisions.**  With `--robot` the contact pipeline is enabled using NEWTON's
own collision detection (`use_mujoco_contacts=False`), which honours newton's
collision-group semantics exactly: tree wood and apples are group −1 (collide
with others, never with each other), leaves 0 (never), robot chassis/wheels 3
and arm links 1 (positive groups hit the −1 tree but not each other).  Only
~400 shapes ever reach the broad phase, so robot-vs-tree contact costs almost
nothing (~21 vs ~22.5 fps) — the robot bumper stops against the trunk and the
arm brushes branches aside instead of ghosting through the canopy.  Contact
capacity is sized for the worst case (`nconmax=2048`, `njmax=8192`): shoving
the robot deep into the canopy produced ~666 simultaneous contacts, which
overflowed the old defaults and exploded the solve.  Three more rules keep
full-speed canopy rams stable (verified across seeds at 3 m/s): the base
drive is **wheel-traction-limited** (`drive_effort` ~900 N — without it the
velocity servo is a 12 kN crusher and pinning a branch against the trunk
injects unbounded energy); **gram-scale twigs (< 10 mm radius) take no rigid
contacts** (a 10^4:1 mass-ratio contact is unresolvable — verified to explode
even at very soft contact gains); and **snapped subtrees go fully inert** at
break time (limp zero-stiffness debris pinned by the bumper was the last
blow-up path): the subtree's wood *and the apples riding it* drop out of
collision, its twigs leave the brush kernel, and the pick force ignores it —
the dead limb falls, freezes at the bottom of its arc, and nothing can
disturb it again (the robot drives straight over the debris).  Wood contact
gains are soft (`ke=600`) — green wood gives.  Twigs are still PUSHED ASIDE by the robot through a dedicated
**brush kernel** (`robot.TwigBrush`): a bounded penalty force between twig
capsules and ~20 proxy spheres on the chassis/arm, applied through ``body_f``
— i.e. through the impulse-cap/fade/brake rails — so it is stable by
construction where a contact constraint is not.

**Ground riding.**  The base joint has a vertical axis with a stiff position
servo whose target tracks the terrain height under the chassis every frame
(`RobotDriver.follow_ground`; 0 on flat ground).  Wheels ride ON bumps
instead of the chassis clipping through them.  Multi-env, every world sits at
the same physics origin (newton's recommended layout) and the viewer spreads
them with a **per-dimension pitch** from the env content's AABB — the robot
pads the footprint in x only, so a square pitch left the x gaps tight and the
y gaps wide; per-dimension pitch keeps the visual gap even along both axes.

**Fast multi-env DR startup.**  `--randomize-envs` builds ONE base env,
`replicate()`s it (array tiling), then patches the per-env continuous values
straight into the finalized model arrays — joint rest transforms, shape
scales/transforms, masses/inertias (closed-form capsule formulas, validated
element-wise against the per-env-builder path), joint gains, colours, tether
and rupture tables.  50 rendered envs: 107 s -> 54 s startup, and cross-env
GL instancing survives.  `TREESIM_SLOW_DR=1` forces the original path.

## 10. Fruit perception (`perception.py`)
Depth-only detection chosen for sim-to-real transfer: apples are the only
sphere-like surfaces in an orchard, and a sphere is detectable from geometry
alone — no colour, texture, or learned model (the geometry-first localisation
used by field harvesters, e.g. Silwal et al. 2017; sensor surveys: Gongal et
al. 2015).  Pipeline (~7 ms of host numpy at 192×144): back-project depth to a
point cloud; segment into depth-continuous patches (`scipy.ndimage.label` with
depth-jump edges); per patch, fit a sphere by linear least squares and gate on
radius (2–5.8 cm), millimetric RMS residual, convexity toward the camera,
pixel-count consistency (a sphere of radius r at distance z can only cover
~π(rf/z)² pixels — kills leaf clusters that happen to fit a small sphere), and
silhouette isolation.  Detection range is capped at 1.35 m — where the sensor
has enough pixels-per-fruit — and the picker adds 3-frame temporal persistence
before committing.  Measured on the sim: sub-cm centre error, precision ~0.9.
Detections are highlighted in the 3-D viewer as green marker spheres at the
estimated world position (`viewer.log_points`); the committed pick target is
orange.  Self-detections (the toe-in camera sees the robot's own rounded
links, which fit apple-sized spheres well) are masked from known kinematics,
exactly as a real system would.  The sensor runs at **5 Hz**
(`robot.camera_every = 12`), a realistic RGB-D detection rate — and the
picker's track gates are sized for that cadence.  **Anti-branch gates**: a
short visible section of a thick limb fits a small sphere alarmingly well, so
every candidate passes (1) a footprint-elongation cap (a sphere cap is
isotropic, a branch section stretches along its axis) and (2) an explicit
sphere-vs-cylinder model comparison (points as consistent around a line as
around the fitted centre = wood).  Tuned on 48 rendered canopy vantages x 3
seeds: wood false positives 14 → 3, precision 0.78 → 0.95, at −13%
single-view recall (the survey orbit re-sees fruit from many angles, so track
persistence recovers it).

## 11. Autonomous harvesting (`picker.py`, `--auto`)
The analytic pick-cycle of field systems (Silwal et al. 2017): a host-side
state machine SCAN (steered by a per-pixel **scene mask** — pixels belonging
to the robot's own gripper/arm or to the ground/terrain are excluded before
any "do I see the tree?" decision, else the toe-in view of its own hand or a
wall of near ground returns reads as a tree and lures the base off into the
field) → ALIGN (base to a 0.62–0.80 m stand-off, facing the fruit)
→ REACH (pre-grasp 16 cm behind the fruit on a horizontal approach axis) →
GRASP (advance, close fingers, engage a stiff hand-to-fruit grip spring whose
force feeds the same stem-pull detector a mouse pull uses) → PULL (hold at
the fruit ~0.4 s while the fingers finish closing — yanking against a
half-closed pincer slipped the fruit out — then retract gently at ~0.13 m/s
along the approach axis until the stem strength is exceeded and the fruit
detaches) → TRANSPORT/DROP (over the bucket glued to the chassis,
release, verify the fruit landed inside).  Arm motion is damped-least-squares
IK (`newton.ik`, LM with analytic jacobians) solved on a private Franka-only
model — solving on the full model would wiggle the tree's ~600 joints — with
per-frame slew limiting into the arm's position servos.  Failures (unreachable,
timeouts, stem too strong) blacklist the target and return to SCAN.  Metrics
(`metrics.py`): fps, grasp/pick/place success, cycle time (sim seconds),
throughput, max pull force, branches snapped, detection precision/recall —
printed and saved as JSON (`--metrics`, default `output/metrics_<seed>.json`
with `--auto`).  Recovery is deliberately fast: per-state progress watchdogs (~1-2 s of no
progress fails the attempt) instead of long timeouts, a phantom-target bail in
GRASP, live servoing on the swaying fruit with integral gravity-droop
compensation (the arm's position servos sag a few cm at the TCP), a
reach-aware stand-off (high fruit needs the base closer), and an ORBIT
behaviour that backs out and circles the tree when the base is blocked from
far-side fruit.  The base stuck watchdog fires after ~0.8 s of sustained
drive command with < 2.5 cm motion, arm watchdogs after ~0.9 s without
progress, and RECOVER reverses a full ~0.5 m before rotating away (a short
reverse re-wedged on the same limb).  Failed targets are blacklisted with an
expiry and retried.  Measured: full cycle ≈ 6.5 sim-s.

**Tree-centre belief.**  The picker keeps a persistent estimate of where the
tree is (an EMA of the tree-pixel centroid, refreshed whenever the canopy is
in view — a one-landmark version of the target maps viewpoint planners
maintain).  When the view goes empty the robot no longer rotates blind: for a
couple of seconds it *coasts the orbit* around the remembered centre (circling
the tree necessarily loses sight of it), then turns back toward the belief
along the shortest arc; only a belief that stays empty when stared at is
dropped and blind search resumes.

**Multi-env autonomy.**  With `--num-envs N`, `--auto` runs a SEPARATE picker
on every world: per-env depth, detections, scene mask and metrics (the tiled
camera already renders all worlds in one GPU call; the host-side sphere-fit is
the per-env cost, ~7 ms each).  Only env 0 draws viewer overlays.  Physics
stays batched; the pickers write through shared host mirrors so N robots cost
N small array uploads per frame, not N device round-trips.  Metrics merge into
one JSON: stand-level summary + per-env summaries + all pick records tagged by
env (`metrics.save_combined`).

**Apple sizes.**  Visible fruit is clamped to a realistic dessert-apple band
(4.8–6.8 cm diameter; the ceiling keeps fruit graspable in the Franka's 8 cm
opening).  The DR "not grown" subset — the trick that varies per-tree fruit
count without breaking world homogeneity — is shrunk to 1.2 mm (sub-pixel at
any working distance; it used to be 1 cm, which read as pea-sized fruit), and
the pickers exclude those bodies from every proximity check.
