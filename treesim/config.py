"""Configuration for procedural tree generation and simulation.

The parameter values follow two sources:

* The botanical ternary-tree L-systems of Prusinkiewicz & Lindenmayer,
  *The Algorithmic Beauty of Plants* (ABoP), §1.10 / Honda's model.
* "Gentle Manipulation of Tree Branches: A Contact-Aware Policy Learning
  Approach", Jacob et al., CoRL 2024 (PMLR v270).  The paper models branches
  as *rigid cylindrical links* connected by *torsional mass-spring-dampers*
  whose stiffness follows a beam-deflection law ``Kp = E*pi*r**4 / (2*l)``
  with ``Kd = Kp/10`` (their formulation 1 and §3.1).

Everything the user asked to be "optional" is a boolean / enum here so a tree
can be built rigid-only, or progressively upgraded with deformability,
breaking and foliage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# L-system morphology
# --------------------------------------------------------------------------- #
@dataclass
class LSystemParams:
    """Parameters of a parametric ternary 0L-system (ABoP / Honda model).

    Production rules (turtle interpretation, see ``lsystem.py``)::

        axiom : !(w0) F(F0) /(roll0) A
        A     -> !(vr) F(F0) [ &(a) F(F0) A ] /(d1)
                              [ &(a) F(F0) A ] /(d2)
                              [ &(a) F(F0) A ]
        F(l)  -> F(l * lr)          # internodes elongate every step
        !(w)  -> !(w * vr)          # widths thicken every step

    Angles are stored in **degrees** (as in ABoP); they are converted to
    radians by the turtle.
    """

    a: float = 18.95        # branching angle (pitch of each child off parent)
    # 137.5 deg = the golden angle, the phyllotactic divergence at the shoot
    # apex; apple shows 2/5 spiral phyllotaxis converging on it [Okabe 2015,
    # Sci. Reports 5:15358].
    d1: float = 137.5       # 1st divergence (roll) angle between child planes
    d2: float = 137.5       # 2nd divergence (roll) angle
    lr: float = 1.109       # internode elongation rate per derivation step
    vr: float = 1.732       # width (radius) increase rate ~ sqrt(3) -> area conserving
    n: int = 8              # number of derivation steps (recursion depth)

    F0: float = 0.05        # base internode length [m] (paper uses arbitrary units)
    w0: float = 1.0         # initial turtle width parameter (relative)
    roll0: float = 45.0     # initial roll of the trunk before first node
    trunk_length_factor: float = 4.0   # axiom trunk internode = F0 * this (paper: 200/50)

    # Radius model: "pipe" uses da-Vinci/Murray pipe-model taper (monotonic,
    # trunk thickest); "lsystem" uses the turtle !() width values directly.
    radius_model: str = "pipe"
    tip_radius: float = 0.0015    # twig radius [m] for the pipe model
    # Pipe/da-Vinci model: parent_r^beta = sum(child_r^beta).  beta=2 is area-
    # preserving (da Vinci's rule / Shinozaki pipe model [Shinozaki et al. 1964,
    # Jap. J. Ecology 14; review Lehnebach et al. 2018, Ann. Bot. 121:773]);
    # ~2.49-3 is Murray's law (hydraulic-optimal), but load-bearing tree wood
    # stays near 2 [McCulloh et al. 2004].  2.0-2.5 is the defensible band.
    pipe_beta: float = 2.3        # pipe-model exponent (2=area-conserving, ~2.5=Murray)

    # Absolute scaling of the finished skeleton so the whole tree is a sane size.
    # Modern trained dwarf/semi-dwarf orchard trees are ~3.0-3.5 m [Robinson,
    # "Modern Apple Training Systems," Cornell/UVM 2006]; the sim uses a more
    # compact tree so most fruit falls within the fixed-base Franka's ~0.85 m
    # reach envelope (a taller tree just adds unpickable high fruit).
    target_height: float = 2.4    # final tree is rescaled to this height [m]
    # Mature M.9 trunk ~5-7.5 cm diameter at ~6 yr (radius ~0.025-0.0375 m)
    # [M.9 rootstock caliper / trunk cross-sectional-area data].
    base_radius: float = 0.035    # trunk radius [m] after rescaling

    # Gaussian domain-randomisation sigma on shape params (paper: sigma=0.1).
    shape_jitter: float = 0.0     # 0 -> deterministic; 0.1 -> paper default

    # ------------------------------------------------------------------ #
    # Stochastic apple-tree generator (used when ``kind == "apple"``).
    # Following MAppleT (Costes et al.): a central leader with tiers of
    # scaffold branches at wide crotch angles, phyllotactic spiral, and
    # gravimorphic droop (branches arch downward).  Ranges are sampled
    # per-seed so every tree is different but plausible.
    # ------------------------------------------------------------------ #
    kind: str = "ternary"                       # "ternary" | "apple"
    # A trained central-leader tree carries several tiers of 3-5 scaffolds
    # (~8-15 total); the sim's fewer, longer primary limbs stand in for those
    # tiers [NMSU H-333; PSU Extension apple training].
    ap_scaffolds: tuple = (3, 6)                # number of main scaffold limbs
    ap_trunk_internodes: tuple = (2, 4)         # leader internodes before 1st scaffold
    # 45-60 deg crotch angle maximizes limb strength and fruit-bud formation;
    # <45 deg gives weak crotches [UW-Madison Extension; Warner & Barden 1991,
    # HortScience 26(10):1266].
    ap_scaffold_pitch: tuple = (42.0, 65.0)     # scaffold angle from vertical [deg]
    ap_lateral_pitch: tuple = (32.0, 58.0)      # sub-branch angle from parent [deg]
    # Centred on the 137.5 deg golden angle (phyllotactic spiral) [Okabe 2015].
    ap_divergence: tuple = (128.0, 148.0)       # phyllotactic roll between branches [deg]
    ap_droop: tuple = (4.0, 13.0)               # gravimorphic down-bend per internode [deg]
    ap_len_decay: tuple = (0.66, 0.82)          # child length / parent length
    ap_children: tuple = (2, 3)                 # laterals spawned per fork
    ap_prune: tuple = (0.08, 0.40)              # probability a bud aborts
    ap_wobble: tuple = (5.0, 16.0)              # random heading wobble per internode [deg]
    ap_internodes_per_axis: tuple = (2, 4)      # internodes between forks
    ap_leader_decay: float = 0.82               # leader weakening per tier
    ap_form_droop_bias: tuple = (0.6, 1.6)      # global droop multiplier (upright..weeping)


def preset(name: str) -> LSystemParams:
    """Return one of the four ABoP ternary classes (Ta..Td).

    Tb is exactly the class used in the CoRL paper (``Sigma_b``).
    """
    name = name.lower()
    table = {
        # name : (a,     d1,     d2,     lr,    vr,    n)
        "ta": (18.95, 94.74, 132.63, 1.109, 1.732, 10),
        "tb": (18.95, 137.5, 137.5, 1.109, 1.732, 8),   # paper's class
        "tc": (22.5, 112.5, 157.5, 1.790, 1.732, 8),
        "td": (36.0, 180.0, 252.0, 1.070, 1.732, 6),
    }
    if name == "apple":
        # central-leader apple tree; ``n`` is the max branch order (recursion depth)
        return LSystemParams(kind="apple", n=4, target_height=2.6, base_radius=0.055,
                             tip_radius=0.004, pipe_beta=2.2)
    if name not in table:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(table) + ['apple']}")
    a, d1, d2, lr, vr, n = table[name]
    return LSystemParams(a=a, d1=d1, d2=d2, lr=lr, vr=vr, n=n)


# --------------------------------------------------------------------------- #
# Physics / dynamics
# --------------------------------------------------------------------------- #
class StiffnessModel(str, Enum):
    BEAM = "beam"              # Kp = E*I/l (Euler-Bernoulli), physically grounded
    RUDIMENTARY = "rudimentary"  # per-level exponential decay (paper's model a)
    RIGID = "rigid"           # extremely stiff joints -> behaves rigid


@dataclass
class PhysicsParams:
    # Material -----------------------------------------------------------------
    # GREEN (living) apple wood.  Apple basic specific gravity 0.61 (ovendry
    # mass / green volume); with green moisture this gives ~850-1000 kg/m^3.
    # [The Wood Database, "Apple" (Malus domestica): 830 kg/m^3 dried, basic
    # SG 0.61; USDA FPL Wood Handbook FPL-GTR-190 Ch.3/5 green-density relation]
    wood_density: float = 850.0        # kg/m^3 (green apple ~ 780-1000)
    # Green apple MOE ~7 GPa: apple is 8.76 GPa at 12% MC (Wood Database),
    # and green MOE runs ~10-25% below the dried value (FPL green-hardwood
    # table; black cherry green 9.0 vs 10.3 GPa at 12%, ~13% lower).
    youngs_modulus: float = 7.0e9      # Pa; green apple ~ 6-8 GPa
    # The CoRL beam model writes Kp = E*pi*r^4/(2 l); standard Euler-Bernoulli
    # cantilever rotational stiffness is E*I/l with I = pi*r^4/4.  ``beam_factor``
    # selects between them:  Kp = beam_factor * E * r^4 / l.
    beam_factor: float = 3.14159265 / 4.0   # = E*I/l  (set to pi/2 for paper)
    # Kd = damping_ratio * Kp (numerical joint damping, not a physical zeta).
    # For reference, measured structural damping of a BARE woody branch is
    # zeta ~= 0.01-0.03, rising to ~0.06-0.075 once foliated [James 2014,
    # "A Study of Branch Dynamics on an Open-Grown Tree," Arboric. & Urban
    # Forestry 40(3):125; Moore & Maguire 2004, Trees 18:195 (internal
    # damping < 0.05)]; the 0.1 here is a solver-stability value tuned so the
    # tree neither creeps nor rings, kept above the physical figure on purpose.
    damping_ratio: float = 0.1         # Kd = damping_ratio * Kp   (paper: 1/10)

    # Rudimentary model (paper's "model a") -----------------------------------
    phi_upper: float = 50.0            # stiffness at the trunk
    phi_lower: float = 0.05            # floor to avoid solver blow-up
    phi_decay: float = 0.45            # multiply per branch level away from trunk

    model: StiffnessModel = StiffnessModel.BEAM

    max_stiffness: float = 5.0e5       # cap Kp for solver stability [N*m/rad]
    min_stiffness: float = 1.0e-4      # floor Kp so twigs are not numerically free
    joint_limit: float = 2.0           # +/- bending soft-limit per dof [rad] (if use_limits)
    use_limits: bool = False           # hard limits = MuJoCo constraints (slow on big trees)
    armature: float = 0.0              # added rotor inertia per dof (stability)

    # Joint type for compliant branches.  D6 (2 bending dofs) is closest to a
    # real branch; "revolute" matches the paper's single-dof PD joints.
    joint_type: str = "d6"             # "d6" | "revolute" | "spherical"

    # Domain randomisation on dynamics (paper: Gaussian sigma=1.0 on rudimentary).
    dynamics_jitter: float = 0.0

    gravity: float = -9.81

    # Aerodynamic drag + a soft ground plane.  Drag makes detached branches/fruit
    # decelerate and settle instead of spinning forever (and damps the whole tree
    # a little, which also improves stability).  The soft ground catches fallen
    # pieces without the cost of the full collision pipeline.
    linear_drag: float = 1.5           # 1/s; F = -linear_drag * m * v
    angular_drag: float = 2.5          # 1/s; tau = -angular_drag * I * w (stops spin)
    ground_z: float = 0.0              # height of the soft ground plane [m]
    # Soft ground is a spring-damper penalty (per unit mass) on bodies below it.
    # ground_damping is kept >= critical (2*sqrt(ground_stiffness) ~= 110 here) so
    # a falling apple is caught and settles WITHOUT bouncing back up.
    ground_stiffness: float = 3000.0   # N/m per kg penalty when below ground
    ground_damping: float = 120.0      # vertical damping on contact (>= critical: no bounce)
    ground_friction: float = 6.0       # horizontal velocity damping on the ground
    # Optional bumpy outdoor terrain (--terrain): a value-noise heightfield
    # (newton.Heightfield), gentle enough to stay driveable, randomized per
    # seed, flattened under the tree so the trunk stays planted.  ONE global
    # static field shared by all envs (like the ground plane), so multi-env
    # batching is unaffected and it costs a single collision shape.
    terrain: bool = False
    # A grassed orchard alley is untilled: agricultural soil "random roughness"
    # (std. dev. of surface elevation) runs ~0.7 cm (no-till/firm sod) to 5 cm
    # (freshly disked); a sod alley sits at the low end [Allmaras/Zobeck tillage
    # roughness literature].  No orchard-floor-specific amplitude is published,
    # so ~3 cm is a defensible mid proxy for a driveable grassed alley.
    terrain_amplitude: float = 0.03    # max bump height [m] (orchard alley ~1-3 cm)
    terrain_wavelength: float = 1.8    # dominant bump size [m]
    terrain_extent: float = 14.0       # half-extent of the field [m]
    # Soft velocity limiter (anti-blowup): bodies faster than this get a strong
    # braking force (inactive below the caps, so normal physics is untouched).
    # This is what stops a pick-clamp-scale yank on a 5 g twig (the viewer
    # allows ~5*g*TREE mass ~ 1000 N on any tree body!) from accelerating it to
    # explosion -> NaN -> "every joint over threshold" total collapse.
    max_speed: float = 25.0            # linear speed cap [m/s]
    max_omega: float = 60.0            # angular speed cap [rad/s]
    brake: float = 120.0               # braking gain above the caps [1/s] (< 1/dt)
    # Power limit on APPLIED forces (pick/external/tether — gravity and the
    # joint springs live inside the solver and are untouched): the component of
    # body_f pushing a body ALONG its velocity fades out between fade_speed and
    # 2*fade_speed, like a hand that can't keep pushing something already
    # flying away.  Static loading (bending/breaking a branch) sees the full
    # force; runaway acceleration of a light twig becomes impossible.
    fade_speed: float = 8.0            # m/s; no body moves this fast normally
    fade_omega: float = 30.0           # rad/s, same idea for applied torques
    # Viewer mouse-pick strength ceiling, in multiples of g (newton default 5).
    # Raised so a firm grab can beat the new, stronger apple stems; the impulse
    # cap + force fade above keep even the hardest yank integrable.
    pick_max_acceleration: float = 16.0
    # mjwarp constraint-solver algorithm: "cg" (matrix-free, ~7x faster on a
    # single big articulation — the default "newton" spends the whole step in
    # one blocked-Cholesky kernel that parallelizes per WORLD) or "newton".
    # Physics-identical on the full regression matrix; see sim._make_solver.
    mj_solver: str = "cg"


# --------------------------------------------------------------------------- #
# Branch breaking / snapping
# --------------------------------------------------------------------------- #
@dataclass
class BreakParams:
    enabled: bool = False
    # A branch ruptures when the bending moment at its base exceeds the modulus
    # of rupture times the section modulus:  M_max = sigma_r * pi * r^3 / 4.
    # GREEN apple MOR ~50-55 MPa: apple is 88.3 MPa at 12% MC (Wood Database),
    # and green MOR runs ~35-40% below dried (FPL green-hardwood table; black
    # cherry green 55 vs 84.8 MPa at 12%).  NOTE: living branches greenstick-
    # fracture / buckle rather than snap cleanly, tolerating stress past linear
    # MOR [Ozden & Ennos 2014, "Why don't branches snap?", Wood Sci. Technol.],
    # so this is deliberately near the LOW end of the green range so the picker
    # meets realistic resistance before a branch gives.
    rupture_stress: float = 5.0e7      # Pa; green apple MOR ~ 45-60 MPa
    safety_factor: float = 1.0
    # Alternatively trigger on bend angle exceeding this (radians); None disables.
    max_bend_angle: float | None = None
    # "detach": broken branch (and its subtree) fully separates and falls
    #   (requires 6-DOF joints, built automatically).
    # "hinge":  broken joint goes limp; the branch droops/hangs (greenstick /
    #   hanging-by-bark), keeping 2-DOF joints (lighter, faster).
    mode: str = "detach"
    linear_stiffness: float = 5.0e4    # near-rigid lock for the 3 linear DOFs (detach mode)
    twist_stiffness_scale: float = 3.0  # twist DOF stiffness = scale * bending Kp
    broken_color: tuple = (0.30, 0.16, 0.10)  # recolour for snapped branches
    # When breaking is enabled, branch joints use this damping ratio (Kd = ratio*Kp)
    # instead of physics.damping_ratio.  It is lower (0.1 -> 0.03) so that once a
    # joint is snapped limp it actually DROOPS/flops under gravity instead of
    # creeping (Kd=0.1*Kp is several times over-critical and a snapped branch then
    # barely sags).  The tree still settles because the aerodynamic angular drag
    # also damps every body; verified the rest pose stays calm at this value.
    hysteresis_frames: int = 3         # consecutive over-threshold frames before a joint snaps
    # free_fall (MuJoCo solver only): on rupture the joint's position-actuator
    # gains (stiffness AND implicit damping) are zeroed IN PLACE in the mjwarp
    # model arrays — no notify, no recompile, no graph recapture — so the broken
    # joint is a genuinely free hinge and the branch swings down at gravity
    # rate instead of creeping against the residual actuator damping.  The
    # branch's aero drag is also scaled down (broken_drag_scale) so air
    # resistance doesn't fake a slow-motion fall.  With free_fall the tree is
    # built with the NORMAL damping ratio (pre-break dynamics identical to a
    # non-breakable tree); the limp_damping_ratio soft-damping hack is only
    # used on the fallback path (non-MuJoCo solvers).
    free_fall: bool = True
    broken_drag_scale: float = 0.2     # LINEAR aero-drag multiplier for snapped subtrees
    # Snapped joints get Coulomb FRICTION at the break (mjwarp dof_frictionloss,
    # written in place like the gains): torn fibers grinding.  Sized as a
    # fraction of the subtree's rest gravity moment, so the branch still falls
    # near gravity rate but loses energy every swing and, once hanging, is
    # PINNED — it drops, settles, and does not pendulum back up.
    break_friction: float = 0.2
    # The friction RAMPS UP after the snap (torn fibers seize): the fall sees
    # ~break_friction, but within ramp_time the torque grows by ramp_max, so
    # the branch drops at speed, swings through once, and is then pinned —
    # it does not pendulum back up or sway forever.
    break_friction_ramp_max: float = 5.0
    break_friction_ramp_time: float = 1.5   # seconds to reach ramp_max
    # FREEZE at the bottom of the fall: the moment a snapped subtree's vertical
    # velocity flips from falling to rising, its joint friction is jumped to
    # freeze_friction x the subtree's gravity moment — the branch simply stays
    # where it landed (no upswing, no residual weirdness), and since the pick
    # force is already dropped on snapped subtrees it can't be interacted with.
    # A branch that never falls (broke in place) freezes after freeze_timeout.
    freeze_at_bottom: bool = True
    freeze_friction: float = 60.0           # x subtree gravity moment (>> 1 = pinned)
    freeze_fall_speed: float = 0.25         # m/s of downward COM speed = "it is falling"
    freeze_timeout: float = 3.0             # s after break: freeze regardless
    # Angular drag on snapped subtrees keeps most of its strength (unlike the
    # linear drag): it barely resists the slow early fall but bleeds off the
    # fast swing through the bottom, killing the upswing.
    broken_ang_drag_scale: float = 0.8
    limp_damping_ratio: float = 0.012  # FALLBACK only (free_fall off/unavailable): branch
                                       # Kd = this * Kp so snapped branches still droop


# --------------------------------------------------------------------------- #
# Rendering / assets
# --------------------------------------------------------------------------- #
@dataclass
class RenderParams:
    wood_tint: tuple = (1.0, 1.0, 1.0)  # per-env wood colour multiplier (DR)


# --------------------------------------------------------------------------- #
# Foliage
# --------------------------------------------------------------------------- #
@dataclass
class FoliageParams:
    enabled: bool = False
    leaves_per_terminal: int = 3
    min_order_for_leaves: int = 3      # only twigs at/after this branch order
    # Apple leaf blade ~4-13 cm long x 3-7 cm wide, elliptic-ovate, L:W ~1.7:1
    # [CFIA "Biology of Malus domestica"; midpoint ~8.5 x 5 cm].
    leaf_length: float = 0.085         # m
    leaf_width: float = 0.050          # m  (L:W ~ 1.7, elliptic-ovate)
    leaf_color: tuple = (0.18, 0.42, 0.12)  # per-env foliage colour (DR varies it)
    leaf_mass: float = 0.002           # kg (only used by the --foliage-physics path)
    physics: bool = False              # give each leaf a compliant petiole joint
    petiole_stiffness: float = 0.02
    petiole_damping: float = 0.002

    def set_density(self, d: float) -> None:
        """Map one 0..~2 'density' dial to concrete leaf parameters.

        The render-only leaves are massless, non-colliding cards that all share
        one geometry, so the viewer INSTANCES them: density is purely visual and
        costs no physics and almost no render time (frame rate is independent of
        how many leaves there are).  0 = bare; ~0.5 = light; ~1 = a lush canopy;
        >1 puts leaves on inner branches too, to obscure the tree's interior.
        """
        self.enabled = d > 0.0
        if d <= 0.0:
            return
        self.leaves_per_terminal = max(1, int(round(2 + d * 3)))
        self.min_order_for_leaves = 2 if d > 1.0 else 3   # inner branches at high density
        # centred on the real ~8.5 x 5 cm apple leaf [CFIA] at the nominal
        # density dial, scaling gently with it (keeps L:W ~ 1.7)
        self.leaf_length = 0.065 + d * 0.033
        self.leaf_width = 0.038 + d * 0.020


# --------------------------------------------------------------------------- #
# Fruit (apples)
# --------------------------------------------------------------------------- #
@dataclass
class FruitParams:
    enabled: bool = False
    min_order: int = 2                  # apples grow on >=2-year-old wood (spurs), outer canopy
    prob_per_spur: float = 0.35         # chance an eligible spur bears fruit
    max_count: int = 60                 # cap total apples (perf)
    # Real dessert apples are 58-92 mm diameter (typical ~70-75 mm) [Stemilt
    # commercial size chart; USDA "medium" ~76 mm].  Capped SMALLER here: the
    # Franka Hand opens only 80 mm [Franka Hand manual], so a real 75 mm apple
    # leaves ~2 mm/side clearance — unpickable in practice.  5-7 cm keeps fruit
    # graspable while staying at the small end of the real commercial range.
    radius: tuple = (0.026, 0.035)      # apple radius [m] (~5.2-7 cm diameter)
    # Typical commercial dessert apple ~150-180 g; grade range 90-400 g
    # [Stemilt size chart; USDA medium apple ~182 g].
    mass: float = 0.16                  # kg (a real apple ~ 0.09-0.40, typ 0.15-0.18)
    # Pedicel ~20-35 mm, diameter ~2-3 mm across cultivars [US plant-patent
    # pomological descriptions, e.g. PP27368 'MORED' 24-27 mm, PP28076 'MAKALI'
    # ~32 mm, PP28359 'MILLY' 25-29 mm].
    stem_length: float = 0.025          # m, pedicel (visual hang below attach)
    # Each apple is an independent FREE rigid body (its own 1-body articulation,
    # NOT jointed into the tree).  At runtime a one-sided spring-damper "tether"
    # holds it at its hang point under the spur.  The tether is applied to the
    # apple ONLY — never reacted back onto the compliant spur — so a heavy apple
    # can never destabilise the thin branch it hangs from.  When a pull stretches
    # the tether past ``detach_force`` of tension the tether is cut: the apple
    # becomes a plain free body and falls under gravity + drag onto the soft
    # ground.  Detaching is just a flag flip (no model edit, no notify, no
    # graph-recapture) so it is free and never hitches the frame rate.
    tether_stiffness: float = 550.0     # N/m holding the apple at its hang point (firm stem)
    tether_damping_ratio: float = 0.9   # tether damping as a fraction of critical
    # Stem strength: a DIRECT pull on the fruit above this many newtons,
    # SUSTAINED for detach_hysteresis frames, snaps it.  The viewer clamps its
    # mouse-pick force on an apple to physics.pick_max_acceleration * g * mass
    # (12 * 9.81 * 0.16 ~= 18.8 N), so the upper bound must stay below that or a
    # ripe apple could never be tugged off.  You have to grab the fruit and
    # pull HARD for ~a tenth of a second — a light tug just bends the branch.
    # Straight-pull detachment force from the harvesting literature: 28.3 N to
    # detach 95% of apples [Bu et al. 2020, Scientia Horticulturae 261:108937];
    # 24.8 N fruit-end vs 19.9 N spur-end for green fruit [Magni/Hussain, PSU
    # biorobotics].  Detachment force DROPS with ripeness as the abscission zone
    # weakens [Parameswarakumar & Gupta 1991, cited in Zhang et al. 2016, Comp.
    # Electron. Agric.], and bending/twisting lowers it further [Bu 2020].  So
    # RIPE harvest fruit (what the picker targets, pulled ~straight) sits at/
    # below the ~20-25 N green-fruit numbers.  Upper bound stays under the
    # viewer pick clamp (pick_max_acceleration*g*mass = 16*9.81*0.16 = 25.1 N).
    detach_force: tuple = (14.0, 23.0)  # N of direct pull that snaps a ripe apple's stem
    detach_hysteresis: int = 6          # consecutive over-threshold frames before it lets go
    # The stem also breaks under sustained TENSION in the stem itself (however
    # applied — e.g. the gripper pulling the fruit while holding it by contact
    # friction, which never shows up as a body force).  Longer hysteresis than
    # the direct-pull path so a whipping branch still never sheds fruit, and a
    # margin factor so gravity + transient tether overshoot on a NEAR-threshold
    # direct pull can't sneak past the intended strength.
    tension_hysteresis: int = 12
    tension_factor: float = 1.2         # tension threshold = factor * detach_force
    # DEPRECATED (kept for compat): detached apples are now plain ballistic
    # free bodies — normal aero drag + the ground land them, and the mouse pick
    # keeps working so you can carry a picked apple around.
    detach_drag: float = 0.0
    # Pulling an apple LOADS ITS BRANCH: the *excess* tether tension (spring force
    # minus the apple's static weight, which the tree already carries implicitly)
    # is reacted onto the spur at the attach point, so tugging a fruit visibly
    # bends the branch toward you, proportional to how hard you pull.  Only the
    # elastic (spring) part is reacted — reacting the damper on the apple's
    # absolute velocity is not passive and can pump energy into the spur.  The
    # reaction is clamped so a detaching yank can never chain-snap the tree.
    branch_reaction: float = 1.0        # 0 = old one-sided tether; 1 = full excess reaction
    reaction_max: float = 10.0          # clamp on the reaction force [N] (safety)
    # How the apple connects to the world: "slide" = 3 translational DOFs (an
    # apple never needs to spin; HALF the free-joint DOF count, which is the
    # whole sim cost) or "free" = full 6-DOF free body (legacy).
    joint: str = "slide"
    # reds / red-orange / green-yellow, spanning the cultivar colour classes:
    # dark red (Red Delicious/Starkrimson, CIELAB L*~41), red-striped (Jonagold
    # L* 58.6, a* +13.4, b* +29.1), green (Granny Smith L* 63.5, a* -17.6,
    # b* +38.0) [Zhang et al., PMC8951106; bag-removal study PMC6269864].
    colors: tuple = ((0.78, 0.10, 0.07), (0.85, 0.30, 0.05),
                     (0.65, 0.62, 0.12))  # reds / red-green / green-yellow


# --------------------------------------------------------------------------- #
# Mobile manipulator (Ridgeback base + Franka arm, IsaacLab's RidgebackFranka)
# --------------------------------------------------------------------------- #
@dataclass
class RobotParams:
    """A RidgebackFranka mobile manipulator in every env, driveable with WASD.

    Modeled the way IsaacLab models it: the Clearpath Ridgeback omnidirectional
    base is NOT simulated wheel-by-wheel — it is driven through three "dummy"
    planar DOFs (world-x, world-y, yaw) with velocity actuators, and the Franka
    arm (real FR3 URDF from newton-assets) is welded on top.  One extra body +
    planar joint + the arm's ~11 links per env: negligible next to the tree.
    """
    enabled: bool = False
    position: tuple = (2.4, 0.0)       # base spawn in the env's local frame [m]
    yaw: float = 3.14159265            # spawn heading [rad] (pi = facing the tree)
    # Clearpath Ridgeback max speed is 1.1 m/s [Clearpath Ridgeback datasheet];
    # real orchard harvesters drive ~0.3-0.5 m/s while picking and up to ~1.4
    # m/s in row transit [MDPI Remote Sens. 14(3):675; arXiv 2507.15484].  The
    # autonomous picker already commands 0.2-0.55 m/s internally; this cap is
    # the WASD teleop / row-transit ceiling.
    drive_speed: float = 1.1           # W/S forward/back target speed [m/s]
    turn_speed: float = 1.5            # A/D yaw rate [rad/s]
    drive_kd: float = 4000.0           # velocity-servo gain for x/y [N per m/s]
    turn_kd: float = 800.0             # velocity-servo gain for yaw [N*m per rad/s]
    # Wheel-traction-scale effort limits on the base drive.  Without them the
    # velocity servo is an unstoppable ~12 kN crusher: pinning a branch between
    # the bumper and the trunk injected unbounded contact energy and NaN'd the
    # sim.  With them the base simply STALLS against obstacles, like a real
    # Ridgeback would.  Friction-limited traction F = mu*m*g: the Ridgeback is
    # 135 kg [Clearpath datasheet] and its Mecanum wheels give mu ~ 0.6-0.8, so
    # F_max ~ 800-1060 N (rubber-on-floor mu ~ 0.7-0.9, derated for Mecanum
    # roller slip) — 900 N sits squarely in that band.
    drive_effort: float = 900.0        # max drive force per axis [N]
    turn_effort: float = 350.0         # max yaw torque [N*m]
    arm_kp: float = 1500.0             # arm position-servo stiffness [N*m/rad]
    arm_kd: float = 120.0              # arm position-servo damping
    # wrist-mounted depth camera (rendered with newton's tiled camera sensor
    # into a live image panel in the viewer)
    camera: bool = True
    camera_width: int = 192
    camera_height: int = 144
    # Intel RealSense D435i (a common wrist RGB-D): depth FOV 87 deg H x 58 deg
    # V, min depth ~0.1 m, usable 0.3-3 m, up to 90 fps [Intel D435i specs].
    # We render a single square-ish cone at 75 deg (between the H and V of the
    # real sensor) for wider scan coverage than the 58 deg vertical alone.
    camera_fov: float = 75.0           # vertical FOV [deg] (real D435i: 87Hx58V)
    # Display/detection range.  Kept SHORT (under the D435i's 3 m usable max):
    # the far plane crushes to black, so the tree only "appears" once the robot
    # gets close, and the near-range contrast is spent where it matters — at
    # manipulation distance, apples read as bright round blobs against foliage.
    camera_range: float = 2.0          # depth display range [m] (D435i: 0.3-3 m)
    camera_every: int = 12             # sensor cadence in frames: 12 = 5 Hz,
                                       # a realistic RGB-D detection rate (and
                                       # the camera+perception stack off the
                                       # per-frame budget); tracks/gates in the
                                       # picker are sized for this cadence


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class TreeConfig:
    lsystem: LSystemParams = field(default_factory=lambda: preset("tb"))
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    breaking: BreakParams = field(default_factory=BreakParams)
    render: RenderParams = field(default_factory=RenderParams)
    foliage: FoliageParams = field(default_factory=FoliageParams)
    fruit: FruitParams = field(default_factory=FruitParams)
    robot: RobotParams = field(default_factory=RobotParams)

    # Global toggles ----------------------------------------------------------
    deformable: bool = False     # False -> all joints welded rigid; True -> compliant
    seed: int = 0
    device: str = "cuda"         # "cuda" | "cpu"

    @classmethod
    def rigid(cls, preset_name: str = "tb", n: int = 5) -> "TreeConfig":
        c = cls(lsystem=preset(preset_name))
        if preset_name != "apple":
            c.lsystem.n = n
        c.deformable = False
        c.physics.model = StiffnessModel.RIGID
        return c

    @classmethod
    def compliant(cls, preset_name: str = "tb", n: int = 5) -> "TreeConfig":
        c = cls(lsystem=preset(preset_name))
        if preset_name != "apple":
            c.lsystem.n = n
        c.deformable = True
        c.physics.model = StiffnessModel.BEAM
        return c
