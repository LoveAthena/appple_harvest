"""Apple placement on the tree.

Apples grow on short fruiting spurs on 2+-year-old wood in the well-lit
mid/outer canopy.  Each becomes (in the builder) a sphere hanging from a short
pedicel (stem) via a compliant joint, so it dangles and sways; the stem is
breakable, so a firm pull (or a whipping branch) snaps the apple off.

This module only chooses placements (pure geometry); the builder creates the
bodies and breakable stem joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .config import FruitParams
from .skeleton import TreeSkeleton


@dataclass
class ApplePlacement:
    parent_seg: int          # spur segment the apple hangs from
    attach: np.ndarray       # world position of the pedicel base
    radius: float
    color: tuple


def place_apples(skel: TreeSkeleton, fp: FruitParams, seed: int = 0) -> list[ApplePlacement]:
    rng = np.random.default_rng(seed + 4242)
    max_order = max(s.order for s in skel.segments)
    thr = min(fp.min_order, max_order)

    # eligible spurs: thin outer/mid wood, prefer well-lit upper-outer canopy
    lo, hi = skel.bounds()
    span = max(hi - lo)
    eligible = [s for s in skel.segments
                if s.order >= thr and s.radius_start < 0.02]
    rng.shuffle(eligible)

    out: list[ApplePlacement] = []
    for s in eligible:
        if len(out) >= fp.max_count:
            break
        # bias toward the outer canopy (distance from trunk axis) and good light
        rad_xy = float(np.hypot(s.midpoint[0] - (lo[0] + hi[0]) / 2,
                                s.midpoint[1] - (lo[1] + hi[1]) / 2))
        outerness = np.clip(rad_xy / (0.5 * span + 1e-6), 0.0, 1.0)
        if rng.random() > fp.prob_per_spur * (0.4 + 0.6 * outerness):
            continue
        # attach near the spur tip; the apple itself hangs below (gravity does that)
        t = rng.uniform(0.5, 1.0)
        attach = s.start + s.direction * (t * s.length)
        col = fp.colors[rng.integers(0, len(fp.colors))]
        # slight per-apple colour variation
        col = tuple(float(np.clip(c + rng.normal(0, 0.04), 0, 1)) for c in col)
        # radius quantized to 2 mm steps: the GL viewer builds one instancer
        # (its own VBO upload every frame) per UNIQUE sphere size, so 40
        # continuous radii = 40 draw groups where 6 size classes render in 6.
        # 2 mm is below anything the perception gates or grasp distinguish.
        r = float(rng.uniform(*fp.radius))
        r = float(np.clip(round(r / 0.002) * 0.002, fp.radius[0], fp.radius[1]))
        out.append(ApplePlacement(parent_seg=s.index, attach=attach, radius=r, color=col))
    return out


# --------------------------------------------------------------------------- #
# Runtime: apples held to their spur by a spring-damper tether.
#
# Each apple is an independent body (built in builder.py; translational "slide"
# joint by default, legacy free body optional).  Every substep this kernel
# pulls each still-attached apple toward its hang point a fixed distance below
# the (moving) spur attach point.
#
# The tether also LOADS THE BRANCH: the reaction of the *elastic* part of the
# tether force, minus the apple's static weight (the rest-pose already accounts
# for that; reacting it would sag every spur and could snap thin ones at rest),
# is applied to the parent spur at the attach point.  So *pulling the fruit
# pulls the branch*, in proportion to the pull force — but idle apples load
# nothing, and the reaction is clamped so a detach-strength yank can never
# chain-snap the tree.  Only the spring term is reacted; the damper acts on the
# apple's absolute velocity, and its reaction is not passive (it can pump
# energy into the spur).
#
# Detachment is decided by how hard the USER pulls the fruit ITSELF: the kernel
# runs after the mouse-pick / external pull is in ``body_f`` but before drag/
# ground, so ``body_f[apple]`` at entry is exactly that pull.  Its magnitude is
# recorded; when it exceeds the apple's ``detach_force`` (for a couple of frames)
# the host flags the apple detached.  Branch motion applies no force to the
# apple, so it is ignored — only a direct tug on the fruit snaps the stem.  A
# detached apple stops being tethered and falls under gravity + the drag/
# soft-ground kernel.  No model edit, notify or graph-recapture is involved,
# so it is cheap and cannot destabilise the tree.
# --------------------------------------------------------------------------- #
@wp.kernel
def _apple_tether(body_q: wp.array(dtype=wp.transform),
                  body_qd: wp.array(dtype=wp.spatial_vector),
                  apple_body: wp.array(dtype=wp.int32),
                  parent_body: wp.array(dtype=wp.int32),
                  offset: wp.array(dtype=wp.vec3),
                  hang_drop: wp.array(dtype=wp.float32),
                  k: float, c: float,
                  body_mass: wp.array(dtype=wp.float32),
                  body_com: wp.array(dtype=wp.vec3),
                  react_scale: float, react_max: float, grav: float,
                  detached: wp.array(dtype=wp.int32),
                  held: wp.array(dtype=wp.int32),
                  hold_body: wp.array(dtype=wp.int32),
                  hold_off: wp.vec3, hold_k: float, hold_c: float,
                  pull: wp.array(dtype=wp.float32),
                  tension: wp.array(dtype=wp.float32),
                  body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    ab = apple_body[i]
    # The pull on this apple = pick/external force already on body_f (this
    # kernel runs before drag/ground add anything) PLUS the gripper's hold
    # spring, if a robot hand is holding it — so a robot pulling the fruit
    # snaps the stem exactly like a mouse pull.  Newton: force = top.
    entry = wp.spatial_top(body_f[ab])
    fh = wp.vec3(0.0, 0.0, 0.0)
    if held[i] != 0:
        hp = wp.transform_point(body_q[hold_body[i]], hold_off)   # grip point, world
        p = wp.transform_get_translation(body_q[ab])
        v = wp.spatial_top(body_qd[ab])
        kh = hold_k
        if detached[i] != 0:
            # while the stem was attached its 20+ N tension dragged the fruit
            # sideways in the grip (finger friction can't hold that shear);
            # once free, recentre it firmly between the pads so it doesn't
            # visibly float beside the gripper
            kh = 2.5 * hold_k
        fh = -kh * (p - hp) - hold_c * v
        body_f[ab] = body_f[ab] + wp.spatial_vector(fh, wp.vec3(0.0, 0.0, 0.0))
    pull[i] = wp.length(entry + fh)
    if detached[i] != 0:
        # Detached: a plain free body (ballistic; the normal aero drag and the
        # ground land it).  If held, the grip assist above carries it.
        return
    pb = parent_body[i]
    attach = wp.transform_point(body_q[pb], offset[i])      # spur attach point, world
    target = attach - wp.vec3(0.0, 0.0, hang_drop[i])       # apple hangs straight below it
    p = wp.transform_get_translation(body_q[ab])
    v = wp.spatial_top(body_qd[ab])                        # apple linear velocity (Newton: top)
    fs = -k * (p - target)                                 # elastic tether force on the apple
    tension[i] = wp.length(fs)                             # stem tension (for rupture)
    f = fs - c * v                                         # + damping toward the hang point
    # one apple per thread -> plain accumulate on the apple, no atomics needed
    body_f[ab] = body_f[ab] + wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))

    if react_scale > 0.0:
        # Reaction on the spur = -(elastic force - static weight hold).  At rest
        # fs == +m*g*z (the tether holds the apple up), so the reaction is zero
        # and the tree's rest pose / rupture margins are EXACTLY as before; only
        # an actual tug (or a big swing) transmits load to the branch.
        fr = -(fs - wp.vec3(0.0, 0.0, body_mass[ab] * grav)) * react_scale
        mag = wp.length(fr)
        if mag > react_max:
            fr = fr * (react_max / mag)
        # wrench about the parent's COM (Newton body_f convention)
        com = wp.transform_point(body_q[pb], body_com[pb])
        # several apples can share a parent across envs -> atomic accumulate
        wp.atomic_add(body_f, pb, wp.spatial_vector(fr, wp.cross(attach - com, fr)))


class AppleField:
    """Holds free-body apples on their spurs with a one-sided tether and detaches
    an apple only when the USER pulls the fruit itself hard enough (a direct
    pick/external force above its stem strength); it then free-falls.  Branch
    motion never detaches it.  Pure forces + a flag: no model edit, no
    ``notify``, no graph recapture, so it is cheap and cannot destabilise the
    tree."""

    def __init__(self, tree, config=None):
        self.tree = tree
        self.model = tree.model
        dev = self.model.device
        d = tree.apple_data
        fr = (config or tree.config).fruit
        self.n = len(d["apple_body"])
        self.apple_body = wp.array(np.asarray(d["apple_body"], dtype=np.int32), device=dev)
        self.parent_body = wp.array(np.asarray(d["parent_body"], dtype=np.int32), device=dev)
        self.offset = wp.array(np.asarray(d["offset"], dtype=np.float32), dtype=wp.vec3, device=dev)
        self.hang_drop = wp.array(np.asarray(d["hang_drop"], dtype=np.float32), device=dev)
        self.detach_force = np.asarray(d["detach_force"], dtype=np.float64)

        self.k = float(fr.tether_stiffness)
        # damping as a fraction of critical for a mass-spring (m = apple mass)
        self.c = float(fr.tether_damping_ratio) * 2.0 * float(np.sqrt(self.k * fr.mass))
        self._hyst = max(int(getattr(fr, "detach_hysteresis", 1)), 1)
        self.detach_drag = float(getattr(fr, "detach_drag", 120.0))
        self.react_scale = float(getattr(fr, "branch_reaction", 0.0))
        self.react_max = float(getattr(fr, "reaction_max", 10.0))
        self.grav = abs(float((config or tree.config).physics.gravity))
        self.body_mass = self.model.body_mass
        self.body_com = self.model.body_com

        self.detached = np.zeros(self.n, dtype=bool)
        self._over = np.zeros(self.n, dtype=np.int32)      # consecutive frames over threshold
        self.broken_count = 0
        self._flag = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._flag_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self._pull = wp.zeros(max(self.n, 1), dtype=wp.float32, device=dev)
        self._tension = wp.zeros(max(self.n, 1), dtype=wp.float32, device=dev)
        self._over_t = np.zeros(self.n, dtype=np.int32)    # tension hysteresis counter
        self._hyst_t = max(int(getattr(fr, "tension_hysteresis", 12)), 1)
        self._tension_factor = float(getattr(fr, "tension_factor", 1.2))
        # gripper hold: a strong spring from a hand-frame grip point to the
        # apple (in-graph; flags/bodies rewritten in place by hold()/release())
        self._held = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._held_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self._hold_body = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._hold_body_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self.hold_off = wp.vec3(0.0, 0.0, 0.10)            # TCP between the fingertips
        # Palm assist, not glue: the fingers hold the fruit by CONTACT
        # friction; this spring centres it in the palm and backstops slips.
        # Detaching works through the stem-TENSION rupture path, so the grip
        # doesn't need to transmit the full 13-20 N as a body force.
        self.hold_k = 500.0
        self.hold_c = 2.0 * float(np.sqrt(self.hold_k * fr.mass))   # ~critical
        self.dev = dev

    def hold(self, i: int, hand_body: int):
        """Attach apple ``i`` to a robot hand (strong grip spring)."""
        self._held_host[int(i)] = 1
        self._hold_body_host[int(i)] = int(hand_body)
        self._held.assign(self._held_host)
        self._hold_body.assign(self._hold_body_host)

    def release(self, i: int):
        self._held_host[int(i)] = 0
        self._held.assign(self._held_host)

    def pull_forces(self) -> np.ndarray:
        """Latest per-apple pull-force readings [N] (host copy)."""
        return self._pull.numpy()[:self.n]

    def apply(self, state):
        """Tether the still-attached apples and record each apple's pull force.
        Call each substep, AFTER the pick/external pull is in body_f but BEFORE
        drag/ground (so the recorded pull is the user's tug, not aero drag)."""
        if self.n == 0:
            return
        wp.launch(_apple_tether, dim=self.n,
                  inputs=[state.body_q, state.body_qd, self.apple_body, self.parent_body,
                          self.offset, self.hang_drop, self.k, self.c,
                          self.body_mass, self.body_com,
                          self.react_scale, self.react_max, self.grav,
                          self._flag, self._held, self._hold_body,
                          self.hold_off, self.hold_k, self.hold_c,
                          self._pull, self._tension, state.body_f],
                  device=self.dev)

    def update(self, state) -> int:
        """Flag apples whose stem load has exceeded their strength: either the
        DIRECT pull force on the fruit (mouse pick / hold spring) for
        ``detach_hysteresis`` frames, or the STEM TENSION itself (e.g. the
        gripper pulling a contact-held fruit) for the longer
        ``tension_hysteresis``.  Returns how many let go this frame."""
        if self.n == 0 or self.detached.all():
            return 0
        pull = self._pull.numpy()[:self.n]
        over = (pull > self.detach_force) & (~self.detached)
        self._over[over] += 1
        self._over[~over] = 0
        tension = self._tension.numpy()[:self.n]
        over_t = (tension > self._tension_factor * self.detach_force) & (~self.detached)
        self._over_t[over_t] += 1
        self._over_t[~over_t] = 0
        newly = ((self._over >= self._hyst) | (self._over_t >= self._hyst_t)) \
            & (~self.detached)
        if not newly.any():
            return 0
        self.detached[newly] = True
        self._flag_host[:self.n][newly] = 1
        self._flag.assign(self._flag_host)
        self.broken_count = int(self.detached.sum())
        return int(newly.sum())
