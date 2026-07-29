"""Branch dynamics: torsional spring-damper stiffness and rupture thresholds.

Follows Jacob et al. (CoRL 2024) §3.1.  Each inter-branch joint is a torsional
mass-spring-damper with stiffness ``Kp`` and damping ``Kd``.  Two models:

* **beam** (physically grounded): ``Kp = beam_factor * E * r**4 / l`` which for
  ``beam_factor = pi/4`` is the Euler-Bernoulli cantilever stiffness ``E*I/l``
  with ``I = pi*r**4/4``.  The paper uses ``Kp = E*pi*r**4/(2 l)`` (set
  ``beam_factor = pi/2``).  ``Kd = damping_ratio * Kp`` (paper: 1/10).
* **rudimentary**: ``Kp`` fixed per branch order, decaying exponentially away
  from the trunk between ``phi_upper`` and ``phi_lower``.

Rupture (branch snapping) uses beam theory: a branch fails when the bending
moment at its base exceeds ``M_max = sigma_r * pi * r**3 / 4`` (modulus of
rupture times the section modulus of a solid circular cross-section).
"""

from __future__ import annotations

import math

import numpy as np

from .config import PhysicsParams, BreakParams, StiffnessModel
from .skeleton import Segment


def stiffness_for(seg: Segment, phys: PhysicsParams,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Return ``(Kp, Kd)`` for the joint at the base of ``seg``.

    The relevant radius/length are those of the *child* branch (``seg``), since
    that is the beam that bends about the joint.
    """
    r = max(seg.radius_start, 1e-4)
    l = max(seg.length, 1e-3)

    if phys.model == StiffnessModel.BEAM:
        kp = phys.beam_factor * phys.youngs_modulus * (r ** 4) / l
    elif phys.model == StiffnessModel.RUDIMENTARY:
        kp = phys.phi_upper * (phys.phi_decay ** seg.order)
        kp = max(kp, phys.phi_lower)
    else:  # RIGID
        kp = phys.max_stiffness

    kd = phys.damping_ratio * kp

    # domain randomisation on dynamics (paper: Gaussian on the rudimentary model)
    if rng is not None and phys.dynamics_jitter > 0.0:
        kp *= max(0.05, 1.0 + rng.normal(0.0, phys.dynamics_jitter))
        kd = phys.damping_ratio * kp

    kp = float(np.clip(kp, phys.min_stiffness, phys.max_stiffness))
    kd = float(max(kd, 0.0))
    return kp, kd


def rupture_moment(seg: Segment, brk: BreakParams) -> float:
    """Maximum bending moment [N*m] the base of ``seg`` can sustain before it
    snaps:  ``M_max = sigma_r * pi * r**3 / 4`` (solid circular section)."""
    r = max(seg.radius_start, 1e-4)
    return brk.safety_factor * brk.rupture_stress * math.pi * (r ** 3) / 4.0


def segment_mass(seg: Segment, density: float) -> float:
    """Capsule mass = cylinder + two hemispherical caps, for reporting."""
    r = seg.mean_radius
    l = seg.length
    cyl = math.pi * r * r * l
    caps = (4.0 / 3.0) * math.pi * r ** 3
    return density * (cyl + caps)


def summarize(skel, phys: PhysicsParams) -> str:
    kps = [stiffness_for(s, phys)[0] for s in skel.segments if s.parent >= 0]
    masses = [segment_mass(s, phys.wood_density) for s in skel.segments]
    if not kps:
        return "no compliant joints"
    return (f"stiffness Kp range {min(kps):.3g}..{max(kps):.3g} N*m/rad | "
            f"total mass {sum(masses):.2f} kg | "
            f"model={phys.model.value}")
