"""Run metrics: fps, pick/place success, throughput, forces, detection quality.

Host-side and O(1) per frame; everything is accumulated in plain python and
written as one JSON at the end (plus a human summary on stdout).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque


class Metrics:
    def __init__(self, path: str | None = None):
        self.path = path
        self.t0 = time.time()
        self._last = None
        self._fps = deque(maxlen=240)
        self.frames = 0
        self.counters: dict = {}
        self.picks: list[dict] = []          # one record per attempted pick
        self._open_pick: dict | None = None
        self.detection = dict(frames=0, true_positives=0, false_positives=0,
                              visible_candidates=0)
        # canopy-zone census: how many reachable apples sit in each
        # (vertical x radial) zone of THIS env's tree, so per-zone pick
        # success can be normalised by the fruit actually present there.
        self.apple_census: dict = {}

    SIM_FPS = 60.0        # nominal sim frame rate: event times are SIM time

    # -- per-frame ------------------------------------------------------------
    def frame(self, wall: bool = True):
        """``wall=False`` advances sim time only (per-env metrics in multi-env
        autonomy share one wall clock; only env 0 samples fps)."""
        if wall:
            now = time.time()
            if self._last is not None:
                dt = now - self._last
                if dt > 0:
                    self._fps.append(1.0 / dt)
            self._last = now
        self.frames += 1

    def count(self, name: str, inc: int = 1):
        self.counters[name] = self.counters.get(name, 0) + inc

    # -- detection quality (called by the picker with ground-truth matching) --
    def detection_frame(self, tp: int, fp: int, candidates: int):
        d = self.detection
        d["frames"] += 1
        d["true_positives"] += tp
        d["false_positives"] += fp
        d["visible_candidates"] += candidates

    def set_apple_census(self, census: dict):
        """Record how many reachable apples live in each canopy zone (called
        once at picker init).  ``census`` maps a zone label ("upper-outer",
        ...) to a count."""
        self.apple_census = dict(census)

    # -- pick lifecycle --------------------------------------------------------
    def pick_start(self, fruit_id: int, target, zone: str | None = None,
                   zone_v: str | None = None, zone_r: str | None = None):
        self._open_pick = dict(
            fruit_id=int(fruit_id), t_start=self._t(),
            target=[float(x) for x in target],
            zone=zone, zone_v=zone_v, zone_r=zone_r,
            grasped=False, detached=False, placed=False,
            max_pull_N=0.0, fail_reason=None,
            t_grasp=None, t_detach=None, t_end=None)

    def pick_pull(self, force_n: float):
        if self._open_pick is not None:
            self._open_pick["max_pull_N"] = max(self._open_pick["max_pull_N"],
                                                float(force_n))

    def pick_event(self, what: str):
        p = self._open_pick
        if p is None:
            return
        t = self._t()
        if what == "grasp":
            p["grasped"], p["t_grasp"] = True, t
        elif what == "detach":
            p["detached"], p["t_detach"] = True, t

    def pick_end(self, placed: bool, fail_reason: str | None = None):
        p = self._open_pick
        if p is None:
            return
        p["placed"] = bool(placed)
        p["fail_reason"] = fail_reason
        p["t_end"] = self._t()
        self.picks.append(p)
        self._open_pick = None

    def _t(self):
        """Event timestamps in SIM seconds (frames / nominal fps), so cycle
        times and throughput describe the simulated system, not the wall clock
        of this particular GPU."""
        return round(self.frames / self.SIM_FPS, 3)

    # -- reporting -------------------------------------------------------------
    def summary(self, sim=None) -> dict:
        fps = sorted(self._fps)
        att = len(self.picks)
        gr = sum(p["grasped"] for p in self.picks)
        de = sum(p["detached"] for p in self.picks)
        pl = sum(p["placed"] for p in self.picks)
        cyc = [p["t_end"] - p["t_start"] for p in self.picks if p["placed"]]
        wall = time.time() - self.t0
        sim_s = self.frames / self.SIM_FPS
        d = self.detection
        out = dict(
            wall_time_s=round(wall, 1),
            sim_time_s=round(sim_s, 1),
            sim_frames=self.frames,
            fps_mean=round(sum(fps) / len(fps), 1) if fps else None,
            fps_median=round(fps[len(fps) // 2], 1) if fps else None,
            picks_attempted=att,
            grasp_success=gr,
            pick_success=de,                       # fruit detached from the stem
            place_success=pl,                      # fruit ended up in the bucket
            total_success_rate=round(pl / att, 3) if att else None,
            mean_cycle_time_s=round(sum(cyc) / len(cyc), 1) if cyc else None,   # SIM time
            throughput_fruit_per_min=round(60.0 * pl / sim_s, 2) if sim_s > 0 else None,
            max_pull_force_N=round(max((p["max_pull_N"] for p in self.picks),
                                       default=0.0), 2),
            detection_precision=(round(d["true_positives"] /
                                       max(d["true_positives"] + d["false_positives"], 1), 3)
                                 if d["frames"] else None),
            detection_recall_visible=(round(d["true_positives"] /
                                            max(d["visible_candidates"], 1), 3)
                                      if d["frames"] else None),
            counters=dict(self.counters),
        )
        if sim is not None:
            if sim.breaker is not None:
                out["branches_snapped"] = int(sim.breaker.broken_count)
            if sim.apples is not None:
                out["apples_detached_total"] = int(sim.apples.broken_count)
        return out

    def save(self, sim=None) -> str | None:
        if self._open_pick is not None:
            self.pick_end(False, "run ended")
        data = dict(summary=self.summary(sim), picks=self.picks,
                    apple_census=self.apple_census,
                    zone_breakdown=zone_breakdown(self.picks, self.apple_census))
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
        return self.path


ZONES_V = ("lower", "middle", "upper")
ZONES_R = ("inner", "outer")


def zone_breakdown(picks: list, census: dict | None = None) -> dict:
    """Per-canopy-zone pick outcomes for the paper's inner/outer x lower/
    middle/upper success table.  Each entry: attempts, grasped, detached,
    placed, and (if a census is given) the reachable apples present there."""
    out: dict = {}
    for p in picks:
        z = p.get("zone")
        if z is None:
            continue
        e = out.setdefault(z, dict(attempts=0, grasped=0, detached=0,
                                   placed=0, apples=0))
        e["attempts"] += 1
        e["grasped"] += int(bool(p.get("grasped")))
        e["detached"] += int(bool(p.get("detached")))
        e["placed"] += int(bool(p.get("placed")))
    if census:
        for z, n in census.items():
            out.setdefault(z, dict(attempts=0, grasped=0, detached=0,
                                   placed=0, apples=0))["apples"] = int(n)
    for z, e in out.items():
        a = e["attempts"]
        e["place_rate"] = round(e["placed"] / a, 3) if a else None
        e["grasp_rate"] = round(e["grasped"] / a, 3) if a else None
    return out


def combined_summary(mets: list, sim=None) -> dict:
    """Merge per-env Metrics (multi-env autonomy: one robot per world) into a
    stand-level summary: successes and counters sum, rates pool, throughput is
    fruit per SIM minute across the whole stand."""
    m0 = mets[0]
    picks = [p for m in mets for p in m.picks]
    att = len(picks)
    pl = sum(p["placed"] for p in picks)
    cyc = [p["t_end"] - p["t_start"] for p in picks if p["placed"]]
    sim_s = m0.frames / Metrics.SIM_FPS
    tp = sum(m.detection["true_positives"] for m in mets)
    fp = sum(m.detection["false_positives"] for m in mets)
    cand = sum(m.detection["visible_candidates"] for m in mets)
    counters: dict = {}
    for m in mets:
        for k, v in m.counters.items():
            counters[k] = counters.get(k, 0) + v
    fps = sorted(m0._fps)
    out = dict(
        envs=len(mets),
        wall_time_s=round(time.time() - m0.t0, 1),
        sim_time_s=round(sim_s, 1),
        sim_frames=m0.frames,
        fps_mean=round(sum(fps) / len(fps), 1) if fps else None,
        picks_attempted=att,
        grasp_success=sum(p["grasped"] for p in picks),
        pick_success=sum(p["detached"] for p in picks),
        place_success=pl,
        total_success_rate=round(pl / att, 3) if att else None,
        mean_cycle_time_s=round(sum(cyc) / len(cyc), 1) if cyc else None,
        throughput_fruit_per_min=round(60.0 * pl / sim_s, 2) if sim_s > 0 else None,
        max_pull_force_N=round(max((p["max_pull_N"] for p in picks), default=0.0), 2),
        detection_precision=round(tp / max(tp + fp, 1), 3) if (tp + fp) else None,
        detection_recall_visible=round(tp / max(cand, 1), 3) if cand else None,
        counters=counters,
    )
    if sim is not None:
        if sim.breaker is not None:
            out["branches_snapped"] = int(sim.breaker.broken_count)
        if sim.apples is not None:
            out["apples_detached_total"] = int(sim.apples.broken_count)
    return out


def save_combined(path: str | None, mets: list, sim=None) -> dict:
    """Write one JSON for a multi-env run: stand-level summary + per-env
    summaries + all pick records tagged by env."""
    for m in mets:
        if m._open_pick is not None:
            m.pick_end(False, "run ended")
    all_picks = [dict(p, env=e) for e, m in enumerate(mets) for p in m.picks]
    census: dict = {}
    for m in mets:
        for z, n in m.apple_census.items():
            census[z] = census.get(z, 0) + int(n)
    data = dict(
        summary=combined_summary(mets, sim),
        envs=[m.summary() for m in mets],
        picks=all_picks,
        apple_census=census,
        zone_breakdown=zone_breakdown(all_picks, census),
    )
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return data["summary"]
