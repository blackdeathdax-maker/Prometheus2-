"""
goals.py — Explicit commitments (contentful-thought substrate).

Design constraints:
  - No LLM. Goals are graph-local structures, not natural-language wishes.
  - A goal is not "whatever focus is on"; focus can serve a goal, or wander.
  - Success / failure are checked against residual, prediction error, and
    optional structural conditions (edge families present/absent).

Lifecycle:
  focus dwells on target >= COMMIT_AFTER_PULSES
    → open commitment (if none exists for that target)
  each pulse: evaluate open goals
    → satisfied  (residual cooled + optional structural ok)
    → failed     (abandoned under stagnation / hard switch with heat left)
    → active     (still pursuing)

Prometheus wiring:
  - pulse: goals.observe_focus(...); goals.tick(...)
  - residual: goals.commitment_boost(node_id) added into focus scoring via
    boost_residual or composite (caller applies)
  - report: goals.report()
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
GOALS_STATE_PATH = os.path.join(_DATA_DIR, "goals_state.json")


@dataclass
class Commitment:
    goal_id: str
    target_id: str
    status: str = "active"  # active | satisfied | failed | suspended
    created_pulse: int = 0
    last_pulse: int = 0
    strength: float = 1.0
    # Structural hints (optional): edge families we hoped to see / avoid
    want_families: List[str] = field(default_factory=list)
    avoid_families: List[str] = field(default_factory=list)
    # Outcomes
    satisfied_pulse: Optional[int] = None
    failed_pulse: Optional[int] = None
    fail_reason: str = ""
    success_reason: str = ""
    dwell_pulses: int = 0  # consecutive pulses focus was on target while active


class GoalModule:
    """Bounded set of explicit commitments."""

    MAX_ACTIVE = 5
    COMMIT_AFTER_PULSES = 12       # focus must dwell this long to open a goal
    SATISFY_RESIDUAL_BELOW = 1.2   # total residual cooled enough
    SATISFY_MIN_DWELL = 8          # must have pursued at least this long
    FAIL_ON_STAGNATION = True
    STRENGTH_DECAY = 0.98         # per pulse while active but not focused
    STRENGTH_FOCUS_BOOST = 0.05
    STRENGTH_CAP = 3.0
    RESIDUAL_BOOST = 1.8          # injected toward active goal targets
    HISTORY_CAP = 40              # kept satisfied/failed records

    def __init__(self):
        self.active: Dict[str, Commitment] = {}
        self.history: List[Commitment] = []
        self._dwell: Dict[str, int] = {}  # target_id -> consecutive focus pulses
        self.load()

    def _gid(self, target_id: str) -> str:
        return f"goal_{target_id}"

    def observe_focus(self, focus_id: Optional[str], pulse: int) -> None:
        """Track dwell; open commitment when dwell crosses threshold."""
        if not focus_id:
            self._dwell.clear()
            return
        # Reset other dwells
        for k in list(self._dwell.keys()):
            if k != focus_id:
                self._dwell[k] = 0
        self._dwell[focus_id] = self._dwell.get(focus_id, 0) + 1

        if self._dwell[focus_id] < self.COMMIT_AFTER_PULSES:
            return
        if focus_id in (None, "SELF", "OTHER"):
            return
        gid = self._gid(focus_id)
        if gid in self.active:
            g = self.active[gid]
            g.dwell_pulses += 1
            g.last_pulse = pulse
            g.strength = min(self.STRENGTH_CAP, g.strength + self.STRENGTH_FOCUS_BOOST)
            return
        if len(self.active) >= self.MAX_ACTIVE:
            # Drop weakest active
            weakest = min(self.active.values(), key=lambda c: c.strength)
            self._close(weakest, status="failed", pulse=pulse, reason="capacity_evict")
        self.active[gid] = Commitment(
            goal_id=gid,
            target_id=focus_id,
            status="active",
            created_pulse=pulse,
            last_pulse=pulse,
            strength=1.0,
            dwell_pulses=1,
        )

    def tick(
        self,
        pulse: int,
        focus_id: Optional[str],
        residual_fn,
        stagnation: bool = False,
        force_switch: bool = False,
        graph=None,
    ) -> Dict:
        """Evaluate active goals. residual_fn(node_id) -> float."""
        satisfied = failed = 0
        for gid, g in list(self.active.items()):
            g.last_pulse = pulse
            on_focus = focus_id == g.target_id
            if on_focus:
                g.dwell_pulses += 1
                g.strength = min(self.STRENGTH_CAP, g.strength + self.STRENGTH_FOCUS_BOOST)
            else:
                g.strength *= self.STRENGTH_DECAY

            r = 0.0
            try:
                r = float(residual_fn(g.target_id) or 0.0)
            except Exception:
                r = 0.0

            # Failure: stagnation / hard switch while this was the focus
            if on_focus and self.FAIL_ON_STAGNATION and (stagnation or force_switch):
                if r > self.SATISFY_RESIDUAL_BELOW:
                    self._close(g, status="failed", pulse=pulse, reason="stagnation_abandon")
                    failed += 1
                    continue

            # Success: pursued long enough and residual cooled
            if (
                g.dwell_pulses >= self.SATISFY_MIN_DWELL
                and r <= self.SATISFY_RESIDUAL_BELOW
                and not on_focus
            ):
                # structural optional check
                if self._structural_ok(g, graph):
                    self._close(g, status="satisfied", pulse=pulse, reason="residual_cooled")
                    satisfied += 1
                    continue

            # Also succeed while still on focus if residual very low and long dwell
            if (
                on_focus
                and g.dwell_pulses >= self.SATISFY_MIN_DWELL * 2
                and r <= self.SATISFY_RESIDUAL_BELOW * 0.5
            ):
                if self._structural_ok(g, graph):
                    self._close(g, status="satisfied", pulse=pulse, reason="resolved_on_focus")
                    satisfied += 1
                    continue

            if g.strength < 0.15:
                self._close(g, status="failed", pulse=pulse, reason="strength_decay")
                failed += 1

        return {
            "active": len(self.active),
            "satisfied": satisfied,
            "failed": failed,
        }

    def _structural_ok(self, g: Commitment, graph) -> bool:
        if graph is None or g.target_id not in getattr(graph, "nodes", {}):
            return True
        if not g.want_families and not g.avoid_families:
            return True
        seen: Set[str] = set()
        try:
            for _u, _v, ed in list(graph.edges(g.target_id, data=True)) + list(
                graph.in_edges(g.target_id, data=True)
            ):
                fam = ed.get("family") or ed.get("relation_type")
                if fam:
                    seen.add(str(fam))
        except Exception:
            return True
        if g.want_families and not any(f in seen for f in g.want_families):
            return False
        if g.avoid_families and any(f in seen for f in g.avoid_families):
            return False
        return True

    def _close(self, g: Commitment, status: str, pulse: int, reason: str) -> None:
        g.status = status
        g.last_pulse = pulse
        if status == "satisfied":
            g.satisfied_pulse = pulse
            g.success_reason = reason
        else:
            g.failed_pulse = pulse
            g.fail_reason = reason
        self.active.pop(g.goal_id, None)
        self.history.append(g)
        if len(self.history) > self.HISTORY_CAP:
            self.history = self.history[-self.HISTORY_CAP :]

    def commitment_boost(self, node_id: str) -> float:
        """Extra residual-like boost for active goal targets (and mild for related)."""
        g = self.active.get(self._gid(node_id))
        if g and g.status == "active":
            return self.RESIDUAL_BOOST * min(1.5, g.strength)
        return 0.0

    def active_target_ids(self) -> List[str]:
        return [g.target_id for g in self.active.values() if g.status == "active"]

    def report(self) -> dict:
        return {
            "active": [
                {
                    "goal_id": g.goal_id,
                    "target_id": g.target_id,
                    "strength": round(g.strength, 3),
                    "dwell_pulses": g.dwell_pulses,
                    "created_pulse": g.created_pulse,
                }
                for g in sorted(self.active.values(), key=lambda x: -x.strength)
            ],
            "recent_history": [
                {
                    "goal_id": g.goal_id,
                    "target_id": g.target_id,
                    "status": g.status,
                    "reason": g.success_reason or g.fail_reason,
                    "pulse": g.satisfied_pulse or g.failed_pulse,
                }
                for g in self.history[-8:]
            ],
        }

    def save(self, path: str = None) -> None:
        path = path or GOALS_STATE_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "active": [asdict(g) for g in self.active.values()],
                "history": [asdict(g) for g in self.history[-self.HISTORY_CAP :]],
                "dwell": dict(self._dwell),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("GoalModule.save failed: %s", e)

    def load(self, path: str = None) -> None:
        path = path or GOALS_STATE_PATH
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.active = {}
            for row in data.get("active") or []:
                c = Commitment(**{k: row[k] for k in Commitment.__dataclass_fields__ if k in row})
                self.active[c.goal_id] = c
            self.history = []
            for row in data.get("history") or []:
                c = Commitment(**{k: row[k] for k in Commitment.__dataclass_fields__ if k in row})
                self.history.append(c)
            self._dwell = {k: int(v) for k, v in (data.get("dwell") or {}).items()}
        except Exception as e:
            logger.warning("GoalModule.load failed: %s", e)
