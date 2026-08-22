"""
goals.py — Explicit commitments bound to nodes or kind-schemas.

Design:
  - No LLM. Goals are graph-local, not natural-language wishes.
  - Focus dwell opens a commitment on the focus target.
  - If the target is an epistemic/somatic schema, the goal binds to that
    kind and tracks member growth + nested child schemas + residual.
  - Success / failure are deterministic checks on residual and structure.

Lifecycle:
  focus dwells on target >= COMMIT_AFTER_PULSES
    -> open commitment
  each pulse: evaluate open goals
    -> satisfied  (residual cooled and/or schema grew enough)
    -> failed     (stagnation / capacity / hard switch with heat left)
    -> active

Prometheus wiring:
  - pulse: goals.observe_focus(...); goals.tick(...); commitment_boost -> residual
  - protection: active_target_ids + schema_closure_ids
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

try:
    from .edge_types import NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA, EDGE_COMPOSED_OF
except Exception:  # pragma: no cover
    NODE_SCHEMA = "schema"
    NODE_EPISTEMIC_SCHEMA = "epistemic_schema"
    EDGE_COMPOSED_OF = "composed-of"


@dataclass
class Commitment:
    goal_id: str
    target_id: str
    status: str = "active"
    created_pulse: int = 0
    last_pulse: int = 0
    strength: float = 1.0
    is_schema_goal: bool = False
    baseline_member_count: int = 0
    baseline_nested_count: int = 0
    last_member_count: int = 0
    last_nested_count: int = 0
    growth_events: int = 0
    want_families: List[str] = field(default_factory=list)
    avoid_families: List[str] = field(default_factory=list)
    satisfied_pulse: Optional[int] = None
    failed_pulse: Optional[int] = None
    fail_reason: str = ""
    success_reason: str = ""
    dwell_pulses: int = 0
    off_focus_pulses: int = 0


class GoalModule:
    """Bounded set of explicit commitments (node or schema targets)."""

    MAX_ACTIVE = 5
    COMMIT_AFTER_PULSES = 8
    SATISFY_RESIDUAL_BELOW = 1.15
    SATISFY_MIN_DWELL = 6
    SCHEMA_GROWTH_MEMBERS = 2
    SCHEMA_GROWTH_NESTED = 1
    SCHEMA_GROWTH_EVENTS = 2
    FAIL_ON_STAGNATION = True
    STAGNATION_OFF_FOCUS_MIN = 40   # must be off-focus this long before stagnation fails
    FORCE_SWITCH_OFF_FOCUS_MIN = 25
    STRENGTH_DECAY = 0.98
    STRENGTH_FOCUS_BOOST = 0.05
    STRENGTH_CAP = 3.0
    RESIDUAL_BOOST = 1.8
    CLOSURE_BOOST_SCALE = 0.55
    HISTORY_CAP = 40

    def __init__(self):
        self.active: Dict[str, Commitment] = {}
        self.history: List[Commitment] = []
        self._dwell: Dict[str, int] = {}
        self._on_event = None  # optional callable(event, target, detail, pulse)
        self.load()

    def _gid(self, target_id: str) -> str:
        return f"goal_{target_id}"

    def _is_schema(self, graph, node_id: str) -> bool:
        if graph is None or node_id not in graph:
            return False
        return graph.nodes.get(node_id, {}).get("node_type") in (
            NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA,
        )

    def _member_ids(self, graph, schema_id: str) -> List[str]:
        if graph is None or schema_id not in graph:
            return []
        return [
            v for _u, v, ed in graph.out_edges(schema_id, data=True)
            if ed.get("relation_type") == EDGE_COMPOSED_OF
        ]

    def _schema_stats(self, graph, schema_id: str) -> tuple:
        members = self._member_ids(graph, schema_id)
        nested = sum(1 for m in members if self._is_schema(graph, m))
        leaves = len(members) - nested
        return leaves, nested

    def schema_closure_ids(self, graph, root_id: str, max_n: int = 48) -> Set[str]:
        if graph is None or not root_id or root_id not in graph:
            return set()
        out: Set[str] = {root_id}

        def walk(sid: str, depth: int) -> None:
            if depth > 2 or len(out) >= max_n or sid not in graph:
                return
            for _u, v, ed in graph.out_edges(sid, data=True):
                if len(out) >= max_n:
                    break
                if ed.get("relation_type") != EDGE_COMPOSED_OF:
                    continue
                if v in out:
                    continue
                out.add(v)
                if self._is_schema(graph, v):
                    walk(v, depth + 1)

        if self._is_schema(graph, root_id):
            walk(root_id, 0)
        return out

    def observe_focus(self, focus_id: Optional[str], pulse: int, graph=None) -> None:
        if not focus_id:
            self._dwell.clear()
            return
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
            if g.is_schema_goal and graph is not None:
                leaves, nested = self._schema_stats(graph, focus_id)
                if leaves > g.last_member_count or nested > g.last_nested_count:
                    g.growth_events += 1
                g.last_member_count = leaves
                g.last_nested_count = nested
            return

        if len(self.active) >= self.MAX_ACTIVE:
            weakest = min(self.active.values(), key=lambda c: c.strength)
            self._close(weakest, status="failed", pulse=pulse, reason="capacity_evict")

        is_schema = self._is_schema(graph, focus_id) if graph is not None else False
        leaves, nested = (0, 0)
        if is_schema and graph is not None:
            leaves, nested = self._schema_stats(graph, focus_id)

        self.active[gid] = Commitment(
            goal_id=gid,
            target_id=focus_id,
            status="active",
            created_pulse=pulse,
            last_pulse=pulse,
            strength=1.0,
            dwell_pulses=1,
            off_focus_pulses=0,
            is_schema_goal=is_schema,
            baseline_member_count=leaves,
            baseline_nested_count=nested,
            last_member_count=leaves,
            last_nested_count=nested,
            growth_events=0,
        )
        print(
            f"Goals: OPEN {gid} schema={is_schema} "
            f"members={leaves} nested={nested} pulse={pulse}"
        )
        if callable(self._on_event):
            try:
                self._on_event(
                    "open", focus_id,
                    detail=f"schema={is_schema};members={leaves};nested={nested}",
                    pulse=pulse,
                )
            except Exception:
                pass

    def tick(
        self,
        pulse: int,
        focus_id: Optional[str],
        residual_fn,
        stagnation: bool = False,
        force_switch: bool = False,
        graph=None,
    ) -> Dict:
        satisfied = failed = 0
        for gid, g in list(self.active.items()):
            g.last_pulse = pulse
            on_focus = focus_id == g.target_id
            # Also treat focus on schema closure as on-goal (Color schema vs Color node)
            if not on_focus and graph is not None and g.is_schema_goal:
                try:
                    if focus_id in self.schema_closure_ids(graph, g.target_id):
                        on_focus = True
                except Exception:
                    pass
            if not on_focus and graph is not None and focus_id:
                # Focus on schema whose dominant parent / name matches target lemma
                try:
                    fd = graph.nodes.get(focus_id, {})
                    if fd.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA):
                        if str(fd.get("name") or "").casefold() == str(g.target_id).casefold():
                            on_focus = True
                        if str(fd.get("dominant_parent") or "").casefold() == str(g.target_id).casefold():
                            on_focus = True
                    # Inverse: goal on schema, focus on lemma name
                    if g.is_schema_goal:
                        gd = graph.nodes.get(g.target_id, {})
                        if str(gd.get("name") or "").casefold() == str(focus_id).casefold():
                            on_focus = True
                except Exception:
                    pass

            if on_focus:
                g.dwell_pulses += 1
                g.off_focus_pulses = 0
                g.strength = min(self.STRENGTH_CAP, g.strength + self.STRENGTH_FOCUS_BOOST)
            else:
                g.off_focus_pulses = int(getattr(g, "off_focus_pulses", 0) or 0) + 1
                g.strength *= self.STRENGTH_DECAY

            r = 0.0
            try:
                r = float(residual_fn(g.target_id) or 0.0)
            except Exception:
                r = 0.0

            if g.is_schema_goal and graph is not None:
                try:
                    closure = self.schema_closure_ids(graph, g.target_id)
                    vals = []
                    for nid in closure:
                        try:
                            vals.append(float(residual_fn(nid) or 0.0))
                        except Exception:
                            pass
                    if vals:
                        r = max(r, sum(vals) / len(vals))
                except Exception:
                    pass
                leaves, nested = self._schema_stats(graph, g.target_id)
                if leaves > g.last_member_count or nested > g.last_nested_count:
                    g.growth_events += 1
                g.last_member_count = leaves
                g.last_nested_count = nested

            if g.dwell_pulses >= self.SATISFY_MIN_DWELL:
                cooled = r <= self.SATISFY_RESIDUAL_BELOW
                grew = False
                if g.is_schema_goal:
                    member_delta = g.last_member_count - g.baseline_member_count
                    nested_delta = g.last_nested_count - g.baseline_nested_count
                    grew = (
                        member_delta >= self.SCHEMA_GROWTH_MEMBERS
                        or nested_delta >= self.SCHEMA_GROWTH_NESTED
                        or g.growth_events >= self.SCHEMA_GROWTH_EVENTS
                    )
                if cooled or grew:
                    reason = []
                    if cooled:
                        reason.append("residual_cooled")
                    if grew:
                        reason.append(
                            "schema_growth(members=%d,nested=%d,events=%d)"
                            % (
                                g.last_member_count - g.baseline_member_count,
                                g.last_nested_count - g.baseline_nested_count,
                                g.growth_events,
                            )
                        )
                    self._close(
                        g, status="satisfied", pulse=pulse,
                        reason="+".join(reason) or "satisfied",
                    )
                    satisfied += 1
                    continue

            off = int(getattr(g, "off_focus_pulses", 0) or 0)
            # Stagnation/explore overrides must NOT kill a fresh commitment
            if (
                self.FAIL_ON_STAGNATION
                and stagnation
                and not on_focus
                and off >= self.STAGNATION_OFF_FOCUS_MIN
                and r > self.SATISFY_RESIDUAL_BELOW * 1.5
            ):
                self._close(g, status="failed", pulse=pulse, reason="stagnation")
                failed += 1
                continue
            if (
                force_switch
                and not on_focus
                and off >= self.FORCE_SWITCH_OFF_FOCUS_MIN
                and r > self.SATISFY_RESIDUAL_BELOW * 2.0
            ):
                self._close(g, status="failed", pulse=pulse, reason="force_switch_heat")
                failed += 1
                continue
            if g.strength < 0.2 and not on_focus and off >= 30:
                self._close(g, status="failed", pulse=pulse, reason="strength_collapse")
                failed += 1
                continue

        return {
            "active": len(self.active),
            "satisfied_this_tick": satisfied,
            "failed_this_tick": failed,
            "targets": [g.target_id for g in self.active.values()],
        }

    def _close(self, g: Commitment, status: str, pulse: int, reason: str) -> None:
        g.status = status
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
        if callable(self._on_event):
            try:
                self._on_event(
                    status, g.target_id,
                    detail=g.success_reason or g.fail_reason,
                    pulse=pulse,
                )
            except Exception:
                pass

    def commitment_boost(self, node_id: str, graph=None) -> float:
        g = self.active.get(self._gid(node_id))
        if g and g.status == "active":
            return self.RESIDUAL_BOOST * min(1.5, g.strength)
        if graph is not None:
            for g in self.active.values():
                if g.status != "active" or not g.is_schema_goal:
                    continue
                if node_id in self.schema_closure_ids(graph, g.target_id):
                    return (
                        self.RESIDUAL_BOOST
                        * self.CLOSURE_BOOST_SCALE
                        * min(1.5, g.strength)
                    )
        return 0.0

    def active_target_ids(self) -> List[str]:
        return [g.target_id for g in self.active.values() if g.status == "active"]

    def protected_ids(self, graph=None) -> Set[str]:
        out: Set[str] = set()
        for g in self.active.values():
            if g.status != "active":
                continue
            out.add(g.target_id)
            if g.is_schema_goal and graph is not None:
                out |= self.schema_closure_ids(graph, g.target_id)
        return out

    def report(self) -> dict:
        return {
            "active": [
                {
                    "goal_id": g.goal_id,
                    "target_id": g.target_id,
                    "is_schema_goal": g.is_schema_goal,
                    "strength": round(g.strength, 3),
                    "dwell_pulses": g.dwell_pulses,
                    "members": g.last_member_count,
                    "nested": g.last_nested_count,
                    "growth_events": g.growth_events,
                    "baseline_members": g.baseline_member_count,
                    "created_pulse": g.created_pulse,
                }
                for g in sorted(self.active.values(), key=lambda x: -x.strength)
            ],
            "recent_history": [
                {
                    "goal_id": g.goal_id,
                    "target_id": g.target_id,
                    "is_schema_goal": g.is_schema_goal,
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
            fields = set(Commitment.__dataclass_fields__)
            self.active = {}
            for row in data.get("active") or []:
                c = Commitment(**{k: row[k] for k in fields if k in row})
                self.active[c.goal_id] = c
            self.history = []
            for row in data.get("history") or []:
                c = Commitment(**{k: row[k] for k in fields if k in row})
                self.history.append(c)
            self._dwell = {k: int(v) for k, v in (data.get("dwell") or {}).items()}
        except Exception as e:
            logger.warning("GoalModule.load failed: %s", e)
