"""
plan.py — Compositional planning over causal traces (Package D).

No LLM. Plans are short chains of graph-local means derived from
causes / enables / prevents edges produced by Package B co-occurrence
and (optionally) user teaching.

A plan is bound to an active goal target:
  goal G
    means[0] enables/causes something useful for G
    means[1] ...
  next_step = first means not yet "done" (low residual or weak edge)

Prometheus uses:
  - suggested_operator: soft bias into OperatorModule scores
  - next_means_id: residual / focus preference
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

CAUSAL_RELS = frozenset({"causes", "enables", "prevents", "results-in"})

# Prefer operators by consequence type (heuristic, not ontology)
OP_FOR_BODY_UP = {
    "energy": "EXPAND",
    "pleasure": "EXPAND",
    "warmth": "EXPAND",
    "heart_rate": "EXPAND",
    "breath": "RELEASE",
}
OP_FOR_BODY_DOWN = {
    "muscle_tension": "SETTLE",
    "sweat_skin": "SETTLE",
    "pain": "SETTLE",
    "heart_rate": "SETTLE",
    "gut": "SETTLE",
}
OP_FOR_WORLD = {
    "object_near": "EXPAND",
    "obstacle": "RELEASE",  # decrease obstacle
    "goal_cue": "RETURN",
}


@dataclass
class PlanStep:
    means_id: str
    relation: str = "enables"
    target_id: str = ""          # body:channel or world:slot or lemma
    suggested_op: str = "EXPAND"
    weight: float = 0.0
    confidence: float = 0.3


@dataclass
class Plan:
    goal_id: str
    steps: List[PlanStep] = field(default_factory=list)
    next_index: int = 0
    pulse_built: int = 0
    note: str = ""

    @property
    def next_step(self) -> Optional[PlanStep]:
        if 0 <= self.next_index < len(self.steps):
            return self.steps[self.next_index]
        return None


class PlanModule:
    """Build and advance short means-end plans for active goals."""

    MAX_STEPS = 4
    MIN_EDGE_WEIGHT = 0.12
    MIN_CONFIDENCE = 0.40       # only high-confidence causes feed plans
    REQUIRE_SCORED = True       # prefer edges stamped from improved outcomes
    ADVANCE_WEIGHT = 0.35       # edge weight high enough to treat step as used

    def __init__(self):
        self.plans: Dict[str, Plan] = {}  # goal_id -> Plan
        self.last_plan: Optional[Plan] = None

    def clear(self) -> None:
        self.plans.clear()
        self.last_plan = None

    def _suggest_op(self, relation: str, target_id: str) -> str:
        t = str(target_id or "")
        if t.startswith("body:"):
            ch = t.split(":", 1)[-1]
            if relation == "prevents" or relation == "causes":
                # causes body:X with negative typical? we only have positive link minting for causes
                return OP_FOR_BODY_DOWN.get(ch, OP_FOR_BODY_UP.get(ch, "EXPAND"))
            return OP_FOR_BODY_UP.get(ch, OP_FOR_BODY_DOWN.get(ch, "EXPAND"))
        if t.startswith("world:"):
            slot = t.split(":", 1)[-1]
            if relation == "prevents" and slot == "obstacle":
                return "RELEASE"
            return OP_FOR_WORLD.get(slot, "EXPAND")
        if relation == "prevents":
            return "SETTLE"
        if relation == "enables":
            return "EXPAND"
        return "RETURN"

    def _collect_causal_edges(self, graph, root_ids: Set[str]) -> List[Tuple[str, str, str, dict]]:
        """Return list of (src, rel, dst, data) for causal edges involving roots."""
        edges = []
        if graph is None:
            return edges
        for src in root_ids:
            if src not in graph:
                continue
            try:
                for _, dst, data in graph.out_edges(src, data=True):
                    rel = (data or {}).get("relation_type") or ""
                    if rel in CAUSAL_RELS:
                        edges.append((src, rel, dst, data or {}))
            except Exception:
                pass
            try:
                for pred, _, data in graph.in_edges(src, data=True):
                    rel = (data or {}).get("relation_type") or ""
                    if rel in CAUSAL_RELS:
                        edges.append((pred, rel, src, data or {}))
            except Exception:
                pass
        return edges

    def build_plan(
        self,
        graph,
        goal_id: str,
        pulse: int = 0,
        closure_ids: Optional[Set[str]] = None,
    ) -> Optional[Plan]:
        """Compose a short plan for goal_id from causal neighborhood."""
        if not goal_id or graph is None:
            return None
        roots = set(closure_ids or ())
        roots.add(goal_id)
        edges = self._collect_causal_edges(graph, roots)
        if not edges:
            # No causal traces yet — empty plan (D waits on B data)
            plan = Plan(goal_id=goal_id, steps=[], pulse_built=pulse, note="no_causal_traces")
            self.plans[goal_id] = plan
            self.last_plan = plan
            return plan

        # Rank edges by weight * confidence; prefer scored-improvement stamps
        ranked = []
        for src, rel, dst, data in edges:
            w = float(data.get("weight") or 0.0)
            c = float(data.get("confidence") or 0.3)
            scored = bool(data.get("scored_improve") or data.get("evidence_credit"))
            if c < self.MIN_CONFIDENCE and not scored:
                continue
            if w < self.MIN_EDGE_WEIGHT and c < self.MIN_CONFIDENCE:
                continue
            if self.REQUIRE_SCORED and not scored and c < 0.55:
                # Legacy unscoring edges need higher confidence to enter plans
                continue
            score = w * (0.5 + c)
            if scored:
                score *= 1.5
            if str(dst).startswith(("body:", "world:")):
                score *= 1.4
            if src == goal_id or dst == goal_id:
                score *= 1.2
            # link-level L if present
            try:
                score += 0.3 * float(data.get("link_L") or 0.0)
            except Exception:
                pass
            ranked.append((score, src, rel, dst, data))
        ranked.sort(key=lambda t: -t[0])
        if not ranked:
            plan = Plan(
                goal_id=goal_id, steps=[], pulse_built=pulse,
                note="no_high_conf_causal",
            )
            self.plans[goal_id] = plan
            self.last_plan = plan
            return plan

        steps: List[PlanStep] = []
        seen_means = set()
        for score, src, rel, dst, data in ranked:
            if len(steps) >= self.MAX_STEPS:
                break
            # means = the knowledge lemma side (not body/world)
            means = src
            target = dst
            if str(src).startswith(("body:", "world:", "felt:", "narr:")):
                means = dst
                target = src
            if means in seen_means or means == goal_id:
                # still allow goal --causes--> body as a step with means=goal
                if not str(target).startswith(("body:", "world:")):
                    continue
            if str(means).startswith(("body:", "world:", "felt:", "epistemic_")):
                continue
            seen_means.add(means)
            steps.append(
                PlanStep(
                    means_id=str(means),
                    relation=rel,
                    target_id=str(target),
                    suggested_op=self._suggest_op(rel, target),
                    weight=float(data.get("weight") or 0.0),
                    confidence=float(data.get("confidence") or 0.3),
                )
            )

        plan = Plan(
            goal_id=goal_id,
            steps=steps,
            next_index=0,
            pulse_built=pulse,
            note=f"steps={len(steps)}",
        )
        self.plans[goal_id] = plan
        self.last_plan = plan
        return plan

    def advance_if_ready(self, plan: Plan, graph=None) -> None:
        """Advance next_index when current means edge is strong enough."""
        step = plan.next_step
        if step is None:
            return
        # If weight already high, treat as established means
        if float(step.weight or 0) >= self.ADVANCE_WEIGHT and float(step.confidence or 0) >= 0.5:
            plan.next_index = min(len(plan.steps), plan.next_index + 1)
            plan.note = f"advanced_to={plan.next_index}"

    def tick(
        self,
        pulse: int,
        graph,
        goal_ids: List[str],
        schema_closure_fn=None,
    ) -> Optional[Plan]:
        """Rebuild/refresh plan for primary active goal."""
        if not goal_ids:
            self.last_plan = None
            return None
        goal_id = goal_ids[0]
        closure = set()
        if schema_closure_fn is not None:
            try:
                closure = set(schema_closure_fn(goal_id) or [])
            except Exception:
                closure = set()
        plan = self.build_plan(graph, goal_id, pulse=pulse, closure_ids=closure)
        if plan:
            self.advance_if_ready(plan, graph=graph)
        return plan

    def suggested_operator(self) -> Optional[str]:
        plan = self.last_plan
        if not plan or not plan.next_step:
            return None
        return plan.next_step.suggested_op

    def next_means_id(self) -> Optional[str]:
        plan = self.last_plan
        if not plan or not plan.next_step:
            return None
        return plan.next_step.means_id

    def report(self) -> dict:
        plan = self.last_plan
        if not plan:
            return {"active": False}
        steps = [
            {
                "means_id": s.means_id,
                "relation": s.relation,
                "target_id": s.target_id,
                "suggested_op": s.suggested_op,
                "weight": round(float(s.weight), 3),
                "confidence": round(float(s.confidence), 3),
            }
            for s in plan.steps
        ]
        nxt = plan.next_step
        return {
            "active": True,
            "goal_id": plan.goal_id,
            "next_index": plan.next_index,
            "next_means_id": nxt.means_id if nxt else None,
            "suggested_op": nxt.suggested_op if nxt else None,
            "note": plan.note,
            "steps": steps,
            "pulse_built": plan.pulse_built,
        }
