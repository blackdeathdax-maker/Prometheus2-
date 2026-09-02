"""
plan.py — Compositional planning over causal traces (Packages D + G).

No LLM. Plans are short chains of graph-local means derived from
causes / enables / prevents edges (scored / high-conf only).

Package G:
  - Depth 2–3 chains (means → intermediate → goal/body)
  - Backward search: what enables / causes the goal target?
  - Forward search: what does the goal cause/enable that is actionable?
  - Stop when edge conf / link_L below floor

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
    """Build and advance short means-end plans for active goals (D + G)."""

    MAX_STEPS = 5
    MAX_DEPTH = 3                 # Package G: chain depth
    MIN_EDGE_WEIGHT = 0.12
    MIN_CONFIDENCE = 0.40
    MIN_LINK_L = -0.5             # floor: very negative link_L stops expansion
    REQUIRE_SCORED = True
    ADVANCE_WEIGHT = 0.35

    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.last_plan: Optional[Plan] = None

    def clear(self) -> None:
        self.plans.clear()
        self.last_plan = None

    def _suggest_op(self, relation: str, target_id: str) -> str:
        t = str(target_id or "")
        if t.startswith("body:"):
            ch = t.split(":", 1)[-1]
            if relation == "prevents" or relation == "causes":
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

    def _edge_ok(self, data: dict) -> bool:
        """High-conf / scored filter; respect link_L floor."""
        data = data or {}
        w = float(data.get("weight") or 0.0)
        c = float(data.get("confidence") or 0.3)
        scored = bool(data.get("scored_improve") or data.get("evidence_credit") or data.get("info_credit"))
        try:
            ll = float(data.get("link_L") or 0.0)
        except Exception:
            ll = 0.0
        if ll < self.MIN_LINK_L:
            return False
        if c < self.MIN_CONFIDENCE and not scored:
            return False
        if w < self.MIN_EDGE_WEIGHT and c < self.MIN_CONFIDENCE:
            return False
        if self.REQUIRE_SCORED and not scored and c < 0.55:
            return False
        return True

    def _edge_score(self, data: dict, src: str, dst: str, goal_id: str) -> float:
        data = data or {}
        w = float(data.get("weight") or 0.0)
        c = float(data.get("confidence") or 0.3)
        scored = bool(data.get("scored_improve") or data.get("evidence_credit") or data.get("info_credit"))
        score = w * (0.5 + c)
        if scored:
            score *= 1.5
        if str(dst).startswith(("body:", "world:")):
            score *= 1.4
        if src == goal_id or dst == goal_id:
            score *= 1.2
        try:
            score += 0.3 * float(data.get("link_L") or 0.0)
        except Exception:
            pass
        return score

    def _neighbors(self, graph, node: str, direction: str = "both"):
        """Yield (src, rel, dst, data) causal edges touching node."""
        if graph is None or node not in graph:
            return
        if direction in ("out", "both"):
            try:
                for _, dst, data in graph.out_edges(node, data=True):
                    rel = (data or {}).get("relation_type") or ""
                    if rel in CAUSAL_RELS and self._edge_ok(data or {}):
                        yield node, rel, dst, data or {}
            except Exception:
                pass
        if direction in ("in", "both"):
            try:
                for pred, _, data in graph.in_edges(node, data=True):
                    rel = (data or {}).get("relation_type") or ""
                    if rel in CAUSAL_RELS and self._edge_ok(data or {}):
                        yield pred, rel, node, data or {}
            except Exception:
                pass

    def _collect_causal_edges(self, graph, root_ids: Set[str]) -> List[Tuple[str, str, str, dict]]:
        edges = []
        if graph is None:
            return edges
        for src in root_ids:
            for e in self._neighbors(graph, src, "both"):
                edges.append(e)
        return edges

    def _is_knowledge(self, nid: str) -> bool:
        s = str(nid or "")
        return not s.startswith(("body:", "world:", "felt:", "narr:", "epistemic_"))

    def _search_chains(
        self, graph, goal_id: str, roots: Set[str]
    ) -> List[Tuple[float, List[Tuple[str, str, str, dict]]]]:
        """Package G: BFS chains up to MAX_DEPTH ending at actionable or goal-linked edges.

        Returns list of (chain_score, [(src,rel,dst,data), ...]).
        """
        chains: List[Tuple[float, List[Tuple[str, str, str, dict]]]] = []
        # Seed: 1-hop from roots (forward + backward)
        frontier: List[Tuple[float, List[Tuple[str, str, str, dict]], Set[str]]] = []
        for r in roots:
            for src, rel, dst, data in self._neighbors(graph, r, "both"):
                sc = self._edge_score(data, src, dst, goal_id)
                path = [(src, rel, dst, data)]
                visited = {r, src, dst}
                frontier.append((sc, path, visited))
                chains.append((sc, path))

        # Extend to depth 2..MAX_DEPTH
        depth = 1
        while depth < self.MAX_DEPTH and frontier:
            nxt = []
            for sc, path, visited in frontier:
                tip_src, _, tip_dst, _ = path[-1]
                # Expand from both ends of the tip edge
                for tip in (tip_src, tip_dst):
                    if str(tip).startswith(("body:", "world:")):
                        continue  # don't chain through body/world
                    for src, rel, dst, data in self._neighbors(graph, tip, "both"):
                        if src in visited and dst in visited:
                            continue
                        if len(path) >= self.MAX_DEPTH:
                            continue
                        sc2 = sc + 0.65 * self._edge_score(data, src, dst, goal_id)
                        path2 = path + [(src, rel, dst, data)]
                        vis2 = set(visited) | {src, dst}
                        nxt.append((sc2, path2, vis2))
                        chains.append((sc2, path2))
            frontier = sorted(nxt, key=lambda t: -t[0])[:24]  # beam
            depth += 1

        chains.sort(key=lambda t: -t[0])
        return chains

    def build_plan(
        self,
        graph,
        goal_id: str,
        pulse: int = 0,
        closure_ids: Optional[Set[str]] = None,
    ) -> Optional[Plan]:
        """Compose a short plan: 1-hop + depth-2–3 chains over scored causal edges."""
        if not goal_id or graph is None:
            return None
        roots = set(closure_ids or ())
        roots.add(goal_id)
        # Also resolve bare lemma if goal is epistemic shell
        gname = str(goal_id)
        if gname.startswith("epistemic_of_"):
            roots.add(gname[len("epistemic_of_"):])

        chains = self._search_chains(graph, goal_id, roots)
        if not chains:
            plan = Plan(
                goal_id=goal_id, steps=[], pulse_built=pulse,
                note="no_high_conf_causal",
            )
            self.plans[goal_id] = plan
            self.last_plan = plan
            return plan

        steps: List[PlanStep] = []
        seen_means: Set[str] = set()
        depth_used = 1
        for chain_score, path in chains:
            if len(steps) >= self.MAX_STEPS:
                break
            depth_used = max(depth_used, len(path))
            # Prefer outermost knowledge means → actionable target
            # Walk path: pick knowledge node as means, body/world or goal-side as target
            for src, rel, dst, data in path:
                means = src
                target = dst
                if not self._is_knowledge(means) and self._is_knowledge(dst):
                    means, target = dst, src
                if not self._is_knowledge(means):
                    # goal itself causing body is ok
                    if means == goal_id or str(target).startswith(("body:", "world:")):
                        pass
                    else:
                        continue
                if means in seen_means:
                    continue
                if str(means).startswith(("felt:", "narr:", "epistemic_")):
                    continue
                seen_means.add(str(means))
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
                if len(steps) >= self.MAX_STEPS:
                    break

        note = f"steps={len(steps)} depth<={depth_used} chains={len(chains)}"
        if not steps:
            note = "no_high_conf_causal"
        plan = Plan(
            goal_id=goal_id,
            steps=steps,
            next_index=0,
            pulse_built=pulse,
            note=note,
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
