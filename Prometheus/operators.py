"""
operators.py — Minimal internal moves + prediction check.

Cognition loop (pulse-time):
  open window → EXPECT family coherence → match/violate → CHOOSE operator → apply

Operators (fixed menu):
  HOLD    stay with focus
  RETURN  retarget focus to active goal family
  EXPAND  one self-study-style step under open parent (caller runs expansion)
  RELEASE soft-clear sticky focus when stagnant / off-goal
  SETTLE  request stabilize bias (caller may set executive)

No LLM. Deterministic scoring from focus, goals, residual, family sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


OPS = ("HOLD", "RETURN", "EXPAND", "RELEASE", "SETTLE")


@dataclass
class PredictResult:
    match: bool = True
    expected_family: str = ""
    reason: str = ""
    off_family_focus: bool = False


@dataclass
class OperatorDecision:
    operator: str = "HOLD"
    scores: Dict[str, float] = field(default_factory=dict)
    predict: Optional[PredictResult] = None
    note: str = ""


class OperatorModule:
    """Score and select one internal operator per pulse."""

    EPISODE_CAP = 120

    def __init__(self):
        self.episodes: List[dict] = []
        self.last_decision: Optional[OperatorDecision] = None
        self.last_predict: Optional[PredictResult] = None

    # ------------------------------------------------------------------
    # Family helpers (graph-local; same spirit as kind_family)
    # ------------------------------------------------------------------
    def family_ids(self, graph, root_id: str, max_n: int = 64) -> Set[str]:
        if not root_id or graph is None or root_id not in graph:
            return set()
        out: Set[str] = {root_id}
        try:
            d0 = graph.nodes.get(root_id, {})
            nm = str(d0.get("name") or d0.get("dominant_parent") or root_id).casefold()
            rid = str(root_id).casefold()
            for n, nd in graph.nodes(data=True):
                if len(out) >= max_n:
                    break
                is_sch = nd.get("node_type") in ("schema", "epistemic_schema") or nd.get("is_schema")
                sn = str(nd.get("name") or nd.get("dominant_parent") or "").casefold()
                if str(n).casefold() in (rid, nm) or (is_sch and sn in (rid, nm)):
                    out.add(n)
                if is_sch and str(n).startswith("epistemic_of_"):
                    tail = str(n)[len("epistemic_of_"):].replace("_", " ").casefold()
                    if tail in (rid, nm):
                        out.add(n)
            # composed-of members of schema family members
            for seed in list(out):
                nd = graph.nodes.get(seed, {})
                if not (nd.get("node_type") in ("schema", "epistemic_schema") or nd.get("is_schema")):
                    continue
                for _u, v, ed in graph.out_edges(seed, data=True):
                    if ed.get("relation_type") == "composed-of" and v in graph:
                        out.add(v)
                        if len(out) >= max_n:
                            return out
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        graph,
        focus_id: Optional[str],
        goal_targets: List[str],
        wm_slots: List[str],
        residual_top: List[str],
    ) -> PredictResult:
        """Expect on-family coherence when a goal is active."""
        if not goal_targets:
            return PredictResult(match=True, reason="no_active_goal")

        # Primary goal family
        g0 = goal_targets[0]
        fam = self.family_ids(graph, g0)
        if not fam:
            fam = {g0}

        def on_family(nid: Optional[str]) -> bool:
            if not nid:
                return False
            if nid in fam:
                return True
            return str(nid).casefold() in {str(x).casefold() for x in fam}

        focus_ok = on_family(focus_id)
        wm_hit = any(on_family(s) for s in (wm_slots or [])[:8])
        res_hit = any(on_family(s) for s in (residual_top or [])[:5])

        if focus_ok or wm_hit or res_hit:
            return PredictResult(
                match=True,
                expected_family=g0,
                reason="family_present_in_focus_wm_or_residual",
                off_family_focus=bool(focus_id and not focus_ok),
            )

        return PredictResult(
            match=False,
            expected_family=g0,
            reason="goal_family_absent_while_committed",
            off_family_focus=bool(focus_id and not focus_ok),
        )

    # ------------------------------------------------------------------
    # Operator selection
    # ------------------------------------------------------------------
    def choose(
        self,
        graph,
        focus_id: Optional[str],
        goal_targets: List[str],
        wm_slots: List[str],
        residual_top: List[str],
        bias: str = "",
        fatigue: float = 0.0,
        stagnation: bool = False,
        lookup_budget_ok: bool = True,
        parent_open: bool = False,
        goal_strength: float = 1.0,
    ) -> OperatorDecision:
        pred = self.predict(graph, focus_id, goal_targets, wm_slots, residual_top)
        self.last_predict = pred

        scores = {op: 0.0 for op in OPS}

        # HOLD: default comfort when coherent
        scores["HOLD"] = 1.0
        if pred.match and focus_id:
            scores["HOLD"] += 1.2
        if bias == "BIAS_STABILIZE":
            scores["HOLD"] += 0.4

        # RETURN: goal active but focus off-family
        if goal_targets and pred.off_family_focus:
            scores["RETURN"] += 2.5
        if goal_targets and not pred.match:
            scores["RETURN"] += 2.0
        if goal_strength > 1.2:
            scores["RETURN"] += 0.3

        # EXPAND: open parent + uncertainty/explore + budget
        if parent_open and lookup_budget_ok:
            scores["EXPAND"] += 1.0
            if bias in ("BIAS_EXPLORE", "FORCE_EXPLORE"):
                scores["EXPAND"] += 1.2
            if pred.match and focus_id:
                scores["EXPAND"] += 0.8  # deepen coherent focus
            if stagnation and parent_open:
                scores["EXPAND"] += 0.6

        # RELEASE: stagnation / weak goal / long mess
        if stagnation and not pred.match:
            scores["RELEASE"] += 1.5
        if goal_targets and goal_strength < 0.45:
            scores["RELEASE"] += 1.2
        if fatigue > 0.85:
            scores["RELEASE"] += 0.4

        # SETTLE: high fatigue or stabilize bias
        if fatigue >= 0.7:
            scores["SETTLE"] += 1.5
        if bias == "BIAS_STABILIZE":
            scores["SETTLE"] += 0.8
        if fatigue >= 0.85:
            scores["SETTLE"] += 0.8

        # Soft mutual exclusion: if RETURN very high, damp EXPAND slightly
        if scores["RETURN"] >= 2.5:
            scores["EXPAND"] *= 0.7

        best = max(scores, key=lambda k: scores[k])
        note = {
            "HOLD": "stay with what is open",
            "RETURN": "pull back to goal family",
            "EXPAND": "look under open parent",
            "RELEASE": "let go of a dead stickiness",
            "SETTLE": "prefer to settle and compress",
        }.get(best, "")

        if not pred.match:
            note = (note + " · prediction violated").strip(" ·")

        dec = OperatorDecision(operator=best, scores=scores, predict=pred, note=note)
        self.last_decision = dec
        return dec

    def record_episode(
        self,
        pulse: int,
        decision: OperatorDecision,
        focus_id: Optional[str] = None,
        goal: Optional[str] = None,
        detail: str = "",
    ) -> None:
        pred = decision.predict
        self.episodes.append({
            "pulse": pulse,
            "operator": decision.operator,
            "focus": focus_id,
            "goal": goal,
            "predict": "match" if (pred and pred.match) else ("violate" if pred else "n/a"),
            "reason": (pred.reason if pred else ""),
            "note": decision.note,
            "detail": detail,
        })
        if len(self.episodes) > self.EPISODE_CAP:
            self.episodes = self.episodes[-self.EPISODE_CAP:]

    def report(self, last_n: int = 12) -> dict:
        dec = self.last_decision
        pred = self.last_predict
        return {
            "last_operator": dec.operator if dec else None,
            "last_note": dec.note if dec else "",
            "last_scores": dict(dec.scores) if dec else {},
            "last_predict": {
                "match": pred.match if pred else None,
                "expected_family": pred.expected_family if pred else "",
                "reason": pred.reason if pred else "",
            },
            "episodes": list(self.episodes[-last_n:]),
        }
