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
    def family_ids(self, graph, root_id: str, max_n: int = 80) -> Set[str]:
        """Structural family around a goal/hub: self, name twins, 1-hop edges."""
        if not root_id or graph is None or root_id not in graph:
            return set()
        out: Set[str] = {root_id}
        try:
            d0 = graph.nodes.get(root_id, {}) or {}
            nm = str(d0.get("name") or root_id).casefold()
            rid = str(root_id).casefold()
            for n, nd in list(graph.nodes(data=True)):
                if len(out) >= max_n:
                    break
                sn = str(nd.get("name") or "").casefold()
                is_sch = nd.get("node_type") in ("schema", "epistemic_schema") or nd.get("is_schema")
                if str(n).casefold() in (rid, nm) or sn in (rid, nm):
                    out.add(n)
                if is_sch and str(n).startswith("epistemic_of_"):
                    tail = str(n)[len("epistemic_of_"):].replace("_", " ").casefold()
                    if tail in (rid, nm) or nm in tail or tail in nm:
                        out.add(n)
            seed = list(out)
            for s in seed:
                if s not in graph:
                    continue
                try:
                    for _, v in list(graph.out_edges(s)):
                        out.add(v)
                        if len(out) >= max_n:
                            return out
                    for u, _ in list(graph.in_edges(s)):
                        out.add(u)
                        if len(out) >= max_n:
                            return out
                except Exception:
                    pass
            for n, nd in list(graph.nodes(data=True)):
                if len(out) >= max_n:
                    break
                if nd.get("kind_of") in (root_id, d0.get("name")):
                    out.add(n)
        except Exception:
            out = {root_id}
        return out

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
        # Soft: focus has any edge into the family
        focus_touching = False
        if focus_id and not focus_ok and graph is not None and focus_id in graph:
            try:
                for _, v in list(graph.out_edges(focus_id)):
                    if on_family(v):
                        focus_touching = True
                        break
                if not focus_touching:
                    for u, _ in list(graph.in_edges(focus_id)):
                        if on_family(u):
                            focus_touching = True
                            break
            except Exception:
                pass
        wm_hit = any(on_family(s) for s in (wm_slots or [])[:8])
        res_hit = any(on_family(s) for s in (residual_top or [])[:5])

        if focus_ok or focus_touching or wm_hit or res_hit:
            return PredictResult(
                match=True,
                expected_family=g0,
                reason="family_present_in_focus_wm_or_residual",
                # only hard-off when focus exists, not in family, and not touching
                off_family_focus=bool(focus_id and not focus_ok and not focus_touching),
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

        # Anti-oscillation: recent RETURN streak → prefer HOLD/EXPAND
        recent_ops = [e.get("operator") for e in self.episodes[-6:]]
        return_streak = 0
        for o in reversed(recent_ops):
            if o == "RETURN":
                return_streak += 1
            else:
                break
        same_return_target = False
        if return_streak >= 2 and goal_targets:
            # last RETURN details
            for e in reversed(self.episodes[-6:]):
                if e.get("operator") == "RETURN":
                    if e.get("goal") and e.get("goal") == goal_targets[0]:
                        same_return_target = True
                    break

        # HOLD: coherent focus or after RETURN streak
        scores["HOLD"] = 1.0
        if pred.match and focus_id and not pred.off_family_focus:
            scores["HOLD"] += 2.0  # truly on-family
        elif pred.match and focus_id:
            scores["HOLD"] += 1.0  # family in WM but focus slightly off
        if bias == "BIAS_STABILIZE":
            scores["HOLD"] += 0.4
        if return_streak >= 2:
            scores["HOLD"] += 2.5  # break RETURN loops
        if return_streak >= 4:
            scores["HOLD"] += 2.0

        # RETURN: only when focus clearly off-family AND prediction violated
        # or off-family with no WM family presence
        if goal_targets and pred.off_family_focus and not pred.match:
            scores["RETURN"] += 3.0
        elif goal_targets and pred.off_family_focus and pred.match:
            # family in WM but focus drifted — mild RETURN once, not forever
            if return_streak == 0:
                scores["RETURN"] += 1.2
            else:
                scores["RETURN"] += 0.2  # damp after first
        if goal_targets and not pred.match and not pred.off_family_focus:
            scores["RETURN"] += 1.0
        if goal_strength > 1.2 and pred.off_family_focus and return_streak == 0:
            scores["RETURN"] += 0.3

        # EXPAND: on-family or mild off with match — deepen schema
        if parent_open or (pred.match and focus_id):
            scores["EXPAND"] += 1.2
        if pred.match and not pred.off_family_focus:
            scores["EXPAND"] += 1.5  # deepen coherent goal family
        if lookup_budget_ok and pred.match:
            scores["EXPAND"] += 0.8
        if bias == "BIAS_EXPLORE" or bias == "BIAS_FORCE_EXPLORE":
            scores["EXPAND"] += 0.6
        if return_streak >= 2:
            scores["EXPAND"] += 1.8  # after RETURN loop, grow instead

        # RELEASE
        if stagnation and not pred.match:
            scores["RELEASE"] += 1.5
        if goal_targets and goal_strength < 0.45:
            scores["RELEASE"] += 1.2
        if fatigue > 0.85:
            scores["RELEASE"] += 0.4
        if return_streak >= 5:
            scores["RELEASE"] += 1.0  # give up sticky RETURN

        # SETTLE
        if fatigue >= 0.7:
            scores["SETTLE"] += 1.5
        if bias == "BIAS_STABILIZE":
            scores["SETTLE"] += 0.8
        if fatigue >= 0.85:
            scores["SETTLE"] += 0.8

        # Mutual exclusion
        if scores["RETURN"] >= 2.5 and return_streak == 0:
            scores["EXPAND"] *= 0.85
        if return_streak >= 2:
            scores["RETURN"] *= 0.3  # hard damp

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
