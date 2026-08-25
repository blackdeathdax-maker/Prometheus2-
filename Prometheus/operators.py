"""
operators.py — Internal moves + multi-layer prediction.

Cognition loop (pulse-time):
  open window → EXPECT (family + body parts) → match/violate → CHOOSE operator → apply

Prediction layers:
  1. Family coherence — focus/WM/residual on active goal family
  2. Body-part expectation — if goal/schema has composed-of body channels,
     those channels should be salient in the live body vector (interoception)

Operators: HOLD · RETURN · EXPAND · RELEASE · SETTLE
No LLM. Deterministic scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


OPS = ("HOLD", "RETURN", "EXPAND", "RELEASE", "SETTLE")


@dataclass
class PredictResult:
    match: bool = True
    expected_family: str = ""
    reason: str = ""
    off_family_focus: bool = False
    # Body / interoceptive layer
    body_match: bool = True
    body_error: float = 0.0
    expected_body: Dict[str, float] = field(default_factory=dict)
    observed_body: Dict[str, float] = field(default_factory=dict)
    body_reason: str = ""
    # Combined
    overall_match: bool = True


@dataclass
class OperatorDecision:
    operator: str = "HOLD"
    scores: Dict[str, float] = field(default_factory=dict)
    predict: Optional[PredictResult] = None
    note: str = ""


class OperatorModule:
    """Score and select one internal operator per pulse."""

    EPISODE_CAP = 120
    # Body channel expected high when linked as part of active goal schema
    BODY_EXPECT_HIGH = 0.58
    BODY_EXPECT_LOW = 0.42  # channels not in part-set can be ignored
    BODY_ERROR_MATCH_MAX = 0.22  # mean abs error under this → body_match

    def __init__(self):
        self.episodes: List[dict] = []
        self.last_decision: Optional[OperatorDecision] = None
        self.last_predict: Optional[PredictResult] = None
        self._op_ring: List[str] = []

    # ------------------------------------------------------------------
    def family_ids(self, graph, root_id: str, max_n: int = 80) -> Set[str]:
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
                if str(n).casefold() in (rid, nm) or sn in (rid, nm):
                    out.add(n)
            # 1-hop
            for _, v in list(graph.out_edges(root_id)):
                out.add(v)
                if len(out) >= max_n:
                    break
            for u, _ in list(graph.in_edges(root_id)):
                out.add(u)
                if len(out) >= max_n:
                    break
        except Exception:
            pass
        return out

    def body_parts_for(self, graph, root_id: str) -> Dict[str, str]:
        """Map body channel name → node id for parts of root (composed-of / part-of)."""
        parts: Dict[str, str] = {}
        if not root_id or graph is None or root_id not in graph:
            return parts
        try:
            from .edge_types import is_body_channel_node, BODY_CHANNELS
        except Exception:
            is_body_channel_node = lambda x: str(x).startswith("body:")
            BODY_CHANNELS = ()

        def note(nid: str):
            if not nid or not is_body_channel_node(nid):
                return
            ch = str(nid)
            if ch.startswith("body:"):
                ch = ch[5:]
            parts[ch] = nid
            # also store under node attr
            try:
                bc = (graph.nodes.get(nid) or {}).get("body_channel")
                if bc:
                    parts[str(bc)] = nid
            except Exception:
                pass

        try:
            for _, v, d in graph.out_edges(root_id, data=True):
                rel = (d or {}).get("relation_type")
                if rel in ("composed-of", "part-of", "associated-with"):
                    note(v)
            for u, _, d in graph.in_edges(root_id, data=True):
                rel = (d or {}).get("relation_type")
                if rel in ("composed-of", "part-of", "associated-with"):
                    note(u)
            # walk epistemic twin epistemic_of_X
            for n in list(self.family_ids(graph, root_id, max_n=40)):
                if n == root_id:
                    continue
                for _, v, d in graph.out_edges(n, data=True):
                    if (d or {}).get("relation_type") in ("composed-of", "part-of"):
                        note(v)
        except Exception:
            pass
        return parts

    def predict(
        self,
        graph,
        focus_id: Optional[str],
        goal_targets: List[str],
        wm_slots: List[str],
        residual_top: List[str],
        body: Optional[Dict[str, float]] = None,
    ) -> PredictResult:
        """Expect family coherence + body-part salience when committed to a goal."""
        body = body or {}

        if not goal_targets:
            return PredictResult(
                match=True,
                overall_match=True,
                reason="no_active_goal",
                body_match=True,
                body_reason="no_goal_no_body_expect",
            )

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
        res_hit = any(on_family(s) for s in (residual_top or [])[:6])

        family_match = bool(focus_ok or focus_touching or wm_hit or res_hit)
        off_family = bool(focus_id and not focus_ok and not focus_touching)

        if family_match:
            fam_reason = "family_present_in_focus_wm_or_residual"
        else:
            fam_reason = "goal_family_absent_while_committed"

        # --- Body layer: parts of goal/focus should read high ---
        parts = self.body_parts_for(graph, g0)
        if focus_id and focus_id != g0:
            parts.update(self.body_parts_for(graph, focus_id))

        expected_body: Dict[str, float] = {}
        observed_body: Dict[str, float] = {}
        errors: List[float] = []
        body_reason = "no_body_parts_linked"

        if parts:
            body_reason = "body_parts_checked"
            for ch, _nid in parts.items():
                # live reading: bio keys are heart_rate etc.
                val = body.get(ch)
                if val is None:
                    val = body.get(f"body:{ch}")
                if val is None:
                    continue
                try:
                    obs = float(val)
                except (TypeError, ValueError):
                    continue
                exp = float(self.BODY_EXPECT_HIGH)
                expected_body[ch] = exp
                observed_body[ch] = obs
                errors.append(abs(obs - exp))
            if errors:
                mean_err = sum(errors) / len(errors)
                body_match = mean_err <= self.BODY_ERROR_MATCH_MAX
                body_reason = (
                    "body_within_expect"
                    if body_match
                    else f"body_mismatch_err={mean_err:.3f}"
                )
                body_error = mean_err
            else:
                body_match = True
                body_error = 0.0
                body_reason = "body_parts_linked_but_no_readings"
        else:
            body_match = True
            body_error = 0.0

        overall = family_match and body_match
        # Family dominates "match" for operator RETURN; body is extra signal
        return PredictResult(
            match=family_match,
            expected_family=str(g0),
            reason=fam_reason,
            off_family_focus=off_family,
            body_match=body_match,
            body_error=float(body_error) if parts else 0.0,
            expected_body=expected_body,
            observed_body=observed_body,
            body_reason=body_reason,
            overall_match=overall,
        )

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
        body: Optional[Dict[str, float]] = None,
    ) -> OperatorDecision:
        pred = self.predict(
            graph, focus_id, goal_targets, wm_slots, residual_top, body=body
        )
        self.last_predict = pred

        scores = {op: 0.0 for op in OPS}
        ring = list(self._op_ring[-16:])
        if not ring:
            ring = [e.get("operator") for e in self.episodes[-16:] if e.get("operator")]

        def streak_of(name: str) -> int:
            n = 0
            for o in reversed(ring):
                if o == name:
                    n += 1
                else:
                    break
            return n

        return_streak = streak_of("RETURN")
        expand_streak = streak_of("EXPAND")
        hold_streak = streak_of("HOLD")
        last_op = ring[-1] if ring else None
        expands_recent = sum(1 for o in ring[-6:] if o == "EXPAND")
        holds_recent = sum(1 for o in ring[-6:] if o == "HOLD")

        force_hold = False
        force_expand = False
        force_return = False

        if last_op == "EXPAND":
            force_hold = True
        if expands_recent >= 2 and last_op != "HOLD":
            force_hold = True
        if (
            hold_streak >= 3
            and pred.match
            and lookup_budget_ok
            and last_op == "HOLD"
            and expands_recent == 0
        ):
            force_expand = True
            force_hold = False
        if goal_targets and pred.off_family_focus and not pred.match and last_op != "RETURN":
            force_return = True
            force_expand = False

        # Base scores
        scores["HOLD"] = 2.0
        if pred.match:
            scores["HOLD"] += 1.0
        if pred.overall_match:
            scores["HOLD"] += 0.5
        if not pred.off_family_focus:
            scores["HOLD"] += 0.8
        if bias == "BIAS_STABILIZE":
            scores["HOLD"] += 0.4
        if expand_streak >= 1:
            scores["HOLD"] += 2.5
        if return_streak >= 1:
            scores["HOLD"] += 1.5
        if hold_streak >= 5:
            scores["HOLD"] *= 0.75
        if hold_streak >= 8:
            scores["HOLD"] *= 0.7
        # Body mismatch while on-family: less pure HOLD, more EXPAND/RETURN curiosity
        if pred.match and not pred.body_match and pred.body_error > 0.15:
            scores["HOLD"] *= 0.85
            scores["EXPAND"] = max(scores.get("EXPAND", 0), 1.5)

        scores["RETURN"] = 0.15
        if goal_targets and pred.off_family_focus and not pred.match:
            scores["RETURN"] = 4.5
        elif goal_targets and pred.off_family_focus and last_op != "RETURN":
            scores["RETURN"] = 1.2
        if return_streak >= 1:
            scores["RETURN"] *= 0.12

        scores["EXPAND"] = 0.3
        if last_op == "EXPAND" or expand_streak >= 1:
            scores["EXPAND"] = 0.05
        elif lookup_budget_ok and pred.match:
            if hold_streak >= 3:
                scores["EXPAND"] = 3.5
            elif hold_streak >= 2:
                scores["EXPAND"] = 2.0
            elif holds_recent >= 3 and expands_recent == 0:
                scores["EXPAND"] = 2.8
            if bias in ("BIAS_EXPLORE", "BIAS_FORCE_EXPLORE"):
                scores["EXPAND"] = max(scores["EXPAND"], 2.5)
            # Body mismatch: deepen structure / re-check parts
            if not pred.body_match:
                scores["EXPAND"] = max(scores["EXPAND"], 2.2)
        if not lookup_budget_ok:
            scores["EXPAND"] *= 0.1
        if not parent_open and scores["EXPAND"] > 1.0:
            scores["EXPAND"] *= 0.5

        scores["RELEASE"] = 0.1
        if stagnation and hold_streak >= 6:
            scores["RELEASE"] = 2.0
        if fatigue > 0.7 and hold_streak >= 4:
            scores["RELEASE"] = max(scores["RELEASE"], 1.5)

        scores["SETTLE"] = 0.2
        if bias == "BIAS_STABILIZE":
            scores["SETTLE"] = 1.2
        if fatigue > 0.55:
            scores["SETTLE"] = max(scores["SETTLE"], 1.0)

        # Hard forces
        if force_return:
            scores["RETURN"] = max(scores["RETURN"], 5.0)
            scores["EXPAND"] *= 0.2
        if force_expand and not force_return:
            scores["EXPAND"] = max(scores["EXPAND"], 4.0)
            scores["HOLD"] *= 0.5
        if force_hold and not force_return and not force_expand:
            scores["HOLD"] = max(scores["HOLD"], 4.0)
            scores["EXPAND"] *= 0.15

        # Pick
        op = max(scores, key=lambda k: scores[k])
        if scores[op] <= 0:
            op = "HOLD"

        note_parts = [pred.reason]
        if pred.body_reason:
            note_parts.append(pred.body_reason)
        if force_return:
            note_parts.append("force_return")
        if force_expand:
            note_parts.append("force_expand")
        if force_hold:
            note_parts.append("force_hold")

        dec = OperatorDecision(
            operator=op,
            scores=dict(scores),
            predict=pred,
            note=";".join(note_parts),
        )
        self.last_decision = dec
        self._op_ring.append(op)
        if len(self._op_ring) > 32:
            self._op_ring = self._op_ring[-32:]
        return dec

    def record_episode(self, pulse: int, decision: OperatorDecision, **extra) -> None:
        pred = decision.predict
        row = {
            "pulse": pulse,
            "operator": decision.operator,
            "note": decision.note,
            "scores": decision.scores,
            "predict": {
                "match": pred.match if pred else None,
                "overall_match": pred.overall_match if pred else None,
                "expected_family": pred.expected_family if pred else None,
                "reason": pred.reason if pred else None,
                "off_family_focus": pred.off_family_focus if pred else None,
                "body_match": pred.body_match if pred else None,
                "body_error": pred.body_error if pred else None,
                "expected_body": pred.expected_body if pred else None,
                "observed_body": pred.observed_body if pred else None,
                "body_reason": pred.body_reason if pred else None,
            },
        }
        row.update(extra)
        self.episodes.append(row)
        if len(self.episodes) > self.EPISODE_CAP:
            self.episodes = self.episodes[-self.EPISODE_CAP :]

    def report(self) -> dict:
        return {
            "last_operator": (self.last_decision.operator if self.last_decision else None),
            "last_predict": (
                {
                    "match": self.last_predict.match,
                    "overall_match": self.last_predict.overall_match,
                    "expected_family": self.last_predict.expected_family,
                    "reason": self.last_predict.reason,
                    "body_match": self.last_predict.body_match,
                    "body_error": self.last_predict.body_error,
                    "expected_body": self.last_predict.expected_body,
                    "observed_body": self.last_predict.observed_body,
                    "body_reason": self.last_predict.body_reason,
                }
                if self.last_predict
                else None
            ),
            "episodes_tail": self.episodes[-12:],
            "op_ring": list(self._op_ring[-12:]),
        }
