"""
operators.py — Internal moves + multi-layer prediction.

Cognition loop (pulse-time):
  open window → EXPECT (family + body trajectory) → match/violate → CHOOSE operator → apply

Prediction layers:
  1. Family coherence — focus/WM/residual on active goal family
  2. Body / interoceptive prediction —
       a. Static part-salience (linked body:* channels expected high)
       b. Short-horizon trajectory prior from recent (operator, body) history
          (“my heart will rise / settle”)

Operators: HOLD · RETURN · EXPAND · RELEASE · SETTLE
No LLM. Deterministic scoring.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

OPS = ("HOLD", "RETURN", "EXPAND", "RELEASE", "SETTLE")

# Canonical body channels (must stay in sync with edge_types.BODY_CHANNELS)
BODY_CHANNELS = (
    "heart_rate",
    "breath",
    "muscle_tension",
    "sweat_skin",
    "gut",
    "energy",
    "warmth",
    "pain",
    "pleasure",
)

# Channels whose elevation is especially relevant for regulation policies
REGULATION_CHANNELS = frozenset({
    "muscle_tension", "sweat_skin", "gut", "heart_rate", "pain",
})

# Package A: operator → small body deltas (applied next pulse via pending delta)
# Values are soft nudges; hormonal/allostatic caps still apply.
OP_BODY_DELTAS: Dict[str, Dict[str, float]] = {
    "SETTLE": {
        "heart_rate": -0.04, "muscle_tension": -0.05, "pain": -0.02,
        "breath": 0.02, "sweat_skin": -0.03,
    },
    "RELEASE": {
        "muscle_tension": -0.06, "sweat_skin": -0.04, "breath": 0.03,
        "heart_rate": -0.02,
    },
    "EXPAND": {
        "energy": 0.05, "heart_rate": 0.03, "warmth": 0.02,
        "muscle_tension": 0.01,
    },
    "HOLD": {
        "muscle_tension": 0.02,  # mild freeze
    },
    "RETURN": {
        "energy": 0.02, "heart_rate": -0.01, "muscle_tension": -0.02,
    },
}

ACT_CONF_START = 0.30
ACT_CONF_MIN = 0.15
ACT_CONF_MAX = 0.95
ACT_CONF_UP = 0.06
ACT_CONF_DOWN = 0.08
ACT_MAX_BODY_DELTA = 0.08


@dataclass
class PredictResult:
    match: bool = True
    expected_family: str = ""
    reason: str = ""
    off_family_focus: bool = False
    # Body / interoceptive layer
    body_match: bool = True
    body_error: float = 0.0
    signed_body_error: float = 0.0          # positive = observed > expected
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

    # ---- static part-salience defaults (kept for backward compatibility) ----
    BODY_EXPECT_HIGH = 0.58
    BODY_EXPECT_LOW = 0.42
    BODY_ERROR_MATCH_MAX = 0.22

    # ---- trajectory prior ----
    HISTORY_LEN = 24                # recent (op, body) pairs
    TRAJ_BLEND = 0.55               # weight of trajectory vs static high expectation
    TRAJ_LR = 0.18                  # how strongly last observed delta influences next prior
    MIN_HISTORY_FOR_TRAJ = 3

    def __init__(self):
        self.episodes: List[dict] = []
        self.last_decision: Optional[OperatorDecision] = None
        # Evidence spine: log-odds L separate from bias B
        try:
            from .evidence import EvidenceModule
            self.evidence = EvidenceModule()
        except Exception:
            try:
                from evidence import EvidenceModule
                self.evidence = EvidenceModule()
            except Exception:
                self.evidence = None
        self._last_L_scores: Dict[str, float] = {}
        self._last_evidence_led: bool = False
        self.last_predict: Optional[PredictResult] = None
        self._op_ring: List[str] = []

        # Short-horizon interoceptive memory: deque of
        # {"op": str, "body": Dict[str, float]}
        self._body_hist: deque = deque(maxlen=self.HISTORY_LEN)
        # Per-channel exponential smoother of recent deltas (operator-agnostic baseline)
        self._delta_ema: Dict[str, float] = {ch: 0.0 for ch in BODY_CHANNELS}

        # Package A + thin C: pending body delta + outcome confidence
        self.pending_body_delta: Dict[str, float] = {}
        # key: "OP:channel" or "OP:world:slot" → confidence
        self.outcome_confidence: Dict[str, float] = {}
        self._pre_act_body: Dict[str, float] = {}
        self._pre_act_world: Dict[str, float] = {}
        self._last_act_op: str = ""
        self._last_act_body_deltas: Dict[str, float] = {}
        self._last_act_world_deltas: Dict[str, float] = {}

    def planned_body_delta(self, op: str, scale: float = 1.0) -> Dict[str, float]:
        """Signed body nudges for this operator (clamped)."""
        op_u = (op or "HOLD").upper()
        raw = dict(OP_BODY_DELTAS.get(op_u, {}))
        out: Dict[str, float] = {}
        for ch, dv in raw.items():
            d = max(-ACT_MAX_BODY_DELTA, min(ACT_MAX_BODY_DELTA, float(dv) * float(scale)))
            if abs(d) > 1e-6:
                out[ch] = d
        return out

    def queue_body_delta(self, op: str, scale: float = 1.0) -> Dict[str, float]:
        """Queue operator body effect for next pulse sense."""
        d = self.planned_body_delta(op, scale=scale)
        # merge into pending (sum, then clamp per channel later)
        for ch, dv in d.items():
            self.pending_body_delta[ch] = float(self.pending_body_delta.get(ch, 0.0)) + dv
        self._last_act_op = (op or "HOLD").upper()
        self._last_act_body_deltas = dict(d)
        return d

    def consume_pending_body_delta(self) -> Dict[str, float]:
        """Take and clear pending body deltas (call once per pulse at sense)."""
        out = dict(self.pending_body_delta)
        self.pending_body_delta = {}
        return out

    def conf_key(self, op: str, target: str) -> str:
        return f"{(op or 'HOLD').upper()}:{target}"

    def get_confidence(self, op: str, target: str) -> float:
        return float(self.outcome_confidence.get(self.conf_key(op, target), ACT_CONF_START))

    def update_outcome_confidence(
        self,
        op: str,
        predicted_deltas: Dict[str, float],
        observed_before: Dict[str, float],
        observed_after: Dict[str, float],
        prefix: str = "",
    ) -> None:
        """Thin C: raise confidence when observed change agrees with planned sign."""
        op_u = (op or "HOLD").upper()
        for ch, pred_d in (predicted_deltas or {}).items():
            if abs(float(pred_d)) < 1e-6:
                continue
            before = float((observed_before or {}).get(ch, 0.0))
            after = float((observed_after or {}).get(ch, before))
            actual = after - before
            key = self.conf_key(op_u, f"{prefix}{ch}" if prefix else ch)
            conf = float(self.outcome_confidence.get(key, ACT_CONF_START))
            # agreement: same sign and non-trivial move
            if pred_d * actual > 0 and abs(actual) >= 0.005:
                conf = min(ACT_CONF_MAX, conf + ACT_CONF_UP)
            elif pred_d * actual < 0 and abs(actual) >= 0.005:
                conf = max(ACT_CONF_MIN, conf - ACT_CONF_DOWN)
            else:
                conf = max(ACT_CONF_MIN, conf - ACT_CONF_DOWN * 0.35)
            self.outcome_confidence[key] = conf

    def confidence_bias_scores(self, body: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Soft score additives from learned outcome confidence (thin C)."""
        body = body or {}
        bias = {op: 0.0 for op in OPS}
        tension = float(body.get("muscle_tension", 0.4) or 0.4)
        pain = float(body.get("pain", 0.1) or 0.1)
        energy = float(body.get("energy", 0.5) or 0.5)
        # If SETTLE has high conf on lowering tension and tension high → boost SETTLE
        if tension > 0.55:
            c = self.get_confidence("SETTLE", "muscle_tension")
            bias["SETTLE"] += 0.8 * (c - 0.3)
            c2 = self.get_confidence("RELEASE", "muscle_tension")
            bias["RELEASE"] += 0.6 * (c2 - 0.3)
        if pain > 0.5:
            c = self.get_confidence("SETTLE", "pain")
            bias["SETTLE"] += 0.7 * (c - 0.3)
        if energy < 0.35:
            c = self.get_confidence("EXPAND", "energy")
            bias["EXPAND"] += 0.6 * (c - 0.3)
        return bias

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
            from .edge_types import is_body_channel_node, BODY_CHANNELS as ET_BODY
        except Exception:
            is_body_channel_node = lambda x: str(x).startswith("body:")
            ET_BODY = BODY_CHANNELS

        def note(nid: str):
            if not nid or not is_body_channel_node(nid):
                return
            ch = str(nid)
            if ch.startswith("body:"):
                ch = ch[5:]
            parts[ch] = nid
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

    # ------------------------------------------------------------------
    # Interoceptive history + trajectory prior
    # ------------------------------------------------------------------
    def _record_body(self, body: Dict[str, float], op: Optional[str] = None) -> None:
        """Call once per pulse after body vector is known (from choose or external)."""
        if not body:
            return
        clean = {}
        for ch in BODY_CHANNELS:
            v = body.get(ch)
            if v is None:
                v = body.get(f"body:{ch}")
            if v is None:
                continue
            try:
                clean[ch] = float(v)
            except (TypeError, ValueError):
                continue
        if not clean:
            return

        # Update delta EMA against previous sample
        if self._body_hist:
            prev = self._body_hist[-1]["body"]
            for ch, val in clean.items():
                if ch in prev:
                    d = val - prev[ch]
                    self._delta_ema[ch] = (
                        (1.0 - self.TRAJ_LR) * self._delta_ema.get(ch, 0.0)
                        + self.TRAJ_LR * d
                    )

        self._body_hist.append({
            "op": (op or (self._op_ring[-1] if self._op_ring else "HOLD")),
            "body": clean,
        })

    def _trajectory_prior(
        self,
        current_body: Dict[str, float],
        linked_channels: Set[str],
        last_op: Optional[str],
    ) -> Dict[str, float]:
        """
        Predict next body vector.
        Hybrid:
          - linked channels: blend static HIGH expectation with trajectory
          - unlinked channels: pure trajectory (or current if no history)
        """
        prior: Dict[str, float] = {}
        if not current_body:
            return prior

        use_traj = len(self._body_hist) >= self.MIN_HISTORY_FOR_TRAJ

        # Operator-conditioned recent delta (prefer same op)
        op_delta: Dict[str, float] = {ch: 0.0 for ch in BODY_CHANNELS}
        if use_traj and last_op:
            same = [h for h in self._body_hist if h.get("op") == last_op]
            if len(same) >= 2:
                # average last few deltas under this operator
                deltas: Dict[str, List[float]] = {ch: [] for ch in BODY_CHANNELS}
                for i in range(1, len(same)):
                    a = same[i - 1]["body"]
                    b = same[i]["body"]
                    for ch in BODY_CHANNELS:
                        if ch in a and ch in b:
                            deltas[ch].append(b[ch] - a[ch])
                for ch, ds in deltas.items():
                    if ds:
                        op_delta[ch] = sum(ds) / len(ds)

        for ch in BODY_CHANNELS:
            cur = current_body.get(ch)
            if cur is None:
                continue
            try:
                cur_f = float(cur)
            except (TypeError, ValueError):
                continue

            # Trajectory component
            if use_traj:
                # prefer operator-conditioned delta, fall back to global EMA
                d = op_delta.get(ch, 0.0)
                if abs(d) < 1e-6:
                    d = self._delta_ema.get(ch, 0.0)
                traj = max(0.0, min(1.0, cur_f + d))
            else:
                traj = cur_f

            if ch in linked_channels:
                # Hybrid: static high expectation + trajectory
                static = float(self.BODY_EXPECT_HIGH)
                prior[ch] = (
                    self.TRAJ_BLEND * traj + (1.0 - self.TRAJ_BLEND) * static
                )
            else:
                # Unlinked: pure trajectory (or stay)
                prior[ch] = traj

        return prior

    # ------------------------------------------------------------------
    def predict(
        self,
        graph,
        focus_id: Optional[str],
        goal_targets: List[str],
        wm_slots: List[str],
        residual_top: List[str],
        body: Optional[Dict[str, float]] = None,
    ) -> PredictResult:
        """Expect family coherence + interoceptive trajectory when committed to a goal."""
        body = body or {}

        # Always keep history (even without a goal) so the prior stays warm
        self._record_body(body)

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

        # --- Body / interoceptive layer ---
        parts = self.body_parts_for(graph, g0)
        if focus_id and focus_id != g0:
            parts.update(self.body_parts_for(graph, focus_id))
        linked = set(parts.keys())

        # Observed body (canonical keys only)
        observed_body: Dict[str, float] = {}
        for ch in BODY_CHANNELS:
            val = body.get(ch)
            if val is None:
                val = body.get(f"body:{ch}")
            if val is None:
                continue
            try:
                observed_body[ch] = float(val)
            except (TypeError, ValueError):
                continue

        last_op = self._op_ring[-1] if self._op_ring else None
        expected_body = self._trajectory_prior(observed_body, linked, last_op)

        errors: List[float] = []
        signed_errors: List[float] = []
        body_reason = "no_body_parts_linked"

        if linked and observed_body:
            body_reason = "body_trajectory_checked"
            for ch in linked:
                if ch not in observed_body or ch not in expected_body:
                    continue
                obs = observed_body[ch]
                exp = expected_body[ch]
                err = abs(obs - exp)
                errors.append(err)
                signed_errors.append(obs - exp)

            if errors:
                mean_err = sum(errors) / len(errors)
                mean_signed = sum(signed_errors) / len(signed_errors)
                body_match = mean_err <= self.BODY_ERROR_MATCH_MAX
                body_error = mean_err
                signed_body_error = mean_signed
                if body_match:
                    body_reason = "body_within_expect"
                else:
                    direction = "over" if mean_signed > 0 else "under"
                    body_reason = f"body_mismatch_err={mean_err:.3f}_{direction}"
            else:
                body_match = True
                body_error = 0.0
                signed_body_error = 0.0
                body_reason = "body_parts_linked_but_no_readings"
        else:
            # Still expose trajectory prior for unlinked inspection / future use
            body_match = True
            body_error = 0.0
            signed_body_error = 0.0
            if not linked:
                body_reason = "no_body_parts_linked"
            else:
                body_reason = "body_parts_linked_but_no_readings"

        overall = family_match and body_match
        return PredictResult(
            match=family_match,
            expected_family=str(g0),
            reason=fam_reason,
            off_family_focus=off_family,
            body_match=body_match,
            body_error=float(body_error),
            signed_body_error=float(signed_body_error),
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
        thread_intent: str = "",
        barren_focus: bool = False,
        pain: float = 0.0,
        pleasure: float = 0.0,
        allostatic_escape: str = "",
        plan_suggested_op: str = "",
    ) -> OperatorDecision:
        pred = self.predict(
            graph, focus_id, goal_targets, wm_slots, residual_top, body=body
        )
        self.last_predict = pred

        scores = {op: 0.0 for op in OPS}
        # --- L: learned log-odds (data) ---
        L_scores = {op: 0.0 for op in OPS}
        ctx = str(focus_id or "")
        # Prefer lemma-like context only
        if ctx.startswith(("body:", "felt:", "epistemic_", "narr:", "world:")):
            ctx = ""
        try:
            if self.evidence is not None:
                self.evidence.decay()
                L_scores = self.evidence.L_scores(context=ctx)
                for op_k, Lv in L_scores.items():
                    # scale L (~[-4,4]) into score space
                    scores[op_k] = scores.get(op_k, 0.0) + float(Lv) * 0.85
        except Exception:
            L_scores = {op: 0.0 for op in OPS}
        self._last_L_scores = dict(L_scores)

        # --- B: admitted bias (not knowledge) ---
        # Thin C: soft bias from outcome confidence
        try:
            for op_k, add in self.confidence_bias_scores(body).items():
                scores[op_k] = scores.get(op_k, 0.0) + float(add)
        except Exception:
            pass
        # Package D: soft bias toward plan's suggested operator
        if plan_suggested_op:
            ps = str(plan_suggested_op).upper()
            if ps in scores:
                scores[ps] += 1.25
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
        # Body mismatch while on-family: less pure HOLD, more EXPAND curiosity
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

        # ---- Interoceptive regulation bias ----
        # High body error on regulation-relevant channels → prefer SETTLE / RELEASE
        if pred.body_error > 0.18 and not pred.body_match:
            reg_elevated = False
            for ch in REGULATION_CHANNELS:
                obs = pred.observed_body.get(ch)
                exp = pred.expected_body.get(ch)
                if obs is not None and exp is not None and obs > exp + 0.12:
                    reg_elevated = True
                    break
            if reg_elevated:
                scores["SETTLE"] = max(scores["SETTLE"], 2.4)
                scores["RELEASE"] = max(scores["RELEASE"], 1.8)
                scores["HOLD"] *= 0.75
                scores["EXPAND"] *= 0.6

        # ---- Active Thread intent + barren focus (Cognitive Continuity) ----
        intent = (thread_intent or "").upper()
        if barren_focus or intent == "HOLD":
            # Stop barren EXPAND streaks; HOLD or RETURN only
            force_hold = True
            force_expand = False
            scores["EXPAND"] *= 0.08
            scores["HOLD"] = max(scores["HOLD"], 4.2)
            if intent == "HOLD":
                note_barren = "thread_hold"
            else:
                note_barren = "barren_focus"
        else:
            note_barren = ""
        if intent == "REGULATE":
            scores["SETTLE"] = max(scores["SETTLE"], 3.0)
            scores["RELEASE"] = max(scores["RELEASE"], 2.2)
            scores["EXPAND"] *= 0.25
            scores["HOLD"] *= 0.85
            force_expand = False
        elif intent == "LEARN":
            if lookup_budget_ok and not barren_focus:
                scores["EXPAND"] = max(scores["EXPAND"], 2.6)
            scores["HOLD"] = max(scores["HOLD"], 1.5)
        elif intent == "EXPLORE":
            if lookup_budget_ok and not barren_focus:
                scores["EXPAND"] = max(scores["EXPAND"], 3.0)
            scores["HOLD"] *= 0.9

        # Anti-HOLD-lock: after 6 HOLDs with LEARN/EXPLORE, allow one EXPAND
        if (
            hold_streak >= 6
            and intent in ("LEARN", "EXPLORE")
            and lookup_budget_ok
            and not barren_focus
            and not force_return
            and intent != "HOLD"
        ):
            force_expand = True
            force_hold = False
            scores["EXPAND"] = max(scores["EXPAND"], 3.8)
            scores["HOLD"] *= 0.55

        # Allostasis & Affect — pain/pleasure policy (mild; caps elsewhere)
        pn = float(pain or 0.0)
        pl = float(pleasure or 0.0)
        if pn >= 0.55:
            scores["SETTLE"] = max(scores["SETTLE"], 3.2)
            scores["RELEASE"] = max(scores["RELEASE"], 2.4)
            scores["EXPAND"] *= 0.35
            force_expand = False
        if pl >= 0.55 and not barren_focus and lookup_budget_ok:
            # Mild EXPAND bias only (pleasure addiction guard)
            scores["EXPAND"] = max(scores["EXPAND"], 2.4)
            scores["HOLD"] = max(scores["HOLD"], 1.2)
        if allostatic_escape == "op_diversity":
            # Break pure SETTLE lock under pain
            scores["HOLD"] = max(scores["HOLD"], 3.5)
            scores["RETURN"] = max(scores["RETURN"], 2.0)
            scores["SETTLE"] *= 0.4
            force_hold = True
            force_expand = False
        if allostatic_escape == "time_clamp":
            scores["SETTLE"] = max(scores["SETTLE"], 3.5)
            scores["RELEASE"] = max(scores["RELEASE"], 3.0)
            scores["EXPAND"] *= 0.2

        # Hard forces
        if force_return:
            scores["RETURN"] = max(scores["RETURN"], 5.0)
            scores["EXPAND"] *= 0.2
        if force_expand and not force_return and not barren_focus:
            scores["EXPAND"] = max(scores["EXPAND"], 4.0)
            scores["HOLD"] *= 0.5
        if force_hold and not force_return and not force_expand:
            scores["HOLD"] = max(scores["HOLD"], 4.0)
            scores["EXPAND"] *= 0.15

        # Pick (serial commitment)
        op = max(scores, key=lambda k: scores[k])
        if scores[op] <= 0:
            op = "HOLD"

        evidence_led = False
        try:
            if self.evidence is not None:
                evidence_led = self.evidence.evidence_led(op, context=ctx)
        except Exception:
            evidence_led = False
        self._last_evidence_led = bool(evidence_led)

        note_parts = [pred.reason]
        if pred.body_reason:
            note_parts.append(pred.body_reason)
        if force_return:
            note_parts.append("force_return")
        if force_expand:
            note_parts.append("force_expand")
        if force_hold:
            note_parts.append("force_hold")
        if note_barren:
            note_parts.append(note_barren)
        if intent:
            note_parts.append(f"intent={intent}")
        # Diagnostics: L vs decision
        try:
            Lv = float(L_scores.get(op, 0.0))
            note_parts.append(f"L_{op}={Lv:.2f}")
            if evidence_led:
                note_parts.append("evidence_led")
            else:
                note_parts.append("bias_or_weak_L")
        except Exception:
            pass

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

        # Re-record with the chosen operator so next prior is conditioned correctly
        if body:
            self._record_body(body, op=op)

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
                "signed_body_error": pred.signed_body_error if pred else None,
                "expected_body": pred.expected_body if pred else None,
                "observed_body": pred.observed_body if pred else None,
                "body_reason": pred.body_reason if pred else None,
            },
        }
        row.update(extra)
        self.episodes.append(row)
        if len(self.episodes) > self.EPISODE_CAP:
            self.episodes = self.episodes[-self.EPISODE_CAP :]

    def credit_evidence(
        self,
        chosen: str,
        improved: Optional[bool],
        context: str = "",
        magnitude: float = 1.0,
    ) -> None:
        """Write-back: update log-odds from scored outcome."""
        if self.evidence is None:
            return
        try:
            self.evidence.update(
                chosen, improved, context=context, magnitude=magnitude
            )
        except Exception:
            pass

    def report(self) -> dict:
        top_conf = sorted(
            self.outcome_confidence.items(), key=lambda kv: -kv[1]
        )[:12]
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
                    "signed_body_error": self.last_predict.signed_body_error,
                    "expected_body": self.last_predict.expected_body,
                    "observed_body": self.last_predict.observed_body,
                    "body_reason": self.last_predict.body_reason,
                }
                if self.last_predict
                else None
            ),
            "episodes_tail": self.episodes[-12:],
            "op_ring": list(self._op_ring[-12:]),
            "body_hist_len": len(self._body_hist),
            "delta_ema": {k: round(v, 4) for k, v in self._delta_ema.items()},
            "pending_body_delta": dict(self.pending_body_delta),
            "last_act_op": self._last_act_op,
            "last_act_body_deltas": dict(self._last_act_body_deltas),
            "last_act_world_deltas": dict(self._last_act_world_deltas),
            "L_scores": dict(getattr(self, "_last_L_scores", {}) or {}),
            "evidence_led": bool(getattr(self, "_last_evidence_led", False)),
            "evidence": (
                self.evidence.report() if getattr(self, "evidence", None) is not None else {}
            ),
            "outcome_confidence_top": [
                {"key": k, "confidence": round(v, 3)} for k, v in top_conf
            ],
        }
