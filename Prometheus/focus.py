"""
focus.py -- §13.y Residual Focus (Goals) & Prediction Error.

Pieces A–E in one module so the orchestrator stays thin:
  A. Residual store (boost / decay / cap)
  B. Single sticky focus thread
  C. Consumer helpers (WM boost, self-study bias, collapse protection)
  D. Schema expected_families (Consolidation EMA)
  E. Prediction error → residual when a schema is active

Goals are not strings. A goal is whatever keeps winning sticky focus
until residual tension falls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .archivist import SELF_NODE, OTHER_NODE, TIER_TRUSTED, TIER_WORKING
from .edge_types import (
    FAMILY_CAUSAL,
    FAMILY_HIERARCHY,
    FAMILY_MEMBERSHIP,
    FAMILY_RESIDUAL,
    FAMILY_ROLE,
    FAMILY_SOCIAL_NORM,
    get_family,
    NODE_SCHEMA,
    NODE_EPISTEMIC_SCHEMA,
)

logger = logging.getLogger(__name__)

# Structural families used for uncertainty + prediction (not RESIDUAL glue).
STRUCTURAL_FAMILIES = frozenset({
    FAMILY_HIERARCHY,
    FAMILY_MEMBERSHIP,
    FAMILY_ROLE,
    FAMILY_CAUSAL,
    FAMILY_SOCIAL_NORM,
})

# Families tracked as schema expectations in v1.
# Growable first: hierarchy/membership are what self-study actually writes.
# Role/causal/social are tracked but down-weighted until write paths mature.
TRACKED_EXPECTATION_FAMILIES = frozenset({
    FAMILY_ROLE,
    FAMILY_CAUSAL,
    FAMILY_SOCIAL_NORM,
    FAMILY_HIERARCHY,
    FAMILY_MEMBERSHIP,
})

# Relative weight on prediction error contribution by family.
# Unfillable families (no producer yet) must not dominate focus forever.
EXPECTATION_FAMILY_WEIGHT = {
    FAMILY_HIERARCHY: 1.0,
    FAMILY_MEMBERSHIP: 1.0,
    FAMILY_SOCIAL_NORM: 0.45,  # only if relational intake fires
    FAMILY_ROLE: 0.25,         # thin write path
    FAMILY_CAUSAL: 0.15,       # almost no producer yet
}


@dataclass
class FocusThread:
    target_id: str
    kind: str = "node"  # "node" | "schema"
    score: float = 0.0
    created_pulse: int = 0
    last_seen_pulse: int = 0
    source_mix: Dict[str, float] = field(default_factory=dict)


class FocusModule:
    """
    §13.y. Owns residuals + one sticky focus thread + prediction hooks.
    Call sites (prometheus.py):
      - on activation: boost_residual(node_id)
      - each pulse after learn/self-study: tick(pulse, ...)
      - consolidation: update_expected_families(graph); decay_residuals(stronger=True)
      - collapse protection: protected_ids()
      - WM / self-study: focus_id, focus_boost_for(node)
    """

    # --- Piece A defaults ---
    RESIDUAL_DECAY = 0.95
    RESIDUAL_DECAY_CONSOLIDATION = 0.85
    RESIDUAL_CAP = 15.0
    RESIDUAL_BOOST = 1.0
    RESIDUAL_FLOOR = 0.05

    # Uncertainty weights
    W_ACT = 1.0
    W_UNC = 0.8
    W_PRED = 1.2
    W_PAR = 0.5

    # --- Piece B defaults ---
    MIN_FOCUS_RESIDENCY = 8
    FOCUS_SWITCH_MARGIN = 0.15  # challenger must beat current by this fraction
    CANDIDATE_POOL_CAP = 40
    # Soft stagnation (cold act): slash pred and force switch opportunity.
    MAX_FOCUS_AGE_STAGNANT = 60
    STAGNANT_PRED_DECAY = 0.15
    STAGNANT_ACT_FLOOR = 0.2
    # Hard max age: force switch regardless of act/pred (breaks hot-node feedback locks).
    MAX_FOCUS_AGE_HARD = 100
    # After leaving a target, block it from winning focus again for this many pulses.
    FOCUS_COOLDOWN_PULSES = 40

    # --- Piece E defaults ---
    # Lower gain so unfilled gaps cannot pin equilibrium against decay forever.
    K_PRED = 0.25
    PRED_EMA_ALPHA = 0.2
    MIN_MEMBERS_FOR_EXPECTATION = 3
    # Only inject prediction residual every N pulses (breaks perfect fixed-point).
    PRED_INJECT_EVERY = 3
    # Soften expected_families that stay missing while schema is focused.
    EXPECTATION_SOFTEN_AFTER = 40   # pulses of focus with family still missing
    EXPECTATION_SOFTEN_FACTOR = 0.92  # per soften tick toward ignoring dead gaps
    EXPECTATION_SOFTEN_FLOOR = 0.12  # below this, family stops generating error

    # Consumer boosts
    WM_FOCUS_BONUS = 6.0
    SELF_STUDY_FOCUS_WEIGHT = 4.0

    def __init__(self):
        self.residuals: Dict[str, float] = {}
        self.r_pred: Dict[str, float] = {}
        self.r_par: Dict[str, float] = {}
        self.thread: Optional[FocusThread] = None
        self.last_error: Dict[str, float] = {}
        self.last_tick_summary: Dict = {}
        self._force_switch: bool = False
        self._stagnation_events: int = 0
        self._hard_age_events: int = 0
        # node_id -> pulse when cooldown expires
        self._cooldown_until: Dict[str, int] = {}
        # schema_id -> {family: consecutive pulses still missing while focused}
        self._missing_streak: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Piece A — residual store
    # ------------------------------------------------------------------
    def boost_residual(self, node_id: str, amount: Optional[float] = None) -> None:
        if not node_id or node_id in (SELF_NODE, OTHER_NODE):
            return
        amt = self.RESIDUAL_BOOST if amount is None else amount
        self.residuals[node_id] = min(
            self.RESIDUAL_CAP,
            self.residuals.get(node_id, 0.0) + amt,
        )

    def add_parental_residual(self, node_id: str, amount: float) -> None:
        if not node_id or node_id in (SELF_NODE, OTHER_NODE):
            return
        self.r_par[node_id] = min(
            self.RESIDUAL_CAP,
            max(0.0, self.r_par.get(node_id, 0.0) + amount),
        )

    def _decay_dict(self, d: Dict[str, float], rate: float) -> None:
        dead = []
        for k, v in d.items():
            nv = v * rate
            if nv < self.RESIDUAL_FLOOR:
                dead.append(k)
            else:
                d[k] = min(self.RESIDUAL_CAP, nv)
        for k in dead:
            del d[k]

    def decay_residuals(self, consolidation: bool = False) -> None:
        rate = self.RESIDUAL_DECAY_CONSOLIDATION if consolidation else self.RESIDUAL_DECAY
        self._decay_dict(self.residuals, rate)
        self._decay_dict(self.r_pred, rate)
        self._decay_dict(self.r_par, rate)

    def total_residual(self, node_id: str) -> float:
        return (
            self.W_ACT * self.residuals.get(node_id, 0.0)
            + self.W_PRED * self.r_pred.get(node_id, 0.0)
            + self.W_PAR * self.r_par.get(node_id, 0.0)
        )

    # ------------------------------------------------------------------
    # Uncertainty from graph (cheap, on candidates only)
    # ------------------------------------------------------------------
    def uncertainty_residual(self, graph, node_id: str) -> float:
        if node_id not in graph:
            return 0.0
        data = graph.nodes[node_id]
        tier = data.get("tier", 0)
        # Map tier to rough trust unit in [0,1]
        if tier >= TIER_TRUSTED:
            trust_unit = 1.0
        elif tier >= TIER_WORKING:
            trust_unit = 0.6
        else:
            trust_unit = 0.25

        families_seen: Set[str] = set()
        for _u, _v, ed in graph.edges(node_id, data=True):
            fam = get_family(ed.get("relation_type", ""), ed.get("family"))
            if fam in STRUCTURAL_FAMILIES:
                families_seen.add(fam)
        for _u, _v, ed in graph.in_edges(node_id, data=True):
            fam = get_family(ed.get("relation_type", ""), ed.get("family"))
            if fam in STRUCTURAL_FAMILIES:
                families_seen.add(fam)

        # sparsity: fewer structural families → higher penalty
        sparsity = 1.0 - (len(families_seen) / max(1, len(STRUCTURAL_FAMILIES)))
        return self.W_UNC * ((1.0 - trust_unit) + 0.5 * sparsity)

    def composite_score(self, graph, node_id: str, basin_anchor_set: Optional[Set[str]] = None) -> Tuple[float, Dict[str, float]]:
        basin_anchor_set = basin_anchor_set or set()
        r_act = self.residuals.get(node_id, 0.0)
        r_pred = self.r_pred.get(node_id, 0.0)
        r_par = self.r_par.get(node_id, 0.0)
        r_unc = self.uncertainty_residual(graph, node_id)
        basin_bonus = 5.0 if node_id in basin_anchor_set else 0.0
        data = graph.nodes.get(node_id, {})
        schema_bonus = 2.0 if (
            data.get("is_schema")
            or data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
        ) else 0.0
        mix = {
            "act": r_act,
            "unc": r_unc,
            "pred": r_pred,
            "par": r_par,
            "basin": basin_bonus,
            "schema": schema_bonus,
        }
        score = (
            self.W_ACT * r_act
            + r_unc
            + self.W_PRED * r_pred
            + self.W_PAR * r_par
            + basin_bonus
            + schema_bonus
        )
        return score, mix

    # ------------------------------------------------------------------
    # Piece B — sticky focus
    # ------------------------------------------------------------------
    def _candidate_ids(self, graph, basin_anchor_set: Set[str]) -> List[str]:
        scored = []
        # Seed from residual keys + basin anchors + current focus
        seeds = set(self.residuals) | set(self.r_pred) | set(self.r_par) | set(basin_anchor_set)
        if self.thread:
            seeds.add(self.thread.target_id)
        for n in seeds:
            if n in graph and n not in (SELF_NODE, OTHER_NODE):
                s, _ = self.composite_score(graph, n, basin_anchor_set)
                if s > self.RESIDUAL_FLOOR:
                    scored.append((n, s))
        # Also consider high-activation nodes so focus can start on a cold residual table
        if len(scored) < 10:
            act_ranked = sorted(
                (
                    (n, d.get("activation", 0.0))
                    for n, d in graph.nodes(data=True)
                    if n not in (SELF_NODE, OTHER_NODE) and not d.get("is_basin")
                ),
                key=lambda t: t[1],
                reverse=True,
            )[:15]
            for n, _a in act_ranked:
                if n not in seeds:
                    s, _ = self.composite_score(graph, n, basin_anchor_set)
                    scored.append((n, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [n for n, _ in scored[: self.CANDIDATE_POOL_CAP]]

    def select_focus(
        self,
        graph,
        pulse: int,
        basin_anchor_set: Optional[Set[str]] = None,
    ) -> Optional[FocusThread]:
        basin_anchor_set = basin_anchor_set or set()
        if graph.number_of_nodes() == 0:
            self.thread = None
            return None

        candidates = self._candidate_ids(graph, basin_anchor_set)
        if not candidates:
            self.thread = None
            return None

        best_id = None
        best_score = -1.0
        best_mix: Dict[str, float] = {}
        best_open_id = None
        best_open_score = -1.0
        best_open_mix: Dict[str, float] = {}
        for n in candidates:
            s, mix = self.composite_score(graph, n, basin_anchor_set)
            if s > best_score:
                best_score = s
                best_id = n
                best_mix = mix
            # Track best candidate not in cooldown (for forced switches)
            if not self._in_cooldown(n, pulse) and s > best_open_score:
                best_open_score = s
                best_open_id = n
                best_open_mix = mix

        if best_id is None:
            self.thread = None
            return None

        data = graph.nodes.get(best_id, {})
        kind = (
            "schema"
            if (data.get("is_schema") or data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA))
            else "node"
        )

        if self.thread is None:
            self.thread = FocusThread(
                target_id=best_id,
                kind=kind,
                score=best_score,
                created_pulse=pulse,
                last_seen_pulse=pulse,
                source_mix=best_mix,
            )
            return self.thread

        current = self.thread
        cur_score, cur_mix = self.composite_score(graph, current.target_id, basin_anchor_set)
        current.score = cur_score
        current.source_mix = cur_mix
        current.last_seen_pulse = pulse

        age = pulse - current.created_pulse
        residency_met = age >= self.MIN_FOCUS_RESIDENCY
        margin_needed = max(abs(cur_score) * self.FOCUS_SWITCH_MARGIN, 0.5)
        challenger_wins = best_id != current.target_id and best_score > cur_score + margin_needed

        # Stagnation escape: cold activation + long age → allow switch without margin
        if self._force_switch:
            # Prefer best non-cooldown candidate; fall back to any other node
            pick = best_open_id if (best_open_id and best_open_id != current.target_id) else None
            if pick is None and best_id is not None and best_id != current.target_id:
                pick = best_id
            if pick is not None:
                pdata = graph.nodes.get(pick, {})
                pkind = (
                    "schema"
                    if (pdata.get("is_schema") or pdata.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA))
                    else "node"
                )
                ps, pmix = self.composite_score(graph, pick, basin_anchor_set)
                self.thread = FocusThread(
                    target_id=pick,
                    kind=pkind,
                    score=ps,
                    created_pulse=pulse,
                    last_seen_pulse=pulse,
                    source_mix=pmix if pick == best_open_id else (best_mix if pick == best_id else pmix),
                )
                self._force_switch = False
                return self.thread
            # No alternative exists — clear flag and keep current one more cycle
            self._force_switch = False

        if residency_met and challenger_wins:
            self._start_cooldown(current.target_id, pulse)
            self.thread = FocusThread(
                target_id=best_id,
                kind=kind,
                score=best_score,
                created_pulse=pulse,
                last_seen_pulse=pulse,
                source_mix=best_mix,
            )
        return self.thread

    # ------------------------------------------------------------------
    # Piece C — consumer helpers
    # ------------------------------------------------------------------
    @property
    def focus_id(self) -> Optional[str]:
        return self.thread.target_id if self.thread else None

    def focus_boost_for(self, node_id: str) -> float:
        """Additive score bonus for WM ranking."""
        if self.thread and node_id == self.thread.target_id:
            return self.WM_FOCUS_BONUS
        return 0.0

    def self_study_weight(self, node_id: str) -> float:
        """Multiplicative-ish weight contribution for self-study selection."""
        if self.thread and node_id == self.thread.target_id:
            return self.SELF_STUDY_FOCUS_WEIGHT
        # Neighbourhood: mild boost if residual is high
        r = self.total_residual(node_id)
        if r > 1.0:
            return 1.0 + min(2.0, r / 5.0)
        return 1.0

    def protected_ids(self) -> Set[str]:
        out: Set[str] = set()
        if self.thread:
            out.add(self.thread.target_id)
        return out

    def neighbourhood_boost_ids(self, graph, max_n: int = 12) -> Set[str]:
        """Focus target + immediate neighbors for self-study pool bias."""
        fid = self.focus_id
        if not fid or fid not in graph:
            return set()
        ids = {fid}
        for _, v in list(graph.out_edges(fid))[:max_n]:
            ids.add(v)
        for u, _ in list(graph.in_edges(fid))[:max_n]:
            ids.add(u)
        return ids

    # ------------------------------------------------------------------
    # Piece D — expected families on schemas
    # ------------------------------------------------------------------
    def update_expected_families(self, graph) -> int:
        """Consolidation-gated EMA of family presence across schema members.
        Returns number of schemas updated."""
        updated = 0
        for sid, data in list(graph.nodes(data=True)):
            if not (
                data.get("is_schema")
                or data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
            ):
                continue
            members = [
                v for _u, v, ed in graph.out_edges(sid, data=True)
                if get_family(ed.get("relation_type", ""), ed.get("family")) == FAMILY_MEMBERSHIP
                or ed.get("relation_type") in ("composed-of", "instance-of")
            ]
            if len(members) < self.MIN_MEMBERS_FOR_EXPECTATION:
                continue

            family_hits = {f: 0 for f in TRACKED_EXPECTATION_FAMILIES}
            for m in members:
                if m not in graph:
                    continue
                seen = set()
                for _u, _v, ed in graph.edges(m, data=True):
                    fam = get_family(ed.get("relation_type", ""), ed.get("family"))
                    if fam in TRACKED_EXPECTATION_FAMILIES:
                        seen.add(fam)
                for _u, _v, ed in graph.in_edges(m, data=True):
                    fam = get_family(ed.get("relation_type", ""), ed.get("family"))
                    if fam in TRACKED_EXPECTATION_FAMILIES:
                        seen.add(fam)
                for f in seen:
                    family_hits[f] += 1

            n = float(len(members))
            observed = {f: family_hits[f] / n for f in TRACKED_EXPECTATION_FAMILIES}
            prev = dict(data.get("expected_families") or {})
            alpha = self.PRED_EMA_ALPHA
            merged = {}
            for f in TRACKED_EXPECTATION_FAMILIES:
                # Down-weight ungrowable families in the prototype itself
                w = EXPECTATION_FAMILY_WEIGHT.get(f, 0.3)
                obs = observed[f] * (0.5 + 0.5 * w)  # hierarchy full; causal muted
                old = float(prev.get(f, obs))
                merged[f] = (1.0 - alpha) * old + alpha * obs
            graph.nodes[sid]["expected_families"] = merged
            updated += 1
        return updated

    # ------------------------------------------------------------------
    # Piece E — prediction error → residual
    # ------------------------------------------------------------------
    def _local_families_present(self, graph, schema_id: str) -> Set[str]:
        present: Set[str] = set()
        if schema_id not in graph:
            return present
        nodes = {schema_id}
        for _u, v, ed in graph.out_edges(schema_id, data=True):
            if get_family(ed.get("relation_type", ""), ed.get("family")) == FAMILY_MEMBERSHIP:
                nodes.add(v)
            fam = get_family(ed.get("relation_type", ""), ed.get("family"))
            if fam in TRACKED_EXPECTATION_FAMILIES:
                present.add(fam)
        for n in list(nodes):
            if n not in graph:
                continue
            for _u, _v, ed in graph.edges(n, data=True):
                fam = get_family(ed.get("relation_type", ""), ed.get("family"))
                if fam in TRACKED_EXPECTATION_FAMILIES:
                    present.add(fam)
            for _u, _v, ed in graph.in_edges(n, data=True):
                fam = get_family(ed.get("relation_type", ""), ed.get("family"))
                if fam in TRACKED_EXPECTATION_FAMILIES:
                    present.add(fam)
        return present

    def prediction_error(self, graph, schema_id: str) -> float:
        """Weighted structural gap. Hierarchy/membership count fully;
        role/causal/social are discounted. Strengths already softened for
        long-missing families are stored back onto the schema node."""
        data = graph.nodes.get(schema_id, {})
        expected = dict(data.get("expected_families") or {})
        if not expected:
            return 0.0
        present = self._local_families_present(graph, schema_id)
        error = 0.0
        changed = False
        streak = self._missing_streak.setdefault(schema_id, {})
        for fam, strength in list(expected.items()):
            try:
                s = float(strength)
            except (TypeError, ValueError):
                continue
            if s < self.EXPECTATION_SOFTEN_FLOOR:
                continue
            if fam in present:
                streak[fam] = 0
                continue
            # Still missing
            streak[fam] = streak.get(fam, 0) + 1
            if streak[fam] >= self.EXPECTATION_SOFTEN_AFTER:
                new_s = max(self.EXPECTATION_SOFTEN_FLOOR * 0.5, s * self.EXPECTATION_SOFTEN_FACTOR)
                if new_s != s:
                    expected[fam] = new_s
                    s = new_s
                    changed = True
            w = EXPECTATION_FAMILY_WEIGHT.get(fam, 0.3)
            if s >= self.EXPECTATION_SOFTEN_FLOOR:
                error += s * w
        if changed:
            graph.nodes[schema_id]["expected_families"] = expected
        return error

    def apply_prediction_to_residuals(self, graph, pulse: int = 0) -> float:
        """If focus is a schema, add prediction error into r_pred.
        Injected only every PRED_INJECT_EVERY pulses so decay can move
        the residual instead of locking a perfect fixed point."""
        fid = self.focus_id
        if not fid or fid not in graph:
            return 0.0
        data = graph.nodes.get(fid, {})
        if not (
            data.get("is_schema")
            or data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
        ):
            return 0.0
        err = self.prediction_error(graph, fid)
        self.last_error[fid] = err
        if err > 0 and (pulse % max(1, self.PRED_INJECT_EVERY) == 0):
            self.r_pred[fid] = min(
                self.RESIDUAL_CAP,
                self.r_pred.get(fid, 0.0) + self.K_PRED * err,
            )
        return err

    def _purge_expired_cooldowns(self, pulse: int) -> None:
        dead = [k for k, until in self._cooldown_until.items() if until <= pulse]
        for k in dead:
            del self._cooldown_until[k]

    def _start_cooldown(self, node_id: str, pulse: int) -> None:
        if not node_id:
            return
        self._cooldown_until[node_id] = pulse + self.FOCUS_COOLDOWN_PULSES

    def _in_cooldown(self, node_id: str, pulse: int) -> bool:
        until = self._cooldown_until.get(node_id)
        return until is not None and until > pulse

    def _force_leave_current(self, pulse: int, reason: str) -> None:
        """Slash residuals on current focus, blacklist it, arm force switch."""
        if self.thread is None:
            return
        fid = self.thread.target_id
        if fid in self.r_pred:
            self.r_pred[fid] *= self.STAGNANT_PRED_DECAY
            if self.r_pred[fid] < self.RESIDUAL_FLOOR:
                del self.r_pred[fid]
        if fid in self.residuals:
            self.residuals[fid] *= self.STAGNANT_PRED_DECAY
            if self.residuals[fid] < self.RESIDUAL_FLOOR:
                del self.residuals[fid]
        self._start_cooldown(fid, pulse)
        self._force_switch = True
        if reason == "hard_age":
            self._hard_age_events += 1
        else:
            self._stagnation_events += 1

    def _maybe_stagnation_escape(self, pulse: int) -> bool:
        """Soft escape: old + relatively cold act.
        Hard escape: age >= MAX_FOCUS_AGE_HARD regardless of act
        (breaks hot-node self-study feedback locks like 'jubilation')."""
        if self.thread is None:
            return False
        age = pulse - self.thread.created_pulse
        fid = self.thread.target_id
        act = self.residuals.get(fid, 0.0)

        # Hard max age — always leaves, even if act is hot
        if age >= self.MAX_FOCUS_AGE_HARD:
            self._force_leave_current(pulse, "hard_age")
            return True

        # Soft stagnation — cold-ish act
        if age >= self.MAX_FOCUS_AGE_STAGNANT and act <= self.STAGNANT_ACT_FLOOR:
            self._force_leave_current(pulse, "stagnant")
            return True

        return False

    # ------------------------------------------------------------------
    # Per-pulse tick (call from prometheus.pulse)
    # ------------------------------------------------------------------
    def tick(
        self,
        graph,
        pulse: int,
        basin_anchor_set: Optional[Set[str]] = None,
    ) -> Dict:
        basin_anchor_set = basin_anchor_set or set()
        self._purge_expired_cooldowns(pulse)
        self.decay_residuals(consolidation=False)
        stagnant = self._maybe_stagnation_escape(pulse)
        err = self.apply_prediction_to_residuals(graph, pulse=pulse)
        thread = self.select_focus(graph, pulse, basin_anchor_set)
        summary = {
            "focus_id": thread.target_id if thread else None,
            "focus_kind": thread.kind if thread else None,
            "focus_score": thread.score if thread else 0.0,
            "prediction_error": err,
            "stagnation_escape": stagnant,
            "stagnation_events": self._stagnation_events,
            "hard_age_events": self._hard_age_events,
            "cooldowns_active": len(self._cooldown_until),
            "residual_count": len(self.residuals),
            "top_residuals": self.top_residuals(8),
        }
        self.last_tick_summary = summary
        return summary

    def top_residuals(self, n: int = 8) -> List[Tuple[str, float]]:
        totals = {}
        keys = set(self.residuals) | set(self.r_pred) | set(self.r_par)
        for k in keys:
            totals[k] = self.total_residual(k)
        return sorted(totals.items(), key=lambda t: t[1], reverse=True)[:n]

    def report(self) -> Dict:
        t = self.thread
        return {
            "focus_id": t.target_id if t else None,
            "focus_kind": t.kind if t else None,
            "focus_score": t.score if t else 0.0,
            "focus_age_pulses": None if not t else max(0, t.last_seen_pulse - t.created_pulse),
            "source_mix": t.source_mix if t else {},
            "top_residuals": self.top_residuals(10),
            "last_prediction_error": dict(self.last_error),
            "residual_count": len(self.residuals),
            "stagnation_events": self._stagnation_events,
            "hard_age_events": self._hard_age_events,
            "cooldowns_active": len(self._cooldown_until),
            "force_switch_armed": self._force_switch,
            "last_tick": dict(self.last_tick_summary),
        }
