"""
evidence.py — Log-odds evidence + threshold choice materials.

Function (project "Parietal" rule):
  L_i accumulates from prediction / outcome hits over time.
  Choice when score_i = L_i + B_i crosses threshold (among candidates).

Bias B is admitted prior (intent, plan, allostasis) — not counted as knowledge.
L is learned data — decayed slowly so old noise does not lock forever.

No brain layout. Same rule for operator choice; link credit can reuse hit/miss API.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Operators are the first candidate set (serial act choice)
DEFAULT_CANDIDATES = ("HOLD", "RETURN", "EXPAND", "RELEASE", "SETTLE")

# Log-odds update (in log space, natural log)
L_HIT = 0.35          # evidence for chosen op when outcome improved
L_MISS = 0.40          # evidence against when worsened
L_WEAK = 0.12          # small nudge on neutral / weak signal
L_DECAY = 0.997        # per-pulse multiply toward 0 (|L| shrinks)
L_MIN = -4.0
L_MAX = 4.0
L_START = 0.0

# Threshold on combined score (L is ~[-4,4], B is separate additive in choose)
# We expose theta for "decided by evidence" diagnostics, not only soft-max.
THETA_EVIDENCE = 0.85  # |L| above this counts as evidence-led if also wins


class EvidenceModule:
    """Track log-odds per candidate; update from scored outcomes."""

    def __init__(self, candidates: Tuple[str, ...] = DEFAULT_CANDIDATES):
        self.candidates = tuple(candidates)
        self.log_odds: Dict[str, float] = {c: L_START for c in self.candidates}
        # Optional contextual L: "focus_lemma|OP" — sparse, for later schema use
        self.ctx_log_odds: Dict[str, float] = {}
        self.last_report: Dict = {}
        self._pulse = 0

    def decay(self) -> None:
        for k in list(self.log_odds.keys()):
            self.log_odds[k] = self._clamp(float(self.log_odds[k]) * L_DECAY)
        if len(self.ctx_log_odds) > 200:
            # prune weakest contextual entries
            ranked = sorted(self.ctx_log_odds.items(), key=lambda kv: abs(kv[1]))
            for key, _ in ranked[:40]:
                self.ctx_log_odds.pop(key, None)
        for k in list(self.ctx_log_odds.keys()):
            self.ctx_log_odds[k] = self._clamp(float(self.ctx_log_odds[k]) * L_DECAY)

    @staticmethod
    def _clamp(x: float) -> float:
        return max(L_MIN, min(L_MAX, float(x)))

    def get_L(self, candidate: str, context: str = "") -> float:
        c = str(candidate).upper()
        base = float(self.log_odds.get(c, L_START))
        if context:
            key = f"{context}|{c}"
            base += 0.55 * float(self.ctx_log_odds.get(key, 0.0))
        return base

    def L_scores(self, context: str = "") -> Dict[str, float]:
        return {c: self.get_L(c, context=context) for c in self.candidates}

    def update(
        self,
        chosen: str,
        improved: Optional[bool],
        context: str = "",
        magnitude: float = 1.0,
    ) -> None:
        """Credit chosen candidate from scored outcome.

        improved True  → hit (raise L_chosen, mild lower others)
        improved False → miss (lower L_chosen)
        improved None  → weak / neutral
        """
        c = str(chosen).upper()
        if c not in self.log_odds:
            self.log_odds[c] = L_START
        mag = max(0.25, min(2.0, float(magnitude or 1.0)))

        if improved is True:
            delta = L_HIT * mag
            self.log_odds[c] = self._clamp(self.log_odds[c] + delta)
            for o in self.candidates:
                if o != c:
                    self.log_odds[o] = self._clamp(self.log_odds[o] - 0.15 * delta)
            if context:
                key = f"{context}|{c}"
                self.ctx_log_odds[key] = self._clamp(
                    float(self.ctx_log_odds.get(key, 0.0)) + delta
                )
        elif improved is False:
            delta = L_MISS * mag
            self.log_odds[c] = self._clamp(self.log_odds[c] - delta)
            if context:
                key = f"{context}|{c}"
                self.ctx_log_odds[key] = self._clamp(
                    float(self.ctx_log_odds.get(key, 0.0)) - delta
                )
        else:
            # neutral: slight decay of confidence in the choice
            self.log_odds[c] = self._clamp(self.log_odds[c] - L_WEAK * 0.25 * mag)

        self.last_report = {
            "chosen": c,
            "improved": improved,
            "context": context or None,
            "L": dict(self.log_odds),
            "magnitude": mag,
        }

    def evidence_led(self, chosen: str, context: str = "") -> bool:
        """True if |L| for winner is past theta (data, not only bias)."""
        return abs(self.get_L(chosen, context=context)) >= THETA_EVIDENCE

    def report(self) -> dict:
        L = {c: round(float(self.log_odds.get(c, 0.0)), 3) for c in self.candidates}
        top = sorted(L.items(), key=lambda kv: -abs(kv[1]))
        ctx_top = sorted(
            ((k, round(v, 3)) for k, v in self.ctx_log_odds.items()),
            key=lambda kv: -abs(kv[1]),
        )[:8]
        return {
            "L": L,
            "top": top[:5],
            "theta": THETA_EVIDENCE,
            "ctx_top": ctx_top,
            "last": dict(self.last_report or {}),
        }


def score_body_improvement(
    before: Dict[str, float],
    after: Dict[str, float],
    *,
    pain_before: float = 0.0,
    pain_after: float = 0.0,
    setpoints: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[bool], float]:
    """Heuristic global score for operator credit (self-domain).

    Returns (improved, magnitude).
    improved: True/False/None (neutral)
    """
    setpoints = setpoints or {}
    # Prefer lower pain, lower tension, lower |error to setpoint| if provided
    keys = ("muscle_tension", "pain", "sweat_skin", "heart_rate", "energy", "pleasure")
    err_before = 0.0
    err_after = 0.0
    n = 0
    for k in keys:
        if k not in before and k not in after:
            continue
        b = float(before.get(k, after.get(k, 0.5)))
        a = float(after.get(k, b))
        sp = setpoints.get(k)
        if sp is None:
            # default comfort priors: pain/tension low, energy/pleasure mid-high
            if k in ("pain", "muscle_tension", "sweat_skin"):
                sp = 0.25
            elif k in ("energy", "pleasure"):
                sp = 0.55
            else:
                sp = 0.45
        err_before += abs(b - float(sp))
        err_after += abs(a - float(sp))
        n += 1
    # pain surface explicit
    err_before += max(0.0, float(pain_before) - 0.3) * 1.5
    err_after += max(0.0, float(pain_after) - 0.3) * 1.5

    if n == 0 and pain_before == 0 and pain_after == 0:
        return None, 0.0
    delta = err_before - err_after  # positive = improved
    if delta > 0.02:
        return True, min(2.0, delta / 0.05)
    if delta < -0.02:
        return False, min(2.0, abs(delta) / 0.05)
    return None, 0.0
