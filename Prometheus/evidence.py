"""
evidence.py — Log-odds evidence + threshold choice materials.

Function (project "Parietal" rule):
  L_i accumulates from prediction / outcome hits over time.
  Choice when score_i = L_i + B_i crosses threshold (among candidates).

Bias B is admitted prior (intent, plan, allostasis) — not counted as knowledge.
L is learned data — decayed so old noise does not lock forever.

Anti-starvation:
  - No (or tiny) rival penalty on hits — rivals need trials to learn
  - Soft floor so L cannot pin at L_MIN forever without recovery
  - Exploration lift for candidates far below the leader
  - Neutral / "within expect" is NOT a hit

No brain layout. Same rule for operator choice; link credit can reuse hit/miss API.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

DEFAULT_CANDIDATES = ("HOLD", "RETURN", "EXPAND", "RELEASE", "SETTLE")

# Log-odds update
L_HIT = 0.28           # raise chosen on real improvement
L_MISS = 0.32          # lower chosen on real worsening
L_NEUTRAL = 0.04       # tiny drift toward 0 on neutral (not a hit)
L_DECAY = 0.996         # per-pulse |L| shrink
L_MIN = -3.0            # softer than -4 (easier recovery)
L_MAX = 3.0
L_START = 0.0
L_RIVAL_PENALTY = 0.0   # was 0.15 — disabled to stop HOLD wipeout
L_SOFT_FLOOR = -1.25    # if below, drift up each decay (recovery)
L_EXPLORE_GAP = 1.5     # if leader - L_i > gap, add explore lift in scores
L_EXPLORE_LIFT = 0.55

THETA_EVIDENCE = 0.85


class EvidenceModule:
    """Track log-odds per candidate; update from scored outcomes."""

    def __init__(self, candidates: Tuple[str, ...] = DEFAULT_CANDIDATES):
        self.candidates = tuple(candidates)
        self.log_odds: Dict[str, float] = {c: L_START for c in self.candidates}
        self.ctx_log_odds: Dict[str, float] = {}
        self.trial_count: Dict[str, int] = {c: 0 for c in self.candidates}
        self.last_report: Dict = {}

    def decay(self) -> None:
        for k in list(self.log_odds.keys()):
            v = float(self.log_odds[k]) * L_DECAY
            # Soft recovery toward floor: don't stay pinned at L_MIN
            if v < L_SOFT_FLOOR:
                v = v + 0.02 * (L_SOFT_FLOOR - v)
            self.log_odds[k] = self._clamp(v)
        if len(self.ctx_log_odds) > 200:
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
        """Raw L plus exploration lift for starved rivals."""
        raw = {c: self.get_L(c, context=context) for c in self.candidates}
        if not raw:
            return raw
        leader = max(raw.values())
        out = {}
        for c, v in raw.items():
            lift = 0.0
            if (leader - v) >= L_EXPLORE_GAP:
                # Stronger lift if almost never tried
                trials = int(self.trial_count.get(c, 0) or 0)
                starve = 1.0 if trials < 3 else 0.6 if trials < 10 else 0.35
                lift = L_EXPLORE_LIFT * starve
            out[c] = v + lift
        return out

    def update(
        self,
        chosen: str,
        improved: Optional[bool],
        context: str = "",
        magnitude: float = 1.0,
    ) -> None:
        """Credit chosen candidate from scored outcome.

        improved True  → hit (raise L_chosen only; no rival wipeout)
        improved False → miss (lower L_chosen)
        improved None  → neutral (drift toward 0, not a hit)
        """
        c = str(chosen).upper()
        if c not in self.log_odds:
            self.log_odds[c] = L_START
        if c not in self.trial_count:
            self.trial_count[c] = 0
        self.trial_count[c] = int(self.trial_count[c]) + 1
        mag = max(0.25, min(2.0, float(magnitude or 1.0)))

        if improved is True:
            delta = L_HIT * mag
            self.log_odds[c] = self._clamp(self.log_odds[c] + delta)
            if L_RIVAL_PENALTY > 0:
                for o in self.candidates:
                    if o != c:
                        self.log_odds[o] = self._clamp(
                            self.log_odds[o] - L_RIVAL_PENALTY * delta
                        )
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
            # Neutral: pull slightly toward 0 (don't treat as success)
            v = float(self.log_odds[c])
            self.log_odds[c] = self._clamp(v - L_NEUTRAL * (1.0 if v > 0 else -1.0) * mag)

        self.last_report = {
            "chosen": c,
            "improved": improved,
            "context": context or None,
            "L": dict(self.log_odds),
            "trials": dict(self.trial_count),
            "magnitude": mag,
        }

    def evidence_led(self, chosen: str, context: str = "") -> bool:
        return abs(self.get_L(chosen, context=context)) >= THETA_EVIDENCE

    def report(self) -> dict:
        L = {c: round(float(self.log_odds.get(c, 0.0)), 3) for c in self.candidates}
        top = sorted(L.items(), key=lambda kv: -abs(kv[1]))
        trials = {c: int(self.trial_count.get(c, 0)) for c in self.candidates}
        ctx_top = sorted(
            ((k, round(v, 3)) for k, v in self.ctx_log_odds.items()),
            key=lambda kv: -abs(kv[1]),
        )[:8]
        return {
            "L": L,
            "top": top[:5],
            "trials": trials,
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
    """Score act outcome. Neutral if within noise — NOT a hit.

    Returns (improved, magnitude).
    """
    setpoints = setpoints or {}
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
            if k in ("pain", "muscle_tension", "sweat_skin"):
                sp = 0.25
            elif k in ("energy", "pleasure"):
                sp = 0.55
            else:
                sp = 0.45
        err_before += abs(b - float(sp))
        err_after += abs(a - float(sp))
        n += 1
    err_before += max(0.0, float(pain_before) - 0.3) * 1.5
    err_after += max(0.0, float(pain_after) - 0.3) * 1.5

    if n == 0 and pain_before == 0 and pain_after == 0:
        return None, 0.0
    delta = err_before - err_after  # positive = improved
    # Wider neutral band so body_within_expect ≠ hit
    if delta > 0.04:
        return True, min(2.0, delta / 0.06)
    if delta < -0.04:
        return False, min(2.0, abs(delta) / 0.06)
    return None, 0.0
