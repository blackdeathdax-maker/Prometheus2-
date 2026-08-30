"""
allostasis.py — Adaptive set-points + pain/pleasure policy (Allostasis & Affect).

Homeostasis kept variables near fixed baselines.
Allostasis moves set-points with context (intent, goals, epoch, fatigue)
and treats pain/pleasure as body-surface channels with hard caps and escapes.

Cognition never sees hormone names. Pain/pleasure are anatomy channels
(body:pain, body:pleasure) — parts only, never is-a, never self-study parents.

Decision 2 (set-point drivers) is tunable via SETPOINT_USE_EPOCH / FATIGUE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Keep in sync with edge_types.BODY_CHANNELS affect subset
AFFECT_CHANNELS = ("pain", "pleasure")

# Default baselines (homeostatic floor under allostasis)
BASELINE = {
    "heart_rate": 0.45,
    "breath": 0.45,
    "muscle_tension": 0.40,
    "sweat_skin": 0.35,
    "gut": 0.40,
    "energy": 0.50,
    "warmth": 0.48,
    "pain": 0.12,
    "pleasure": 0.20,
}

# Tunable: decision 2 — revisit to narrow drivers
SETPOINT_USE_EPOCH = True
SETPOINT_USE_FATIGUE = True

# Safety
CHANNEL_LO = 0.0
CHANNEL_HI = 1.0
MAX_DELTA_PER_PULSE = 0.08
PAIN_ESCAPE_LEVEL = 0.85
PAIN_ESCAPE_PULSES = 18
PAIN_SOFT_CLAMP = 0.72
SETTLE_STREAK_ESCAPE = 5


def _clamp(x: float, lo: float = CHANNEL_LO, hi: float = CHANNEL_HI) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class AllostasisState:
    setpoints: Dict[str, float] = field(default_factory=dict)
    error: Dict[str, float] = field(default_factory=dict)  # observed - setpoint
    pain: float = 0.12
    pleasure: float = 0.20
    high_pain_streak: int = 0
    settle_under_pain_streak: int = 0
    last_escape: str = ""
    note: str = ""


class AllostasisModule:
    """Owns adaptive set-points, affect channels, caps, escapes."""

    def __init__(self):
        self.state = AllostasisState(
            setpoints=dict(BASELINE),
            pain=BASELINE["pain"],
            pleasure=BASELINE["pleasure"],
        )
        self._prev_body: Dict[str, float] = {}

    def compute_setpoints(
        self,
        intent: str = "HOLD",
        has_goals: bool = False,
        epoch: str = "",
        fatigue: float = 0.0,
    ) -> Dict[str, float]:
        sp = dict(BASELINE)
        intent = (intent or "HOLD").upper()
        fat = float(fatigue or 0.0)
        ep = str(epoch or "")

        if intent == "LEARN" and has_goals:
            sp["energy"] = _clamp(sp["energy"] + 0.08)
            sp["heart_rate"] = _clamp(sp["heart_rate"] + 0.05)
            sp["pleasure"] = _clamp(sp["pleasure"] + 0.06)  # expect some reward
            sp["pain"] = _clamp(sp["pain"] - 0.02)
        elif intent == "EXPLORE":
            sp["energy"] = _clamp(sp["energy"] + 0.06)
            sp["heart_rate"] = _clamp(sp["heart_rate"] + 0.04)
        elif intent == "REGULATE":
            sp["muscle_tension"] = _clamp(sp["muscle_tension"] - 0.10)
            sp["pain"] = _clamp(sp["pain"] - 0.08)  # want lower pain
            sp["heart_rate"] = _clamp(sp["heart_rate"] - 0.05)
            sp["breath"] = _clamp(sp["breath"] - 0.04)
            sp["pleasure"] = _clamp(sp["pleasure"] + 0.04)  # relief sought
        elif intent == "HOLD":
            pass

        # Decision 2 drivers (can disable later)
        if SETPOINT_USE_EPOCH and ep == "Childhood":
            sp["pain"] = _clamp(min(sp["pain"], 0.55))  # lower ceiling via setpoint bias
            sp["pleasure"] = _clamp(sp["pleasure"] + 0.03)
        if SETPOINT_USE_FATIGUE and fat > 0.55:
            sp["energy"] = _clamp(sp["energy"] - 0.08 * (fat - 0.55))
            sp["pain"] = _clamp(sp["pain"] + 0.04 * (fat - 0.55))

        self.state.setpoints = sp
        return sp

    def rate_limit(self, channel: str, new_val: float) -> float:
        prev = float(self._prev_body.get(channel, new_val))
        delta = _clamp(new_val - prev, -MAX_DELTA_PER_PULSE, MAX_DELTA_PER_PULSE)
        return _clamp(prev + delta)

    def update_affect(
        self,
        body: Dict[str, float],
        *,
        body_error_max: float = 0.0,
        barren_focus: bool = False,
        goal_growth: bool = False,
        goal_failed: bool = False,
        conflict: float = 0.0,
        user_reinforce: bool = False,
        last_op: str = "",
    ) -> AllostasisState:
        """Derive pain/pleasure, apply caps, track escapes. Mutates body dict for affect keys."""
        st = self.state
        # Synthetic drivers → desired raw affect
        pain_raw = float(body.get("pain", st.pain) or st.pain)
        pleas_raw = float(body.get("pleasure", st.pleasure) or st.pleasure)

        # Cost drivers
        if body_error_max > 0.12:
            pain_raw += min(0.15, (body_error_max - 0.12) * 0.45)
        if barren_focus:
            pain_raw += 0.04
        if goal_failed:
            pain_raw += 0.10
        if conflict > 0.25:
            pain_raw += min(0.12, conflict * 0.25)

        # Reward drivers
        if goal_growth:
            pleas_raw += 0.10
        if user_reinforce:
            pleas_raw += 0.08
        if last_op in ("SETTLE", "RELEASE") and body_error_max < 0.15:
            pleas_raw += 0.05
            pain_raw -= 0.04

        # Natural decay toward soft baseline
        pain_raw = pain_raw * 0.92 + BASELINE["pain"] * 0.08
        pleas_raw = pleas_raw * 0.88 + BASELINE["pleasure"] * 0.12

        pain = self.rate_limit("pain", _clamp(pain_raw))
        pleasure = self.rate_limit("pleasure", _clamp(pleas_raw))

        # Escapes
        st.last_escape = ""
        if pain >= PAIN_ESCAPE_LEVEL:
            st.high_pain_streak += 1
        else:
            st.high_pain_streak = 0

        if last_op in ("SETTLE", "RELEASE") and pain > 0.55:
            st.settle_under_pain_streak += 1
        else:
            st.settle_under_pain_streak = 0

        if st.high_pain_streak >= PAIN_ESCAPE_PULSES:
            pain = min(pain, PAIN_SOFT_CLAMP)
            st.last_escape = "time_clamp"
            st.high_pain_streak = 0
            st.note = "pain_time_escape"
        elif st.settle_under_pain_streak >= SETTLE_STREAK_ESCAPE:
            st.last_escape = "op_diversity"
            st.settle_under_pain_streak = 0
            st.note = "pain_settle_escape"
        else:
            st.note = ""

        st.pain = pain
        st.pleasure = pleasure
        body["pain"] = pain
        body["pleasure"] = pleasure
        self._prev_body["pain"] = pain
        self._prev_body["pleasure"] = pleasure

        # Allostatic error vs setpoints
        sp = st.setpoints or BASELINE
        err = {}
        for ch, obs in body.items():
            if ch in sp:
                err[ch] = float(obs) - float(sp[ch])
        st.error = err
        return st

    def force_regulate(self) -> bool:
        return self.state.pain >= 0.55 or (
            abs(self.state.error.get("pain", 0)) > 0.20
            and self.state.pain > float(self.state.setpoints.get("pain", 0.12))
        )

    def force_escape_expand_hold(self) -> bool:
        """True when op-diversity escape should bias away from pure SETTLE."""
        return self.state.last_escape == "op_diversity"

    def report(self) -> dict:
        st = self.state
        return {
            "pain": round(st.pain, 4),
            "pleasure": round(st.pleasure, 4),
            "setpoints": {k: round(v, 4) for k, v in (st.setpoints or {}).items()},
            "error": {k: round(v, 4) for k, v in (st.error or {}).items()},
            "high_pain_streak": st.high_pain_streak,
            "last_escape": st.last_escape,
            "note": st.note,
            "setpoint_drivers": {
                "epoch": SETPOINT_USE_EPOCH,
                "fatigue": SETPOINT_USE_FATIGUE,
            },
        }
