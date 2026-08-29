"""
active_thread.py — Short-horizon "what I'm doing now".

In-memory only (persistence deferred until live head).
Not narrative, not a goal — the shared spine for operators, goals, and stream.

Intent is *derived* each pulse from body_error, barren focus, goals, and bias:
  HOLD | EXPLORE | REGULATE | LEARN
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


INTENTS = ("HOLD", "EXPLORE", "REGULATE", "LEARN")

# Body channels that drive REGULATE intent (keep in sync with operators)
REGULATION_CHANNELS = frozenset({
    "heart_rate", "breath", "muscle_tension", "sweat_skin", "gut",
})


@dataclass
class ActiveThread:
    focus_id: Optional[str] = None
    goal_ids: List[str] = field(default_factory=list)
    intent: str = "HOLD"
    body_expect: Dict[str, float] = field(default_factory=dict)
    body_error: Dict[str, float] = field(default_factory=dict)  # signed per channel
    max_abs_body_error: float = 0.0
    last_ops: Deque[str] = field(default_factory=lambda: deque(maxlen=16))
    opened_pulse: int = 0
    age: int = 0
    barren_focus: bool = False
    note: str = ""

    def clear(self) -> None:
        self.focus_id = None
        self.goal_ids = []
        self.intent = "HOLD"
        self.body_expect = {}
        self.body_error = {}
        self.max_abs_body_error = 0.0
        self.barren_focus = False
        self.note = ""
        # keep last_ops for cadence continuity


class ActiveThreadModule:
    """Owns the single live ActiveThread; updated every pulse."""

    # Soft abandon: no focus and no goals for this many pulses → clear
    ABANDON_PULSES = 12
    BODY_ERROR_REGULATE = 0.18
    BODY_ERROR_SOFT = 0.12

    def __init__(self):
        self.thread = ActiveThread()
        self._idle_pulses = 0

    def update(
        self,
        pulse: int,
        focus_id: Optional[str],
        goal_ids: Optional[List[str]] = None,
        body_expect: Optional[Dict[str, float]] = None,
        body_error: Optional[Dict[str, float]] = None,
        last_op: Optional[str] = None,
        barren_focus: bool = False,
        bias: str = "",
        lookup_budget_ok: bool = True,
    ) -> ActiveThread:
        t = self.thread
        goals = list(goal_ids or [])

        if focus_id or goals:
            # Reset age clock on real focus change (soak fix)
            if focus_id and focus_id != t.focus_id:
                t.opened_pulse = pulse
                t.age = 0
            elif t.opened_pulse == 0:
                t.opened_pulse = pulse
            t.focus_id = focus_id
            t.goal_ids = goals
            t.age = max(0, pulse - int(t.opened_pulse or pulse))
            self._idle_pulses = 0
        else:
            self._idle_pulses += 1
            if self._idle_pulses >= self.ABANDON_PULSES:
                t.clear()
                t.opened_pulse = 0
                t.age = 0
                return t

        if body_expect is not None:
            t.body_expect = dict(body_expect)
        if body_error is not None:
            t.body_error = dict(body_error)
            t.max_abs_body_error = max(
                (abs(float(v)) for v in t.body_error.values()),
                default=0.0,
            )
        t.barren_focus = bool(barren_focus)
        if last_op:
            t.last_ops.append(str(last_op))

        t.intent = self._derive_intent(
            t, bias=bias, lookup_budget_ok=lookup_budget_ok
        )
        return t

    def _derive_intent(
        self,
        t: ActiveThread,
        bias: str = "",
        lookup_budget_ok: bool = True,
    ) -> str:
        # 1) Body regulation dominates when error is high on regulation channels
        reg_err = 0.0
        for ch, err in (t.body_error or {}).items():
            if ch in REGULATION_CHANNELS:
                reg_err = max(reg_err, abs(float(err)))
        if reg_err >= self.BODY_ERROR_REGULATE:
            t.note = "body_error_regulate"
            return "REGULATE"
        if t.max_abs_body_error >= self.BODY_ERROR_REGULATE:
            t.note = "body_error_global"
            return "REGULATE"

        # 2) Barren focus → HOLD (don't EXPAND again)
        if t.barren_focus:
            t.note = "barren_hold"
            return "HOLD"

        # 3) Open goals with room to grow → LEARN
        if t.goal_ids and lookup_budget_ok:
            t.note = "goal_learn"
            return "LEARN"

        # 4) Executive bias
        if bias in ("BIAS_EXPLORE", "BIAS_FORCE_EXPLORE"):
            t.note = "bias_explore"
            return "EXPLORE"
        if bias == "BIAS_STABILIZE":
            t.note = "bias_hold"
            return "HOLD"

        # 5) Default
        if t.goal_ids:
            t.note = "goal_default_learn"
            return "LEARN"
        t.note = "default_hold"
        return "HOLD"

    def report(self) -> dict:
        t = self.thread
        return {
            "focus_id": t.focus_id,
            "goal_ids": list(t.goal_ids),
            "intent": t.intent,
            "max_abs_body_error": round(float(t.max_abs_body_error), 4),
            "body_error": {k: round(float(v), 4) for k, v in (t.body_error or {}).items()},
            "body_expect": {k: round(float(v), 4) for k, v in (t.body_expect or {}).items()},
            "last_ops": list(t.last_ops),
            "opened_pulse": t.opened_pulse,
            "age": t.age,
            "barren_focus": t.barren_focus,
            "note": t.note,
        }
