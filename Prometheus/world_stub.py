"""
world_stub.py — Minimal internal world state for action→consequence (Package A).

Not physics, not tools, not files. A few [0,1] slots the agent can change
by operators and re-sense next pulse. Causal co-occurrence (Package B) will
observe focus + act → slot/body change windows later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

WORLD_SLOTS = (
    "object_near",   # something salient nearby / available to engage
    "obstacle",      # friction / block
    "goal_cue",      # cue toward active goal target
)

# Operator → world deltas (small, capped elsewhere)
OP_WORLD_DELTAS: Dict[str, Dict[str, float]] = {
    "SETTLE":  {"obstacle": -0.04, "object_near": -0.01},
    "RELEASE": {"obstacle": -0.05, "object_near": 0.02},
    "EXPAND":  {"object_near": 0.06, "goal_cue": 0.03, "obstacle": 0.01},
    "HOLD":    {},  # freeze: no change (handled by skip)
    "RETURN":  {"goal_cue": 0.05, "object_near": 0.02, "obstacle": -0.02},
}

MAX_DELTA = 0.08


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class WorldStub:
    """Tiny internal world the agent can act on and re-observe."""

    slots: Dict[str, float] = field(default_factory=dict)
    last_op: str = ""
    last_deltas: Dict[str, float] = field(default_factory=dict)
    pulse_applied: int = 0

    def __post_init__(self):
        if not self.slots:
            self.slots = {
                "object_near": 0.35,
                "obstacle": 0.30,
                "goal_cue": 0.25,
            }

    def observe(self) -> Dict[str, float]:
        return {k: float(self.slots.get(k, 0.0)) for k in WORLD_SLOTS}

    def apply_operator(self, op: str, pulse: int = 0, scale: float = 1.0) -> Dict[str, float]:
        """Apply operator world nudge. HOLD freezes (no change)."""
        op_u = (op or "HOLD").upper()
        self.last_op = op_u
        self.pulse_applied = int(pulse or 0)
        if op_u == "HOLD":
            self.last_deltas = {}
            return self.observe()
        raw = dict(OP_WORLD_DELTAS.get(op_u, {}))
        applied: Dict[str, float] = {}
        for slot, dv in raw.items():
            if slot not in self.slots:
                self.slots[slot] = 0.3
            d = max(-MAX_DELTA, min(MAX_DELTA, float(dv) * float(scale)))
            self.slots[slot] = _clamp(self.slots[slot] + d)
            applied[slot] = d
        self.last_deltas = applied
        return self.observe()

    def report(self) -> dict:
        return {
            "slots": self.observe(),
            "last_op": self.last_op,
            "last_deltas": dict(self.last_deltas),
            "pulse_applied": self.pulse_applied,
        }
