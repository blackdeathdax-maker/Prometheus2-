"""
Fast neuromodulator bus (cognitive necessities only).

Timescale stack:
  Fast  — this module (salience, encode, alert, settle)
  Medium — hormonal.py cortisol/adrenaline class
  Slow  — hormonal slow layer + epoch baselines

Opacity: cognition never reads these names. Effects appear as
focus/encode gates and small body-channel gusts only.

Extended: conflict / ambivalence from synthesizer.get_conflict_score()
raises alert + salience and lowers settle (harder to stay locked on one
focus while two basins compete).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
MODULATOR_STATE_PATH = os.path.join(_DATA_DIR, "modulators_state.json")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


class FastModulators:
    """
    Four necessity gates only:
      salience — this matters / residual growth
      encode   — bind co-occurrence / edges stick
      alert    — shift / interrupt / urgency
      settle   — stay / raise switch cost
    """

    DECAY = 0.12
    BASELINE = {
        "salience": 0.35,
        "encode": 0.40,
        "alert": 0.30,
        "settle": 0.45,
    }
    # Cap total body gust so map stays readable
    BODY_GUST_CAP = 0.18

    # Conflict (ambivalence) influence — tuned as placeholders
    CONFLICT_ALERT_GAIN = 0.35
    CONFLICT_SALIENCE_GAIN = 0.25
    CONFLICT_SETTLE_PENALTY = 0.40

    def __init__(self):
        self.levels: Dict[str, float] = dict(self.BASELINE)
        self.last_body_delta: Dict[str, float] = {}
        self._last_conflict = 0.0
        self.load_state()

    def decay_toward_baseline(self) -> None:
        for k, base in self.BASELINE.items():
            cur = self.levels.get(k, base)
            self.levels[k] = cur * (1.0 - self.DECAY) + base * self.DECAY

    def pulse(self, event: str, amount: float = 0.08) -> None:
        """Named cognitive events → modulator nudges (still no emotion taxonomy)."""
        amount = _clamp(amount, 0.0, 0.35)
        table = {
            "prediction_error": {"salience": 1.0, "alert": 0.7, "encode": 0.4},
            "approval": {"salience": 0.9, "settle": 0.5, "encode": 0.5},
            "disapproval": {"alert": 0.8, "salience": 0.6, "settle": -0.3},
            "user_input": {"salience": 0.5, "encode": 0.4, "alert": 0.3},
            "focus_stagnant": {"alert": 0.6, "settle": -0.4, "salience": -0.2},
            "sleep_enter": {"settle": 0.8, "salience": -0.5, "alert": -0.4, "encode": 0.3},
            "sleep_exit": {"alert": 0.3, "salience": 0.2},
            "novelty": {"alert": 0.7, "salience": 0.5},
            "self_study_hit": {"encode": 0.35, "salience": 0.25},
            "conflict": {"alert": 0.9, "salience": 0.6, "settle": -0.7},  # mixed affect
        }
        mix = table.get(event)
        if not mix:
            return
        for k, w in mix.items():
            if k not in self.levels:
                continue
            self.levels[k] = _clamp(self.levels[k] + w * amount)

    def apply_conflict(self, conflict_score: float) -> None:
        """Continuous ambivalence signal from synthesizer.get_conflict_score().
        Raises alert + salience, lowers settle. Safe boundary input only.
        """
        c = _clamp(conflict_score)
        self._last_conflict = c
        if c < 0.08:
            return
        self.levels["alert"] = _clamp(
            self.levels["alert"] + self.CONFLICT_ALERT_GAIN * c
        )
        self.levels["salience"] = _clamp(
            self.levels["salience"] + self.CONFLICT_SALIENCE_GAIN * c
        )
        self.levels["settle"] = _clamp(
            self.levels["settle"] - self.CONFLICT_SETTLE_PENALTY * c
        )

    def apply_medium_bias(self, hormones: Optional[Dict[str, float]] = None) -> None:
        """Medium climate slightly biases fast baselines (not the reverse)."""
        if not hormones:
            return
        cor = float(hormones.get("cortisol", 0.5))
        adr = float(hormones.get("adrenaline", 0.5))
        ser = float(hormones.get("serotonin", 0.5))
        # high stress climate → alert up, settle down
        self.levels["alert"] = _clamp(self.levels["alert"] + 0.04 * (cor + adr - 1.0))
        self.levels["settle"] = _clamp(self.levels["settle"] + 0.04 * (ser - cor))

    def residual_gain(self) -> float:
        return 0.7 + 1.1 * self.levels.get("salience", 0.35)

    def encode_gain(self) -> float:
        return 0.55 + 1.0 * self.levels.get("encode", 0.4)

    def switch_cost_mult(self) -> float:
        # high settle → harder to leave focus; high alert → easier
        settle = self.levels.get("settle", 0.45)
        alert = self.levels.get("alert", 0.3)
        return _clamp(0.6 + 0.9 * settle - 0.5 * alert, 0.35, 1.8)

    def alert_level(self) -> float:
        return self.levels.get("alert", 0.3)

    def body_delta(self) -> Dict[str, float]:
        """Phenomenological gusts only — added on top of hormone→body map."""
        s = self.levels.get("salience", 0.35)
        e = self.levels.get("encode", 0.4)
        a = self.levels.get("alert", 0.3)
        t = self.levels.get("settle", 0.45)
        # Conflict adds a little gut/tension texture
        c = self._last_conflict
        raw = {
            "heart_rate": 0.55 * a + 0.20 * s - 0.15 * t + 0.15 * c,
            "breath": 0.50 * a + 0.15 * s - 0.20 * t + 0.10 * c,
            "muscle_tension": 0.45 * a + 0.25 * s - 0.25 * t + 0.20 * c,
            "sweat_skin": 0.40 * a + 0.20 * s + 0.10 * c,
            "gut": 0.35 * a - 0.15 * t + 0.10 * s + 0.25 * c,
            "energy": 0.35 * s + 0.25 * e - 0.15 * a - 0.10 * c,
            "warmth": 0.30 * t + 0.20 * e - 0.15 * a - 0.10 * c,
        }
        # center around 0 and cap magnitude
        out = {}
        for k, v in raw.items():
            # shift so baseline modulators ≈ 0 delta
            v = v - 0.25
            out[k] = _clamp(v, -self.BODY_GUST_CAP, self.BODY_GUST_CAP)
        self.last_body_delta = dict(out)
        return out

    def report(self) -> dict:
        return {
            "levels": {k: round(v, 3) for k, v in self.levels.items()},
            "residual_gain": round(self.residual_gain(), 3),
            "encode_gain": round(self.encode_gain(), 3),
            "switch_cost_mult": round(self.switch_cost_mult(), 3),
            "last_conflict": round(self._last_conflict, 3),
            "last_body_delta": {k: round(v, 3) for k, v in self.last_body_delta.items()},
        }

    def save_state(self, path: str = None) -> None:
        path = path or MODULATOR_STATE_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({"levels": self.levels}, f, indent=2)
        except OSError as e:
            logger.warning("FastModulators.save_state failed: %s", e)

    def load_state(self, path: str = None) -> None:
        path = path or MODULATOR_STATE_PATH
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for k, v in (data.get("levels") or {}).items():
                if k in self.BASELINE:
                    self.levels[k] = _clamp(v)
        except Exception as e:
            logger.warning("FastModulators.load_state failed: %s", e)
