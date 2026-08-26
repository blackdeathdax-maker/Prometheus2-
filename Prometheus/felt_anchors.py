"""
felt_anchors.py — Stable felt *places* over PAD basins.

Each place is a dwell-backed identity: basin key + running body_mean.
Naming is optional and late; binding to schemas lives in schema_felt.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BasinKey = Tuple[float, float, float]


def _key_id(key: BasinKey) -> str:
    a, v, d = key
    raw = f"{float(a):.1f}_{float(v):.1f}_{float(d):.1f}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"felt_{h}"


def _ema(old: float, new: float, alpha: float = 0.12) -> float:
    return (1.0 - alpha) * float(old) + alpha * float(new)


@dataclass
class FeltPlace:
    anchor_id: str
    basin_key: BasinKey
    dwell: int = 0
    revisits: int = 0
    named: bool = False
    name: Optional[str] = None
    body_mean: Dict[str, float] = field(default_factory=dict)

    def as_report(self) -> dict:
        return {
            "id": self.anchor_id,
            "basin_key": [float(x) for x in self.basin_key],
            "dwell": int(self.dwell),
            "revisits": int(self.revisits),
            "named": bool(self.named),
            "name": self.name,
            "body_mean": {k: round(float(v), 3) for k, v in self.body_mean.items()},
        }


class FeltAnchorStore:
    """Observe live PAD bins → stable places with body fingerprints."""

    BODY_KEYS = (
        "heart_rate", "breath", "muscle_tension", "sweat_skin",
        "gut", "energy", "warmth", "respiration_rate",
    )

    def __init__(self):
        self.places: Dict[str, FeltPlace] = {}
        self._current_id: Optional[str] = None
        self._last_key: Optional[BasinKey] = None

    def observe(self, basin_key, raw_body: Optional[Dict[str, float]] = None) -> Optional[FeltPlace]:
        if basin_key is None:
            return self.current()
        try:
            key = (
                float(basin_key[0]),
                float(basin_key[1]),
                float(basin_key[2]),
            )
        except Exception:
            return self.current()

        # Normalize to 1 decimal to match GRID_RESOLUTION
        key = (round(key[0], 1), round(key[1], 1), round(key[2], 1))
        aid = _key_id(key)
        place = self.places.get(aid)
        if place is None:
            place = FeltPlace(anchor_id=aid, basin_key=key, dwell=0, revisits=0)
            self.places[aid] = place
        else:
            if self._current_id != aid:
                place.revisits += 1

        place.dwell += 1
        body = raw_body or {}
        for k in self.BODY_KEYS:
            if k not in body:
                continue
            try:
                val = float(body[k])
            except (TypeError, ValueError):
                continue
            if k in place.body_mean:
                place.body_mean[k] = _ema(place.body_mean[k], val)
            else:
                place.body_mean[k] = val

        self._current_id = aid
        self._last_key = key
        return place

    def current(self) -> Optional[FeltPlace]:
        if not self._current_id:
            return None
        return self.places.get(self._current_id)

    def try_name_current(self, text: str) -> bool:
        """Optional short user label for the live place (not required for binding)."""
        cur = self.current()
        if cur is None or not text:
            return False
        label = str(text).strip()[:48]
        if not label:
            return False
        cur.named = True
        cur.name = label
        return True

    def report(self) -> dict:
        anchors = sorted(
            self.places.values(),
            key=lambda p: (-p.dwell, -p.revisits, p.anchor_id),
        )
        cur = self.current()
        return {
            "current_id": cur.anchor_id if cur else None,
            "current_name": cur.name if cur else None,
            "anchor_count": len(self.places),
            "anchors": [p.as_report() for p in anchors[:40]],
        }

    def save_state(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "current_id": self._current_id,
                "places": {
                    aid: {
                        "anchor_id": p.anchor_id,
                        "basin_key": list(p.basin_key),
                        "dwell": p.dwell,
                        "revisits": p.revisits,
                        "named": p.named,
                        "name": p.name,
                        "body_mean": p.body_mean,
                    }
                    for aid, p in self.places.items()
                },
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("FeltAnchorStore.save_state failed: %s", e)

    def load_state(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.places = {}
            for aid, row in (data.get("places") or {}).items():
                key = tuple(row.get("basin_key") or (0.5, 0.0, 0.5))
                key = (float(key[0]), float(key[1]), float(key[2]))
                self.places[aid] = FeltPlace(
                    anchor_id=row.get("anchor_id") or aid,
                    basin_key=key,
                    dwell=int(row.get("dwell") or 0),
                    revisits=int(row.get("revisits") or 0),
                    named=bool(row.get("named")),
                    name=row.get("name"),
                    body_mean=dict(row.get("body_mean") or {}),
                )
            self._current_id = data.get("current_id")
        except Exception as e:
            logger.warning("FeltAnchorStore.load_state failed: %s", e)
