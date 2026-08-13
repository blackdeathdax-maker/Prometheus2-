"""
felt_anchors.py -- Linkable felt-state identities over PAD geometry.

Internal geometry stays (A, V, D) bins on the synthesizer.
This module gives each stable region a durable anchor id that schemas
can bind to — without pre-assigning emotion names like "anger".

Names are earned only when evidence appears (user word while in-state,
or optional later rules). Coordinates never become the public name.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BasinKey = Tuple[float, float, float]

# Phenomenological body only — never endocrine keys
MIN_BASIN_DISTANCE = 0.22  # PAD L2 — new anchors must be this far from existing
REFINE_DWELL = 3.0         # or revisit enough to refine in place

BODY_CHANNELS = frozenset({
    "heart_rate", "breath", "respiration_rate", "muscle_tension",
    "sweat_skin", "gut", "energy", "warmth",
})


def _key_tuple(key) -> BasinKey:
    return (float(key[0]), float(key[1]), float(key[2]))


def _anchor_id(key: BasinKey) -> str:
    """Stable id from coordinates — short, not a display name."""
    raw = f"{key[0]:.3f}|{key[1]:.3f}|{key[2]:.3f}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"felt_{digest}"


@dataclass
class FeltAnchor:
    anchor_id: str
    basin_key: BasinKey
    dwell: float = 0.0
    revisits: int = 0
    named: bool = False
    name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # Rolling body snapshot averages when occupied (for schema grounding later)
    body_mean: Dict[str, float] = field(default_factory=dict)
    body_samples: int = 0


class FeltAnchorStore:
    """
    Call observe() each pulse with current basin key + optional raw body dict.
    Call mark_stable() from Consolidation when synthesizer stabilizes a basin.
    Naming: try_name(word) only while that basin is current and word is lemma-like.
    """

    MIN_REVISITS_FOR_STABLE = 3
    NAME_MAX_LEN = 32

    def __init__(self):
        self.anchors: Dict[str, FeltAnchor] = {}
        self.by_basin: Dict[BasinKey, str] = {}
        self.current_id: Optional[str] = None

    def _pad_dist(self, a: BasinKey, b: BasinKey) -> float:
        return (
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        ) ** 0.5

    def _nearest_anchor(self, key: BasinKey):
        best_id, best_d = None, 1e9
        for aid, a in self.anchors.items():
            d = self._pad_dist(key, a.basin_key)
            if d < best_d:
                best_d, best_id = d, aid
        return best_id, best_d

    def observe(self, basin_key, raw_body: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Map continuous PAD to sparse anchors — refuse near-duplicates."""
        key = _key_tuple(basin_key)
        aid = self.by_basin.get(key)
        if aid is None:
            near_id, near_d = self._nearest_anchor(key)
            if near_id is not None and near_d < MIN_BASIN_DISTANCE:
                # Snap to existing place — basins stay further apart
                aid = near_id
                self.by_basin[key] = aid  # alias this grid cell to the anchor
            else:
                aid = _anchor_id(key)
                self.anchors[aid] = FeltAnchor(anchor_id=aid, basin_key=key)
                self.by_basin[key] = aid
        a = self.anchors[aid]
        if self.current_id != aid:
            a.revisits += 1
        a.dwell += 1.0
        a.last_seen_at = datetime.now().isoformat()
        self.current_id = aid
        if raw_body:
            n = a.body_samples
            for k, v in raw_body.items():
                if k not in BODY_CHANNELS:
                    continue
                # Prefer canonical breath key
                key = "breath" if k == "respiration_rate" else k
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                prev = a.body_mean.get(key, fv)
                a.body_mean[key] = (prev * n + fv) / (n + 1) if n else fv
            a.body_samples = n + 1
        return aid

    def mark_stable_keys(self, stabilized_keys) -> int:
        """Ensure anchors exist for synthesizer.stabilized_basins keys."""
        n = 0
        for key in stabilized_keys:
            key = _key_tuple(key if not isinstance(key, str) else key)
            # stabilized_basins may be dict key -> name
            if isinstance(key, str):
                continue
            aid = self.observe(key)
            if aid:
                n += 1
        return n

    def try_name_current(self, word: str) -> bool:
        """Earn a name for the current felt anchor from a short user/dictionary term."""
        if not self.current_id or self.current_id not in self.anchors:
            return False
        w = (word or "").strip()
        if not w or len(w) > self.NAME_MAX_LEN or len(w.split()) > 3:
            return False
        low = w.lower()
        if low.startswith(("i ", "i'", "the ", "a ")):
            return False
        a = self.anchors[self.current_id]
        if a.revisits < 1 and a.dwell < 3:
            return False
        a.name = w
        a.named = True
        return True

    def get(self, anchor_id: str) -> Optional[FeltAnchor]:
        return self.anchors.get(anchor_id)

    def current(self) -> Optional[FeltAnchor]:
        return self.anchors.get(self.current_id) if self.current_id else None

    def report(self, top_n: int = 12) -> Dict:
        rows = sorted(
            self.anchors.values(),
            key=lambda a: a.dwell,
            reverse=True,
        )[:top_n]
        return {
            "current_id": self.current_id,
            "current_name": (self.current().name if self.current() else None),
            "anchor_count": len(self.anchors),
            "anchors": [
                {
                    "id": a.anchor_id,
                    "basin_key": list(a.basin_key),
                    "dwell": round(a.dwell, 1),
                    "revisits": a.revisits,
                    "named": a.named,
                    "name": a.name,
                    "body_mean": {k: round(v, 3) for k, v in a.body_mean.items()},
                }
                for a in rows
            ],
        }


    def save_state(self, path: str) -> None:
        import json
        import os
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "current_id": self.current_id,
                "anchors": {
                    aid: {
                        "anchor_id": a.anchor_id,
                        "basin_key": list(a.basin_key),
                        "dwell": a.dwell,
                        "revisits": a.revisits,
                        "named": a.named,
                        "name": a.name,
                        "created_at": a.created_at,
                        "last_seen_at": a.last_seen_at,
                        "body_mean": dict(a.body_mean),
                        "body_samples": a.body_samples,
                    }
                    for aid, a in self.anchors.items()
                },
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            logger.warning("FeltAnchorStore.save_state failed: %s", e)

    def load_state(self, path: str) -> None:
        import json
        import os
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.anchors.clear()
            self.by_basin.clear()
            for aid, row in (data.get("anchors") or {}).items():
                key = tuple(float(x) for x in row["basin_key"])
                a = FeltAnchor(
                    anchor_id=row.get("anchor_id") or aid,
                    basin_key=key,
                    dwell=float(row.get("dwell") or 0),
                    revisits=int(row.get("revisits") or 0),
                    named=bool(row.get("named")),
                    name=row.get("name"),
                    created_at=row.get("created_at") or datetime.now().isoformat(),
                    last_seen_at=row.get("last_seen_at") or datetime.now().isoformat(),
                    body_mean=dict(row.get("body_mean") or {}),
                    body_samples=int(row.get("body_samples") or 0),
                )
                self.anchors[a.anchor_id] = a
                self.by_basin[key] = a.anchor_id
            self.current_id = data.get("current_id")
            if self.current_id and self.current_id not in self.anchors:
                self.current_id = None
        except Exception as e:
            logger.warning("FeltAnchorStore.load_state failed: %s", e)
