"""
Long-term interest themes — orientation across sleep, not momentary focus.

Promoted from recurrence of:
  - sticky focus / high residual
  - narrative element weight
  - parental marks
  - schema↔felt binds

Biases short-term curiosity; does not invent hardcoded goals.
Opacity: themes are node/schema ids + weights, not slogan goals.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
LTI_PATH = os.path.join(_DATA_DIR, "long_term_interest.json")


class LongTermInterest:
    THEME_CAP = 40
    PROMOTE_FLOOR = 2.5       # narrative weight or residual total to seed
    DECAY = 0.97              # per consolidation
    BOOST_FROM_NARRATIVE = 0.35
    BOOST_FROM_FOCUS = 0.25
    BOOST_FROM_PARENTAL = 0.4
    BOOST_FROM_FELT_BIND = 0.2
    WEIGHT_CAP = 12.0
    CURIOSITY_BIAS = 3.0      # multiplier when theme in self-study pool

    def __init__(self):
        self.themes: Dict[str, float] = {}  # node_id -> weight
        self.load()

    def promote(
        self,
        focus_id: Optional[str] = None,
        residual_totals: Optional[Dict[str, float]] = None,
        narrative_elements: Optional[dict] = None,
        parental_nodes: Optional[List[str]] = None,
        felt_bound_schemas: Optional[List[str]] = None,
    ) -> dict:
        """Consolidation-time: short-term heat → long-term theme weight."""
        residual_totals = residual_totals or {}
        promoted = 0
        reinforced = 0

        def bump(nid: str, amt: float):
            nonlocal promoted, reinforced
            if not nid or nid in ("SELF", "OTHER"):
                return
            prev = self.themes.get(nid, 0.0)
            self.themes[nid] = min(self.WEIGHT_CAP, prev + amt)
            if prev <= 0:
                promoted += 1
            else:
                reinforced += 1

        if focus_id and residual_totals.get(focus_id, 0) >= 1.0:
            bump(focus_id, self.BOOST_FROM_FOCUS * min(3.0, residual_totals.get(focus_id, 1.0) / 5.0))

        for nid, tot in residual_totals.items():
            if tot >= self.PROMOTE_FLOOR:
                bump(nid, self.BOOST_FROM_FOCUS * 0.5)

        if narrative_elements:
            for el in narrative_elements.values():
                w = float(el.get("weight") or 0)
                if w < self.PROMOTE_FLOOR:
                    continue
                for n in el.get("linked_nodes") or []:
                    bump(n, self.BOOST_FROM_NARRATIVE * min(1.0, w / 5.0))

        for n in parental_nodes or []:
            bump(n, self.BOOST_FROM_PARENTAL)

        for n in felt_bound_schemas or []:
            bump(n, self.BOOST_FROM_FELT_BIND)

        # decay all
        dead = []
        for k in list(self.themes.keys()):
            self.themes[k] *= self.DECAY
            if self.themes[k] < 0.15:
                dead.append(k)
        for k in dead:
            del self.themes[k]

        # cap count: keep strongest
        if len(self.themes) > self.THEME_CAP:
            ranked = sorted(self.themes.items(), key=lambda t: -t[1])[: self.THEME_CAP]
            self.themes = dict(ranked)

        return {
            "themes": len(self.themes),
            "promoted": promoted,
            "reinforced": reinforced,
            "pruned": len(dead),
        }

    def theme_ids(self) -> Set[str]:
        return set(self.themes.keys())

    def weight(self, node_id: str) -> float:
        return float(self.themes.get(node_id, 0.0))

    def curiosity_multiplier(self, node_id: str) -> float:
        w = self.weight(node_id)
        if w <= 0:
            return 1.0
        return 1.0 + self.CURIOSITY_BIAS * min(1.0, w / self.WEIGHT_CAP)

    def top_themes(self, n: int = 12) -> List[dict]:
        ranked = sorted(self.themes.items(), key=lambda t: -t[1])[:n]
        return [{"id": k, "weight": round(v, 3)} for k, v in ranked]

    def report(self) -> dict:
        return {
            "theme_count": len(self.themes),
            "top": self.top_themes(15),
        }

    def save(self, path: str = None) -> None:
        path = path or LTI_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({"themes": self.themes}, f, indent=2)
        except OSError as e:
            logger.warning("LongTermInterest.save failed: %s", e)

    def load(self, path: str = None) -> None:
        path = path or LTI_PATH
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.themes = {k: float(v) for k, v in (data.get("themes") or {}).items()}
        except Exception as e:
            logger.warning("LongTermInterest.load failed: %s", e)
