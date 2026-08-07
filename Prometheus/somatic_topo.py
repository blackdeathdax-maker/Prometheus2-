"""
somatic_topo.py -- Somatic topographic map (§2.1a landscape).

Not the raw hormone gauges (Debug JSON). This module records:
  - basin nodes in PAD space (from synthesizer keys)
  - transition edges when felt state moves between basins

Consolidation soft-decays rare transitions. Report is read-only for UI.
Optional later: focus/schema neighborhood bias from adjacency.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BasinKey = Tuple[float, float, float]


def _key_id(key: BasinKey) -> str:
    return f"basin_{key[0]}_{key[1]}_{key[2]}"


class SomaticTopo:
    """
    Transition graph over PAD basin keys.

    Call record(key) every pulse after synthesizer.update_from_core.
    Call consolidate() from Consolidation.
    """

    TRANSITION_DECAY = 0.92          # per consolidation
    TRANSITION_FLOOR = 0.15          # drop below this
    MIN_TRANSITION_TO_REPORT = 0.5

    def __init__(self):
        self.prev_key: Optional[BasinKey] = None
        # dwell counts this session (mirrors grid; useful if grid not passed)
        self.dwell: Dict[BasinKey, float] = defaultdict(float)
        # directed transition weight prev -> cur
        self.transitions: Dict[Tuple[BasinKey, BasinKey], float] = defaultdict(float)
        self.total_records = 0
        self.total_moves = 0

    def record(self, key: BasinKey) -> None:
        """Call once per pulse with synthesizer.get_current_basin_key()."""
        if key is None:
            return
        # normalize to tuple of floats
        key = (float(key[0]), float(key[1]), float(key[2]))
        self.dwell[key] += 1.0
        self.total_records += 1
        if self.prev_key is not None and self.prev_key != key:
            self.transitions[(self.prev_key, key)] += 1.0
            self.total_moves += 1
        self.prev_key = key

    def consolidate(self) -> Dict[str, int]:
        """Decay weak transitions; drop floor. Returns summary counts."""
        dead = []
        for edge, w in self.transitions.items():
            nw = w * self.TRANSITION_DECAY
            if nw < self.TRANSITION_FLOOR:
                dead.append(edge)
            else:
                self.transitions[edge] = nw
        for edge in dead:
            del self.transitions[edge]
        return {
            "transitions_remaining": len(self.transitions),
            "transitions_dropped": len(dead),
            "basins_touched": len(self.dwell),
        }

    def neighbors(self, key: BasinKey, min_w: float = 0.5) -> List[Tuple[BasinKey, float]]:
        key = (float(key[0]), float(key[1]), float(key[2]))
        out = []
        for (a, b), w in self.transitions.items():
            if a == key and w >= min_w:
                out.append((b, w))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def report(self, top_n: int = 12) -> Dict:
        """UI-facing topographic summary (not raw hormones)."""
        basins = sorted(
            (
                {
                    "id": _key_id(k),
                    "arousal": k[0],
                    "valence": k[1],
                    "dominance": k[2],
                    "dwell": round(v, 2),
                }
                for k, v in self.dwell.items()
            ),
            key=lambda d: d["dwell"],
            reverse=True,
        )[:top_n]

        edges = sorted(
            (
                {
                    "from": _key_id(a),
                    "to": _key_id(b),
                    "weight": round(w, 2),
                }
                for (a, b), w in self.transitions.items()
                if w >= self.MIN_TRANSITION_TO_REPORT
            ),
            key=lambda d: d["weight"],
            reverse=True,
        )[:top_n]

        current = None
        if self.prev_key is not None:
            current = {
                "id": _key_id(self.prev_key),
                "arousal": self.prev_key[0],
                "valence": self.prev_key[1],
                "dominance": self.prev_key[2],
            }

        return {
            "current_basin": current,
            "basins_touched": len(self.dwell),
            "moves": self.total_moves,
            "records": self.total_records,
            "top_basins_by_dwell": basins,
            "top_transitions": edges,
        }
"""
Integration (Prometheus.py):

  from .somatic_topo import SomaticTopo
  # __init__:
  self.somatic_topo = SomaticTopo()

  # pulse(), after synthesizer.update_from_core(...):
  self.somatic_topo.record(self.synthesizer.get_current_basin_key())

  # _run_consolidation(), after consolidate_basins():
  topo_summary = self.somatic_topo.consolidate()
  print(f"Consolidation: somatic topo {topo_summary}")

  def get_somatic_topo_report(self) -> dict:
      return self.somatic_topo.report()

app.py Debug (below raw somatic JSON):

  st.subheader("Somatic topography (basin map)")
  st.caption("Basins and transitions — not raw hormone gauges.")
  if hasattr(prom, "get_somatic_topo_report"):
      st.json(prom.get_somatic_topo_report())
"""
