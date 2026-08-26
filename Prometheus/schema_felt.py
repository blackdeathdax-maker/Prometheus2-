"""
schema_felt.py — Schema ↔ felt-place binding via co-occurrence.

While a schema is active (focus / goal / WM) and a felt place is current,
counts accumulate. Past threshold, promote writes graph edges and exposes
bound_schemas in the UI.

No emotion taxonomy. Any schema can bind to any place.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Relation used on the knowledge graph (not is-a)
EDGE_FELT_BIND = "felt-bind"
EDGE_ASSOCIATED = "associated-with"


class SchemaFeltBinder:
    """Co-occurrence counter + promotion to graph binds."""

    def __init__(self, threshold: int = 3):
        self.threshold = int(threshold)
        # schema_id -> {anchor_id: count}
        self.cooccur: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # schema_id -> set of promoted anchor ids
        self.binds: Dict[str, Set[str]] = defaultdict(set)
        # reverse: anchor_id -> [schema_ids]
        self._anchor_index: Dict[str, Set[str]] = defaultdict(set)

    def note(self, schema_ids, anchor_id: Optional[str]) -> None:
        if not anchor_id or not schema_ids:
            return
        for sid in schema_ids:
            if not sid:
                continue
            sid = str(sid)
            self.cooccur[sid][anchor_id] = int(self.cooccur[sid].get(anchor_id, 0)) + 1
            # Soft auto-promote when threshold hit (also runs at consolidation)
            if self.cooccur[sid][anchor_id] >= self.threshold:
                self.binds[sid].add(anchor_id)
                self._anchor_index[anchor_id].add(sid)

    def schemas_for_anchor(self, anchor_id: str) -> List[str]:
        if not anchor_id:
            return []
        # Promoted binds first, then near-threshold cooccur
        out = list(self._anchor_index.get(anchor_id) or [])
        if out:
            return sorted(out)
        near = []
        for sid, amap in self.cooccur.items():
            c = int(amap.get(anchor_id, 0))
            if c >= max(1, self.threshold - 1):
                near.append(sid)
        return sorted(near)

    def promote(self, graph) -> dict:
        """Write felt-bind edges for pairs at/above threshold. Returns summary."""
        promoted = 0
        reinforced = 0
        if graph is None:
            return {"promoted": 0, "reinforced": 0, "binds": 0}

        for sid, amap in list(self.cooccur.items()):
            for anchor_id, count in list(amap.items()):
                if int(count) < self.threshold:
                    continue
                already = anchor_id in self.binds.get(sid, ())
                self.binds[sid].add(anchor_id)
                self._anchor_index[anchor_id].add(sid)

                # Ensure nodes exist lightly
                try:
                    if sid not in graph:
                        continue
                    if anchor_id not in graph:
                        graph.add_node(
                            anchor_id,
                            name=anchor_id,
                            node_type="felt_place",
                            is_felt_place=True,
                        )
                    # Edge schema --felt-bind--> place
                    data = graph.get_edge_data(sid, anchor_id) or {}
                    # networkx may return dict of keys
                    exists = False
                    try:
                        if graph.has_edge(sid, anchor_id):
                            exists = True
                    except Exception:
                        pass
                    if not exists:
                        graph.add_edge(
                            sid,
                            anchor_id,
                            relation_type=EDGE_FELT_BIND,
                            weight=min(1.0, 0.35 + 0.05 * int(count)),
                            source="schema_felt",
                        )
                        promoted += 1
                    else:
                        # bump weight if possible
                        try:
                            ed = graph[sid][anchor_id]
                            if isinstance(ed, dict) and "weight" in ed:
                                ed["weight"] = min(1.0, float(ed.get("weight", 0.4)) + 0.02)
                            elif isinstance(ed, dict):
                                # multi-edge
                                for k, attr in ed.items():
                                    if isinstance(attr, dict):
                                        attr["weight"] = min(
                                            1.0, float(attr.get("weight", 0.4)) + 0.02
                                        )
                                        attr["relation_type"] = attr.get(
                                            "relation_type", EDGE_FELT_BIND
                                        )
                                        break
                        except Exception:
                            pass
                        reinforced += 1
                except Exception as e:
                    logger.warning("schema_felt promote edge failed: %s", e)

        return {
            "promoted": promoted,
            "reinforced": reinforced,
            "binds": sum(len(v) for v in self.binds.values()),
            "pairs_over_threshold": sum(
                1
                for amap in self.cooccur.values()
                for c in amap.values()
                if int(c) >= self.threshold
            ),
        }

    def report(self) -> dict:
        top_pairs = []
        for sid, amap in self.cooccur.items():
            for aid, c in amap.items():
                top_pairs.append(
                    {
                        "schema": sid,
                        "anchor": aid,
                        "count": int(c),
                        "bound": aid in self.binds.get(sid, ()),
                    }
                )
        top_pairs.sort(key=lambda r: -r["count"])
        return {
            "threshold": self.threshold,
            "bind_count": sum(len(v) for v in self.binds.values()),
            "pair_count": sum(len(v) for v in self.cooccur.values()),
            "top_pairs": top_pairs[:30],
            "binds": {s: sorted(list(a)) for s, a in self.binds.items()},
        }

    def save_state(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "threshold": self.threshold,
                "cooccur": {s: dict(m) for s, m in self.cooccur.items()},
                "binds": {s: sorted(list(a)) for s, a in self.binds.items()},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("SchemaFeltBinder.save_state failed: %s", e)

    def load_state(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.threshold = int(data.get("threshold") or self.threshold)
            self.cooccur = defaultdict(lambda: defaultdict(int))
            for s, m in (data.get("cooccur") or {}).items():
                for a, c in m.items():
                    self.cooccur[s][a] = int(c)
            self.binds = defaultdict(set)
            self._anchor_index = defaultdict(set)
            for s, arr in (data.get("binds") or {}).items():
                for a in arr:
                    self.binds[s].add(a)
                    self._anchor_index[a].add(s)
            # rebuild near-threshold into index for UI
            for s, m in self.cooccur.items():
                for a, c in m.items():
                    if int(c) >= self.threshold:
                        self.binds[s].add(a)
                        self._anchor_index[a].add(s)
        except Exception as e:
            logger.warning("SchemaFeltBinder.load_state failed: %s", e)
