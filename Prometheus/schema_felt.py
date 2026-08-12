"""
schema_felt.py -- Implicit bind between schemas and felt anchors.

No hard-coded emotion map. Binding is earned by co-occurrence:
  schema active (focus or WM) + current felt anchor occupied
  → count++
  after threshold, schema.primary_felt_anchor is set and a light
  reverse index is kept on the anchor.

Coordinates stay internal; anchors stay addressable.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Relation label stored on schema node attributes (and optional graph edge)
GROUNDS_IN = "grounds-in"
DEFAULT_THRESHOLD = 3


class SchemaFeltBinder:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        # schema_id -> {anchor_id: count}
        self.cooccur: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # anchor_id -> set of schema_ids that met threshold
        self.anchor_schemas: Dict[str, Set[str]] = defaultdict(set)

    def note(self, schema_ids, anchor_id: Optional[str]) -> None:
        if not anchor_id or not schema_ids:
            return
        for sid in schema_ids:
            if not sid:
                continue
            self.cooccur[sid][anchor_id] += 1

    def promote(self, graph) -> Dict[str, int]:
        """Write primary_felt_anchor onto schemas that crossed threshold.
        Also stamps a soft grounds-in attribute (no hormone names).
        Returns counts for logging."""
        promoted = 0
        updated = 0
        for sid, anchors in list(self.cooccur.items()):
            if sid not in graph:
                continue
            best_a, best_c = None, 0
            for a, c in anchors.items():
                if c > best_c:
                    best_a, best_c = a, c
            if best_a is None or best_c < self.threshold:
                continue
            data = graph.nodes[sid]
            prev = data.get("primary_felt_anchor")
            data["felt_cooccur"] = dict(anchors)
            data["primary_felt_anchor"] = best_a
            data["felt_bind_count"] = int(best_c)
            data["grounds_in"] = best_a  # linkable felt place, not a chemical
            self.anchor_schemas[best_a].add(sid)
            updated += 1
            if prev != best_a:
                promoted += 1
        return {"bindings_updated": updated, "bindings_new_primary": promoted}

    def schemas_for_anchor(self, anchor_id: str):
        return sorted(self.anchor_schemas.get(anchor_id, set()))

    def report(self, top_n: int = 20) -> dict:
        rows = []
        for sid, anchors in self.cooccur.items():
            if not anchors:
                continue
            best_a = max(anchors.items(), key=lambda t: t[1])
            rows.append({
                "schema_id": sid,
                "top_anchor": best_a[0],
                "count": best_a[1],
                "ready": best_a[1] >= self.threshold,
            })
        rows.sort(key=lambda r: -r["count"])
        return {
            "threshold": self.threshold,
            "tracked_schemas": len(self.cooccur),
            "promoted_anchors": len(self.anchor_schemas),
            "top": rows[:top_n],
        }

    def save_state(self, path: str) -> None:
        import json
        import os
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "threshold": self.threshold,
                "cooccur": {sid: dict(anchors) for sid, anchors in self.cooccur.items()},
                "anchor_schemas": {a: sorted(list(s)) for a, s in self.anchor_schemas.items()},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("SchemaFeltBinder.save_state failed: %s", e)

    def load_state(self, path: str) -> None:
        import json
        import os
        from collections import defaultdict
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.threshold = int(data.get("threshold") or self.threshold)
            self.cooccur = defaultdict(lambda: defaultdict(int))
            for sid, anchors in (data.get("cooccur") or {}).items():
                for aid, c in anchors.items():
                    self.cooccur[sid][aid] = int(c)
            self.anchor_schemas = defaultdict(set)
            for aid, sids in (data.get("anchor_schemas") or {}).items():
                self.anchor_schemas[aid] = set(sids)
        except Exception as e:
            logger.warning("SchemaFeltBinder.load_state failed: %s", e)
