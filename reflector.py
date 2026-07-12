import hashlib
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from .archivist import SELF_NODE, TIER_PROVISIONAL, TIER_WORKING


class OverrideSignal:
    def __init__(self, command: str, reason: str):
        self.command = command
        self.reason = reason


# §2.1b item 4: co-occurrence stabilization threshold for Schema Node
# formation. Same tuning-placeholder category as basin stabilization
# (§10 item 13) -- not yet numeric in the spec.
SCHEMA_STABILIZATION_THRESHOLD = 3

_RELATIONAL_TYPES = ("responsible-for", "violates", "temporal-contrast", "concerns-other")


class ReflectorModule:
    """
    Visible layer (§7 / §4A). Reads the finished state of the graph and
    chronos's history and produces insight *about* it -- metacognition,
    not cognition. Three responsibilities per §4A:
      1. Structural self-report (observe/evaluate, pre-existing).
      2. Regulatory self-awareness (regulatory_self_report) -- new.
      3. Complex-schema detection (detect_schemas) -- new, §2.1b.
    All Consolidation-gated except the structural spinning/stagnant check,
    which still runs every pulse to steer the live bias signal (unchanged
    behavior from before).
    """

    SPINNING_THROUGHPUT = 0.3
    SPINNING_VARIANCE = 0.15
    STAGNANT_THROUGHPUT = 0.2
    STAGNANT_VARIANCE = 0.05

    def __init__(self, chronos, archivist):
        self.chronos = chronos
        self.archivist = archivist
        self.pulse_count = 0
        self.last_schema_scan_pulse = 0
        # Instance attribute, not just the module-level constant -- lets
        # the Debug tab's sliders tune this live. Same "not yet
        # numerically tuned" placeholder as everywhere else (§10).
        self.SCHEMA_STABILIZATION_THRESHOLD = SCHEMA_STABILIZATION_THRESHOLD

    # ------------------------------------------------------------------
    # 1. Structural self-report (pre-existing, unchanged)
    # ------------------------------------------------------------------
    def observe(self):
        self.pulse_count += 1
        summary = self.chronos.get_state_summary()
        throughput = len(self.archivist.graph.edges()) / max(1, self.pulse_count)
        variance = abs(summary.get("tension_acceleration", 0))
        tier_counts = Counter(
            d.get("tier", TIER_PROVISIONAL) for _n, d in self.archivist.graph.nodes(data=True)
        )
        return {
            "throughput": throughput,
            "variance": variance,
            "trend": summary.get("urgency_trend", 0),
            "tier_distribution": {
                "provisional": tier_counts.get(0, 0),
                "working": tier_counts.get(1, 0),
                "trusted": tier_counts.get(2, 0),
            },
            "node_count": self.archivist.graph.number_of_nodes(),
            "edge_count": self.archivist.graph.number_of_edges(),
        }

    def evaluate(self) -> Optional[OverrideSignal]:
        metrics = self.observe()
        if metrics["throughput"] < self.SPINNING_THROUGHPUT and metrics["variance"] > self.SPINNING_VARIANCE:
            return OverrideSignal("FORCE_RESET", "Spinning detected")
        if metrics["throughput"] < self.STAGNANT_THROUGHPUT and metrics["variance"] < self.STAGNANT_VARIANCE:
            return OverrideSignal("FORCE_EXPLORE", "Stagnant")
        return None

    def issue_directive(self, current_bias: str) -> str:
        signal = self.evaluate()
        if signal:
            print(f"Reflector Override: {signal.command} ({signal.reason})")
            return signal.command
        return current_bias

    # ------------------------------------------------------------------
    # 2. Regulatory self-awareness (§4.5 aggregation, §4A item 2).
    # Consolidation-gated -- call from prometheus.py only during
    # Consolidation.
    # ------------------------------------------------------------------
    def regulatory_self_report(self, top_n: int = 5) -> Dict:
        capable = [
            (n, d.get("regulatory_efficacy", 0.5))
            for n, d in self.archivist.graph.nodes(data=True)
            if d.get("tier", TIER_PROVISIONAL) >= TIER_WORKING
        ]
        capable.sort(key=lambda t: t[1], reverse=True)
        return {
            "regulation_capable_count": len(capable),
            "most_effective": capable[:top_n],
            "least_effective": capable[-top_n:] if capable else [],
        }

    # ------------------------------------------------------------------
    # 3. Complex-schema detection (§2.1b, §4A item 3). Consolidation-gated,
    # same clock as trust promotion/demotion and regulatory efficacy.
    # ------------------------------------------------------------------
    def detect_schemas(self) -> List[str]:
        """
        Scans SELF/OTHER-anchored relational edges (`responsible-for`,
        `violates`, `temporal-contrast`, `concerns-other`, §2.1b) and
        cross-references each against the felt state chronos.py had
        logged at the nearest preceding pulse, to find recurring
        co-occurrence of a stabilized basin with a *consistent* relational
        edge pattern. Reflector has no advance knowledge that a given
        combination "means" guilt or pride -- it only counts recurrence of
        the (felt_state, relation_set) pair itself. Returns the list of
        newly-created Schema Node ids.
        """
        graph = self.archivist.graph
        pair_events: Dict[tuple, List[str]] = {}  # (felt_state, relation_set) -> [event_node, ...]

        # Group relational edges by target event node so multi-relation
        # events (e.g. responsible-for + violates on the same node) count
        # as one combined pattern, not two separate ones.
        event_relations: Dict[str, List[tuple]] = {}
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation_type")
            if rel in _RELATIONAL_TYPES and u in (SELF_NODE, "OTHER"):
                event_relations.setdefault(v, []).append((rel, data.get("created_at")))

        for event_node, rels in event_relations.items():
            relation_set = frozenset(r for r, _ts in rels)
            timestamps = [ts for _r, ts in rels if ts]
            felt_state = self._felt_state_near(timestamps[0] if timestamps else None)
            if felt_state is None or felt_state == "Unformed":
                continue
            key = (felt_state, relation_set)
            pair_events.setdefault(key, []).append(event_node)

        created = []
        for (felt_state, relation_set), event_nodes in pair_events.items():
            if len(event_nodes) < self.SCHEMA_STABILIZATION_THRESHOLD:
                continue
            schema_id = self._schema_id(felt_state, relation_set)
            if schema_id in graph:
                continue  # already formed
            graph.add_node(
                schema_id,
                source="schema",
                tier=TIER_WORKING,
                last_reinforced=datetime.now(),
                regulatory_efficacy=0.5,
                tier0_cycles=0,
                is_schema=True,
                named=False,
                name=None,
                basin=felt_state,
                relation_types=sorted(relation_set),
            )
            graph.add_edge(schema_id, felt_state if felt_state in graph else SELF_NODE,
                            relation_type="associated-with", source="schema", placement="explicit",
                            created_at=datetime.now().isoformat())
            for en in event_nodes:
                graph.add_edge(schema_id, en, relation_type="associated-with", source="schema",
                                placement="explicit", created_at=datetime.now().isoformat())
            created.append(schema_id)

        # No self.archivist.save() here (§4C) -- detect_schemas() is one
        # sub-step of prometheus.py's Consolidation pass; the orchestrator
        # checkpoints once, after every sub-step (trust pass, re-parenting,
        # schema detection, efficacy) has run, not after each individually.
        return created

    def name_schema(self, schema_id: str, word: str):
        """§2.1b item 4a: a Schema Node earns a name only if/when the
        agent's actual dictionary/user input happens to link a word to it
        -- never pre-assigned. Delegates to archivist.py, which owns the
        graph mutation directly (kept here too since app.py's manual
        "Name it" UI control calls reflector.name_schema -- this is a
        thin pass-through, not a second implementation, so the two paths
        can't drift out of sync)."""
        self.archivist.name_schema(schema_id, word)

    def schema_count(self) -> int:
        """Used by prometheus.py for the §6.2 Adolescence->Maturity gate."""
        return sum(1 for _n, d in self.archivist.graph.nodes(data=True) if d.get("is_schema"))

    def _felt_state_near(self, timestamp_iso: Optional[str]) -> Optional[str]:
        if not timestamp_iso:
            return None
        try:
            target = datetime.fromisoformat(timestamp_iso)
        except ValueError:
            return None
        best = None
        best_delta = None
        for entry in self.chronos.history:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
            except (KeyError, ValueError):
                continue
            if ts > target:
                continue
            delta = (target - ts).total_seconds()
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = entry.get("felt_state")
        return best

    @staticmethod
    def _schema_id(felt_state: str, relation_set: frozenset) -> str:
        digest = hashlib.sha1(f"{felt_state}|{sorted(relation_set)}".encode()).hexdigest()[:8]
        return f"schema_{digest}"
