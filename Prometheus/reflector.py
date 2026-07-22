import hashlib
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx

from .archivist import SELF_NODE, TIER_PROVISIONAL, TIER_WORKING
from .edge_types import (
    RELATIONAL_EDGE_TYPES, EDGE_COMPOSED_OF, EDGE_IS_A, NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA,
)


class OverrideSignal:
    def __init__(self, command: str, reason: str):
        self.command = command
        self.reason = reason


# §2.1b item 4: co-occurrence stabilization threshold for Schema Node
# formation. Same tuning-placeholder category as basin stabilization
# (§10 item 13) -- not yet numeric in the spec.
SCHEMA_STABILIZATION_THRESHOLD = 3

# §13.3, new: epistemic (knowledge-cluster) schema formation. Same
# tuning-placeholder status as everything else in this design (§10).
EPISTEMIC_MIN_CLUSTER_SIZE = 3
EPISTEMIC_NAME_MIN_COVERAGE = 2  # how many cluster members a parsed is-a parent must cover before it's recognized as earning the cluster's name (§13.3.1)



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
        self.EPISTEMIC_MIN_CLUSTER_SIZE = EPISTEMIC_MIN_CLUSTER_SIZE
        self.EPISTEMIC_NAME_MIN_COVERAGE = EPISTEMIC_NAME_MIN_COVERAGE

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
            if rel in RELATIONAL_EDGE_TYPES and u in (SELF_NODE, "OTHER"):
                event_relations.setdefault(v, []).append((rel, data))

        for event_node, rels in event_relations.items():
            relation_set = frozenset(r for r, _d in rels)
            felt_state = self._resolve_felt_state(rels)
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
                node_type=NODE_SCHEMA,
                named=False,
                name=None,
                basin=felt_state,
                relation_types=sorted(relation_set),
            )
            # composed-of (§6A, this revision), not associated-with: this
            # is a permanent structural fact about what the schema is made
            # of, not a tentative co-occurrence placement -- keeping it
            # distinct also keeps it out of _trust_score's corroboration
            # count (TRUST_BEARING_EDGE_TYPES is categorical-only) and out
            # of archivist.reparenting_candidates()'s associated-with scan.
            graph.add_edge(schema_id, felt_state if felt_state in graph else SELF_NODE,
                            relation_type=EDGE_COMPOSED_OF, source="schema", placement="explicit",
                            created_at=datetime.now().isoformat())
            for en in event_nodes:
                graph.add_edge(schema_id, en, relation_type=EDGE_COMPOSED_OF, source="schema",
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

    # ------------------------------------------------------------------
    # §13.3 Epistemic (knowledge-cluster) Schema formation -- new,
    # Consolidation-gated (same clock as everything else). Tier 1 only in
    # this pass: clusters form directly from base graph nodes. The
    # `abstraction_level` field is present from the start so recursive
    # Tier 2+ (schemas clustering from other schemas) can be added later
    # without a data-migration -- but only level 1 is actually exercised
    # here, deliberately, rather than attempting the full recursive system
    # in one unvalidated leap.
    # ------------------------------------------------------------------
    def detect_epistemic_clusters(self) -> List[str]:
        """
        Groups nodes whose co-activation has stabilized (archivist.
        stabilized_co_activation_pairs()) into cluster candidates via
        connected components of a temporary co-activation graph --
        deterministic, inspectable graph theory (same standard already
        used for cycle handling in the original §13.3 proposal), not an
        opaque clustering library. A candidate becomes a real (unnamed)
        Epistemic Schema Node if it has at least EPISTEMIC_MIN_CLUSTER_SIZE
        members and doesn't already share a single dominant parent
        (skipped in that case -- that parent already names the group,
        creating a second, competing structure for the same thing would
        be redundant, not useful). Returns the list of newly-created
        schema ids.
        """
        pairs = self.archivist.stabilized_co_activation_pairs()
        if not pairs:
            return []

        co_graph = nx.Graph()
        co_graph.add_edges_from(pairs)

        graph = self.archivist.graph
        created = []
        for component in nx.connected_components(co_graph):
            if len(component) < self.EPISTEMIC_MIN_CLUSTER_SIZE:
                continue
            members = sorted(component)

            if self._has_dominant_shared_parent(members):
                continue  # already named via ordinary hierarchy -- redundant to cluster

            cluster_id = self._epistemic_cluster_id(members)
            if cluster_id in graph:
                continue  # already formed, nothing new to do

            graph.add_node(
                cluster_id,
                source="schema",
                tier=TIER_WORKING,
                last_reinforced=datetime.now(),
                regulatory_efficacy=0.5,
                tier0_cycles=0,
                activation=0.0,
                valence_coloring=0.0,
                node_type=NODE_EPISTEMIC_SCHEMA,
                abstraction_level=1,
                named=False,
                name=None,
                member_count=len(members),
            )
            for member in members:
                graph.add_edge(cluster_id, member, relation_type=EDGE_COMPOSED_OF,
                                source="schema", placement="explicit",
                                created_at=datetime.now().isoformat())
            created.append(cluster_id)

        return created

    def _has_dominant_shared_parent(self, members: List[str], min_coverage: Optional[int] = None) -> bool:
        """True if a single existing is-a parent already covers enough of
        `members` that clustering them again would be redundant. Reuses
        the same "does a parent word cover >= K members" check that also
        underlies naming (§13.3.1) -- kept as a shared helper so the two
        can't silently diverge into different definitions of "covers.\""""
        min_coverage = self.EPISTEMIC_NAME_MIN_COVERAGE if min_coverage is None else min_coverage
        graph = self.archivist.graph
        parent_counts: Counter = Counter()
        for member in members:
            if member not in graph:
                continue
            for u, _v, edata in graph.in_edges(member, data=True):
                if edata.get("relation_type") == EDGE_IS_A:
                    parent_counts[u] += 1
        return bool(parent_counts) and max(parent_counts.values()) >= min_coverage

    def try_name_epistemic_schemas(self) -> int:
        """§13.3.1's resolved naming rule, implemented: an epistemic
        schema earns a name only when a dictionary-pattern-parsed is-a
        assertion ties back to enough of its members -- never generated
        by the system. Consolidation-time scan (not a live trigger) is
        the simplest correct implementation of the same rule §2.1b item
        4a and §6.1 already use elsewhere, just checked periodically
        instead of on every placement. Returns the count newly named."""
        graph = self.archivist.graph
        named_count = 0
        for node, data in list(graph.nodes(data=True)):
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA or data.get("named"):
                continue
            members = [
                v for _u, v, edata in graph.out_edges(node, data=True)
                if edata.get("relation_type") == EDGE_COMPOSED_OF
            ]
            if not members:
                continue
            parent_counts: Counter = Counter()
            for member in members:
                if member not in graph:
                    continue
                for u, _v, edata in graph.in_edges(member, data=True):
                    if edata.get("relation_type") == EDGE_IS_A:
                        parent_counts[u] += 1
            if not parent_counts:
                continue
            best_parent, coverage = parent_counts.most_common(1)[0]
            if coverage >= self.EPISTEMIC_NAME_MIN_COVERAGE:
                graph.nodes[node]["name"] = best_parent
                graph.nodes[node]["named"] = True
                named_count += 1
        return named_count

    @staticmethod
    def _epistemic_cluster_id(members: List[str]) -> str:
        digest = hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:8]
        return f"epistemic_{digest}"

    def epistemic_schema_report(self, top_n: int = 5) -> Dict:
        """
        Diagnostic (§13.3, new): same "make it checkable" pattern as every
        other new mechanism this session. Shows real cluster-candidate
        progress -- how many co-activation pairs exist, how many have
        stabilized, and how close any live candidate component is to
        EPISTEMIC_MIN_CLUSTER_SIZE -- plus a summary of formed schemas
        (named/unnamed). Read-only, never mutates the graph.
        """
        pairs = self.archivist.stabilized_co_activation_pairs()
        co_graph = nx.Graph()
        co_graph.add_edges_from(pairs)
        candidates = []
        for component in nx.connected_components(co_graph):
            candidates.append({
                "size": len(component),
                "threshold": self.EPISTEMIC_MIN_CLUSTER_SIZE,
                "remaining": max(0, self.EPISTEMIC_MIN_CLUSTER_SIZE - len(component)),
                "members": sorted(component)[:5],
            })
        candidates.sort(key=lambda c: c["remaining"])

        schemas = [
            (n, d) for n, d in self.archivist.graph.nodes(data=True)
            if d.get("node_type") == NODE_EPISTEMIC_SCHEMA
        ]
        return {
            "total_co_activation_pairs": len(self.archivist.co_activation),
            "stabilized_pairs": len(pairs),
            "candidate_clusters": candidates[:top_n],
            "schemas_formed": len(schemas),
            "schemas_named": sum(1 for _n, d in schemas if d.get("named")),
        }

    def schema_candidate_report(self, top_n: int = 5) -> Dict:
        """
        Diagnostic, read-only mirror of detect_schemas()'s grouping logic
        (§2.1b) -- shows how close the system is to forming a Schema Node
        without waiting for one to actually stabilize, and without
        mutating the graph. Added because "No stable Schema Nodes formed
        yet" gave zero visibility into *why* -- whether relational edges
        exist at all, how many are being silently dropped for occurring
        before any felt-state basin had stabilized (§2.1a's "Unformed"
        case, permanently excluded from candidacy, not retried later),
        and how close any surviving (felt_state, relation_set) pair
        actually is to SCHEMA_STABILIZATION_THRESHOLD. Safe to call every
        Reflection-tab render, not just at Consolidation -- it never
        creates or modifies a node.
        """
        graph = self.archivist.graph
        event_relations: Dict[str, List[tuple]] = {}
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation_type")
            if rel in RELATIONAL_EDGE_TYPES and u in (SELF_NODE, "OTHER"):
                event_relations.setdefault(v, []).append((rel, data))

        pair_events: Dict[tuple, List[str]] = {}
        dropped_unformed = 0
        for event_node, rels in event_relations.items():
            relation_set = frozenset(r for r, _d in rels)
            felt_state = self._resolve_felt_state(rels)
            if felt_state is None or felt_state == "Unformed":
                dropped_unformed += 1
                continue
            key = (felt_state, relation_set)
            pair_events.setdefault(key, []).append(event_node)

        candidates = []
        for (felt_state, relation_set), event_nodes in pair_events.items():
            count = len(event_nodes)
            candidates.append({
                "felt_state": felt_state,
                "relation_types": sorted(relation_set),
                "count": count,
                "threshold": self.SCHEMA_STABILIZATION_THRESHOLD,
                "remaining": max(0, self.SCHEMA_STABILIZATION_THRESHOLD - count),
            })
        candidates.sort(key=lambda c: c["remaining"])

        return {
            "total_relational_event_nodes": len(event_relations),
            "dropped_unformed_felt_state": dropped_unformed,
            "candidate_pairs": candidates[:top_n],
        }

    def activation_report(self, top_n: int = 10) -> Dict:
        """
        Diagnostic (§11 pull-forward, this revision): surfaces real
        activation numbers instead of leaving "is focus/working-memory
        actually working" as something only inferable from whether the
        Graph tab subjectively looks focused. Added directly in response
        to a real bug found this way -- felt_state_anchors had been
        growing unbounded and silently swamping the top-K activation
        filter, which was invisible without a way to see actual numbers.
        Read-only, safe to call every Reflection-tab render.
        """
        graph = self.archivist.graph
        activations = [
            (n, d.get("activation", 0.0), d.get("node_type", "standard"))
            for n, d in graph.nodes(data=True)
        ]
        activations.sort(key=lambda t: t[1], reverse=True)
        nonzero = [a for a in activations if a[1] > 0.0]
        return {
            "total_nodes": len(activations),
            "nodes_with_nonzero_activation": len(nonzero),
            "top_active": activations[:top_n],
        }

    def valence_coloring_report(self, top_n: int = 5) -> Dict:
        """
        Diagnostic (§13.2, new): surfaces real valence_coloring numbers,
        same "make it checkable, not just eyeballed" pattern as
        activation_report/self_other_report. A node's coloring only ever
        moves through prometheus.give_parental_reaction()'s co-occurrence
        mechanism -- this reports what's actually accumulated, split into
        most-positive and most-negative, so the mirror-neuron-style
        learning is directly observable. Read-only.
        """
        graph = self.archivist.graph
        colored = [
            (n, d.get("valence_coloring", 0.0))
            for n, d in graph.nodes(data=True)
            if d.get("valence_coloring", 0.0) != 0.0
        ]
        colored.sort(key=lambda t: t[1], reverse=True)
        return {
            "total_colored_nodes": len(colored),
            "most_positive": colored[:top_n],
            "most_negative": colored[-top_n:][::-1] if colored else [],
        }

    def self_other_report(self, recent_n: int = 5) -> Dict:
        """
        Diagnostic (this revision, in response to "SELF never seems to
        expand"): SELF and OTHER only ever gain new edges through
        relational detection (§2.1b), triggered by typed input matching
        specific keyword patterns -- self-study and regulation both
        exclude SELF/OTHER by design (they're axioms/placeholders, not
        dictionary concepts that grow via hyponym expansion or get used
        as coping strategies). Three of the four relation types
        (responsible-for/violates/temporal-contrast) route through SELF;
        only concerns-other routes through OTHER -- so third-person-heavy
        input will visibly grow OTHER while SELF looks comparatively
        frozen, which can read as "SELF is broken" when it's actually
        just receiving a different mix of triggering phrasing. Surfaces
        raw per-type edge counts so this is checkable directly instead of
        inferred from the Graph tab. Read-only.
        """
        graph = self.archivist.graph

        def _edge_summary(anchor: str) -> Dict:
            counts = Counter()
            recent = []
            for _u, v, data in graph.out_edges(anchor, data=True):
                rel = data.get("relation_type")
                if rel in RELATIONAL_EDGE_TYPES:
                    counts[rel] += 1
                    recent.append((data.get("created_at", ""), rel, v))
            recent.sort(key=lambda t: t[0], reverse=True)
            return {
                "total": sum(counts.values()),
                "by_type": dict(counts),
                "most_recent": recent[:recent_n],
            }

        return {
            "self": _edge_summary(SELF_NODE),
            "other": _edge_summary("OTHER"),
        }

    def _resolve_felt_state(self, rels: List[tuple]) -> Optional[str]:
        """Resolves the felt state active when a set of relational edges
        was created. Prefers `felt_state_at_creation`, stamped directly on
        the edge at creation time by association.link_relational() (this
        revision) -- reliable, since it's the ground truth at the moment
        of creation, not a reconstruction. Falls back to the old
        timestamp-nearest-neighbor lookup via _felt_state_near() only for
        edges that predate this fix (e.g. relational edges already saved
        in an existing graph checkpoint) and therefore don't carry the
        stamped field. See archivist.link()'s docstring for why the old
        path alone was unreliable: it could only ever find the *previous*
        tick's felt state (never the current tick's, since _ingest()
        always runs before that same tick's chronos.record_pulse()), or
        nothing at all on the very first pulse ever / right after a
        felt-state transition -- silently and permanently dropping a
        relational edge from schema candidacy even when a real, named
        felt state was active. `rels` is a list of (relation_type, edge_data)
        tuples for one event node, as built by detect_schemas/
        schema_candidate_report's grouping loop."""
        for _rel, data in rels:
            stamped = data.get("felt_state_at_creation")
            if stamped:
                return stamped
        # Fallback: no edge in this group was stamped (pre-fix data) --
        # reconstruct from the earliest available timestamp, same as
        # before this revision.
        timestamps = [d.get("created_at") for _r, d in rels if d.get("created_at")]
        if not timestamps:
            return None
        return self._felt_state_near(timestamps[0])

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
