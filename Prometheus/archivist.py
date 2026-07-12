import json
import logging
import os
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
EPISTEMIC_GRAPH_PATH = os.path.join(_DATA_DIR, "epistemic_graph.json")

# Trust tiers per spec §3.1
TIER_PROVISIONAL = 0
TIER_WORKING = 1
TIER_TRUSTED = 2

# §2.1b item 1: the one deliberate axiom in the design -- SELF is seeded
# directly into Trusted, not earned, because no experience can precede
# having a self to relate things to.
SELF_NODE = "SELF"

# --- Trust scoring weights (§3.2). None of these are numerically tuned in
# the spec (§10 item 4 is flagged as the single highest-priority remaining
# item) -- these are documented placeholders, not claimed-final values.
SOURCE_WEIGHT = {"dictionary": 0.6, "user": 0.3, "self_generated": 0.2}
DIVERSITY_WEIGHT = 0.25
EDGE_COUNT_WEIGHT = 0.05
EDGE_COUNT_CAP = 10
WORKING_THRESHOLD = 0.6
TRUSTED_THRESHOLD = 1.2

# Hysteresis (§3.3: "N consecutive consolidation passes, not a single
# pass"). Same tuning-placeholder status as the thresholds above.
PROMOTION_HYSTERESIS_N = 2
DEMOTION_HYSTERESIS_N = 2

# §10 item 19 concrete pruning rule: still Tier 0 after N consolidation
# cycles with no reinforcement -> eligible for pruning.
PRUNE_TIER0_CYCLES = 5


class ArchivistModule:
    """
    Visible layer (§7). Pruning thresholds, trust-tier bookkeeping,
    re-parenting evaluation, and regulatory efficacy scoring all belong
    here per the module responsibility table. Trust-tier promotion/demotion
    itself is only ever *executed* here when prometheus.py calls
    run_consolidation_pass() during the Consolidation state (§3.3) -- this
    class never decides *when* to run a pass, only what happens during one.
    """

    def __init__(self):
        # MultiDiGraph, not DiGraph: §2.1b requires an event node to carry
        # *more than one* simultaneous relational edge type from SELF at
        # once (its own example: "I shouldn't have done that" flags both
        # `responsible-for` and `violates` on the same node). A plain
        # DiGraph silently collapses repeated add_edge(u, v, ...) calls
        # into a single overwritten edge, which would make co-occurring
        # relation types on one event node structurally unrepresentable --
        # exactly the case Schema Node detection (§2.1b, §4A) depends on.
        self.graph = nx.MultiDiGraph()
        self.load()
        self._seed_self_node()

    # ------------------------------------------------------------------
    # §2.1b item 1 -- the one non-emergent exception in the whole design.
    # ------------------------------------------------------------------
    def _seed_self_node(self):
        if SELF_NODE not in self.graph:
            self.graph.add_node(
                SELF_NODE,
                last_reinforced=datetime.now(),
                source="axiom",
                tier=TIER_TRUSTED,
                regulatory_efficacy=0.5,
                tier0_cycles=0,
            )
            self.save()

    # ------------------------------------------------------------------
    # Growth / storage
    # ------------------------------------------------------------------
    def store(self, entity: str, metadata: Dict = None, source: str = "user", tier: int = TIER_PROVISIONAL):
        """
        source: 'dictionary' | 'user' | 'self_generated' (§2.2/§3.2) --
        used for trust-weighting and to exclude self-generated edges from
        the diversity signal (§9 risk 5).
        """
        if entity not in self.graph:
            self.graph.add_node(
                entity,
                last_reinforced=datetime.now(),
                source=source,
                tier=tier,
                regulatory_efficacy=0.5,
                tier0_cycles=0,
            )
        else:
            self.graph.nodes[entity]["last_reinforced"] = datetime.now()

        if metadata:
            for rel, target in metadata.get("relations", {}).items():
                if target not in self.graph:
                    self.graph.add_node(
                        target, source=source, tier=TIER_PROVISIONAL,
                        last_reinforced=datetime.now(), regulatory_efficacy=0.5,
                        tier0_cycles=0,
                    )
                self.graph.add_edge(entity, target, relation_type=rel, source=source,
                                     created_at=datetime.now().isoformat())
        self.save()

    def link(self, node_a: str, node_b: str, relation_type: str, source: str = "user",
              placement: str = "explicit"):
        """
        General typed-edge creator used by association.py's hierarchy
        placement (§2.3). `placement` records whether this edge came from
        explicit dictionary-pattern parsing or the co-occurrence fallback
        -- re-parenting (§2.3 mechanism 3) only ever reconsiders
        co-occurrence placements, never explicit ones.
        """
        for n in (node_a, node_b):
            if n not in self.graph:
                self.graph.add_node(n, source=source, tier=TIER_PROVISIONAL,
                                     last_reinforced=datetime.now(),
                                     regulatory_efficacy=0.5, tier0_cycles=0)
        self.graph.add_edge(node_a, node_b, relation_type=relation_type, source=source,
                             placement=placement, created_at=datetime.now().isoformat())
        self.graph.nodes[node_a]["last_reinforced"] = datetime.now()
        self.save()

    def flag_negation(self, node: str):
        """§3.4 mechanism 1: explicit negation/correction detected by
        sensory.py against a recently-active node. Demotion itself still
        only happens at Consolidation (one tier, gradual) -- this just
        records the flag."""
        if node in self.graph:
            self.graph.nodes[node]["negated_flag"] = True
            self.save()

    # ------------------------------------------------------------------
    # §3 Trust scoring -- Consolidation-gated only. prometheus.py must
    # only call this from the Consolidation state.
    # ------------------------------------------------------------------
    def _trust_score(self, node: str) -> float:
        data = self.graph.nodes[node]
        base = SOURCE_WEIGHT.get(data.get("source", "user"), 0.3)

        incident_sources = set()
        edge_count = 0
        for _u, _v, edata in list(self.graph.in_edges(node, data=True)) + list(self.graph.out_edges(node, data=True)):
            edge_count += 1
            esrc = edata.get("source", "user")
            if esrc != "self_generated":  # §2.2/§9 risk 5: excluded from diversity signal
                incident_sources.add(esrc)

        diversity = len(incident_sources)
        score = base + diversity * DIVERSITY_WEIGHT + min(edge_count, EDGE_COUNT_CAP) * EDGE_COUNT_WEIGHT
        return score

    def _tier_for_score(self, score: float) -> int:
        if score >= TRUSTED_THRESHOLD:
            return TIER_TRUSTED
        if score >= WORKING_THRESHOLD:
            return TIER_WORKING
        return TIER_PROVISIONAL

    def run_consolidation_pass(self) -> Dict[str, int]:
        """
        Executes one Consolidation-gated trust evaluation pass (§3.3):
        promotion/demotion via hysteresis, explicit-negation demotion
        (§3.4 mechanism 1), non-reinforcement decay (§3.4 mechanism 2),
        and tier0_cycles bookkeeping for pruning (§10 item 19). Must only
        be called by prometheus.py while in the Consolidation state.
        Returns a small summary dict for logging/dashboard use.
        """
        promotions, demotions = 0, 0
        for node in list(self.graph.nodes):
            if node == SELF_NODE:
                continue  # permanent axiom, never re-evaluated (§2.1b item 1)
            data = self.graph.nodes[node]
            current = data.get("tier", TIER_PROVISIONAL)

            # Non-reinforcement decay (§3.4 mechanism 2): track cycles
            # since this node was last touched.
            data["tier0_cycles"] = data.get("tier0_cycles", 0) + 1 if current == TIER_PROVISIONAL else 0

            # Explicit negation demotes one tier immediately upon the next
            # consolidation pass, gradual (one tier), then clears the flag.
            if data.pop("negated_flag", False) and current > TIER_PROVISIONAL:
                data["tier"] = current - 1
                demotions += 1
                data["_promo_streak"] = 0
                data["_demo_streak"] = 0
                continue

            score = self._trust_score(node)
            target = self._tier_for_score(score)

            if target > current:
                if data.get("_promo_target") == target:
                    data["_promo_streak"] = data.get("_promo_streak", 0) + 1
                else:
                    data["_promo_target"] = target
                    data["_promo_streak"] = 1
                data["_demo_streak"] = 0
                if data["_promo_streak"] >= PROMOTION_HYSTERESIS_N:
                    data["tier"] = min(current + 1, TIER_TRUSTED)  # one tier at a time
                    data["_promo_streak"] = 0
                    promotions += 1
            elif target < current:
                data["_demo_streak"] = data.get("_demo_streak", 0) + 1
                data["_promo_streak"] = 0
                if data["_demo_streak"] >= DEMOTION_HYSTERESIS_N:
                    data["tier"] = max(current - 1, TIER_PROVISIONAL)  # one tier at a time (§3.4)
                    data["_demo_streak"] = 0
                    demotions += 1
            else:
                data["_promo_streak"] = 0
                data["_demo_streak"] = 0

        self.save()
        return {"promotions": promotions, "demotions": demotions}

    # ------------------------------------------------------------------
    # §4.5 Regulatory efficacy -- separate score from epistemic trust,
    # evaluated during Consolidation only.
    # ------------------------------------------------------------------
    def eligible_regulation_nodes(self, anchor_nodes: Optional[List[str]] = None) -> List[str]:
        """§4.2 node selection: Working or Trusted tier only, optionally
        restricted to a set of anchor nodes connected to the current felt
        state's stabilized basin."""
        candidates = anchor_nodes if anchor_nodes is not None else list(self.graph.nodes)
        return [
            n for n in candidates
            if n in self.graph and self.graph.nodes[n].get("tier", TIER_PROVISIONAL) >= TIER_WORKING
        ]

    def update_regulatory_efficacy(self, node: str, worked: bool, step: float = 0.05):
        """Called during Consolidation (§4.5) after checking whether felt-
        state intensity dropped faster than baseline decay alone would
        predict following a regulation attempt."""
        if node not in self.graph:
            return
        eff = self.graph.nodes[node].get("regulatory_efficacy", 0.5)
        eff = eff + step if worked else eff - step
        self.graph.nodes[node]["regulatory_efficacy"] = max(0.0, min(1.0, eff))
        self.save()

    # ------------------------------------------------------------------
    # §2.3 mechanism 3 -- re-parenting evaluation, Consolidation-gated.
    # ------------------------------------------------------------------
    def reparenting_candidates(self, min_corroboration: int = 3) -> List[str]:
        """Nodes placed via the co-occurrence fallback (`associated-with`,
        placement='cooccurrence') that have since accumulated enough
        independent corroboration to justify re-evaluating their parent.
        Returns node names only -- association.py owns deciding *what* the
        firmer parent should be (it has the dictionary-pattern parser);
        this just flags who's eligible."""
        candidates = []
        for node in self.graph.nodes:
            if node == SELF_NODE:
                continue
            in_edges = list(self.graph.in_edges(node, data=True))
            cooccurrence_parent = any(
                d.get("placement") == "cooccurrence" and d.get("relation_type") == "associated-with"
                for _u, _v, d in in_edges
            )
            if not cooccurrence_parent:
                continue
            sources = {d.get("source", "user") for _u, _v, d in in_edges if d.get("source") != "self_generated"}
            if len(sources) >= min_corroboration:
                candidates.append(node)
        return candidates

    def reparent(self, node: str, new_parent: str, relation_type: str = "is-a"):
        """Executes a re-parent: drops the old co-occurrence edge(s) into
        `node`, adds the new, firmer typed edge."""
        if node not in self.graph or new_parent not in self.graph:
            return
        to_remove = [
            (u, v) for u, v, d in list(self.graph.in_edges(node, data=True))
            if d.get("placement") == "cooccurrence"
        ]
        for u, v in to_remove:
            self.graph.remove_edge(u, v)
        self.link(new_parent, node, relation_type, source="reparent", placement="explicit")

    # ------------------------------------------------------------------
    # §10 item 19 -- Pruning's concrete trigger.
    # ------------------------------------------------------------------
    def prune(self) -> int:
        """Removes nodes still Tier 0 after PRUNE_TIER0_CYCLES consolidation
        cycles with no reinforcement -- the "still Tier 0 after N
        consolidation cycles" rule the spec names as the missing mechanism,
        as opposed to a raw salience score (the alternative it explicitly
        left undecided). Only prometheus.py should call this, and only
        while in the Pruning state."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if n != SELF_NODE
            and d.get("tier", TIER_PROVISIONAL) == TIER_PROVISIONAL
            and d.get("tier0_cycles", 0) >= PRUNE_TIER0_CYCLES
        ]
        for n in to_remove:
            self.graph.remove_node(n)
        if to_remove:
            self.save()
        return len(to_remove)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            data = nx.readwrite.json_graph.node_link_data(self.graph)
            with open(EPISTEMIC_GRAPH_PATH, "w") as f:
                json.dump(data, f, default=str)
        except OSError as e:
            logger.warning("ArchivistModule.save failed: %s", e)

    def load(self):
        if os.path.exists(EPISTEMIC_GRAPH_PATH):
            try:
                with open(EPISTEMIC_GRAPH_PATH, "r") as f:
                    data = json.load(f)
                self.graph = nx.readwrite.json_graph.node_link_graph(data)
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                logger.warning(
                    "ArchivistModule.load failed (%s); starting with an empty graph instead of crashing.",
                    e,
                )
                self.graph = nx.MultiDiGraph()

    def retrieve(self, key: str, bias: str = None):
        """Minimal existing behavior preserved; returns nodes matching key
        as a substring, most-recently-reinforced first."""
        matches = [n for n in self.graph.nodes if key.lower() in str(n).lower()]
        matches.sort(
            key=lambda n: self.graph.nodes[n].get("last_reinforced", datetime.min),
            reverse=True,
        )
        return matches
