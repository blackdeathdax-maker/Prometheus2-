import json
import logging
import os
from collections import Counter
from itertools import combinations
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional

from .edge_types import (
    TRUST_BEARING_EDGE_TYPES, NODE_STANDARD, NODE_BASIN, NODE_SCHEMA, NODE_SELF,
    NODE_EPISTEMIC_SCHEMA, EDGE_ASSOCIATED_WITH, EDGE_IS_A, EDGE_PART_OF,
    EDGE_COMPOSED_OF, EDGE_INSTANCE_OF, EDGE_ABSTRACTED_FROM, EDGE_ELABORATES,
    get_family, EXCLUSIVE_FAMILIES, FAMILY_RESIDUAL, FAMILY_MEMBERSHIP,
)
from .edge_types import is_body_channel_node, BODY_CHANNEL_NODE_IDS, body_channel_node_id, BODY_CHANNELS, is_felt_place_node, is_narrative_graph_node, is_somatic_infrastructure

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
EPISTEMIC_GRAPH_PATH = os.path.join(_DATA_DIR, "epistemic_graph.json")
CO_ACTIVATION_PATH = os.path.join(_DATA_DIR, "co_activation.json")

# Trust tiers per spec §3.1
TIER_PROVISIONAL = 0
TIER_WORKING = 1
TIER_TRUSTED = 2

# §2.1b item 1: the one deliberate axiom in the design -- SELF is seeded
# directly into Trusted, not earned, because no experience can precede
# having a self to relate things to.
SELF_NODE = "SELF"

# §2.1b concerns-other placeholder: generic OTHER entity node for relational
# edges involving someone other than SELF (jealousy, embarrassment, social
# emotions generally).
OTHER_NODE = "OTHER"

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
PRUNE_TIER0_CYCLES = 12  # longer life so co-activation can mature to Working/schemas

# Activation / working-memory layer (new, this revision -- a v1-scoped
# pull-forward of §11's deferred activation-based working-memory concept,
# not the full v2 vision: no archiving to cold storage, no rehydration, no
# per-consumer hot/cold decision. Just a per-node activation score, boosted
# on touch and decayed at Consolidation (same clock as everything else),
# used for two things: (a) self-study preferring active nodes over uniform
# randomness, (b) the Graph tab rendering only a bounded neighborhood
# instead of the entire live graph. Same "not yet numerically tuned"
# placeholder status as every other constant here (§10).
ACTIVATION_BOOST = 1.0
ACTIVATION_DECAY_RATE = 0.6  # fraction retained per Consolidation pass
ACTIVATION_CAP = 10.0

# Co-activation tracking (§13.3, new) -- the detection signal epistemic
# schema clustering depends on. Whenever a group of nodes is touched in
# the same event (a self-study cycle's target+children, an ingestion's
# node+anchor), every pair in that group gets a co-activation count
# bumped. Sparse by construction: only pairs that actually co-occur in a
# real event ever get an entry, never the full N² pairs across the graph
# -- same sparsity property the knowledge graph itself already has (169
# edges among 35 nodes in one production sample, not 35²). Decayed at
# Consolidation, same clock as everything else, same shape as activation
# decay and basin dwell-time decay -- "one clock, not several."
CO_ACTIVATION_DECAY_RATE = 0.6
CO_ACTIVATION_STABILIZATION_THRESHOLD = 5  # stricter: fewer garbage clusters from casual co-touch
CO_ACTIVATION_PRUNE_FLOOR = 0.2  # below this after decay, drop the pair entirely -- same shape as synthesizer.py's basin DESTABILIZATION_FLOOR

# §13.4 Graph Collapse & Abstraction Layer -- starter placeholders per
# §13.4.9, same "not yet numerically tuned" status as everything else.
N_COLLAPSE = 6                     # neglect cycles before absorption (spec range 4-8)
D_COLLAPSE = 4                     # depth from nearest schema root (spec range 3-5)
COLLAPSE_ACTIVATION_FLOOR = 1.0    # out of ACTIVATION_CAP=10.0 -- "low", additional required condition alongside neglect (see run_collapse_pass's docstring for why AND, not OR)
REHYDRATE_EDGE_MOVE_FRACTION = 0.25  # spec range 0.0-0.5


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
        # Tunable thresholds as instance attributes, not just module-level
        # constants -- this is what lets the Debug tab's sliders mutate a
        # live instance (e.g. `archivist.WORKING_THRESHOLD = 0.5`) without
        # any code change taking effect only on next restart. Defaults
        # mirror the module-level constants above (still the documented
        # "not yet numerically tuned" placeholders, §10) -- this is a
        # mechanism for tuning them live, not a claim that these specific
        # values are now correct.
        self.DIVERSITY_WEIGHT = DIVERSITY_WEIGHT
        self.EDGE_COUNT_WEIGHT = EDGE_COUNT_WEIGHT
        self.EDGE_COUNT_CAP = EDGE_COUNT_CAP
        self.WORKING_THRESHOLD = WORKING_THRESHOLD
        self.TRUSTED_THRESHOLD = TRUSTED_THRESHOLD
        self.PROMOTION_HYSTERESIS_N = PROMOTION_HYSTERESIS_N
        self.DEMOTION_HYSTERESIS_N = DEMOTION_HYSTERESIS_N
        self.PRUNE_TIER0_CYCLES = PRUNE_TIER0_CYCLES
        self.ACTIVATION_BOOST = ACTIVATION_BOOST
        self.ACTIVATION_DECAY_RATE = ACTIVATION_DECAY_RATE
        self.ACTIVATION_CAP = ACTIVATION_CAP
        self.CO_ACTIVATION_DECAY_RATE = CO_ACTIVATION_DECAY_RATE
        self.CO_ACTIVATION_STABILIZATION_THRESHOLD = CO_ACTIVATION_STABILIZATION_THRESHOLD
        self.CO_ACTIVATION_PRUNE_FLOOR = CO_ACTIVATION_PRUNE_FLOOR
        self.N_COLLAPSE = N_COLLAPSE
        self.D_COLLAPSE = D_COLLAPSE
        self.COLLAPSE_ACTIVATION_FLOOR = COLLAPSE_ACTIVATION_FLOOR
        self.REHYDRATE_EDGE_MOVE_FRACTION = REHYDRATE_EDGE_MOVE_FRACTION
        # Sparse -- only pairs that actually co-occurred in a real event
        # ever get an entry (§13.3's own docstring above explains why).
        # Keyed by a sorted 2-tuple of node ids, not a frozenset, so it's
        # directly JSON-serializable for persistence without a custom
        # encoder.
        self.co_activation: Counter = Counter()

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
        self._seed_body_channels()
        self._seed_other_node()

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
                node_type=NODE_SELF,
                activation=ACTIVATION_CAP,  # SELF is always maximally active -- never excluded from working memory by low activation
                valence_coloring=0.0,
                # Pre-linguistic identity: filled every pulse by _sync_self_felt
                felt_arousal=0.0,
                felt_valence=0.0,
                felt_dominance=0.0,
                last_felt_label="Unformed",
                last_felt_key=(0.0, 0.0, 0.0),
                is_identity_hub=True,
            )
            self.save()

    # ------------------------------------------------------------------
    # §2.1b concerns-other support: seed OTHER placeholder node.
    # ------------------------------------------------------------------
    def _seed_other_node(self):
        """Seeds the OTHER node for relational edges (§2.1b, §4A).
        OTHER represents "any entity other than SELF" and is used by
        concerns-other edges (jealousy, embarrassment, social emotions).
        Not an axiom like SELF (not permanently Trusted), but initialized
        at Working tier so it doesn't get pruned before accumulating edges."""
        if OTHER_NODE not in self.graph:
            self.graph.add_node(
                OTHER_NODE,
                last_reinforced=datetime.now(),
                source="schema",
                tier=TIER_WORKING,
                regulatory_efficacy=0.5,
                tier0_cycles=0,
                node_type=NODE_STANDARD,
                activation=0.0,
                valence_coloring=0.0,
            )
            self.save()

    # ------------------------------------------------------------------
    # Growth / storage
    # ------------------------------------------------------------------
    def _seed_body_channels(self):
        """Fixed somatic surface nodes. Linkable into epistemic graph as
        parts only; never expanded, never is-a hierarchy participants.
        """
        from .edge_types import BODY_CHANNELS, body_channel_node_id
        for ch in BODY_CHANNELS:
            nid = body_channel_node_id(ch)
            if nid not in self.graph:
                self.graph.add_node(
                    nid,
                    last_reinforced=datetime.now(),
                    source="axiom",
                    tier=TIER_TRUSTED,
                    regulatory_efficacy=0.5,
                    tier0_cycles=0,
                    node_type=NODE_BASIN,
                    activation=0.0,
                    valence_coloring=0.0,
                    is_body_channel=True,
                    is_felt_place=True,
                    body_channel=ch,
                    growable=False,
                )
            else:
                nd = self.graph.nodes[nid]
                nd["is_body_channel"] = True
                nd["is_felt_place"] = True
                nd["growable"] = False
                nd["body_channel"] = ch


    def repair_identity_edges(self) -> dict:
        """Strip illegal is-a involving SELF / body / felt / narr; one-shot hygiene."""
        removed = 0
        g = self.graph
        to_remove = []
        for u, v, k, d in list(g.edges(keys=True, data=True)):
            rel = (d or {}).get("relation_type")
            if rel != "is-a":
                continue
            if (
                u == SELF_NODE or v == SELF_NODE
                or is_somatic_infrastructure(u)
                or is_somatic_infrastructure(v)
            ):
                to_remove.append((u, v, k))
        for u, v, k in to_remove:
            try:
                g.remove_edge(u, v, k)
                removed += 1
            except Exception:
                try:
                    g.remove_edge(u, v)
                    removed += 1
                except Exception:
                    pass
        return {"removed_is_a": removed}

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
                node_type=NODE_STANDARD,
                activation=0.0,
                valence_coloring=0.0,
            )
        else:
            self.graph.nodes[entity]["last_reinforced"] = datetime.now()

        if metadata:
            for rel, target in metadata.get("relations", {}).items():
                if target not in self.graph:
                    self.graph.add_node(
                        target, source=source, tier=TIER_PROVISIONAL,
                        last_reinforced=datetime.now(), regulatory_efficacy=0.5,
                        tier0_cycles=0, node_type=NODE_STANDARD, activation=0.0,
                valence_coloring=0.0,
                    )
                self.graph.add_edge(entity, target, relation_type=rel, source=source,
                                     created_at=datetime.now().isoformat())
        # No self.save() here (§4C): persistence is Consolidation-gated
        # only. store() runs constantly during Learning (every ingested
        # term, every self-study expansion) -- a disk write per call would
        # both violate "checkpoint at Consolidation, one clock" and be a
        # real performance cost as the graph grows. Prometheus.py's
        # _run_consolidation() calls archivist.save() once, after every
        # sub-step (trust pass, re-parenting, schema detection, efficacy)
        # has run.

    def ensure_basin_node(self, basin_id: str, pad_coordinates: tuple, dwell_density: float = 0.0):
        """
        Creates or updates a real graph node for a stabilized felt-state
        basin (§2.1a/§6A). Bug fix, this revision: synthesizer.py's
        stabilized_basins was always just a string mapping
        ((a,v,d) -> basin_id) -- nothing anywhere ever called
        graph.add_node() for it. §6A specifies basin nodes as a real
        node_type with pad_coordinates/dwell_density/stabilized fields,
        and prometheus_dashboard.py already has rendering logic ready for
        them (diamond shape, valence-colored) -- but since no basin node
        ever actually existed, reflector.detect_schemas()'s "link the
        Schema Node back to its component basin" fallback
        (`felt_state if felt_state in graph else SELF_NODE`) always took
        the SELF_NODE branch, silently, every time. Every schema's
        component-basin link was quietly attaching to SELF instead of a
        real basin entity -- misrepresenting what the schema is actually
        composed of, and dumping extra incoming edges onto SELF that
        don't belong there.

        Basin nodes deliberately don't carry `tier` (§6A: "Not applicable
        to basin/schema/self node types -- trust tiers represent
        epistemic corroboration of facts; basins/schemas represent
        recurrence, a different kind of evidence") and are exempt from
        both the trust-tier consolidation pass and pruning, the same
        treatment Schema Nodes and SELF already get.

        Called from prometheus.py's Consolidation pass, once per
        currently-stabilized basin, right after
        synthesizer.consolidate_basins() runs -- so any basin
        reflector.detect_schemas() might reference later in the same pass
        already has a real node to link to.
        """
        if basin_id not in self.graph:
            self.graph.add_node(
                basin_id,
                source="system",
                node_type=NODE_BASIN,
                last_reinforced=datetime.now(),
                activation=0.0,
                valence_coloring=0.0,
                pad_coordinates=list(pad_coordinates),
                dwell_density=dwell_density,
                stabilized=True,
                regulatory_efficacy=0.5,  # unused for basin nodes, kept only for schema-uniformity with other node_types' field set
            )
        else:
            self.graph.nodes[basin_id]["pad_coordinates"] = list(pad_coordinates)
            self.graph.nodes[basin_id]["dwell_density"] = dwell_density
            self.graph.nodes[basin_id]["last_reinforced"] = datetime.now()
        # No self.save() here (§4C) -- same reasoning as store()/link();
        # called from the Consolidation pass, which checkpoints once at
        # the end via prometheus.py's _run_consolidation().

    def link(self, node_a: str, node_b: str, relation_type: str, source: str = "user",
              placement: str = "explicit", felt_state: Optional[str] = None):
        """
        General typed-edge creator used by association.py's hierarchy
        placement (§2.3). `placement` records whether this edge came from
        explicit dictionary-pattern parsing or the co-occurrence fallback
        -- re-parenting (§2.3 mechanism 3) only ever reconsiders
        co-occurrence placements, never explicit ones.

        `felt_state` (new, this revision): the felt state active at the
        moment of creation, stamped directly onto the edge as
        `felt_state_at_creation` when supplied. Fixes a real bug found in
        reflector.py's schema detection: felt state was previously always
        reconstructed after the fact via chronos._felt_state_near()'s
        nearest-preceding-timestamp lookup, but since prometheus.py's
        pulse() always calls _ingest() (which creates relational edges)
        before that same tick's chronos.record_pulse(), an edge's own
        timestamp is always earlier than its own tick's chronos entry --
        the lookup could only ever find the *previous* tick's felt state,
        or nothing at all on the very first pulse ever / right after a
        felt-state transition, silently and permanently dropping a
        relational edge from schema candidacy even though a real, named
        felt state was active when it was created. Stamping ground truth
        directly at creation is far more reliable than reconstructing it
        approximately afterward. Optional and backward-compatible: edges
        created without this (including everything in an existing saved
        graph) fall back to the old timestamp-based reconstruction.
        """
        for n in (node_a, node_b):
            if n not in self.graph:
                self.graph.add_node(n, source=source, tier=TIER_PROVISIONAL,
                                     last_reinforced=datetime.now(),
                                     regulatory_efficacy=0.5, tier0_cycles=0,
                                     node_type=NODE_STANDARD, activation=0.0, valence_coloring=0.0)
        
        # --- Hierarchy direction guards (domain-agnostic) ---
        
        # Somatic + identity infrastructure:
        # - body/felt/narr: never participate in is-a
        # - SELF is never an is-a parent or child of knowledge lemmas
        # - body/felt only as PARTS of epistemic wholes (composed-of / part-of)
        if relation_type == "is-a":
            if is_somatic_infrastructure(node_a) or is_somatic_infrastructure(node_b):
                return
            if node_a == SELF_NODE or node_b == SELF_NODE:
                return  # identity is not a hypernym/hyponym of Anger etc.
        if relation_type in ("part-of", "composed-of"):
            # composed-of: whole -> part
            # part-of:     part -> whole
            if relation_type == "composed-of":
                if is_body_channel_node(node_a) or is_felt_place_node(node_a):
                    return  # body/felt cannot be the whole
                if node_a == SELF_NODE and is_body_channel_node(node_b):
                    return  # SELF senses body via associated-with, not composed-of body as only path — allow narr parts only
            if relation_type == "part-of":
                if is_body_channel_node(node_b) or is_felt_place_node(node_b):
                    return  # cannot be part of a body channel / felt place
                if node_b == SELF_NODE and is_body_channel_node(node_a):
                    return  # body is not "part of SELF" via part-of hierarchy; use associated-with
        if relation_type == "is-a":
            a_data = self.graph.nodes.get(node_a, {}) or {}
            b_data = self.graph.nodes.get(node_b, {}) or {}
            a_schema = a_data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or a_data.get("is_schema")
            b_schema = b_data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or b_data.get("is_schema")
            # Refuse lemma ↔ schema is-a (membership is composed-of)
            if a_schema or b_schema:
                return
        if relation_type == "composed-of":
            a_data = self.graph.nodes.get(node_a, {}) or {}
            b_data = self.graph.nodes.get(node_b, {}) or {}
            a_schema = a_data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or a_data.get("is_schema")
            # only schemas may emit composed-of
            if not a_schema:
                # if target is schema and source is not, skip inverted
                if b_data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or b_data.get("is_schema"):
                    return
        # Idempotent: MultiDiGraph otherwise stacks duplicate same-type edges
        # every pulse (WM is-a scaffolding / re-link), which draws "flower" graphs.
        if self.graph.has_edge(node_a, node_b):
            try:
                for _k, attr in (self.graph.get_edge_data(node_a, node_b) or {}).items():
                    if isinstance(attr, dict) and attr.get("relation_type") == relation_type:
                        self.graph.nodes[node_a]["last_reinforced"] = datetime.now()
                        if node_b in self.graph.nodes:
                            self.graph.nodes[node_b]["last_reinforced"] = datetime.now()
                        return  # already linked this way
            except Exception:
                pass
        edge_kwargs = dict(relation_type=relation_type, source=source,
                            placement=placement, created_at=datetime.now().isoformat())
        if felt_state is not None:
            edge_kwargs["felt_state_at_creation"] = felt_state
        self.graph.add_edge(node_a, node_b, **edge_kwargs)
        self.graph.nodes[node_a]["last_reinforced"] = datetime.now()
        # No self.save() here (§4C) -- see store()'s comment. link() is
        # called on every hierarchy placement and every relational edge,
        # same Learning-time frequency problem.


    def dedupe_parallel_edges(self) -> int:
        """Collapse MultiDiGraph parallel edges that share relation_type."""
        g = self.graph
        removed = 0
        # snapshot edge keys
        pairs = list(g.edges(keys=True, data=True))
        seen = {}  # (u,v,rel) -> keep key
        to_remove = []
        for u, v, k, data in pairs:
            rel = data.get("relation_type", "")
            key = (u, v, rel)
            if key in seen:
                to_remove.append((u, v, k))
            else:
                seen[key] = k
        for u, v, k in to_remove:
            try:
                g.remove_edge(u, v, key=k)
                removed += 1
            except Exception:
                pass
        return removed

    def name_schema(self, schema_id: str, word: str) -> bool:
        """§2.1b item 4a: a Schema Node earns a name only if/when the
        agent's actual dictionary/user input happens to link a word to it
        -- never pre-assigned. Lives here (not reflector.py) because it's
        a direct graph mutation on data archivist.py already owns; calling
        out to reflector for a two-field write was an unnecessary
        cross-module hop that also never existed as a wired reference.
        Returns True if the write happened, False if the node wasn't an
        eligible unnamed schema."""
        if schema_id in self.graph and self.graph.nodes[schema_id].get("is_schema") \
                and not self.graph.nodes[schema_id].get("named", False):
            self.graph.nodes[schema_id]["name"] = word
            self.graph.nodes[schema_id]["named"] = True
            # No self.save() here (§4C) -- see store()'s comment.
            return True
        return False

    def flag_negation(self, node: str):
        """§3.4 mechanism 1: explicit negation/correction detected by
        sensory.py against a recently-active node. Demotion itself still
        only happens at Consolidation (one tier, gradual) -- this just
        records the flag."""
        if node in self.graph:
            self.graph.nodes[node]["negated_flag"] = True
            # No self.save() here (§4C) -- the flag is consumed at the
            # next Consolidation pass regardless; nothing durable is lost
            # by waiting for that pass to checkpoint it.

    # ------------------------------------------------------------------
    # §3 Trust scoring -- Consolidation-gated only. prometheus.py must
    # only call this from the Consolidation state.
    # ------------------------------------------------------------------
    def _trust_score(self, node: str) -> float:
        data = self.graph.nodes[node]
        base = SOURCE_WEIGHT.get(data.get("source", "user"), 0.3)  # source-weight dict itself not yet slider-exposed

        incident_sources = set()
        edge_count = 0
        for _u, _v, edata in list(self.graph.in_edges(node, data=True)) + list(self.graph.out_edges(node, data=True)):
            # §3.2/§10: only categorical edges (is-a/part-of/associated-with)
            # count as epistemic corroboration. Relational edges
            # (responsible-for/violates/temporal-contrast/concerns-other)
            # and composed-of edges represent recurrence-of-pattern or
            # structural composition, not independent confirmation of a
            # fact -- a node frequently on the receiving end of `violates`
            # edges was previously drifting toward Trusted purely for
            # showing up often in guilt-shaped schema patterns, which has
            # nothing to do with whether the node is true.
            if edata.get("relation_type") not in TRUST_BEARING_EDGE_TYPES:
                continue
            edge_count += 1
            esrc = edata.get("source", "user")
            if esrc != "self_generated":  # §2.2/§9 risk 5: excluded from diversity signal
                incident_sources.add(esrc)

        diversity = len(incident_sources)
        score = base + diversity * self.DIVERSITY_WEIGHT + min(edge_count, self.EDGE_COUNT_CAP) * self.EDGE_COUNT_WEIGHT
        return score

    def _tier_for_score(self, score: float) -> int:
        if score >= self.TRUSTED_THRESHOLD:
            return TIER_TRUSTED
        if score >= self.WORKING_THRESHOLD:
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
            if node not in self.graph:
                continue
            if node == SELF_NODE:
                continue  # permanent axiom, never re-evaluated (§2.1b item 1)
            data = self.graph.nodes[node]
            if (data.get("is_schema") or data.get("node_type") in (NODE_BASIN, NODE_EPISTEMIC_SCHEMA)):
                # Bug fix: Schema Nodes were falling through to the generic
                # trust-tier formula, using a "schema" source tag that isn't
                # in SOURCE_WEIGHT at all (silently defaulting to 0.3, the
                # same base as an unconfirmed user assertion). A Schema Node
                # represents a pattern validated by its OWN stabilization
                # mechanism (§2.1b's hysteresis-over-N-recurrences at
                # creation) -- subjecting it a second time to the
                # epistemic-fact trust system is a category error, and
                # concretely meant a schema's score could drift below the
                # promotion threshold (e.g. after a WORKING_THRESHOLD slider
                # adjustment) and get demoted, then eventually pruned --
                # silently erasing a validated emotional pattern through a
                # mechanism that was never meant to touch it. Same treatment
                # as SELF_NODE: exempt entirely, not just given a favorable
                # score.
                #
                # Basin nodes (this revision, ensure_basin_node) get the
                # identical exemption for the identical reason: they carry
                # no `tier` field at all (§6A -- "not applicable to
                # basin/schema/self node types"), so without this check
                # they'd default to TIER_PROVISIONAL here, start
                # accumulating tier0_cycles every pass (since they're
                # rarely touched by ordinary categorical corroboration),
                # and eventually get silently pruned by prune()'s
                # Tier-0-for-N-cycles rule -- deleting a graph node for a
                # basin that synthesizer.py still considers genuinely
                # stabilized, an inconsistency between the two.
                continue
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
                if data["_promo_streak"] >= self.PROMOTION_HYSTERESIS_N:
                    data["tier"] = min(current + 1, TIER_TRUSTED)  # one tier at a time
                    data["_promo_streak"] = 0
                    promotions += 1
            elif target < current:
                data["_demo_streak"] = data.get("_demo_streak", 0) + 1
                data["_promo_streak"] = 0
                if data["_demo_streak"] >= self.DEMOTION_HYSTERESIS_N:
                    data["tier"] = max(current - 1, TIER_PROVISIONAL)  # one tier at a time (§3.4)
                    data["_demo_streak"] = 0
                    demotions += 1
            else:
                data["_promo_streak"] = 0
                data["_demo_streak"] = 0

        # No self.save() here (§4C) -- this is one of several sub-steps
        # prometheus.py's _run_consolidation() runs; re-parenting and
        # schema detection happen after this and mutate the same graph,
        # so saving now would just be redone (or missed) depending on
        # ordering. The orchestrator checkpoints once, after everything.
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

    def categorical_out_degree(self, node: str) -> int:
        """Count of categorical (is-a/part-of/associated-with) out-edges
        from `node` -- the "how many children does this hierarchy parent
        already have" signal. Previously only existed as a local closure
        duplicated inside Prometheus.py's _select_self_study_target;
        centralized here, this revision, because association.place_node()
        now needs the identical check too -- see place_node()'s docstring
        for the cap-bypass bug this closes (dictionary-pattern-parsed
        placements were never capped at all, only the co-occurrence
        fallback was, even though self-study routes through the parsed
        path most of the time)."""
        if node not in self.graph:
            return 0
        return sum(
            1 for _u, _v, edata in self.graph.out_edges(node, data=True)
            if edata.get("relation_type") in TRUST_BEARING_EDGE_TYPES
        )

    def update_regulatory_efficacy(self, node: str, worked: bool, step: float = 0.05):
        """Called during Consolidation (§4.5) after checking whether felt-
        state intensity dropped faster than baseline decay alone would
        predict following a regulation attempt."""
        if node not in self.graph:
            return
        eff = self.graph.nodes[node].get("regulatory_efficacy", 0.5)
        eff = eff + step if worked else eff - step
        self.graph.nodes[node]["regulatory_efficacy"] = max(0.0, min(1.0, eff))
        # No self.save() here (§4C) -- called once per regulating node,
        # potentially several times per Consolidation pass; the
        # orchestrator's single end-of-consolidation checkpoint covers it.

    # ------------------------------------------------------------------
    # Activation / working memory (new, this revision -- §11 pull-forward,
    # v1-scoped: no archiving/rehydration, just a live per-node score).
    # ------------------------------------------------------------------
    def bump_activation(self, node: str, amount: Optional[float] = None):
        """Boosts a node's activation on touch (ingestion, self-study
        expansion, regulation anchor use). Capped so repeated rapid
        touches within one Consolidation window can't produce an
        ever-growing score -- the point is "currently relevant," not
        "historically most-touched ever," which decay_activation's
        per-pass shrinkage already handles on its own timescale."""
        if node not in self.graph:
            return
        amount = self.ACTIVATION_BOOST if amount is None else amount
        current = self.graph.nodes[node].get("activation", 0.0)
        self.graph.nodes[node]["activation"] = min(self.ACTIVATION_CAP, current + amount)
        # §13.4.4: a touch resets neglect tracking -- this is the "not
        # been directly reinforced" clock the collapse eligibility check
        # reads. Reset here rather than in a separate method so every
        # existing bump_activation() call site (real input, self-study,
        # regulation anchor use) automatically counts as reinforcement
        # without needing to be individually updated.
        self.graph.nodes[node]["neglect_cycles"] = 0
        # No self.save() here (§4C) -- touched constantly during Learning,
        # same frequency problem as store()/link(); checkpointed once at
        # end-of-Consolidation like everything else.

    def decay_activation(self):
        """Consolidation-gated (same clock as trust/efficacy/basins/
        schemas, per the design's own "one clock, not several" principle)
        -- activation shrinks toward zero for anything not recently
        touched, so working-memory membership reflects current relevance,
        not permanent historical importance. SELF is exempt (seeded at
        ACTIVATION_CAP, §2.1b item 1's axiom status extended here: it
        should never silently fall out of the focused Graph-tab view)."""
        for node, data in list(self.graph.nodes(data=True)):
            if node == SELF_NODE:
                continue
            data["activation"] = data.get("activation", 0.0) * self.ACTIVATION_DECAY_RATE
            # §13.4.4 neglect tracking: one more Consolidation pass has
            # elapsed without a fresh touch (bump_activation resets this
            # to 0 the moment one occurs). Same pass, same node loop as
            # activation decay -- one sweep, not a second one.
            data["neglect_cycles"] = data.get("neglect_cycles", 0) + 1

    # ------------------------------------------------------------------
    # Valence coloring (§13.2, new): mirror-neuron-style implicit
    # emotional coloring of nodes. NOT trust, NOT epistemic tier -- a
    # third, independent per-node property (same orthogonality pattern as
    # §2.4's trust-vs-depth and §4.5's trust-vs-regulatory-efficacy).
    # Deliberately never set directly from a word or category -- the only
    # way a node's coloring moves is repeated co-occurrence between "this
    # node was the active felt-state anchor" and "a reaction happened,"
    # via prometheus.py's give_parental_reaction(). No fixed valence
    # lookup table exists anywhere in this design; a node's coloring is
    # entirely a record of what it has actually been paired with.
    # ------------------------------------------------------------------
    def nudge_valence_coloring(self, node: str, delta: float, cap: float = 1.0):
        """Bounded accumulator, same clamping shape as regulatory_efficacy
        (§4.5) -- small deltas, cannot run away, symmetric positive/
        negative range since this represents a valence axis (like
        synthesizer.py's own -1..1 valence projection), not a magnitude."""
        if node not in self.graph:
            return
        current = self.graph.nodes[node].get("valence_coloring", 0.0)
        self.graph.nodes[node]["valence_coloring"] = max(-cap, min(cap, current + delta))
        # No self.save() here (§4C) -- same reasoning as bump_activation();
        # fires live, potentially every reaction, checkpointed once at the
        # next Consolidation pass like everything else.

    # ------------------------------------------------------------------
    # §13.4 Graph Collapse & Abstraction Layer -- new this revision.
    # Owner per the spec's own statement: archivist.py (graph mutation),
    # called from prometheus.py's _run_consolidation(). Distinct from
    # reflector.py's merge_duplicate_epistemic_schemas() (an unrelated
    # mechanism, deduplicating schema nodes -- renamed earlier this
    # session specifically to keep "collapse" unambiguous for this
    # section going forward).
    # ------------------------------------------------------------------

    _SCHEMA_NODE_TYPES = (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
    _PARENT_SEARCH_IN_EDGE_TYPES = frozenset({EDGE_IS_A, EDGE_PART_OF, EDGE_COMPOSED_OF, EDGE_INSTANCE_OF})
    _DEPTH_SEARCH_EDGE_TYPES = frozenset({
        EDGE_IS_A, EDGE_PART_OF, EDGE_COMPOSED_OF, EDGE_INSTANCE_OF,
        EDGE_ABSTRACTED_FROM, EDGE_ELABORATES,
    })

    def _is_schema_node(self, node: str) -> bool:
        data = self.graph.nodes.get(node, {})
        return bool(data.get("is_schema")) or data.get("node_type") in self._SCHEMA_NODE_TYPES

    def _find_collapse_parent(self, node: str) -> Optional[str]:
        """§13.4.4: "P is a stable parent of C via MEMBERSHIP/composed-of,
        a clear HIERARCHY chain, or an existing ABSTRACTION/abstracted-
        from record." Checks C's in-edges for is-a/part-of/composed-of/
        instance-of (P -> C, the existing parent->child edge direction
        convention throughout this codebase) first; falls back to C's
        own out-edges for abstracted-from (C -> P, per §13.4.4 step 5's
        own direction) -- a node re-collapsing after a prior rehydration
        already has this record and should return to the same parent
        rather than searching for a new one. Returns the first match;
        real data rarely has more than one candidate parent edge type
        present at once, and §13.4's own algorithm doesn't specify a
        tiebreak when it does -- documented here as a first-pass choice,
        not a claimed-final policy."""
        if node not in self.graph:
            return None
        for u, _v, edata in self.graph.in_edges(node, data=True):
            if edata.get("relation_type") in self._PARENT_SEARCH_IN_EDGE_TYPES:
                return u
        for _u, v, edata in self.graph.out_edges(node, data=True):
            if edata.get("relation_type") == EDGE_ABSTRACTED_FROM:
                return v
        return None

    def _depth_from_schema_root(self, node: str, max_search: Optional[int] = None) -> int:
        """BFS over HIERARCHY/MEMBERSHIP/ABSTRACTION edges (both
        directions -- depth is a distance, not a direction-sensitive
        query) to the nearest schema-tagged node. Capped at max_search
        hops (default D_COLLAPSE + 2) since anything beyond that already
        trivially satisfies the "depth >= D_COLLAPSE" neglect condition;
        no need to search further once that's already established, and
        capping keeps this cheap on a large graph even though it runs
        inside a per-node Consolidation loop."""
        if node not in self.graph:
            return 0
        max_search = (self.D_COLLAPSE + 2) if max_search is None else max_search
        if self._is_schema_node(node):
            return 0

        visited = {node}
        frontier = [node]
        depth = 0
        while frontier and depth < max_search:
            depth += 1
            next_frontier = []
            for n in frontier:
                neighbors = (
                    [u for u, _v, d in self.graph.in_edges(n, data=True)
                     if d.get("relation_type") in self._DEPTH_SEARCH_EDGE_TYPES]
                    + [v for _u, v, d in self.graph.out_edges(n, data=True)
                       if d.get("relation_type") in self._DEPTH_SEARCH_EDGE_TYPES]
                )
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    if self._is_schema_node(neighbor):
                        return depth
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
            frontier = next_frontier
        return max_search  # no schema root found within range -- treat as "far enough"

    def collapse_eligible(self, node: str, protected_nodes: Optional[set] = None) -> Optional[str]:
        """§13.4.4 eligibility check. Returns the parent to collapse into,
        or None if not eligible. `protected_nodes` is caller-supplied
        (§13.4.2/§13.4.14 item 1: the shared protection query unioning
        conversational anchors, narrative-linked nodes, Active Thread's
        focus, and any active Goal's target lives in prometheus.py, which
        has access to all those modules -- archivist.py stays graph-
        mutation-only and just takes the resulting set as a parameter,
        same pattern eligible_regulation_nodes() already uses for its own
        caller-supplied anchor list). SELF/OTHER are always protected
        regardless of what's passed, as a hard floor.

        Design choice, stated explicitly since the source spec left it
        ambiguous ("Optional: C's activation below a floor"): implemented
        here as an ADDITIONAL required condition (AND with the neglect
        check), not a third alternative. This is the more conservative
        reading -- something still meaningfully active shouldn't collapse
        just because a cycle counter ticked over, even if it also happens
        to sit deep in the hierarchy. Easy to loosen to OR later if this
        turns out to be too conservative once observed under real load."""
        protected_nodes = protected_nodes or set()
        if node in (SELF_NODE, OTHER_NODE) or node in protected_nodes:
            return None
        if node not in self.graph:
            return None
        if self._is_schema_node(node):
            return None

        parent = self._find_collapse_parent(node)
        if parent is None or parent not in self.graph:
            return None

        data = self.graph.nodes[node]
        neglected = (data.get("neglect_cycles", 0) >= self.N_COLLAPSE
                     or self._depth_from_schema_root(node) >= self.D_COLLAPSE)
        if not neglected:
            return None
        if data.get("activation", 0.0) > self.COLLAPSE_ACTIVATION_FLOOR:
            return None

        return parent

    def run_collapse_pass(self, protected_nodes: Optional[set] = None, current_pulse: int = 0) -> Dict[str, int]:
        """§13.4.4's main driver -- Consolidation-gated, called from
        prometheus.py's _run_consolidation() after schema detection
        (§13.4.10: "Collapse runs after schema detection so new schemas
        can claim members before leaves are absorbed"). Iterates eligible
        (node, parent) pairs and executes the rewire-then-remove
        algorithm. Returns a summary dict for logging, matching the
        existing convention (run_consolidation_pass() etc)."""
        protected_nodes = protected_nodes or set()
        candidates = []
        for node in list(self.graph.nodes):
            parent = self.collapse_eligible(node, protected_nodes)
            if parent:
                candidates.append((node, parent))

        collapsed = 0
        conflicts = 0
        for child, parent in candidates:
            # A node could have been removed already this pass if it was
            # itself absorbed as part of an earlier pair's rewiring (rare,
            # but possible if a chain collapses in one pass) -- re-check
            # both endpoints still exist before proceeding.
            if child not in self.graph or parent not in self.graph:
                continue
            result = self._collapse_node(child, parent, current_pulse)
            if result is not None:
                collapsed += 1
                conflicts += result
        return {"collapsed": collapsed, "conflicts": conflicts, "candidates_considered": len(candidates)}

    def _collapse_node(self, child: str, parent: str, current_pulse: int) -> Optional[int]:
        """Executes §13.4.4's algorithm steps 1-6 for one (child, parent)
        pair (step 7, ordinary Tier-0 prune, stays a separate existing
        call in prometheus.py's consolidation sequence -- not duplicated
        here). Returns the number of exclusive-family conflicts
        encountered (for the caller's summary), or None if the child
        turned out to no longer exist (defensive; shouldn't happen given
        run_collapse_pass's own re-check, kept here too since this method
        could in principle be called directly)."""
        if child not in self.graph or parent not in self.graph:
            return None
        graph = self.graph
        child_data = dict(graph.nodes[child])
        conflicts = 0

        # Steps 2-3: rewire every edge incident on C, except the edge(s)
        # that directly define the P-relationship itself (§13.4.4's own
        # "do not create P composed-of P; fold into membership summary"
        # rule, generalized to any family -- any edge that would become a
        # P->P self-loop after rewiring is dropped here, not carried
        # forward, regardless of which family it belongs to).
        out_edges = list(graph.out_edges(child, keys=True, data=True))
        in_edges = list(graph.in_edges(child, keys=True, data=True))

        for _u, v, _k, edata in out_edges:
            if v == parent:
                continue  # the defining C->P edge (e.g. abstracted-from from a prior collapse) -- dropped, not rewired
            conflicts += self._rewire_edge(child, parent, v, edata, direction="out")

        for u, _v, _k, edata in in_edges:
            if u == parent:
                continue  # the defining P->C edge (is-a/part-of/composed-of) -- dropped, not rewired
            conflicts += self._rewire_edge(child, parent, u, edata, direction="in")

        # Step 3 (scalar evidence transfer): activation transfers directly
        # (capped, same shape as bump_activation's own cap). Trust/
        # diversity signals are NOT separately transferred -- P's trust
        # score is recomputed live from its own incident edges at the
        # next trust pass (_trust_score()), and C's edges were just
        # rewired onto P above, so the corroboration those edges
        # represent is already reflected without a redundant manual step.
        # Regulatory efficacy: only nudged if C actually had regulation
        # history (efficacy != the never-used 0.5 default), and only by a
        # small fraction toward C's value -- a coping mechanism's track
        # record shouldn't overwrite the parent's own history wholesale.
        parent_data = graph.nodes[parent]
        parent_data["activation"] = min(
            self.ACTIVATION_CAP, parent_data.get("activation", 0.0) + child_data.get("activation", 0.0)
        )
        child_efficacy = child_data.get("regulatory_efficacy", 0.5)
        if child_efficacy != 0.5:
            parent_efficacy = parent_data.get("regulatory_efficacy", 0.5)
            parent_data["regulatory_efficacy"] = parent_efficacy + 0.15 * (child_efficacy - parent_efficacy)

        # Step 4: membership summary on P. Step 5 (ABSTRACTION edge): per
        # the spec's own alternative ("identity may be stored as an
        # attribute on P if node C is fully removed") -- since C IS fully
        # removed here (step 6 below), the absorbed-member record below
        # already fully captures the abstracted-from relationship; no
        # separate graph edge is created, since C won't exist to be its
        # source. Rehydration (§13.4.6) recreates C and writes the real
        # inverse edges at that point, when both nodes actually exist.
        absorbed = parent_data.setdefault("absorbed", [])
        absorbed.append({
            "id": child,
            "source": child_data.get("source", "user"),
            "absorbed_pulse": current_pulse,
            "absorbed_at": datetime.now().isoformat(),
            "node_data": {
                k: v for k, v in child_data.items()
                if k not in ("absorbed",) and not isinstance(v, (list, dict))
            },
            "primary_relations_summary": {
                "out_edge_count": len(out_edges),
                "in_edge_count": len(in_edges),
            },
        })

        # Step 6: remove C. MultiDiGraph.remove_node() cleans up any
        # remaining incident edges automatically -- everything meaningful
        # was already rewired above, so nothing is silently lost by this.
        graph.remove_node(child)
        return conflicts

    def _rewire_edge(self, child: str, parent: str, other: str, edata: dict, direction: str) -> int:
        """One edge's rewiring per §13.4.4's edge-rewiring table. `other`
        is the non-child endpoint; `direction` is "out" (child was the
        source: parent -> other after rewrite) or "in" (child was the
        target: other -> parent after rewrite). Returns 1 if an exclusive
        -family conflict was encountered and resolved, else 0.

        Conflict-resolution evidence heuristic (design choice, stated
        explicitly -- the source spec names the policy "keep higher-
        evidence choice" without specifying what counts as evidence):
        explicit placement beats co-occurrence placement; a genuine tie
        keeps whatever P already has, since P is the more-established
        node by construction (it's the collapse target, not the
        collapsed leaf). Flagged as a §13.4.9-category placeholder, not a
        claimed-final policy."""
        if other == parent:
            return 0  # would create a self-loop -- drop, not rewire (same rule as the defining edge above)
        if other not in self.graph:
            return 0  # dangling reference, nothing to rewire onto

        relation_type = edata.get("relation_type", EDGE_ASSOCIATED_WITH)
        family = get_family(relation_type, edata.get("family"))
        u, v = (parent, other) if direction == "out" else (other, parent)

        # Bug fix while implementing: rewired_from must be set on every
        # rewired edge, not just the rare exclusive-family-conflict case
        # -- rehydrate()'s "move back edges that are clearly child-
        # specific" step (§13.4.6 step 4) reads this to find candidates,
        # and without it on the common (non-conflict) path there would
        # never be anything for that step to find in the ordinary case.
        rewired_from = {"_original_child": child, "original_relation_type": relation_type}

        existing = None
        for _u2, _v2, _k2, ed2 in self.graph.edges(keys=True, data=True):
            if _u2 == u and _v2 == v and get_family(ed2.get("relation_type"), ed2.get("family")) == family:
                existing = ed2
                break

        conflict = 0
        if existing is not None:
            if existing.get("relation_type") == relation_type:
                # Same family, same choice -- merge/reinforce rather than
                # duplicate.
                existing["collapse_reinforced"] = existing.get("collapse_reinforced", 0) + 1
                existing["last_reinforced"] = datetime.now().isoformat()
                return 0
            if family in EXCLUSIVE_FAMILIES:
                # Same family, different choice, exclusive -- conflict.
                incoming_explicit = edata.get("placement") == "explicit"
                existing_explicit = existing.get("placement") == "explicit"
                if incoming_explicit and not existing_explicit:
                    # Incoming wins: demote the existing edge to RESIDUAL,
                    # add the incoming one at full strength.
                    existing["relation_type"] = EDGE_ASSOCIATED_WITH
                    existing["family"] = FAMILY_RESIDUAL
                    existing["conflict"] = True
                    self.graph.add_edge(u, v, relation_type=relation_type, family=family,
                                         source=edata.get("source", "user"), placement=edata.get("placement"),
                                         created_at=datetime.now().isoformat(), rewired_from=rewired_from)
                else:
                    # Existing wins (tie or existing already explicit):
                    # incoming gets added demoted to RESIDUAL instead of
                    # silently dropped -- §13.4.5's "never lost: that a
                    # relational fact existed."
                    self.graph.add_edge(u, v, relation_type=EDGE_ASSOCIATED_WITH, family=FAMILY_RESIDUAL,
                                         source=edata.get("source", "user"), placement=edata.get("placement"),
                                         created_at=datetime.now().isoformat(), conflict=True,
                                         rewired_from=rewired_from)
                conflict = 1
            # else: same family, different choice, but a multi-cardinality
            # family (MEMBERSHIP/CAUSAL/TEMPORAL/RESIDUAL) -- just add
            # below, no conflict; multiple choices are allowed to coexist.

        if existing is None or family not in EXCLUSIVE_FAMILIES:
            self.graph.add_edge(u, v, relation_type=relation_type, family=family,
                                 source=edata.get("source", "user"), placement=edata.get("placement"),
                                 created_at=datetime.now().isoformat(), rewired_from=rewired_from)
        return conflict

    def rehydrate(self, child_id: str, parent_id: str, edge_move_fraction: Optional[float] = None) -> bool:
        """§13.4.6. Recreates an absorbed child from its parent's
        membership record. Returns True on success, False if no matching
        absorbed record exists (not an error -- callers should treat this
        as "nothing to rehydrate," e.g. association.place_node() checking
        speculatively before creating a fresh node).

        `edge_move_fraction` (default REHYDRATE_EDGE_MOVE_FRACTION):
        fraction of P's *own* edges to consider moving back onto the
        rehydrated C, when an edge is clearly child-specific. v1
        implementation of "clearly child-specific" is deliberately
        conservative: only edges whose stored data still names this exact
        child (via `rewired_from`, written during collapse's own conflict
        -handling path above) are moved -- genuinely ambiguous edges stay
        on P, consistent with §13.4.6's own policy of preferring parent
        grain unless there's real evidence an edge belongs to the leaf."""
        if parent_id not in self.graph:
            return False
        parent_data = self.graph.nodes[parent_id]
        absorbed = parent_data.get("absorbed", [])
        record = next((r for r in absorbed if r.get("id") == child_id), None)
        if record is None:
            return False

        edge_move_fraction = self.REHYDRATE_EDGE_MOVE_FRACTION if edge_move_fraction is None else edge_move_fraction

        # Step 2: recreate C with restored provenance, tier capped at
        # Working unless the record's own data says otherwise (§13.4.6:
        # "restored provenance (tier <= Working unless evidence says
        # otherwise)").
        restored = dict(record.get("node_data", {}))
        restored["tier"] = min(restored.get("tier", TIER_WORKING), TIER_WORKING)
        restored["last_reinforced"] = datetime.now()
        restored["neglect_cycles"] = 0
        restored.setdefault("activation", 0.0)
        restored.setdefault("regulatory_efficacy", 0.5)
        restored.setdefault("valence_coloring", 0.0)
        self.graph.add_node(child_id, **restored)

        # Step 3: restore MEMBERSHIP.
        self.graph.add_edge(parent_id, child_id, relation_type=EDGE_COMPOSED_OF, family=FAMILY_MEMBERSHIP,
                             source="rehydration", placement="explicit",
                             created_at=datetime.now().isoformat())

        # Step 4: optionally move edges back that are clearly child-
        # specific (rewired_from data naming this child).
        if edge_move_fraction > 0:
            candidates = [
                (u, v, k, ed) for u, v, k, ed in self.graph.edges(keys=True, data=True)
                if (u == parent_id or v == parent_id) and isinstance(ed.get("rewired_from"), dict)
                and ed["rewired_from"].get("_original_child") == child_id
            ]
            move_count = int(len(candidates) * edge_move_fraction)
            for u, v, k, ed in candidates[:move_count]:
                new_u = child_id if u == parent_id else u
                new_v = child_id if v == parent_id else v
                self.graph.add_edge(new_u, new_v, **{kk: vv for kk, vv in ed.items() if kk != "rewired_from"})
                self.graph.remove_edge(u, v, key=k)

        # Step 5: ABSTRACTION inverse -- P elaborates C, now that both
        # nodes genuinely exist as real graph entities.
        self.graph.add_edge(parent_id, child_id, relation_type=EDGE_ELABORATES, family="ABSTRACTION",
                             source="rehydration", placement="explicit",
                             created_at=datetime.now().isoformat())

        # Step 6: leave P's summarised edges intact -- do NOT remove the
        # absorbed record; §13.4.6 is explicit that another collapse must
        # remain safe afterward, so the record needs to persist for that
        # future collapse to reuse. Only remove this one entry from the
        # list so it doesn't get double-processed by a future rehydration
        # call for the same child.
        parent_data["absorbed"] = [r for r in absorbed if r.get("id") != child_id]

        return True

    def rehydrate_for_parent(self, parent_id: str, max_children: int = 2) -> int:
        """Phase B: bring back a few absorbed children under an active parent.
        Preserves small-cortex defaults; only expands what focus needs.
        """
        if parent_id not in self.graph:
            return 0
        absorbed = list(self.graph.nodes[parent_id].get("absorbed") or [])
        if not absorbed:
            return 0
        # Prefer most recently absorbed
        def sort_key(r):
            return r.get("absorbed_pulse") or r.get("absorbed_at") or ""
        absorbed_sorted = sorted(absorbed, key=sort_key, reverse=True)
        n = 0
        for rec in absorbed_sorted[: max(0, int(max_children))]:
            cid = rec.get("id")
            if not cid:
                continue
            if cid in self.graph:
                continue
            if self.rehydrate(cid, parent_id):
                n += 1
        return n


    def working_memory_nodes(self, top_k: int = 40, always_include: Optional[List[str]] = None,
                              max_relational_neighbors: int = 20) -> set:
        """Returns the set of node ids that should count as 'in focus' --
        the top_k highest-activation nodes, plus SELF/OTHER (always
        relevant regardless of activation), plus SELF/OTHER's relational-
        edge neighbors, plus any caller-supplied always_include set (e.g.
        prometheus.py's current felt-state anchors, which the archivist
        has no way to know about on its own since felt state lives in
        synthesizer.py).

        Bug fix, this revision: SELF/OTHER were always force-included
        here, but their relational-edge *neighbors* were not -- and
        prometheus_dashboard.py's render_graph_html() only draws an edge
        when BOTH endpoints are in the rendered subset. Since a
        relational-edge event node is typically a one-off full-sentence
        node (never a WordNet dictionary word, so self-study essentially
        never re-selects it), its activation decays to near-zero quickly
        after creation and it soon falls out of the top-K set at any real
        scale. The result: SELF renders as a lonely, disconnected node on
        the Graph tab even when it has real, confirmed edges (checkable
        via reflector.self_other_report(), which reads the graph directly
        and was never affected by this) -- reading as "SELF never grows"
        when the actual problem was purely a rendering blind spot, not a
        data one. Bounded to the most-recent `max_relational_neighbors`
        per anchor (not unconditionally all of them), consistent with
        every other "bounded, not unbounded" structure in this design
        (chronos's log, felt_state_anchors's window, §4C/Addendum 6).

        Used for two things (§11 pull-forward): (a) prometheus.py's
        self-study target selection can prefer this set over the full
        eligible pool, giving self-study something closer to genuine
        attention/focus instead of near-uniform randomness; (b)
        prometheus_dashboard.py's Graph-tab rendering can filter to just
        this neighborhood instead of the entire live graph, which is the
        actual fix for rendering cost/readability at scale that §11
        originally proposed this mechanism for.
        """
        ranked = sorted(
            self.graph.nodes(data=True),
            key=lambda item: item[1].get("activation", 0.0),
            reverse=True,
        )
        result = {n for n, _d in ranked[:top_k]}
        result.add(SELF_NODE)
        result.add(OTHER_NODE)

        for anchor_node in (SELF_NODE, OTHER_NODE):
            if anchor_node not in self.graph:
                continue
            neighbor_edges = (
                list(self.graph.out_edges(anchor_node, data=True))
                + list(self.graph.in_edges(anchor_node, data=True))
            )
            neighbor_edges.sort(key=lambda e: e[2].get("created_at", ""), reverse=True)
            for u, v, _d in neighbor_edges[:max_relational_neighbors]:
                result.add(v if u == anchor_node else u)

        if always_include:
            result.update(n for n in always_include if n in self.graph)
        return result

    # ------------------------------------------------------------------
    # Co-activation (§13.3, new) -- the detection signal epistemic schema
    # clustering depends on. Kept deliberately separate from `activation`
    # (a per-node score): this is per-PAIR, tracking which nodes get
    # touched together, not just how often each is touched alone.
    # ------------------------------------------------------------------
    def record_co_activation(self, nodes: List[str], weight: float = 1.0):
        """Bumps co-activation for pairs within `nodes`.

        weight < 1: gated/weak evidence (off-basin, parent closed)
        weight > 1: pedagogical / lesson boost
        """
        real_nodes = [n for n in nodes if n in self.graph]
        if len(real_nodes) < 2:
            return
        unique = sorted(set(real_nodes))
        w = float(weight) if weight is not None else 1.0
        if w <= 0:
            return
        for a, b in combinations(unique, 2):
            self.co_activation[(a, b)] = self.co_activation.get((a, b), 0) + w
        self.record_schema_co_activation(unique, amount=w)



    def merge_case_variant_lemmas(self) -> int:
        """Collapse Color/color (and similar) into one lemma node.

        Prefer the variant with higher activation, user source, or Title case
        if tied. Rewire edges from loser → winner.
        """
        graph = self.graph
        by_low = {}
        for n, d in list(graph.nodes(data=True)):
            if d.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA, NODE_BASIN):
                continue
            if d.get("is_schema"):
                continue
            key = str(n).casefold().strip()
            if not key or " " in key and len(key) > 24:
                continue
            by_low.setdefault(key, []).append(n)
        merged = 0
        for key, ids in by_low.items():
            if len(ids) < 2:
                continue
            def score(nid):
                d = graph.nodes.get(nid, {})
                return (
                    10.0 if d.get("source") == "user" else 0.0,
                    float(d.get("activation") or 0.0),
                    1.0 if str(nid)[:1].isupper() else 0.0,
                    -len(str(nid)),
                )
            ids_sorted = sorted(ids, key=score, reverse=True)
            winner = ids_sorted[0]
            for loser in ids_sorted[1:]:
                if loser not in graph or winner not in graph:
                    continue
                # rewire edges
                for u, v, k, ed in list(graph.in_edges(loser, keys=True, data=True)):
                    if u == winner:
                        continue
                    try:
                        graph.add_edge(u, winner, key=k, **dict(ed))
                    except Exception:
                        pass
                for u, v, k, ed in list(graph.out_edges(loser, keys=True, data=True)):
                    if v == winner:
                        continue
                    try:
                        graph.add_edge(winner, v, key=k, **dict(ed))
                    except Exception:
                        pass
                # copy useful attrs
                ld = graph.nodes[loser]
                wd = graph.nodes[winner]
                if ld.get("source") == "user":
                    wd["source"] = "user"
                if ld.get("pedagogical"):
                    wd["pedagogical"] = True
                wd["activation"] = max(float(wd.get("activation") or 0), float(ld.get("activation") or 0))
                try:
                    graph.remove_node(loser)
                    merged += 1
                except Exception:
                    pass
        return merged

    def kind_family(self, node_id: str) -> set:
        """Lemma + kind-schema ids that name the same concept (Color ↔ epistemic_of_Color).

        Used so phase windows, focus, and goals treat them as one open target.
        """
        if not node_id or node_id not in self.graph:
            return set()
        out = {node_id}
        d = self.graph.nodes.get(node_id, {})
        nt = d.get("node_type")
        name = str(d.get("name") or d.get("dominant_parent") or "").strip()
        low_name = name.casefold() if name else ""
        low_id = str(node_id).casefold()

        def is_schema(nid: str) -> bool:
            nd = self.graph.nodes.get(nid, {})
            return bool(
                nd.get("is_schema")
                or nd.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
            )

        # Schema → lemma(s)
        if is_schema(node_id):
            lemma_candidates = []
            if name:
                lemma_candidates.append(name)
            if str(node_id).startswith("epistemic_of_"):
                lemma_candidates.append(str(node_id)[len("epistemic_of_"):].replace("_", " "))
            if d.get("dominant_parent"):
                lemma_candidates.append(str(d.get("dominant_parent")))
            for lem in lemma_candidates:
                if not lem:
                    continue
                if lem in self.graph:
                    out.add(lem)
                ll = lem.casefold()
                for n in self.graph.nodes:
                    if str(n).casefold() == ll:
                        out.add(n)
            # sibling schemas with same name/parent
            for n, nd in self.graph.nodes(data=True):
                if not is_schema(n):
                    continue
                sn = str(nd.get("name") or nd.get("dominant_parent") or "")
                if low_name and sn.casefold() == low_name:
                    out.add(n)
                if str(n).startswith("epistemic_of_"):
                    tail = str(n)[len("epistemic_of_"):].replace("_", " ").casefold()
                    if low_name and tail == low_name:
                        out.add(n)
                    if low_id.startswith("epistemic_of_") and tail == low_id[len("epistemic_of_"):].replace("_", " "):
                        out.add(n)
        else:
            # Lemma → schemas that cover it or are named after it
            for sid in self.schemas_covering(node_id):
                out.add(sid)
            for n, nd in self.graph.nodes(data=True):
                if not is_schema(n):
                    continue
                sn = str(nd.get("name") or nd.get("dominant_parent") or "")
                if sn.casefold() == low_id:
                    out.add(n)
                if str(n).startswith("epistemic_of_"):
                    tail = str(n)[len("epistemic_of_"):].replace("_", " ").casefold()
                    if tail == low_id:
                        out.add(n)
        return out

    def schemas_covering(self, node_id: str) -> List[str]:
        """Return schema node ids that claim `node_id` via composed-of (in-edge)."""
        if node_id not in self.graph:
            return []
        out = []
        for u, _v, ed in self.graph.in_edges(node_id, data=True):
            if ed.get("relation_type") != EDGE_COMPOSED_OF:
                continue
            nd = self.graph.nodes.get(u, {})
            if nd.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or nd.get("is_schema"):
                out.append(u)
        return out

    def record_schema_co_activation(self, leaf_nodes: List[str], amount: float = 1.0) -> int:
        """When leaf nodes co-occur, bump co-activation between the schemas
        that cover them. Returns number of schema pairs bumped.

        Pure structural lift — no new clustering library, no LLM. Gives
        detect_epistemic_tier2() stabilized pairs among schema nodes so
        hierarchical stacking is not starved.
        """
        schema_set = set()
        for leaf in leaf_nodes:
            for sid in self.schemas_covering(leaf):
                schema_set.add(sid)
        # Also treat any schema ids already in the leaf list as schemas
        for n in leaf_nodes:
            nd = self.graph.nodes.get(n, {})
            if nd.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA) or nd.get("is_schema"):
                schema_set.add(n)
        schemas = sorted(schema_set)
        if len(schemas) < 2:
            return 0
        pairs = 0
        for a, b in combinations(schemas, 2):
            key = (a, b) if a < b else (b, a)
            self.co_activation[key] = self.co_activation.get(key, 0) + amount
            pairs += 1
        return pairs

    def decay_co_activation(self):
        """Consolidation-gated (same clock as activation decay, basin
        decay, trust evaluation -- "one clock, not several"). Shrinks
        every tracked pair's count, and drops pairs that fall below
        CO_ACTIVATION_PRUNE_FLOOR entirely -- same shape as synthesizer.
        py's basin dwell-time decay (DESTABILIZATION_FLOOR), so a pair
        that stops co-occurring genuinely fades rather than accumulating
        forever. Also silently drops any pair referencing a node that no
        longer exists (pruned since it was recorded) -- keeps this
        structure from quietly outliving the nodes it describes."""
        for pair in list(self.co_activation.keys()):
            a, b = pair
            if a not in self.graph or b not in self.graph:
                del self.co_activation[pair]
                continue
            self.co_activation[pair] *= self.CO_ACTIVATION_DECAY_RATE
            if self.co_activation[pair] < self.CO_ACTIVATION_PRUNE_FLOOR:
                del self.co_activation[pair]

    def stabilized_co_activation_pairs(self) -> List[tuple]:
        """Pairs whose co-activation count has crossed the stabilization
        threshold -- the raw material reflector.detect_epistemic_clusters()
        groups into connected components. Exposed as its own method (not
        inlined into the detector) so it's independently diagnosable, same
        "make it checkable" pattern as every other new mechanism this
        session (activation_report, valence_coloring_report, etc.)."""
        return [
            pair for pair, count in self.co_activation.items()
            if count >= self.CO_ACTIVATION_STABILIZATION_THRESHOLD
        ]

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
        for node in list(self.graph.nodes):
            if node == SELF_NODE:
                continue
            in_edges = list(self.graph.in_edges(node, data=True))
            cooccurrence_parent = any(
                d.get("placement") == "cooccurrence" and d.get("relation_type") == EDGE_ASSOCIATED_WITH
                for _u, _v, d in in_edges
            )
            if not cooccurrence_parent:
                continue
            sources = {d.get("source", "user") for _u, _v, d in in_edges if d.get("source") != "self_generated"}
            if len(sources) >= min_corroboration:
                candidates.append(node)
        return candidates

    def reparent(self, node: str, new_parent: str, relation_type: str = EDGE_IS_A):
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
        # Nodes in stabilized co-activation pairs are schema fuel — keep them.
        protected = set()
        try:
            for a, b in self.stabilized_co_activation_pairs():
                protected.add(a)
                protected.add(b)
        except Exception:
            pass
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if n != SELF_NODE
            and n != OTHER_NODE
            and n not in protected
            and not d.get("is_schema")
            and not d.get("is_other")
            and d.get("node_type") not in (NODE_BASIN, NODE_EPISTEMIC_SCHEMA, NODE_SCHEMA)
            and d.get("tier", TIER_PROVISIONAL) == TIER_PROVISIONAL
            and d.get("tier0_cycles", 0) >= self.PRUNE_TIER0_CYCLES
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
        """Persist graph to disk. Uses a graph *copy* so NetworkX
        node_link_data never iterates a structure mid-mutation (semi-live
        / consolidation interleaving caused RuntimeError: dictionary
        changed size during iteration)."""
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            # Snapshot — do not serialize the live MultiDiGraph in place
            g = self.graph.copy()
            data = nx.readwrite.json_graph.node_link_data(g)
            with open(EPISTEMIC_GRAPH_PATH, "w") as f:
                json.dump(data, f, default=str)
            self._save_co_activation()
        except RuntimeError as e:
            # Rare race: retry one snapshot
            logger.warning("ArchivistModule.save RuntimeError (%s); retrying once", e)
            try:
                g = self.graph.copy()
                data = nx.readwrite.json_graph.node_link_data(g)
                with open(EPISTEMIC_GRAPH_PATH, "w") as f:
                    json.dump(data, f, default=str)
                self._save_co_activation()
            except Exception as e2:
                logger.warning("ArchivistModule.save retry failed: %s", e2)
        except OSError as e:
            logger.warning("ArchivistModule.save failed: %s", e)

    def _save_co_activation(self):
        """Separate file, same pattern as synthesizer.py's basin_state.json
        -- tuple keys aren't valid JSON object keys, encoded as a
        delimited string (same technique, not a new one)."""
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            encoded = {f"{a}|||{b}": count for (a, b), count in self.co_activation.items()}
            with open(CO_ACTIVATION_PATH, "w") as f:
                json.dump(encoded, f)
        except OSError as e:
            logger.warning("ArchivistModule._save_co_activation failed: %s", e)

    def load(self):
        if os.path.exists(EPISTEMIC_GRAPH_PATH):
            try:
                with open(EPISTEMIC_GRAPH_PATH, "r") as f:
                    data = json.load(f)
                self.graph = nx.readwrite.json_graph.node_link_graph(data)
                self._deserialize_timestamps()
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                logger.warning(
                    "ArchivistModule.load failed (%s); starting with an empty graph instead of crashing.",
                    e,
                )
                self.graph = nx.MultiDiGraph()
        if os.path.exists(CO_ACTIVATION_PATH):
            try:
                with open(CO_ACTIVATION_PATH, "r") as f:
                    encoded = json.load(f)
                for key_str, count in encoded.items():
                    a, b = key_str.split("|||", 1)
                    self.co_activation[(a, b)] = count
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning(
                    "ArchivistModule.load co_activation failed (%s); starting fresh.", e,
                )

    def _deserialize_timestamps(self):
        """save() writes `last_reinforced` via json.dump's `default=str`,
        which serializes datetime objects to ISO strings -- but
        node_link_graph() never converts them back on load. Every node
        created *this session* gets a real datetime.now() (store()/
        link()), so a loaded graph ends up with a mix of string and
        datetime timestamps on the same field, and any comparison across
        both (e.g. _most_active_node()'s sort, retrieve()'s sort) raises
        `TypeError: '<' not supported between instances of 'str' and
        'datetime.datetime'`. Fixing this once here, at the load boundary,
        means every downstream consumer can assume a real datetime rather
        than defensively handling both types at every comparison site."""
        for _n, data in self.graph.nodes(data=True):
            lr = data.get("last_reinforced")
            if isinstance(lr, str):
                try:
                    data["last_reinforced"] = datetime.fromisoformat(lr)
                except ValueError:
                    data["last_reinforced"] = datetime.min

    def retrieve(self, key: str, bias: str = None):
        """Minimal existing behavior preserved; returns nodes matching key
        as a substring, most-recently-reinforced first."""
        matches = [n for n in self.graph.nodes if key.lower() in str(n).lower()]
        matches.sort(
            key=lambda n: self.graph.nodes[n].get("last_reinforced", datetime.min),
            reverse=True,
        )
        return matches
