import hashlib
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx

from .archivist import SELF_NODE, TIER_PROVISIONAL, TIER_WORKING, TIER_TRUSTED
from .edge_types import (
    RELATIONAL_EDGE_TYPES, EDGE_COMPOSED_OF, EDGE_IS_A, EDGE_PART_OF, EDGE_ASSOCIATED_WITH,
    NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA, NODE_BASIN,
    get_family, FAMILY_HIERARCHY, FAMILY_MEMBERSHIP, FAMILY_CAUSAL, FAMILY_SOCIAL_NORM,
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
EPISTEMIC_MIN_CLUSTER_SIZE = 4
EPISTEMIC_NAME_MIN_COVERAGE = 2  # how many cluster members a parsed is-a parent must cover before it's recognized as earning the cluster's name (§13.3.1)

# Schema quality gates (repair pass — sense before promotion)
EPISTEMIC_MIN_COHERENCE = 0.28          # mean pairwise token/hypernym overlap
EPISTEMIC_MIN_LEMMA_RATIO = 0.55         # fraction of members that look like lemmas not sentences
EPISTEMIC_NAME_MIN_FREQ = 1             # times a candidate label appears as member-ish
EPISTEMIC_NAME_MIN_CONTEXTS = 3         # distinct sources or basins before naming
EPISTEMIC_UNNAMED_MAX_CYCLES = 5
# Taxonomic affinity: co-activation alone is not enough for kind-schemas.
AFFINITY_THRESHOLD = 2.5          # min weighted score to keep a pair in cluster graph
AFFINITY_SHARED_PARENT = 1.5      # bonus if nodes share an is-a parent
AFFINITY_DIRECT_IS_A = 2.0        # one is-a the other
AFFINITY_PART_OF = 1.2
AFFINITY_ASSOCIATED = 0.25        # bare co-occurrence placement — weak
AFFINITY_CAUSAL = 0.4             # thematic, not kind
AFFINITY_SOCIAL = 0.15
AFFINITY_COACT_SCALE = 0.35       # per stabilized co-activation count unit
        # consolidations unnamed+stagnant → dissolve wrapper

# §13 naming hygiene: graph node *ids* must stay short/stable; human-readable
# glosses live on attributes (name / definition), not in the id string.
_MAX_ID_FRAGMENT_LEN = 40
_MAX_DISPLAY_NAME_LEN = 80


def _slug_id_fragment(label: str) -> str:
    """Turn an arbitrary parent label into a safe, short id fragment.
    Long WordNet glosses become a short slug + hash so ids stay navigable.
    """
    import re
    raw = (label or "").strip()
    if not raw:
        return "unknown"
    # Prefer already-short lemma-like labels
    if len(raw) <= _MAX_ID_FRAGMENT_LEN and " " not in raw and raw.isascii():
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", raw)
        return safe[:_MAX_ID_FRAGMENT_LEN] or "unknown"
    # Multi-word / gloss: take first 3 meaningful tokens + short hash
    stop = {
        "a", "an", "the", "of", "or", "and", "to", "in", "on", "for", "with",
        "as", "by", "from", "that", "which", "who", "is", "are", "was", "were",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", raw.lower())
    keep = [tok for tok in tokens if tok not in stop][:3]
    slug = "_".join(keep) if keep else "gloss"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:6]
    frag = f"{slug}_{digest}"
    return frag[:_MAX_ID_FRAGMENT_LEN]


def _display_name(label: str) -> str:
    """Human-facing name: keep short labels; truncate long glosses."""
    raw = (label or "").strip()
    if len(raw) <= _MAX_DISPLAY_NAME_LEN:
        return raw
    return raw[: _MAX_DISPLAY_NAME_LEN - 1].rstrip() + "…"




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
        self.EPISTEMIC_MIN_COHERENCE = EPISTEMIC_MIN_COHERENCE
        self.EPISTEMIC_MIN_LEMMA_RATIO = EPISTEMIC_MIN_LEMMA_RATIO
        self.EPISTEMIC_NAME_MIN_FREQ = EPISTEMIC_NAME_MIN_FREQ
        self.EPISTEMIC_NAME_MIN_CONTEXTS = EPISTEMIC_NAME_MIN_CONTEXTS
        self.EPISTEMIC_UNNAMED_MAX_CYCLES = EPISTEMIC_UNNAMED_MAX_CYCLES
        self.AFFINITY_THRESHOLD = AFFINITY_THRESHOLD
        self.AFFINITY_SHARED_PARENT = AFFINITY_SHARED_PARENT
        self.AFFINITY_DIRECT_IS_A = AFFINITY_DIRECT_IS_A
        self.AFFINITY_PART_OF = AFFINITY_PART_OF
        self.AFFINITY_ASSOCIATED = AFFINITY_ASSOCIATED
        self.AFFINITY_CAUSAL = AFFINITY_CAUSAL
        self.AFFINITY_SOCIAL = AFFINITY_SOCIAL
        self.AFFINITY_COACT_SCALE = AFFINITY_COACT_SCALE

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

        Deliberately NOT tier-gated to Working/Trusted event nodes, unlike
        detect_epistemic_clusters() (§14's "Option B" agreement). This is
        a considered exception, not an inconsistency: an event node here
        is typically a single raw sentence (association.place_node() uses
        the whole message as the node name, §2.2), which structurally
        almost never accumulates the diverse corroboration §3.2's trust
        formula requires to promote past Provisional -- nothing else ever
        asserts "I shouldn't have done that" is a fact needing
        confirmation. Requiring Working+ tier here would very likely make
        somatic schema formation impossible in practice, not just rarer,
        defeating the mechanism rather than refining it. Somatic
        formation's existing bar -- a stabilized felt state AND a
        *consistent*, recurring relational-edge pattern -- is already a
        real, earned requirement of a different kind than epistemic trust,
        and is left as the sole gate here.
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
                basin=felt_state,  # felt place label when known — not a hormone
                relation_types=sorted(relation_set),
                somatic=True,  # Phase B: experience schema, not WordNet cluster
                activation=0.5,
            )
            # Members = event nodes that earned this pattern (experience)
            for en in event_nodes:
                if en in graph:
                    graph.add_edge(
                        schema_id, en, relation_type=EDGE_COMPOSED_OF,
                        source="schema", placement="explicit",
                        created_at=datetime.now().isoformat(),
                    )
            # Soft link to SELF: this is about the agent's lived relations
            if SELF_NODE in graph:
                graph.add_edge(
                    SELF_NODE, schema_id, relation_type="associated-with",
                    source="schema", placement="somatic",
                    created_at=datetime.now().isoformat(),
                )
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

    # ------------------------------------------------------------------
    # Schema quality: coherence, lemma filter, delayed naming, expire
    # ------------------------------------------------------------------
    @staticmethod
    def _is_lemma_like(label: str) -> bool:
        """True for short concept-like labels; false for sentences/glosses."""
        if not label or label in (SELF_NODE,):
            return False
        s = str(label).strip()
        if len(s) > 28:
            return False
        words = s.split()
        if len(words) > 2:
            return False
        low = s.lower()
        if low.startswith(("i ", "i'", "it was", "it is", "she ", "he ", "they ",
                            "we ", "you ", "this ", "that ", "there ", "because ",
                            "when ", "what ", "how ", "why ")):
            return False
        if s.startswith("epistemic_") or s.startswith("schema_") or s.startswith("felt_"):
            return False
        # WordNet gloss fingerprints — never human schema titles
        gloss_markers = (
            "consisting of", "characterized by", "used to", "a person who",
            "the act of", "the state of", "or pair of", "or distance",
            "making the", "infusion of", "unbroken expanse", "rolled steel",
            "ground coffee", "parallel bars",
        )
        if any(m in low for m in gloss_markers):
            return False
        # Dictionary style: "X of Y" long forms
        if len(words) >= 3 and words[1] in ("of", "or", "and", "for"):
            return False
        return True

    @staticmethod
    def _tokens(label: str) -> set:
        stop = {
            "a", "an", "the", "of", "or", "and", "to", "in", "on", "for", "with",
            "as", "by", "from", "that", "which", "who", "is", "are", "was", "were",
            "my", "me", "i",
        }
        return {t for t in re.findall(r"[a-z0-9]+", str(label).lower()) if t not in stop and len(t) > 1}

    def _pair_similarity(self, a: str, b: str) -> float:
        """Token Jaccard; optional WordNet hypernym bonus if available."""
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        jacc = inter / union if union else 0.0
        # Light WordNet hypernym overlap on first token
        try:
            from nltk.corpus import wordnet as wn
            wa = wn.synsets(next(iter(ta)))
            wb = wn.synsets(next(iter(tb)))
            if wa and wb:
                ha = set(wa[0].closure(lambda s: s.hypernyms()))
                hb = set(wb[0].closure(lambda s: s.hypernyms()))
                # include self
                ha.add(wa[0]); hb.add(wb[0])
                if ha & hb:
                    jacc = min(1.0, jacc + 0.25)
        except Exception:
            pass
        return jacc

    def cluster_coherence(self, members: List[str]) -> float:
        """Mean pairwise similarity over lemma-like members (fallback: all)."""
        core = [m for m in members if self._is_lemma_like(m)]
        if len(core) < 2:
            core = list(members)
        if len(core) < 2:
            return 0.0
        total = 0.0
        n = 0
        for i in range(len(core)):
            for j in range(i + 1, len(core)):
                total += self._pair_similarity(core[i], core[j])
                n += 1
        return total / n if n else 0.0

    def lemma_ratio(self, members: List[str]) -> float:
        if not members:
            return 0.0
        return sum(1 for m in members if self._is_lemma_like(m)) / len(members)

    def _member_context_diversity(self, members: List[str]) -> int:
        """Count distinct sources and felt_state_at_creation on incident edges."""
        graph = self.archivist.graph
        contexts = set()
        for m in members:
            if m not in graph:
                continue
            data = graph.nodes.get(m, {})
            src = data.get("source")
            if src:
                contexts.add(f"src:{src}")
            for _u, _v, ed in list(graph.in_edges(m, data=True)) + list(graph.out_edges(m, data=True)):
                if ed.get("source"):
                    contexts.add(f"esrc:{ed.get('source')}")
                if ed.get("felt_state_at_creation"):
                    contexts.add(f"felt:{ed.get('felt_state_at_creation')}")
        return len(contexts)

    def _name_candidate(self, members: List[str]) -> Optional[str]:
        """Best lemma-like member to use as human name, or None."""
        freq = Counter(m for m in members if self._is_lemma_like(m))
        if not freq:
            return None
        ranked = sorted(freq.items(), key=lambda t: (-t[1], len(str(t[0]))))
        for term, count in ranked:
            if count >= self.EPISTEMIC_NAME_MIN_FREQ:
                return str(term)
        return None

    def try_name_epistemic_schemas(self) -> int:
        """Delayed naming pass: only name when coherence + diversity + candidate."""
        # Always collapse same-name duplicates first (self-heal lab graphs)
        self.merge_schemas_sharing_name()
        graph = self.archivist.graph
        named = 0
        for node, data in list(graph.nodes(data=True)):
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            if data.get("named"):
                continue
            members = [
                v for _u, v, ed in graph.out_edges(node, data=True)
                if ed.get("relation_type") == EDGE_COMPOSED_OF
            ]
            if len(members) < self.EPISTEMIC_MIN_CLUSTER_SIZE:
                continue
            coh = self.cluster_coherence(members)
            if coh < self.EPISTEMIC_MIN_COHERENCE:
                continue
            if self._member_context_diversity(members) < self.EPISTEMIC_NAME_MIN_CONTEXTS:
                continue
            candidate = self._name_candidate(members)
            if not candidate:
                continue
            # One human name → one epistemic schema
            taken_by = None
            want = str(candidate).strip().casefold()
            for other, odata in list(graph.nodes(data=True)):
                if other == node:
                    continue
                if odata.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                    continue
                if not odata.get("named"):
                    continue
                oname = odata.get("name")
                if oname and str(oname).strip().casefold() == want:
                    taken_by = other
                    break
            if taken_by is not None:
                data["name_blocked_by"] = taken_by
                continue
            data["name"] = candidate
            data["named"] = True
            data["unnamed_cycles"] = 0
            named += 1
        # Scrub legacy gloss titles that slipped through earlier rules
        self.scrub_invalid_schema_names()
        self.merge_schemas_sharing_name()
        return named

    def merge_schemas_sharing_name(self) -> int:
        """Collapse multiple schema nodes that share the same display name.

        Matches on normalized name even if named flag is inconsistent.
        Rewires composed-of members to the richest survivor, then removes
        the duplicates. Safe to call often (no-op when no dups).
        """
        graph = self.archivist.graph
        schema_types = {NODE_EPISTEMIC_SCHEMA, NODE_SCHEMA, "epistemic_schema", "schema"}
        by_name = {}
        for node, data in list(graph.nodes(data=True)):
            ntype = data.get("node_type")
            if ntype not in schema_types and not (
                str(node).startswith("epistemic_") or str(node).startswith("schema_")
            ):
                continue
            name = data.get("name")
            if not name or not str(name).strip():
                continue
            key = str(name).strip().casefold()
            by_name.setdefault(key, []).append(node)

        merged = 0
        for _key, nodes in by_name.items():
            # unique ids only
            nodes = list(dict.fromkeys(nodes))
            if len(nodes) < 2:
                continue

            def richness(nid):
                d = graph.nodes.get(nid, {})
                members = 0
                try:
                    members = sum(
                        1 for _u, _v, ed in graph.out_edges(nid, data=True)
                        if ed.get("relation_type") == EDGE_COMPOSED_OF
                    )
                except Exception:
                    pass
                return (
                    members,
                    float(d.get("activation", 0) or 0),
                    1 if d.get("named") else 0,
                )

            nodes_sorted = sorted(nodes, key=richness, reverse=True)
            keep = nodes_sorted[0]
            for drop in nodes_sorted[1:]:
                if drop not in graph or keep not in graph:
                    continue
                # Move membership edges
                try:
                    for _u, v, ed in list(graph.out_edges(drop, data=True)):
                        if ed.get("relation_type") != EDGE_COMPOSED_OF:
                            continue
                        already = any(
                            vv == v and eed.get("relation_type") == EDGE_COMPOSED_OF
                            for _uu, vv, eed in graph.out_edges(keep, data=True)
                        )
                        if not already and v in graph:
                            graph.add_edge(
                                keep, v, relation_type=EDGE_COMPOSED_OF,
                                source="schema", placement="merge",
                            )
                    # Re-point any edges that targeted the drop node
                    for u, _v, ed in list(graph.in_edges(drop, data=True)):
                        if u == keep:
                            continue
                        et = ed.get("relation_type", "associated-with")
                        if not graph.has_edge(u, keep):
                            graph.add_edge(u, keep, **{k: v for k, v in ed.items()})
                except Exception:
                    pass
                try:
                    if drop in graph:
                        graph.remove_node(drop)
                        merged += 1
                except Exception:
                    pass
        return merged


    def scrub_invalid_schema_names(self) -> int:
        """Un-name epistemic schemas whose title fails lemma-like checks.
        Gloss stays in definition if missing."""
        graph = self.archivist.graph
        scrubbed = 0
        for node, data in list(graph.nodes(data=True)):
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            if not data.get("named"):
                continue
            name = data.get("name")
            if name and self._is_lemma_like(str(name)):
                continue
            # Demote to unnamed; preserve text as definition
            if name and not data.get("definition"):
                data["definition"] = name
            data["name"] = None
            data["named"] = False
            scrubbed += 1
        return scrubbed

    def expire_unnamed_epistemic_schemas(self) -> int:
        """Dissolve stagnant unnamed schema wrappers; members and their edges remain."""
        graph = self.archivist.graph
        dissolved = 0
        for node, data in list(graph.nodes(data=True)):
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            if data.get("named"):
                data["unnamed_cycles"] = 0
                continue
            cycles = int(data.get("unnamed_cycles", 0)) + 1
            data["unnamed_cycles"] = cycles
            # Reinforced recently? treat as not stagnant
            # last_reinforced is datetime — if member_count grew this pass, growth path resets via caller
            if cycles < self.EPISTEMIC_UNNAMED_MAX_CYCLES:
                continue
            members = [
                v for _u, v, ed in list(graph.out_edges(node, data=True))
                if ed.get("relation_type") == EDGE_COMPOSED_OF
            ]
            coh = self.cluster_coherence(members) if members else 0.0
            # Still improving coherence — keep probation
            prev = float(data.get("last_coherence", 0.0))
            data["last_coherence"] = coh
            if coh > prev + 0.02 and cycles < self.EPISTEMIC_UNNAMED_MAX_CYCLES * 2:
                continue
            # Dissolve wrapper only
            if node in graph:
                graph.remove_node(node)
                dissolved += 1
        return dissolved



    def promote_dictionary_nodes_to_working(self) -> int:
        """Dictionary-original Provisional nodes → Working (schema fuel)."""
        graph = self.archivist.graph
        n = 0
        for node, data in graph.nodes(data=True):
            if data.get("node_type") in (NODE_EPISTEMIC_SCHEMA, NODE_SCHEMA, NODE_BASIN):
                continue
            if data.get("source") == "dictionary" and data.get("tier", TIER_PROVISIONAL) < TIER_WORKING:
                data["tier"] = TIER_WORKING
                n += 1
        return n


    # ------------------------------------------------------------------
    # Taxonomic affinity (weighted relations → logical kind-schemas)
    # ------------------------------------------------------------------
    def _is_a_parents(self, node: str) -> set:
        graph = self.archivist.graph
        if node not in graph:
            return set()
        parents = set()
        for u, _v, ed in graph.in_edges(node, data=True):
            if ed.get("relation_type") == EDGE_IS_A:
                parents.add(u)
        for _u, v, ed in graph.out_edges(node, data=True):
            # some placements use node -is-a-> parent as out-edge
            if ed.get("relation_type") == EDGE_IS_A:
                parents.add(v)
        return parents

    def _shared_hubs(self, a: str, b: str) -> set:
        """Nodes that both a and b link to via is-a / part-of / associated-with."""
        graph = self.archivist.graph
        if a not in graph or b not in graph:
            return set()
        rel_ok = {EDGE_IS_A, EDGE_PART_OF, EDGE_ASSOCIATED_WITH, EDGE_COMPOSED_OF}

        def neighbors(n):
            out = set()
            for u, v, ed in list(graph.in_edges(n, data=True)) + list(graph.out_edges(n, data=True)):
                if ed.get("relation_type") in rel_ok:
                    out.add(u if v == n else v)
            return out

        return neighbors(a) & neighbors(b) - {a, b}

    def _best_edge_weight(self, a: str, b: str) -> float:
        """Strongest typed relation weight between a and b (either direction)."""
        graph = self.archivist.graph
        best = 0.0
        if a not in graph or b not in graph:
            return best
        edges = []
        try:
            for u, v in ((a, b), (b, a)):
                data = graph.get_edge_data(u, v)
                if not data:
                    continue
                # MultiDiGraph: {key: attr_dict}
                if isinstance(data, dict):
                    for val in data.values():
                        if isinstance(val, dict) and "relation_type" in val:
                            edges.append(val)
                        elif isinstance(val, dict):
                            for sub in val.values():
                                if isinstance(sub, dict):
                                    edges.append(sub)
        except Exception:
            pass
        for attr in edges:
            if not isinstance(attr, dict):
                continue
            rel = attr.get("relation_type", "")
            fam = attr.get("family") or get_family(rel)
            if rel == EDGE_IS_A or fam == FAMILY_HIERARCHY:
                best = max(best, self.AFFINITY_DIRECT_IS_A)
            elif rel == EDGE_PART_OF or rel == EDGE_COMPOSED_OF or fam == FAMILY_MEMBERSHIP:
                best = max(best, self.AFFINITY_PART_OF)
            elif fam == FAMILY_CAUSAL or rel in ("causes", "enables", "results-in", "prevents"):
                best = max(best, self.AFFINITY_CAUSAL)
            elif fam == FAMILY_SOCIAL_NORM or rel in ("concerns-other", "violates", "responsible-for"):
                best = max(best, self.AFFINITY_SOCIAL)
            elif rel == EDGE_ASSOCIATED_WITH:
                best = max(best, self.AFFINITY_ASSOCIATED)
            else:
                best = max(best, 0.2)
        return best

    def pair_affinity(self, a: str, b: str, co_count: float = 1.0) -> float:
        """Weighted score for whether (a,b) may join a taxonomic kind-schema.

        affinity = co_act_scale * count + edge_weight + shared_parent_bonus
        Taxonomically unrelated pairs with only weak associated-with stay low.
        """
        if a == b:
            return 0.0
        score = self.AFFINITY_COACT_SCALE * float(co_count)
        score += self._best_edge_weight(a, b)
        pa, pb = self._is_a_parents(a), self._is_a_parents(b)
        if pa and pb and (pa & pb):
            score += self.AFFINITY_SHARED_PARENT
        # One parent of the other (siblings under expansion target)
        if a in pb or b in pa:
            score += self.AFFINITY_SHARED_PARENT * 0.8
        # Shared structural neighbor (same self-study parent / hub via is-a or associated-with)
        shared_hub = self._shared_hubs(a, b)
        if shared_hub:
            score += self.AFFINITY_SHARED_PARENT * 0.9
        # Hard veto: both have is-a parents, no overlap, no direct hierarchy edge
        if pa and pb and not (pa & pb) and self._best_edge_weight(a, b) < self.AFFINITY_PART_OF:
            # allow if pure co-act is very strong AND lemma-similar
            if self._pair_similarity(a, b) < 0.25:
                return 0.0
            score *= 0.35
        return score

    def taxonomic_coactivation_pairs(self) -> list:
        """Stabilized co-activation pairs that pass affinity threshold."""
        raw = self.archivist.stabilized_co_activation_pairs()
        counts = getattr(self.archivist, "co_activation", {})
        kept = []
        for a, b in raw:
            key = (a, b) if a < b else (b, a)
            cnt = float(counts.get(key, counts.get((a, b), counts.get((b, a), 5))))
            aff = self.pair_affinity(a, b, co_count=cnt)
            if aff >= self.AFFINITY_THRESHOLD:
                kept.append((a, b, aff))
        return kept

    def prune_garbage_epistemic_schemas(self) -> int:
        """Remove low-quality epistemic schemas: low coherence, low lemma ratio,
        or majority sentence-like members. Members are left intact.
        """
        graph = self.archivist.graph
        removed = 0
        for node in list(graph.nodes()):
            data = graph.nodes.get(node, {})
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            members = [
                v for _u, v, ed in graph.out_edges(node, data=True)
                if ed.get("relation_type") == EDGE_COMPOSED_OF
            ]
            if not members:
                graph.remove_node(node)
                removed += 1
                continue
            coh = self.cluster_coherence(members)
            lr = self.lemma_ratio(members)
            bad = (
                (not data.get("named") and coh < self.EPISTEMIC_MIN_COHERENCE * 0.85)
                or lr < self.EPISTEMIC_MIN_LEMMA_RATIO * 0.9
                or len(members) < 2
            )
            if bad:
                graph.remove_node(node)
                removed += 1
        return removed

    def detect_epistemic_clusters(self) -> List[str]:
        """
        Groups nodes whose co-activation has stabilized (archivist.
        stabilized_co_activation_pairs()) into cluster candidates via
        connected components of a temporary co-activation graph --
        deterministic, inspectable graph theory (same standard already
        used for cycle handling in the original §13.3 proposal), not an
        opaque clustering library. Returns the list of newly-CREATED
        schema ids (growing an existing schema's membership doesn't count
        as "new" here, same as before).

        Bug fix, this session ("epistemic schema collapse and naming"):
        cluster identity previously came from a hash of the exact member
        set (_epistemic_cluster_id's old implementation). As a cluster
        grows -- the ordinary, expected case, since co-activation keeps
        accumulating every tick -- that hash changes completely, so a
        *new* schema node got created for the grown cluster instead of
        the existing one updating in place. The old node was never
        removed or superseded; it just sat there, decaying, orphaned.
        Confirmed in production: 4 separate epistemic_schema elements
        with descending weights turned out to be successive growth
        snapshots of the same evolving cluster, each permanently frozen
        as its own node.

        This also explains why naming rarely landed on the cases that
        most needed it: _has_dominant_shared_parent() only ran at
        FORMATION time, to decide whether to skip creating a schema at
        all (a real parent already "names" the group). It was never
        rechecked for schemas already formed earlier, before their parent
        had caught up in coverage -- so a schema could become genuinely
        redundant with an already-named hierarchy parent later on, and
        nothing ever collapsed it into that name; it just stayed an
        anonymous hash forever.

        Fixed by giving a cluster's identity to its dominant is-a parent
        directly, once that parent covers >= EPISTEMIC_NAME_MIN_COVERAGE
        members ("the most logical naming is the parent node's original
        name" -- there's no need for separate naming logic at all once
        identity IS the parent): the schema id becomes
        `epistemic_of_{parent}`, named immediately at creation using the
        parent's own existing label, and growing membership under the
        same parent updates that SAME node (new composed-of edges added,
        member_count bumped) rather than spawning a duplicate. Only
        clusters with NO dominant parent (genuine cross-cutting co-
        activation with no existing hierarchy explaining it) still get an
        anonymous hash id and stay unnamed -- exactly the case that
        legitimately has no existing word to borrow.

        Tier-gated to Working/Trusted members only (§14, "Option B" --
        agreed the same session as a fix for over-eager formation:
        344 epistemic schemas formed from just 2000 pulses in production,
        essentially clustering raw touch-recurrence alone). Co-activation
        is still recorded and decayed on every touch regardless of tier
        (§13.3's existing mechanism, unchanged) -- what's new is that
        cluster FORMATION additionally requires the co-touching nodes to
        have already earned Working+ tier through ordinary trust
        evaluation (§3), a compound bar (recurring co-touch AND earned
        corroboration) rather than raw co-touch-count alone. A stabilized
        pair between two still-Provisional nodes is real data and stays
        tracked; it just isn't eligible to form a schema until both sides
        have separately earned cortical status.
        """
        weighted = self.taxonomic_coactivation_pairs()
        if not weighted:
            return []

        co_graph = nx.Graph()
        for a, b, aff in weighted:
            co_graph.add_edge(a, b, affinity=aff)

        graph = self.archivist.graph
        created = []
        for component in nx.connected_components(co_graph):
            # Tier gate: only Working/Trusted members are eligible to
            # participate in a schema at all -- filter before the size
            # check, not after, so a component that's only "big enough"
            # because of uncorroborated Provisional padding is correctly
            # excluded, not trimmed-and-accepted.
            members = sorted(
                n for n in component
                if graph.nodes.get(n, {}).get("tier", TIER_PROVISIONAL) >= TIER_WORKING
            )
            if len(members) < self.EPISTEMIC_MIN_CLUSTER_SIZE:
                continue

            dominant_parent, _coverage = self._dominant_parent(members)

            # Quality gates: reject nonsense clusters before they enter cortex
            coh = self.cluster_coherence(members)
            lr = self.lemma_ratio(members)
            if coh < self.EPISTEMIC_MIN_COHERENCE:
                continue
            if lr < self.EPISTEMIC_MIN_LEMMA_RATIO:
                continue
            # Reject clusters dominated by long / self_generated sentence nodes
            sentenceish = 0
            for m in members:
                md = graph.nodes.get(m, {})
                lab = str(m)
                if md.get("source") == "self_generated" and len(lab.split()) > 2:
                    sentenceish += 1
                elif len(lab) > 40 or len(lab.split()) > 4:
                    sentenceish += 1
            if sentenceish / max(1, len(members)) > 0.4:
                continue

            cluster_id = self._epistemic_cluster_id(members, dominant_parent)

            if cluster_id in graph:
                # Growth path: the same schema (identified by parent, or
                # by member-hash for parent-less clusters) already exists
                # -- add any newly-qualifying members and refresh
                # metadata, rather than treating "already exists" as
                # "nothing to do." This is the actual collapse fix: a
                # cluster that keeps growing keeps updating ONE node.
                existing_members = {
                    v for _u, v, edata in graph.out_edges(cluster_id, data=True)
                    if edata.get("relation_type") == EDGE_COMPOSED_OF
                }
                new_members = [m for m in members if m not in existing_members]
                for member in new_members:
                    graph.add_edge(cluster_id, member, relation_type=EDGE_COMPOSED_OF,
                                    source="schema", placement="explicit",
                                    created_at=datetime.now().isoformat())
                if new_members:
                    graph.nodes[cluster_id]["member_count"] = len(existing_members) + len(new_members)
                    graph.nodes[cluster_id]["last_reinforced"] = datetime.now()
                    graph.nodes[cluster_id]["last_coherence"] = coh
                    # Growth counts as activity — slow unnamed expiry
                    if not graph.nodes[cluster_id].get("named"):
                        prev_u = int(graph.nodes[cluster_id].get("unnamed_cycles", 0))
                        graph.nodes[cluster_id]["unnamed_cycles"] = max(0, prev_u - 1)
                continue

            # Always create unnamed; naming is a separate strict pass.
            node_kwargs = dict(
                source="schema",
                tier=TIER_WORKING,
                last_reinforced=datetime.now(),
                regulatory_efficacy=0.5,
                tier0_cycles=0,
                activation=0.0,
                valence_coloring=0.0,
                node_type=NODE_EPISTEMIC_SCHEMA,
                abstraction_level=1,
                member_count=len(members),
                name=None,
                named=False,
                unnamed_cycles=0,
                last_coherence=coh,
                candidate_parent=dominant_parent,
            )
            if dominant_parent:
                node_kwargs["definition"] = dominant_parent

            graph.add_node(cluster_id, **node_kwargs)
            for member in members:
                graph.add_edge(cluster_id, member, relation_type=EDGE_COMPOSED_OF,
                                source="schema", placement="explicit",
                                created_at=datetime.now().isoformat())
            created.append(cluster_id)

        return created

    def _dominant_parent(self, members: List[str], min_coverage: Optional[int] = None) -> tuple:
        """Returns (best_parent, coverage) -- the is-a parent covering the
        most of `members`, and how many it covers. Returns (None, 0) if
        no member has an is-a parent at all. `min_coverage` lets a caller
        check against a different bar than EPISTEMIC_NAME_MIN_COVERAGE
        without needing a second near-duplicate method (used by
        merge_duplicate_epistemic_schemas() below, same threshold by
        default but kept as a parameter for that reuse)."""
        min_coverage = self.EPISTEMIC_NAME_MIN_COVERAGE if min_coverage is None else min_coverage
        graph = self.archivist.graph
        parent_counts: Counter = Counter()
        for member in members:
            if member not in graph:
                continue
            for u, _v, edata in graph.in_edges(member, data=True):
                if edata.get("relation_type") == EDGE_IS_A:
                    parent_counts[u] += 1
        if not parent_counts:
            return None, 0
        best_parent, coverage = parent_counts.most_common(1)[0]
        if coverage < min_coverage:
            return None, coverage
        return best_parent, coverage

    def merge_duplicate_epistemic_schemas(self) -> int:
        """Migration/ongoing-cleanup pass (this session): handles two
        cases detect_epistemic_clusters()'s own formation-time logic
        can't reach on its own --
          1. Anonymous-hash schemas created before this fix existed
             (production already has these -- confirmed 4 duplicate
             epistemic_schema elements from one evolving cluster).
          2. A schema that legitimately had no dominant parent at
             formation time, but gained one later (e.g. a member received
             a NEW is-a edge via re-parenting, §2.3 mechanism 3, after the
             schema already formed).
        In both cases: once a dominant parent now covers the schema's
        members, fold it into the parent-identified schema (creating one
        if needed) and remove the redundant anonymous node -- same
        "absorb into existing structure, don't silently duplicate"
        principle self_narrative.py's own absorption mechanism already
        uses (§16.4), applied here to a different structure. Consolidation
        -gated, called from prometheus.py's _run_consolidation() alongside
        detect_epistemic_clusters(). Returns the count collapsed."""
        graph = self.archivist.graph
        collapsed = 0
        for node in list(graph.nodes):
            data = graph.nodes.get(node, {})
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            # Parent-identified schemas (epistemic_of_X) are already
            # correctly collapsed by construction -- only anonymous-hash
            # ones are migration candidates.
            if not node.startswith("epistemic_") or node.startswith("epistemic_of_"):
                continue

            members = [
                v for _u, v, edata in graph.out_edges(node, data=True)
                if edata.get("relation_type") == EDGE_COMPOSED_OF
            ]
            if not members:
                continue
            dominant_parent, _coverage = self._dominant_parent(members)
            if not dominant_parent:
                continue  # still a genuine no-parent cluster -- leave it as-is

            target_id = f"epistemic_of_{_slug_id_fragment(dominant_parent)}"
            if target_id not in graph:
                graph.add_node(
                    target_id,
                    source="schema", tier=TIER_WORKING,
                    last_reinforced=datetime.now(), regulatory_efficacy=0.5,
                    tier0_cycles=0, activation=data.get("activation", 0.0),
                    valence_coloring=data.get("valence_coloring", 0.0),
                    node_type=NODE_EPISTEMIC_SCHEMA, abstraction_level=1,
                    name=None, definition=dominant_parent, named=False, unnamed_cycles=0, member_count=0,
                )
            existing_members = {
                v for _u, v, edata in graph.out_edges(target_id, data=True)
                if edata.get("relation_type") == EDGE_COMPOSED_OF
            }
            for member in members:
                if member in existing_members:
                    continue
                graph.add_edge(target_id, member, relation_type=EDGE_COMPOSED_OF,
                                source="schema", placement="explicit",
                                created_at=datetime.now().isoformat())
            graph.nodes[target_id]["member_count"] = len(existing_members | set(members))

            graph.remove_node(node)
            collapsed += 1
        return collapsed

    # ------------------------------------------------------------------
    # Hierarchical stacking (Tier 2+) and schema promotion — new
    # ------------------------------------------------------------------
    # Tier 1 = schemas whose members are ordinary nodes (abstraction_level=1).
    # Tier 2 = schemas whose members are themselves schemas (or a mix with
    # majority schema members). Higher levels follow the same rule.
    # Deterministic: connected components over co-activation among schema
    # nodes, plus shared dominant parent when available. Naming still
    # earned (never invented).
    TIER2_MIN_SCHEMA_MEMBERS = 2
    SCHEMA_PROMOTE_MIN_MEMBERS = 4
    SCHEMA_PROMOTE_MIN_COHERENCE = 0.30
    SCHEMA_PROMOTE_MIN_CYCLES = 3   # consolidations as named+stable before Trusted

    def detect_epistemic_tier2(self) -> List[str]:
        """Form higher-abstraction schemas from co-active Tier-1 (or higher)
        epistemic schemas. Returns newly created higher-level schema ids.

        A Tier-2 schema is created when >= TIER2_MIN_SCHEMA_MEMBERS existing
        epistemic schemas co-activate (via the same stabilized co-activation
        pairs already used for Tier 1) and pass coherence. Members of the
        new schema are the lower schemas themselves (composed-of), not their
        leaf nodes — stacking, not flattening.
        """
        graph = self.archivist.graph
        schema_ids = [
            n for n, d in graph.nodes(data=True)
            if d.get("node_type") == NODE_EPISTEMIC_SCHEMA
            and int(d.get("abstraction_level", 1) or 1) >= 1
            and d.get("tier", TIER_PROVISIONAL) >= TIER_WORKING
        ]
        if len(schema_ids) < self.TIER2_MIN_SCHEMA_MEMBERS:
            return []

        pairs = self.archivist.stabilized_co_activation_pairs()
        # Restrict co-activation graph to schema nodes only
        schema_set = set(schema_ids)
        schema_pairs = [(a, b) for a, b in pairs if a in schema_set and b in schema_set]
        if not schema_pairs:
            # Fallback: schemas that share a dominant parent or overlapping
            # member neighborhoods count as weakly co-located
            parent_groups: Dict[str, List[str]] = {}
            for sid in schema_ids:
                parent = graph.nodes[sid].get("candidate_parent") or graph.nodes[sid].get("definition")
                if parent:
                    parent_groups.setdefault(str(parent), []).append(sid)
            created = []
            for parent, group in parent_groups.items():
                if len(group) < self.TIER2_MIN_SCHEMA_MEMBERS:
                    continue
                created.extend(self._form_tier2_schema(group, dominant_parent=parent))
            return created

        co_graph = nx.Graph()
        co_graph.add_edges_from(schema_pairs)
        created = []
        for component in nx.connected_components(co_graph):
            members = sorted(component)
            if len(members) < self.TIER2_MIN_SCHEMA_MEMBERS:
                continue
            # Coherence among schema members: use their names/definitions
            coh = self.cluster_coherence(members)
            if coh < self.EPISTEMIC_MIN_COHERENCE * 0.7:  # slightly softer for meta-clusters
                continue
            dominant_parent, _cov = self._dominant_parent(
                # expand one level of leaves for parent detection
                self._expand_schema_members(members)
            )
            created.extend(self._form_tier2_schema(members, dominant_parent=dominant_parent))
        return created

    def _expand_schema_members(self, schema_ids: List[str]) -> List[str]:
        """One level of composed-of leaves under the given schemas."""
        graph = self.archivist.graph
        leaves = []
        for sid in schema_ids:
            for _u, v, ed in graph.out_edges(sid, data=True):
                if ed.get("relation_type") == EDGE_COMPOSED_OF:
                    leaves.append(v)
        return leaves

    def _form_tier2_schema(self, member_schemas: List[str], dominant_parent: Optional[str] = None) -> List[str]:
        """Create or grow one higher-level epistemic schema whose members
        are the given lower schemas. Sets abstraction_level = max(child)+1.
        """
        graph = self.archivist.graph
        if len(member_schemas) < self.TIER2_MIN_SCHEMA_MEMBERS:
            return []

        levels = [
            int(graph.nodes.get(m, {}).get("abstraction_level", 1) or 1)
            for m in member_schemas if m in graph
        ]
        if not levels:
            return []
        new_level = max(levels) + 1
        # Cap stacking depth to avoid runaway towers in early runs
        if new_level > 4:
            return []

        if dominant_parent:
            cluster_id = f"epistemic_L{new_level}_of_{_slug_id_fragment(str(dominant_parent))}"
        else:
            digest = hashlib.sha1("|".join(sorted(member_schemas)).encode()).hexdigest()[:8]
            cluster_id = f"epistemic_L{new_level}_{digest}"

        if cluster_id in graph:
            existing = {
                v for _u, v, ed in graph.out_edges(cluster_id, data=True)
                if ed.get("relation_type") == EDGE_COMPOSED_OF
            }
            added = 0
            for m in member_schemas:
                if m not in existing and m in graph:
                    graph.add_edge(
                        cluster_id, m, relation_type=EDGE_COMPOSED_OF,
                        source="schema", placement="explicit",
                        created_at=datetime.now().isoformat(),
                    )
                    added += 1
            if added:
                graph.nodes[cluster_id]["member_count"] = len(existing) + added
                graph.nodes[cluster_id]["last_reinforced"] = datetime.now()
            return []

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
            abstraction_level=new_level,
            member_count=len(member_schemas),
            name=None,
            named=False,
            unnamed_cycles=0,
            candidate_parent=dominant_parent,
            definition=dominant_parent,
            is_meta_schema=True,
        )
        for m in member_schemas:
            if m in graph:
                graph.add_edge(
                    cluster_id, m, relation_type=EDGE_COMPOSED_OF,
                    source="schema", placement="explicit",
                    created_at=datetime.now().isoformat(),
                )
        return [cluster_id]

    def promote_stable_schemas(self) -> Dict[str, int]:
        """Promote named, coherent, sufficiently large epistemic schemas
        from Working → Trusted, and reinforce hierarchy edges to their
        dominant parent when one exists. Returns counts.
        """
        graph = self.archivist.graph
        promoted = 0
        parent_linked = 0
        for node, data in list(graph.nodes(data=True)):
            if data.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            if data.get("tier", TIER_PROVISIONAL) >= TIER_TRUSTED:
                # still try parent link
                pass
            else:
                members = [
                    v for _u, v, ed in graph.out_edges(node, data=True)
                    if ed.get("relation_type") == EDGE_COMPOSED_OF
                ]
                named = bool(data.get("named"))
                coh = float(data.get("last_coherence") or self.cluster_coherence(members) if members else 0)
                stable_cycles = int(data.get("stable_named_cycles", 0))
                if named:
                    data["stable_named_cycles"] = stable_cycles + 1
                    stable_cycles += 1
                else:
                    data["stable_named_cycles"] = 0
                    continue

                if (
                    len(members) >= self.SCHEMA_PROMOTE_MIN_MEMBERS
                    and coh >= self.SCHEMA_PROMOTE_MIN_COHERENCE
                    and stable_cycles >= self.SCHEMA_PROMOTE_MIN_CYCLES
                ):
                    data["tier"] = TIER_TRUSTED
                    data["last_reinforced"] = datetime.now()
                    promoted += 1

            # Stack under dominant parent if missing is-a / associated link
            parent = data.get("candidate_parent") or data.get("definition")
            if parent and parent in graph:
                has_up = any(
                    ed.get("relation_type") in (EDGE_IS_A, "associated-with", EDGE_COMPOSED_OF)
                    for _u, v, ed in graph.out_edges(node, data=True)
                    if v == parent
                ) or any(
                    ed.get("relation_type") == EDGE_IS_A
                    for u, _v, ed in graph.in_edges(node, data=True)
                    if u == parent
                )
                if not has_up:
                    # schema is-a parent concept when parent covers it
                    graph.add_edge(
                        node, parent, relation_type=EDGE_IS_A,
                        source="schema", placement="explicit",
                        created_at=datetime.now().isoformat(),
                    )
                    parent_linked += 1
        return {"promoted_to_trusted": promoted, "parent_links_added": parent_linked}

    def hierarchy_report(self, top_n: int = 15) -> Dict:
        """Diagnostic: abstraction levels, named/trusted counts, sample stack."""
        graph = self.archivist.graph
        by_level: Dict[int, int] = {}
        named = trusted = total = 0
        samples = []
        for n, d in graph.nodes(data=True):
            if d.get("node_type") != NODE_EPISTEMIC_SCHEMA:
                continue
            total += 1
            lvl = int(d.get("abstraction_level", 1) or 1)
            by_level[lvl] = by_level.get(lvl, 0) + 1
            if d.get("named"):
                named += 1
            if d.get("tier", 0) >= TIER_TRUSTED:
                trusted += 1
            members = [
                v for _u, v, ed in graph.out_edges(n, data=True)
                if ed.get("relation_type") == EDGE_COMPOSED_OF
            ]
            samples.append({
                "id": n,
                "level": lvl,
                "named": bool(d.get("named")),
                "name": d.get("name"),
                "tier": d.get("tier"),
                "members": len(members),
                "parent": d.get("candidate_parent") or d.get("definition"),
            })
        samples.sort(key=lambda r: (-r["level"], -r["members"]))
        return {
            "total_epistemic_schemas": total,
            "by_abstraction_level": dict(sorted(by_level.items())),
            "named": named,
            "trusted": trusted,
            "top": samples[:top_n],
        }

    @staticmethod
    def _epistemic_cluster_id(members: List[str], dominant_parent: Optional[str] = None) -> str:
        """Stable cluster id. Parent-based ids use a *short slug*, not the
        raw parent string — long WordNet glosses must not become node ids
        (naming hygiene). Full parent text is stored on the node as
        `name` / `definition` at creation time."""
        if dominant_parent:
            return f"epistemic_of_{_slug_id_fragment(dominant_parent)}"
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
        weighted = self.taxonomic_coactivation_pairs()
        pairs = self.archivist.stabilized_co_activation_pairs()
        co_graph = nx.Graph()
        for a, b, aff in weighted:
            co_graph.add_edge(a, b, affinity=aff)
        candidates = []
        graph = self.archivist.graph
        for component in nx.connected_components(co_graph):
            raw = sorted(component)
            working = sorted(
                n for n in raw
                if graph.nodes.get(n, {}).get("tier", TIER_PROVISIONAL) >= TIER_WORKING
            )
            coh = self.cluster_coherence(working) if len(working) >= 2 else 0.0
            lr = self.lemma_ratio(working) if working else 0.0
            candidates.append({
                "size": len(raw),
                "working_size": len(working),
                "threshold": self.EPISTEMIC_MIN_CLUSTER_SIZE,
                "remaining": max(0, self.EPISTEMIC_MIN_CLUSTER_SIZE - len(working)),
                "coherence": round(coh, 3),
                "lemma_ratio": round(lr, 3),
                "members": working[:5] if working else raw[:5],
                "blocked": (
                    "need_working_tier" if len(working) < self.EPISTEMIC_MIN_CLUSTER_SIZE
                    else ("low_coherence" if coh < self.EPISTEMIC_MIN_COHERENCE
                          else ("low_lemma_ratio" if lr < self.EPISTEMIC_MIN_LEMMA_RATIO
                                else "ready"))
                ),
            })
        candidates.sort(key=lambda c: (0 if c["blocked"] == "ready" else 1, c["remaining"]))

        schemas = [
            (n, d) for n, d in self.archivist.graph.nodes(data=True)
            if d.get("node_type") == NODE_EPISTEMIC_SCHEMA
        ]
        return {
            "total_co_activation_pairs": len(self.archivist.co_activation),
            "stabilized_pairs": len(pairs),
            "taxonomic_pairs": len(weighted),
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
