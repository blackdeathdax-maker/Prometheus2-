import logging
from datetime import datetime
from typing import Dict, List, Optional

from .archivist import SELF_NODE, TIER_PROVISIONAL
from .edge_types import EDGE_ASSOCIATED_WITH, EDGE_IS_A, EDGE_CONCERNS_OTHER
from .sensory import SensoryModule

logger = logging.getLogger(__name__)


class AssociationEngine:
    """
    Visible layer (§7). Grows the knowledge/schema web. Implements §2.3's
    two hierarchy-placement paths (dictionary-pattern parsing primary,
    co-occurrence fallback secondary) plus re-parenting, and creates the
    relational edges to SELF that §2.1b's complex-schema detection needs
    (`responsible-for`, `violates`, `temporal-contrast`, `concerns-other`).
    """

    def __init__(self, archivist, sensory: Optional[SensoryModule] = None):
        self.archivist = archivist
        self.sensory = sensory or SensoryModule()

    # ------------------------------------------------------------------
    # Generic explicit edge (kept for backward compatibility / manual use)
    # ------------------------------------------------------------------
    def link(self, node_a: str, node_b: str, relation_type: str = EDGE_ASSOCIATED_WITH, source: str = "user"):
        self.archivist.link(node_a, node_b, relation_type, source=source, placement="explicit")
        print(f"Linked: {node_a} --{relation_type}--> {node_b}")

    # ------------------------------------------------------------------
    # §2.3 hierarchy placement -- the main entry point for ingesting a new
    # term with its definition/context.
    # ------------------------------------------------------------------
    def place_node(self, term: str, definition: str = "", source: str = "user",
                    context_node: Optional[str] = None, max_parent_children: Optional[int] = None) -> Dict:
        """
        Places `term` into the knowledge web using §2.3's two paths:
          1. Dictionary-pattern parsing on `definition`, if it yields a
             parseable parent -> typed is-a/part-of edge (primary).
          2. Co-occurrence fallback: attaches to whichever node was most
             active (context_node, or else most-recently-reinforced node
             in the graph) with an associated-with edge -- never mislabeled
             as is-a (§2.3 mechanism 2).
        Returns a small dict describing what happened, for logging/tests.

        `max_parent_children` (new, this revision): fixes a real bug found
        from production data -- a dense, uncapped "starburst" hub forming
        despite self-study's degree cap (Prometheus.py's
        _select_self_study_target). The cap only ever gated which node got
        SELECTED as a self-study expansion target; it never actually
        constrained what parent an edge attached to, because self-study
        always supplies a real WordNet gloss as `definition`, so path 1
        (dictionary-pattern parsing) usually succeeds and attaches the new
        child to whatever parent word the GLOSS mentions -- completely
        independent of, and uncapped relative to, the selected target. If
        several self-study-generated words' glosses happened to reference
        the same common word, that word became an unbounded hub, exactly
        reproducing the runaway-growth pattern the cap was built to
        prevent, just through the other path. When supplied, this checks
        the parsed parent's current categorical out-degree
        (archivist.categorical_out_degree) before committing to it; if the
        parent is already at capacity, falls through to the co-occurrence
        anchor instead of attaching there uncapped. None (default)
        preserves the old, uncapped behavior -- real user/dictionary input
        via _ingest() never passes this (it always calls with
        definition="", so path 1 never fires for it regardless), so this
        is scoped to self-study only.
        """
        self.archivist.store(term, source=source, tier=TIER_PROVISIONAL)

        parsed = self.sensory.parse_hierarchy(definition) if definition else None
        if parsed:
            parent, edge_type = parsed
            at_capacity = (
                max_parent_children is not None
                and self.archivist.categorical_out_degree(parent) >= max_parent_children
            )
            if not at_capacity:
                self.archivist.link(parent, term, edge_type, source=source, placement="explicit")
                return {"term": term, "parent": parent, "edge_type": edge_type, "placement": "explicit"}
            # Parsed parent is already at capacity -- fall through to the
            # co-occurrence anchor below instead of attaching here uncapped.

        # Co-occurrence fallback (§2.3 mechanism 2).
        anchor = context_node or self._most_active_node(exclude=term)
        if anchor:
            self.archivist.link(anchor, term, EDGE_ASSOCIATED_WITH, source=source, placement="cooccurrence")
            return {"term": term, "parent": anchor, "edge_type": EDGE_ASSOCIATED_WITH, "placement": "cooccurrence"}

        return {"term": term, "parent": None, "edge_type": None, "placement": "isolated"}

    def _most_active_node(self, exclude: Optional[str] = None) -> Optional[str]:
        """"Most active" = highest recent corroboration, approximated here
        as most-recently-reinforced node (§2.3 mechanism 2). A richer
        version would prefer whichever node is dominant in the *current*
        felt-state context -- callers that have that (prometheus.py, via
        synthesizer's current felt state) should pass context_node
        explicitly instead of relying on this fallback."""
        candidates = [
            (n, d.get("last_reinforced", datetime.min))
            for n, d in self.archivist.graph.nodes(data=True)
            if n != exclude
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[1], reverse=True)
        return candidates[0][0]

    # ------------------------------------------------------------------
    # §2.3 mechanism 3 -- re-parenting, Consolidation-gated. archivist.py
    # identifies *who* is eligible; this decides *where* they should move
    # to, since it owns the dictionary-pattern parser.
    # ------------------------------------------------------------------
    def run_reparenting_pass(self, definitions: Optional[Dict[str, str]] = None) -> int:
        """Called by prometheus.py during Consolidation only.

        Previously a silent no-op: this only ever re-parented a node if a
        `definitions` map supplied its gloss text, but no caller ever
        passed one (Prometheus.py's _run_consolidation() calls this with
        no arguments), so every eligible candidate fell through to
        `continue` and nothing was ever re-parented despite §2.3
        mechanism 3 being documented as resolved.

        Fixed by using sensory.lookup_hypernym() as the primary path --
        WordNet's own taxonomy already gives the authoritative broader
        category for a term (e.g. "blue" -> "color") directly, with no
        gloss to parse and no pattern-matching needed. The `definitions`
        dict is kept as a secondary path (tried first, if supplied) for
        callers that have cached definition text and want parse_hierarchy
        applied to it instead/first -- e.g. a future dictionary source
        other than WordNet.
        """
        definitions = definitions or {}
        moved = 0
        for node in self.archivist.reparenting_candidates():
            definition = definitions.get(node)
            if definition:
                parsed = self.sensory.parse_hierarchy(definition)
                if parsed:
                    new_parent, edge_type = parsed
                    self.archivist.reparent(node, new_parent, edge_type)
                    moved += 1
                    continue

            hypernym = self.sensory.lookup_hypernym(node)
            if hypernym and hypernym != node:
                self.archivist.reparent(node, hypernym, EDGE_IS_A)
                moved += 1
        return moved

    # ------------------------------------------------------------------
    # §2.1b relational edges to SELF -- feeds reflector.py's complex-schema
    # detection. Called from prometheus.py's tick loop whenever sensory.py
    # detects candidates in incoming text.
    # ------------------------------------------------------------------
    def link_relational(self, event_node: str, relation_types: List[str], source: str = "user",
                         felt_state: Optional[str] = None):
        """`concerns-other` links to a generic OTHER placeholder entity
        rather than SELF, since by definition it involves someone other
        than SELF; the other three types link SELF -> event_node.

        `felt_state` (new, this revision): passed through to
        archivist.link() to stamp `felt_state_at_creation` directly on
        each relational edge -- see archivist.link()'s docstring for why
        this replaces the previous after-the-fact timestamp
        reconstruction, which was silently dropping edges."""
        for rel in relation_types:
            if rel == EDGE_CONCERNS_OTHER:
                self.archivist.link("OTHER", event_node, EDGE_CONCERNS_OTHER, source=source,
                                     placement="explicit", felt_state=felt_state)
            else:
                self.archivist.link(SELF_NODE, event_node, rel, source=source,
                                     placement="explicit", felt_state=felt_state)

    # ------------------------------------------------------------------
    # §2.1b item 4a: Schema Node naming trigger. Called whenever a term
    # is placed that might correspond to an existing unnamed schema.
    # ------------------------------------------------------------------
    def try_name_schemas(self, term: str, current_felt_state: Optional[str] = None):
        """
        After placing a new term, check whether any unnamed Schema Node is
        tied to the felt state active *right now* -- if so, the term being
        used in that moment is what earns the schema its name. This
        implements §2.1b item 4a: "Schema Node earns a name only if/when
        the agent's actual dictionary/user input happens to link a word to
        it -- never pre-assigned," via the same felt-state-to-node
        co-occurrence mechanism used for basic basin naming (§6.1), not an
        independent heuristic.

        Fixes two prior bugs: (1) `self.archivist.reflector` was never a
        real attribute -- archivist.py now owns name_schema() directly,
        since it's just a graph mutation on data archivist already owns;
        (2) the old check compared `term` against the schema's `basin`
        field as a raw string (`"basin_0.5_0.2_0.6"`), which is a
        coordinate-derived ID a word can never be a substring of, so it
        could never actually fire. This checks the schema's `basin`
        against the current felt-state key instead, which is the same
        object (both `synthesizer.py`'s stabilized-basin ID), so an actual
        match is possible.
        """
        if not current_felt_state or current_felt_state == "Unformed":
            return
        graph = self.archivist.graph
        for node, data in graph.nodes(data=True):
            if data.get("is_schema") and not data.get("named", False):
                if data.get("basin") == current_felt_state:
                    self.archivist.name_schema(node, term)
                    logger.info(f"Schema {node} named as '{term}' (felt state: {current_felt_state})")

    # ------------------------------------------------------------------
    # Kept for compatibility with earlier callers; delegates to
    # sensory.py's (now multi-result) detector so the two never drift into
    # different labels for the same pattern again.
    # ------------------------------------------------------------------
    def detect_relational_candidate(self, text: str) -> List[str]:
        return self.sensory.detect_relational(text)
