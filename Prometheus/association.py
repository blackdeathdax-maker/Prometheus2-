import logging
from datetime import datetime
from typing import Dict, List, Optional

from prometheus.archivist import SELF_NODE, TIER_PROVISIONAL
from prometheus.sensory import SensoryModule

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
    def link(self, node_a: str, node_b: str, relation_type: str = "associated-with", source: str = "user"):
        self.archivist.link(node_a, node_b, relation_type, source=source, placement="explicit")
        print(f"Linked: {node_a} --{relation_type}--> {node_b}")

    # ------------------------------------------------------------------
    # §2.3 hierarchy placement -- the main entry point for ingesting a new
    # term with its definition/context.
    # ------------------------------------------------------------------
    def place_node(self, term: str, definition: str = "", source: str = "user",
                    context_node: Optional[str] = None) -> Dict:
        """
        Places `term` into the knowledge web using §2.3's two paths:
          1. Dictionary-pattern parsing on `definition`, if it yields a
             parseable parent -> typed is-a/part-of edge (primary).
          2. Co-occurrence fallback: attaches to whichever node was most
             active (context_node, or else most-recently-reinforced node
             in the graph) with an associated-with edge -- never mislabeled
             as is-a (§2.3 mechanism 2).
        Returns a small dict describing what happened, for logging/tests.
        """
        self.archivist.store(term, source=source, tier=TIER_PROVISIONAL)

        parsed = self.sensory.parse_hierarchy(definition) if definition else None
        if parsed:
            parent, edge_type = parsed
            self.archivist.link(parent, term, edge_type, source=source, placement="explicit")
            return {"term": term, "parent": parent, "edge_type": edge_type, "placement": "explicit"}

        # Co-occurrence fallback (§2.3 mechanism 2).
        anchor = context_node or self._most_active_node(exclude=term)
        if anchor:
            self.archivist.link(anchor, term, "associated-with", source=source, placement="cooccurrence")
            return {"term": term, "parent": anchor, "edge_type": "associated-with", "placement": "cooccurrence"}

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
        """Called by prometheus.py during Consolidation only. `definitions`
        is an optional {node: definition_text} map (e.g. from a dictionary
        cache) used to try to find a firmer parent for eligible nodes;
        without it, eligible nodes are simply left as-is (no data to
        re-parent from) rather than guessing."""
        definitions = definitions or {}
        moved = 0
        for node in self.archivist.reparenting_candidates():
            definition = definitions.get(node)
            if not definition:
                continue
            parsed = self.sensory.parse_hierarchy(definition)
            if parsed:
                new_parent, edge_type = parsed
                self.archivist.reparent(node, new_parent, edge_type)
                moved += 1
        return moved

    # ------------------------------------------------------------------
    # §2.1b relational edges to SELF -- feeds reflector.py's complex-schema
    # detection. Called from prometheus.py's tick loop whenever sensory.py
    # detects candidates in incoming text.
    # ------------------------------------------------------------------
    def link_relational(self, event_node: str, relation_types: List[str], source: str = "user"):
        """`concerns-other` links to a generic OTHER placeholder entity
        rather than SELF, since by definition it involves someone other
        than SELF; the other three types link SELF -> event_node."""
        for rel in relation_types:
            if rel == "concerns-other":
                self.archivist.link("OTHER", event_node, "concerns-other", source=source, placement="explicit")
            else:
                self.archivist.link(SELF_NODE, event_node, rel, source=source, placement="explicit")

    # ------------------------------------------------------------------
    # Kept for compatibility with earlier callers; delegates to
    # sensory.py's (now multi-result) detector so the two never drift into
    # different labels for the same pattern again.
    # ------------------------------------------------------------------
    def detect_relational_candidate(self, text: str) -> List[str]:
        return self.sensory.detect_relational(text)
