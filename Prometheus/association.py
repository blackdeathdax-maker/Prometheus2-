import logging
from typing import Dict, List, Optional

from .archivist import SELF_NODE, OTHER_NODE, TIER_PROVISIONAL, TIER_WORKING
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

    # Bug fix, this revision: archivist.py's categorical_out_degree() was
    # added specifically because "association.place_node() now needs the
    # identical check too... dictionary-pattern-parsed placements were
    # never capped at all, only the co-occurrence fallback was" -- but
    # neither path here actually called it yet. Without this, a single
    # common WordNet parent (e.g. "color", "person", "event") reachable
    # via dictionary-pattern parsing could accumulate unlimited children
    # from real user/dictionary input over time, even though self-study's
    # own target selection (Prometheus.py's hard_cap=3 default) was
    # already protected from doing the same thing. Same tuning-placeholder
    # category and same default as self-study's cap (§10) -- not claimed
    # to be a numerically final value, just no longer literally unbounded.
    PARENT_OUT_DEGREE_CAP = 3

    def __init__(self, archivist, sensory: Optional[SensoryModule] = None):
        self.archivist = archivist
        self.sensory = sensory or SensoryModule()
        # Optional: callable(term, source, context_node) -> bool
        # Prometheus sets this to reason-gate dictionary node formation.
        self.lookup_gate = None

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

    def teach_relation(self, child: str, parent: str, relation_type: str = "is-a", source: str = "user"):
        """Explicit user-taught edge: ensure both nodes exist, link child→parent.
        Returns dict with node ids and edge type.
        """
        child = (child or "").strip().lower()
        parent = (parent or "").strip().lower()
        if not child or not parent or child == parent:
            return None
        # place without definition to avoid recursive hierarchy parse noise
        self.place_node(child, definition="", source=source, context_node=None)
        self.place_node(parent, definition="", source=source, context_node=None)
        # Direction: child is-a parent  =>  edge child → parent (or parent→child?)
        # Existing hierarchy from parse_hierarchy links parent→term when
        # term is defined as "a parent". For explicit "yellow is a color",
        # semantic is yellow —is-a→ color. Match is-a direction as child→parent.
        self.archivist.link(
            child, parent, relation_type, source=source, placement="explicit",
        )
        # mark user-linked for WM
        for n in (child, parent):
            if n in self.archivist.graph:
                self.archivist.graph.nodes[n]["user_linked"] = True
                self.archivist.graph.nodes[n]["source"] = source
        return {"child": child, "parent": parent, "relation_type": relation_type}


    def link_self_attribute(self, attribute: str, edge_type: str = "associated-with", source: str = "user"):
        """SELF reflection: place attribute node and link SELF → attribute.
        identity → associated-with; composition → composed-of when possible.
        """
        from .archivist import SELF_NODE
        attr = (attribute or "").strip().lower()
        if not attr:
            return None
        # shorten multiword slightly for node id stability
        term = attr
        self.place_node(term, definition="", source=source, context_node=SELF_NODE)
        if SELF_NODE not in self.archivist.graph:
            self.archivist.store(SELF_NODE, source="system")
        rel = edge_type if edge_type in ("associated-with", "composed-of", "part-of") else "associated-with"
        self.archivist.link(SELF_NODE, term, rel, source=source, placement="explicit")
        if term in self.archivist.graph:
            self.archivist.graph.nodes[term]["user_linked"] = True
            self.archivist.graph.nodes[term]["self_attribute"] = True
            self.archivist.graph.nodes[term]["self_attr_kind"] = (
                "composition" if rel == "composed-of" else "identity"
            )
        return {"self": SELF_NODE, "attribute": term, "relation_type": rel}

    def place_node(self, term: str, definition: str = "", source: str = "user",
                    context_node: Optional[str] = None,
                    max_parent_children: Optional[int] = None,
                    force_lookup: bool = False) -> Dict:
        """
        Places `term` into the knowledge web using §2.3's two paths:
          1. Dictionary-pattern parsing on `definition`, if it yields a
             parseable parent -> typed is-a/part-of edge (primary).
          2. Co-occurrence fallback: attaches to whichever node was most
             active (context_node, or else most-recently-reinforced node
             in the graph) with an associated-with edge -- never mislabeled
             as is-a (§2.3 mechanism 2).
        Both paths respect an out-degree cap (bug fix, this revision --
        see the class docstring): a parent already at capacity is skipped
        rather than accumulating unbounded children, whichever path found
        it. If the parsed parent is full, this falls through to the
        co-occurrence path rather than abandoning placement entirely --
        losing the *stronger* is-a evidence to a degree cap is preferable
        to leaving the term isolated.

        `max_parent_children` lets a caller supply a different cap than
        the class default (e.g. Prometheus.py's self-study passes its own
        SELF_STUDY_SOFT_CAP, since self-study's own target-selection
        already reasons about out-degree at a different threshold than
        general placement, §5.1/§13). Defaults to PARENT_OUT_DEGREE_CAP
        when the caller doesn't specify one, so existing callers (e.g.
        prometheus.py's _ingest(), which never passes this) are unaffected.

        Returns a small dict describing what happened, for logging/tests.
        """
        cap = self.PARENT_OUT_DEGREE_CAP if max_parent_children is None else max_parent_children
        # Quality gate: self_generated / expansion terms should look concept-like.
        # Full user sentences still allowed (source=user) so dialogue is recorded.
        term_s = str(term).strip()
        if source == "dictionary" and callable(self.lookup_gate) and not force_lookup:
            try:
                if not self.lookup_gate(term_s, source, context_node):
                    return {"term": None, "skipped": "lookup_denied", "reason": "no_reason_to_lookup"}
            except Exception as e:
                # Fail OPEN on gate errors so self-study never hard-locks
                try:
                    print(f"lookup_gate error ({term_s!r}): {e}")
                except Exception:
                    pass
        if source in ("self_generated", "schema", "social", "dictionary"):
            words = term_s.split()
            # Allow multiword WordNet lemmas; still block sentence-like strings
            max_words = 5 if source == "dictionary" else 4
            max_len = 56 if source == "dictionary" else 48
            if len(term_s) > max_len or len(words) > max_words:
                return {"term": None, "skipped": "garbage_label", "reason": "too_sentence_like"}
            low = term_s.lower()
            if low.startswith(("i ", "it was", "it is", "the act of", "a person who", "the state of")):
                return {"term": None, "skipped": "garbage_label", "reason": "sentence_opener"}
            # Reject punctuation-heavy fragments
            if any(ch in term_s for ch in ".?!;:"):
                return {"term": None, "skipped": "garbage_label", "reason": "sentence_punct"}
        # Dictionary-original material is already an external authority
        # (§3.1 / §2.2): start at Working so epistemic clustering is not
        # permanently starved while co-activation accumulates on Tier-0 only.
        start_tier = TIER_WORKING if source in ("dictionary", "schema") else TIER_PROVISIONAL
        self.archivist.store(term, source=source, tier=start_tier)

        parsed = self.sensory.parse_hierarchy(definition) if definition else None
        if parsed:
            parent, edge_type = parsed
            if self.archivist.categorical_out_degree(parent) < cap:
                self.archivist.link(parent, term, edge_type, source=source, placement="explicit")
                return {"term": term, "parent": parent, "edge_type": edge_type, "placement": "explicit"}
            logger.info(f"Parsed parent '{parent}' at out-degree cap; falling back to co-occurrence for '{term}'.")

        # Co-occurrence fallback (§2.3 mechanism 2).
        anchor = context_node if context_node and self._has_room(context_node, cap) else None
        if anchor is None:
            anchor = self._most_active_node(exclude=term, cap=cap)
        if anchor:
            self.archivist.link(anchor, term, "associated-with", source=source, placement="cooccurrence")
            return {"term": term, "parent": anchor, "edge_type": "associated-with", "placement": "cooccurrence"}

        return {"term": term, "parent": None, "edge_type": None, "placement": "isolated"}

    def _has_room(self, node: str, cap: Optional[int] = None) -> bool:
        """Shared out-degree cap check (bug fix, this revision -- see the
        class docstring and PARENT_OUT_DEGREE_CAP). `cap` lets callers
        that received their own max_parent_children (place_node) pass it
        through instead of always using the class default."""
        effective_cap = self.PARENT_OUT_DEGREE_CAP if cap is None else cap
        return self.archivist.categorical_out_degree(node) < effective_cap

    def _most_active_node(self, exclude: Optional[str] = None, cap: Optional[int] = None) -> Optional[str]:
        """"Most active" = highest current activation score (§11 pull-
        forward), used as the co-occurrence fallback anchor (§2.3
        mechanism 2) when no context_node was supplied and no dictionary-
        pattern parent was parseable.

        Bug fix, this revision: this previously sorted on a field called
        `last_conversational_touch`, set via an `archivist.
        touch_conversational()` method. Both were introduced as a fix for
        an earlier reported bug ("working memory wanders from user
        prompts" -- self-study's constant reinforcement of
        `last_reinforced` was letting the co-occurrence anchor drift onto
        whatever self-study last touched between real messages). That fix
        predates this revision's activation/working-memory layer
        (archivist.bump_activation/decay_activation, §11 pull-forward),
        which was built independently and never wired to touch the same
        field -- `touch_conversational()` and `last_conversational_touch`
        no longer exist anywhere in archivist.py at all. Since this
        method's filter required that field to be set above its
        datetime.min default, and nothing was setting it anymore, this
        method silently returned None on every call -- any new node with
        no explicit context_node and no parseable definition (routine
        during early Childhood or any `Unformed`-felt-state period, which
        the spec itself flags can last a long time) fell all the way
        through to `placement: "isolated"` and was never attached to the
        graph at all, regardless of how conversationally recent other
        nodes were.

        Fixed by switching to `activation` -- this branch's own current
        "currently relevant" signal, already the canonical ranking used
        by `archivist.working_memory_nodes()` and `working_memory.py`'s
        candidate scoring, so this fallback now agrees with the rest of
        the design instead of reading a field only it still wrote to.
        prometheus.py's `_ingest()` calls `bump_activation()` on every
        real-input node and its anchor, so activation still tracks
        genuine conversational recency, not just self-study's smaller,
        separately-tunable `ACTIVATION_BOOST_SELF_STUDY` bump.

        The `source != "self_generated"` filter is kept as a second,
        independent guard against self-study's own newly-created nodes
        specifically, and requiring activation > 0 excludes nodes that
        have simply never been touched -- consistent with the original
        protective intent (self-study shouldn't be able to make an
        unrelated node look like "what the user was just talking
        about"), just expressed through the field this branch actually
        maintains.

        SELF/OTHER are also excluded, consistent with self-study's own
        `has_room()` check (Prometheus.py's `_select_self_study_target`,
        `n not in (SELF_NODE, OTHER_NODE)`) and the rest of this design's
        treatment of them as relational-edge-only axioms (§2.1b), not
        general dictionary/co-occurrence attachment targets. Without this,
        SELF's permanent max-activation seeding (archivist._seed_self_node,
        `activation=ACTIVATION_CAP` -- correct and necessary for its own
        purpose, staying visible in the §11 working-memory top-K filter)
        would otherwise make it the default fallback anchor for nearly
        any isolated node, which was never its intended role here."""
        candidates = [
            (n, d.get("activation", 0.0))
            for n, d in self.archivist.graph.nodes(data=True)
            if n != exclude and n not in (SELF_NODE, OTHER_NODE)
            and d.get("source") != "self_generated" and self._has_room(n, cap)
        ]
        candidates = [(n, a) for n, a in candidates if a > 0.0]
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
    def link_relational(self, event_node: str, relation_types: List[str], source: str = "user",
                         felt_state: Optional[str] = None,
                         other_ids: Optional[List[str]] = None):
        """Create relational / role / causal edges detected by sensory.py.

        - Social-norm / temporal (responsible-for, violates, temporal-contrast)
          → SELF → event_node
        - concerns-other → specific named other(s) if provided, else OTHER
        - ROLE family (agent / patient / instrument) → SELF → event_node
        - CAUSAL family → SELF → event_node

        `other_ids`: optional list from OthersRegistry.process_text();
        when present, concerns-other edges attach to those nodes instead
        of only the generic OTHER placeholder.

        `felt_state`: threaded through to archivist.link() so the edge
        carries felt_state_at_creation (ground truth at creation time).
        """
        from .edge_types import (
            EDGE_CONCERNS_OTHER,
            ROLE_EDGE_TYPES, CAUSAL_EDGE_TYPES,
        )
        specific_others = [o for o in (other_ids or []) if o and o != OTHER_NODE]
        for rel in relation_types:
            if rel == EDGE_CONCERNS_OTHER or rel == "concerns-other":
                targets = specific_others if specific_others else [OTHER_NODE]
                for other in targets:
                    # other ↔ event (the social concern)
                    self.archivist.link(
                        other, event_node, rel, source=source,
                        placement="explicit", felt_state=felt_state,
                    )
                    # SELF always participates: I am relating to this other/event
                    self.archivist.link(
                        SELF_NODE, event_node, rel, source=source,
                        placement="explicit", felt_state=felt_state,
                    )
                    # durable self↔other acquaintance edge (associated-with)
                    if other != OTHER_NODE:
                        self.archivist.link(
                            SELF_NODE, other, "associated-with", source=source,
                            placement="explicit", felt_state=felt_state,
                        )
            elif rel in ROLE_EDGE_TYPES or rel in CAUSAL_EDGE_TYPES:
                self.archivist.link(
                    SELF_NODE, event_node, rel, source=source,
                    placement="explicit", felt_state=felt_state,
                )
                # If a specific other is present, also bind them as participant
                for other in specific_others:
                    self.archivist.link(
                        other, event_node, rel, source=source,
                        placement="explicit", felt_state=felt_state,
                    )
                    self.archivist.link(
                        SELF_NODE, other, "associated-with", source=source,
                        placement="explicit", felt_state=felt_state,
                    )
            else:
                # Classic social-norm / temporal edges still anchored on SELF
                self.archivist.link(
                    SELF_NODE, event_node, rel, source=source,
                    placement="explicit", felt_state=felt_state,
                )
                for other in specific_others:
                    self.archivist.link(
                        SELF_NODE, other, "associated-with", source=source,
                        placement="explicit", felt_state=felt_state,
                    )

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
