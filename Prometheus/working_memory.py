"""
working_memory.py -- §14, new.

Directed Working Memory, Hippocampal Priority & Emotional Gating. A
moment-to-moment attentional layer sitting ON TOP OF the trust-tier
structure (§14's own spec text distinguishes this explicitly): trust tier
is about a piece of content's *epistemic maturity* (has it earned
corroboration); this module is about what's actually "in mind" right now,
out of everything that exists, regardless of tier.

Fixed maximum composition: SELF (permanent) + current basin (privileged)
+ up to MAX_SCHEMA_SLOTS (7) schema-cluster slots. Schema clusters
(somatic §2.1b or epistemic §13.3) are the primary unit held, not
individual low-level nodes -- though before any real cluster exists, a
single node can stand in as a degenerate one-member "proto-schema"
(§14.7), and real clusters take priority over proto-schemas covering the
same territory once they form.

Per the Core Emergence Principle, this module only ever reads
synthesizer.py's already-synthesized output (basin key: arousal, valence,
dominance; felt state) -- never raw bio hormones directly.
"""
from typing import Dict, List, Optional

from .archivist import SELF_NODE, OTHER_NODE, TIER_WORKING
from .edge_types import NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA, EDGE_COMPOSED_OF


class WorkingMemoryModule:
    """
    §14. Computes working-memory contents on demand (a ranked selection
    over current candidates), not maintained as complex incremental
    state -- callers ask "what's in mind right now" and get a fresh,
    cheap answer each time, since the underlying signals (activation,
    tier, basin) are already maintained elsewhere.
    """

    # §14.1/§14.3 -- hard ceiling, never exceeded regardless of emotional state.
    MAX_SCHEMA_SLOTS = 7
    # §14.3 soft lower bound -- never fewer than SELF + basin + 1 schema,
    # even under maximum narrowing.
    MIN_SCHEMA_SLOTS = 1
    # §14.3 -- floor under high-negative/high-arousal narrowing (SELF +
    # basin + 2 schema slots), tightened from the original "1-3" range to
    # a concrete number during design discussion.
    HIGH_NEGATIVE_SLOT_FLOOR = 2
    # Below this arousal level, full baseline capacity applies regardless
    # of valence -- homeostasis/low intensity, per §14.3.
    LOW_AROUSAL_THRESHOLD = 0.3
    # §14.3 -- positive/low-arousal states get mild narrowing at most, not
    # the hard negative-valence narrowing curve.
    POSITIVE_NARROWING_MAX = 1

    # §14.2 -- epoch-weighted admission. Probability mass a slot's ranking
    # favors user-input-linked content over self-study-generated content,
    # per developmental stage. Childhood: almost entirely user-driven.
    # Adolescence: transitional. Maturity: no longer monopolizing, but
    # never fully displaced (MATURITY_RESERVED_SLOTS below).
    CHILDHOOD_USER_PRIORITY = 0.95
    ADOLESCENCE_USER_PRIORITY = 0.6
    MATURITY_USER_PRIORITY = 0.3
    # §14.2 -- "user input retains a permanent reserved/high-priority
    # channel (never fully displaced)" even in Maturity. Implemented as a
    # reserved slot count exclusively eligible for the best-ranked
    # user-linked candidate, filled first, before the general ranking
    # fills the rest.
    MATURITY_RESERVED_SLOTS = 1

    # §14.3 "exploratory expansion" under calm/positive states -- NOT a
    # slot-count increase (the ceiling never moves, per the resolved
    # design discussion). Implemented as a softening of the epoch's
    # user-priority weighting toward neutral (0.5), giving self-study-
    # sourced content a fairer shot at a couple of slots without
    # reserving them outright. How much softening applies scales with how
    # calm the current state is (see _effective_user_priority).
    CALM_PRIORITY_SOFTENING_MAX = 0.3

    # §14.4 -- schemas/nodes that are themselves among the current felt
    # state's recent anchors (i.e. actually co-occurring with this basin,
    # not just generically active) get a flat bonus in the ranking.
    BASIN_COOCCURRENCE_BONUS = 5.0

    # Bug fix (this session): _score_candidate previously used a bare
    # `10.0` literal for the user-priority bonus spread, not a named
    # tunable like every other constant in this module. Worse than a
    # style inconsistency: 10.0 is the same order of magnitude as
    # archivist.ACTIVATION_CAP (also 10.0), so the two terms could
    # compete on comparable footing at the extremes -- a fully-decayed
    # real (user-linked) node in Childhood scores 0.95*10.0=9.5, while a
    # maxed-out self-study node scores 10.0(activation)+0.05*10.0=10.5,
    # letting self-study content win a slot despite CHILDHOOD_USER_
    # PRIORITY's own docstring stating "almost entirely user-driven."
    # USER_PRIORITY_WEIGHT=20.0 raises the spread enough that Childhood's
    # 0.95 priority reliably dominates the FULL activation range (worst-
    # case real=19.0 vs best-case self-study=11.0), while deliberately
    # leaving Adolescence's weaker 0.6 priority genuinely contestable
    # (worst-case real=12.0 vs best-case self-study=18.0, self-study can
    # still win) -- consistent with §14.2's own "transitional... no
    # longer monopolizing" framing for that epoch. Still a §10-category
    # tuning placeholder, not a claimed-final value -- verified only to
    # actually deliver the *qualitative* guarantee the docstrings already
    # claimed, not tuned beyond that.
    USER_PRIORITY_WEIGHT = 20.0

    def __init__(self, archivist, synthesizer, focus=None):
        self.archivist = archivist
        self.synthesizer = synthesizer
        self.focus = focus  # §13.y optional; set by Prometheus after FocusModule init

        # Instance attributes, not just module-level constants -- same
        # pattern as every other tunable in this design, so the Debug
        # tab's sliders can adjust them live.
        self.MAX_SCHEMA_SLOTS = WorkingMemoryModule.MAX_SCHEMA_SLOTS
        self.MIN_SCHEMA_SLOTS = WorkingMemoryModule.MIN_SCHEMA_SLOTS
        self.HIGH_NEGATIVE_SLOT_FLOOR = WorkingMemoryModule.HIGH_NEGATIVE_SLOT_FLOOR
        self.LOW_AROUSAL_THRESHOLD = WorkingMemoryModule.LOW_AROUSAL_THRESHOLD
        self.POSITIVE_NARROWING_MAX = WorkingMemoryModule.POSITIVE_NARROWING_MAX
        self.CHILDHOOD_USER_PRIORITY = WorkingMemoryModule.CHILDHOOD_USER_PRIORITY
        self.ADOLESCENCE_USER_PRIORITY = WorkingMemoryModule.ADOLESCENCE_USER_PRIORITY
        self.MATURITY_USER_PRIORITY = WorkingMemoryModule.MATURITY_USER_PRIORITY
        self.MATURITY_RESERVED_SLOTS = WorkingMemoryModule.MATURITY_RESERVED_SLOTS
        self.CALM_PRIORITY_SOFTENING_MAX = WorkingMemoryModule.CALM_PRIORITY_SOFTENING_MAX
        self.BASIN_COOCCURRENCE_BONUS = WorkingMemoryModule.BASIN_COOCCURRENCE_BONUS
        self.USER_PRIORITY_WEIGHT = WorkingMemoryModule.USER_PRIORITY_WEIGHT

    # ------------------------------------------------------------------
    # §14.3 Emotional Gating of Capacity
    # ------------------------------------------------------------------
    def compute_capacity(self) -> int:
        """
        Returns the current effective schema-slot count (<= MAX_SCHEMA_
        SLOTS), narrowed by emotional intensity. Reads only synthesizer's
        already-synthesized basin key (arousal, valence, dominance) --
        never a raw hormone -- per the Core Emergence Principle.
        """
        arousal, valence, _dominance = self.synthesizer.get_current_basin_key()

        if arousal < self.LOW_AROUSAL_THRESHOLD:
            return self.MAX_SCHEMA_SLOTS  # homeostasis/low intensity -- full baseline

        severity = min(1.0, (arousal - self.LOW_AROUSAL_THRESHOLD) / max(1e-6, 1.0 - self.LOW_AROUSAL_THRESHOLD))

        if valence < 0:
            # Negative + aroused: narrow hard toward the high-negative
            # floor, scaled by how aroused.
            span = self.MAX_SCHEMA_SLOTS - self.HIGH_NEGATIVE_SLOT_FLOOR
            capacity = self.MAX_SCHEMA_SLOTS - round(span * severity)
            return max(self.MIN_SCHEMA_SLOTS, capacity)
        else:
            # Positive + aroused: mild narrowing only, never below the
            # high-negative floor (positive arousal shouldn't narrow as
            # harshly as distress, §14.3).
            capacity = self.MAX_SCHEMA_SLOTS - round(self.POSITIVE_NARROWING_MAX * severity)
            return max(self.HIGH_NEGATIVE_SLOT_FLOOR, capacity)

    def _is_calm(self) -> bool:
        """Calm/positive-low-arousal check for §14.3's exploratory
        admission softening (not a capacity change -- see
        _effective_user_priority)."""
        arousal, valence, _dominance = self.synthesizer.get_current_basin_key()
        return valence >= 0 and arousal < self.LOW_AROUSAL_THRESHOLD * 2

    # ------------------------------------------------------------------
    # §14.2 Epoch-Dependent Hippocampal Priority (+ §14.3's calm-state
    # admission-restriction softening, which is NOT a slot-count change)
    # ------------------------------------------------------------------
    def _base_user_priority(self, epoch_value: str) -> float:
        return {
            "Childhood": self.CHILDHOOD_USER_PRIORITY,
            "Adolescence": self.ADOLESCENCE_USER_PRIORITY,
            "Maturity": self.MATURITY_USER_PRIORITY,
        }.get(epoch_value, self.CHILDHOOD_USER_PRIORITY)

    def _effective_user_priority(self, epoch_value: str) -> float:
        """§14.3: 'exploratory expansion' under calm/positive states means
        the admission restriction on a couple of slots loosens -- NOT
        that the slot count increases (compute_capacity's ceiling is
        untouched). Implemented as pulling the epoch's user-priority
        weighting partway toward neutral (0.5) when calm, so self-study
        content gets a fairer shot at a couple of slots without an
        explicit per-slot carve-out."""
        base = self._base_user_priority(epoch_value)
        if not self._is_calm():
            return base
        return base + (0.5 - base) * self.CALM_PRIORITY_SOFTENING_MAX

    # ------------------------------------------------------------------
    # Candidate identification / scoring
    # ------------------------------------------------------------------
    def _schema_members(self, schema_id: str) -> List[str]:
        return [
            v for _u, v, edata in self.archivist.graph.out_edges(schema_id, data=True)
            if edata.get("relation_type") == EDGE_COMPOSED_OF
        ]

    def is_user_linked(self, node: str) -> bool:
        """Whether a node (or, for a schema, any of its members) traces
        back to real user input rather than pure self-study generation.
        A schema formed even partly from real input counts as
        user-linked -- §14.2's admission priority is about responsiveness
        to genuine interaction, not purity of origin."""
        data = self.archivist.graph.nodes.get(node, {})
        if data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA):
            return any(
                self.archivist.graph.nodes.get(m, {}).get("source") == "user"
                for m in self._schema_members(node)
            )
        return data.get("source") == "user"

    def _score_candidate(self, node: str, epoch_value: str, basin_anchor_set: set) -> float:
        data = self.archivist.graph.nodes.get(node, {})
        score = data.get("activation", 0.0)

        user_priority = self._effective_user_priority(epoch_value)
        if self.is_user_linked(node):
            score += user_priority * self.USER_PRIORITY_WEIGHT
        else:
            score += (1.0 - user_priority) * self.USER_PRIORITY_WEIGHT

        # §14.4: schemas/nodes actually co-occurring with the CURRENT
        # basin (not just generically active) get an extra bonus, on top
        # of whatever candidacy they already have.
        if node in basin_anchor_set:
            score += self.BASIN_COOCCURRENCE_BONUS

        # §13.y: sticky focus boost
        if self.focus is not None:
            score += self.focus.focus_boost_for(node)

        return score

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _identity_key(self, node_id: str) -> str:
        """Collapse duplicate labels in WM: same name → same identity."""
        data = self.archivist.graph.nodes.get(node_id, {})
        name = data.get("name")
        if name and str(name).strip():
            return "name:" + str(name).strip().casefold()
        return "id:" + str(node_id)

    def _dedupe_slot_ids(self, node_ids):
        """Keep first occurrence per identity key."""
        seen = set()
        out = []
        for n in node_ids:
            key = self._identity_key(n)
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
        return out

    def get_working_memory(self, epoch_value: str, basin_anchors: Optional[List[str]] = None) -> Dict:
        """
        Returns the current working-memory contents:
        {"self": SELF_NODE, "basin_anchors": [...], "capacity": int,
         "slots": [node_id, ...]}

        `basin_anchors` are the current felt state's recently-touched
        nodes (prometheus.py's felt_state_anchors for the active basin
        key) -- used both as the proto-schema candidate pool (§14.7, when
        no real schema covers that territory yet) and for the basin-
        co-occurrence scoring bonus (§14.4).
        """
        basin_anchors = basin_anchors or []
        capacity = self.compute_capacity()
        basin_anchor_set = set(basin_anchors)

        graph = self.archivist.graph
        real_schemas = [
            n for n, d in graph.nodes(data=True)
            if d.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA)
        ]

        # §14.7: proto-schema territory already covered by a real cluster
        # is excluded from standing in twice -- the real cluster takes
        # priority, exactly as designed.
        covered = set()
        for s in real_schemas:
            covered.update(self._schema_members(s))

        candidates = list(real_schemas)
        for n in basin_anchors:
            if n not in covered and n in graph and n not in (SELF_NODE, OTHER_NODE):
                candidates.append(n)
        candidates = list(dict.fromkeys(candidates))  # de-dup, preserve order

        scored = [(n, self._score_candidate(n, epoch_value, basin_anchor_set)) for n in candidates]

        # §14.2 Maturity reserved channel: "never fully displaced" --
        # reserve MATURITY_RESERVED_SLOTS exclusively for the best-ranked
        # user-linked candidate before the general ranking fills the rest,
        # so user-linked content can never be squeezed out entirely even
        # if internal-interest scores dominate everywhere else.
        slots: List[str] = []
        if epoch_value == "Maturity" and self.MATURITY_RESERVED_SLOTS > 0:
            user_linked_scored = sorted(
                (t for t in scored if self.is_user_linked(t[0])),
                key=lambda t: t[1], reverse=True,
            )
            reserved = [n for n, _s in user_linked_scored[: self.MATURITY_RESERVED_SLOTS]]
            slots.extend(reserved)
            scored = [t for t in scored if t[0] not in reserved]

        scored.sort(key=lambda t: t[1], reverse=True)
        capacity = max(0, int(capacity))
        for n, _s in scored:
            if len(slots) >= capacity:
                break
            if n in slots:
                continue
            if any(self._identity_key(x) == self._identity_key(n) for x in slots):
                continue
            slots.append(n)

        slots = self._dedupe_slot_ids(slots)[:capacity]

        return {
            "self": SELF_NODE,
            "basin_anchors": basin_anchors,
            "capacity": capacity,
            "slots": slots,
        }

    def reachable_nodes(self, working_memory: Dict) -> set:
        """Expands working memory's slots (schema ids or proto-schema
        node ids) into the full set of individual nodes they cover --
        for a schema, its members; for a proto-schema, just itself.
        Public wrapper so callers (prometheus.py's self-study targeting)
        don't need to reach into _schema_members directly."""
        graph = self.archivist.graph
        reachable = set()
        for slot in working_memory.get("slots", []):
            data = graph.nodes.get(slot, {})
            if data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA):
                reachable.update(self._schema_members(slot))
            else:
                reachable.add(slot)
        return reachable

    # ------------------------------------------------------------------
    # §14.6 item 2 -- dead-end detection. GENUINELY UNRESOLVED per the
    # design spec, not just untuned. What follows is an explicitly-
    # flagged PROXY, not the resolved mechanism -- see the spec's own
    # honesty about this rather than silently treating it as solved.
    # ------------------------------------------------------------------
    def is_dead_end(self, working_memory: Dict, archivist_has_room_fn, barren_set: set) -> bool:
        """
        Proxy for "does nothing currently in working memory have any
        unexplored productive next step" (§14.2's Childhood suppression
        rule depends on this). NOT the resolved mechanism -- §14.6 item 2
        explicitly flags real dead-end detection as an open design
        question requiring its own pass, not a number to tune. This proxy
        checks each schema slot's member nodes (or the node itself, for a
        proto-schema) against the SAME per-node room/barren checks
        self-study already uses -- it does not attempt anything genuinely
        working-memory-level (e.g. reasoning about whether the *cluster
        as a whole* has unexplored territory beyond its individual
        members). Treat this as a stand-in that unblocks Childhood's
        suppression rule for now, not as a claim that dead-end detection
        is actually solved.
        """
        for slot in working_memory.get("slots", []):
            data = self.archivist.graph.nodes.get(slot, {})
            if data.get("node_type") in (NODE_SCHEMA, NODE_EPISTEMIC_SCHEMA):
                members = self._schema_members(slot)
            else:
                members = [slot]
            for m in members:
                if m in barren_set:
                    continue
                if archivist_has_room_fn(m):
                    return False  # something still has an unexplored next step
        return True
