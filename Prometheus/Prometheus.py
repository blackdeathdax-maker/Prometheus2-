import os
import random
from typing import List, Optional

from .hormonal import BioSystem, Epoch
from .archivist import ArchivistModule, TIER_WORKING, TIER_TRUSTED, SELF_NODE, OTHER_NODE
from .edge_types import CATEGORICAL_EDGE_TYPES
from .executive import ExecutiveModule
from .synthesizer import SynthesizerModule
from .reflector import ReflectorModule
from .chronos import ChronosModule
from .sensory import SensoryModule
from .association import AssociationEngine
from .stimulus import SyntheticStimulusEngine


class Prometheus:
    """
    Orchestrator (§7). Owns cross-layer epoch transition checks and tick
    sequencing. Per the Core Emergence Principle, this class must only
    condition its decisions on visible-layer felt states (synthesizer's
    output) -- never on bio._hormones or bio.get_raw_variables() directly.
    """

    # Fatigue state-cycling thresholds with hysteresis margins (spec §5).
    T1 = 0.4
    T2 = 0.8
    HYSTERESIS = 0.05

    # Fatigue growth (per tick, scaled by urgency) and per-state recovery
    # rates. Consolidation recovers more than Pruning -- it's the
    # restorative state, Pruning is the costly one (fixed bug: Consolidation
    # previously applied zero recovery at all, trapping the system in a
    # permanent Consolidation<->Pruning oscillation that made Learning, and
    # therefore all graph growth, unreachable after the first few ticks).
    # All three remain undecided tuning placeholders (§10) -- named here
    # specifically so the Debug tab's sliders can adjust them live.
    FATIGUE_GROWTH_RATE = 0.2
    FATIGUE_RECOVERY_CONSOLIDATION = 0.5
    FATIGUE_RECOVERY_PRUNING = 0.7

    # Regulation spike threshold (§4.1) and dampening cap (§4.4). Not yet
    # numerically tuned per spec §10 item 8 -- placeholders, documented.
    # Thresholds are on synthesizer.py's arousal-axis intensity signal
    # (0.0-1.0), NOT raw somatic.urgency (§ Core Emergence Principle).
    REGULATION_SPIKE_THRESHOLD = 0.7
    REGULATION_HYSTERESIS = 0.05  # §4.1: "same hysteresis-band pattern as fatigue T1/T2"
    REGULATION_DAMPENING_CAP = 0.4
    REGULATION_FATIGUE_COST = 0.05  # §4.6: regulation draws on the fatigue economy

    # Self-study (§5.1) hormonal reward bump -- scaled down deliberately
    # relative to externally-triggered deltas so it reads as gentle
    # background texture, not a significant event (§5.1, §9 risk 7).
    SELF_STUDY_DOPAMINE_BUMP = 0.03

    # Hormonal reaction to real input -- new, this revision. §5.1 has
    # always described self-study's dopamine bump as "scaled down
    # deliberately relative to externally-triggered deltas," but no
    # externally-triggered delta mechanism existed anywhere in the code:
    # _ingest() ran sensory/association/chronos logic and never touched
    # bio._hormones at all. Ordinary conversation produced zero hormonal
    # response -- the PAD landscape had nothing to disturb it except
    # decay-toward-baseline, self-study's faint trickle, and manual
    # Stimulus triggers, which is almost certainly why felt-state movement
    # read as flat. Fixed via _react_to_input() (§ Core Emergence
    # Principle note there): deterministic, rule-based reaction keyed off
    # signals sensory.py already computes (message length as an intensity
    # proxy, detected relational/negation edges as emotional-salience
    # signals) -- no new NLP/sentiment inference, consistent with the
    # engine's no-black-box constraint. Deliberately larger than
    # self-study's bump, restoring the size relationship §5.1 always
    # assumed but which the code never actually implemented. Same
    # "not yet numerically tuned" placeholder status as everything else
    # (§10) -- these are first-pass values, not claimed-final.
    ENGAGEMENT_DOPAMINE_BUMP = 0.08
    ENGAGEMENT_AROUSAL_SCALE = 0.05       # scaled by message length, capped
    ENGAGEMENT_LENGTH_NORMALIZER = 100.0  # chars; length_factor = min(len/this, 1.0)
    RELATIONAL_CORTISOL_BUMP = 0.05       # violates / responsible-for: stress/guilt-adjacent
    RELATIONAL_AROUSAL_BUMP = 0.04        # concerns-other: social salience
    TEMPORAL_CONTRAST_DOPAMINE_DELTA = 0.03  # temporal-contrast: bittersweet/nostalgia-adjacent
    NEGATION_CORTISOL_BUMP = 0.05         # being corrected is mildly stressful

    # Self-study saturation fix (this revision). Retry a few different
    # candidates within the same tick before giving up entirely, rather
    # than wasting the whole tick on one dead-end pick. The soft cap is an
    # escape valve: once the strict out_degree<3 pool (see
    # _select_self_study_target) is fully exhausted -- every remaining
    # non-barren node already at the cap -- allow a bounded amount of
    # further growth rather than permanently halting, without reopening
    # unlimited runaway-hub growth the strict cap exists to prevent.
    SELF_STUDY_MAX_ATTEMPTS = 3
    SELF_STUDY_SOFT_CAP = 6

    # Activation / working-memory rendering default (§11 pull-forward,
    # this revision). How many top-activation nodes the Graph tab renders
    # by default, before a "show full graph" opt-in override.
    WORKING_MEMORY_DEFAULT_SIZE = 40
    # Self-study's own activation touch, deliberately smaller than
    # archivist.ACTIVATION_BOOST (the default used for real input) --
    # same gentler-than-external-input pattern as SELF_STUDY_DOPAMINE_BUMP.
    ACTIVATION_BOOST_SELF_STUDY = 0.4

    # §6.1 / §6.2 gate parameters. Same "not yet numerically tuned"
    # category as everything else in §10 -- placeholders, documented.
    NAMING_WINDOW = 20
    NAMING_MIN_OCCURRENCES = 5
    NAMING_CONSISTENCY_THRESHOLD = 0.7
    SCHEMA_NODES_REQUIRED_FOR_MATURITY = 3

    def __init__(self):
        self.bio = BioSystem()
        self.archivist = ArchivistModule()
        self.executive = ExecutiveModule(self.bio, self.archivist)
        self.chronos = ChronosModule()
        self.synthesizer = SynthesizerModule()
        self.reflector = ReflectorModule(self.chronos, self.archivist)
        self.sensory = SensoryModule()
        self.association = AssociationEngine(self.archivist, self.sensory)
        self.stimulus = SyntheticStimulusEngine(self.bio, self.archivist, self.reflector)

        self.pulse_count = 0
        self.fatigue = 0.0
        self.state = "Learning"  # Learning, Consolidation, Pruning

        # Per-basin anchor nodes accumulated as input is ingested under a
        # given felt state (§4.2's "stable felt-state -> node anchor
        # established in Childhood"). {basin_key: [node, ...]}
        self.felt_state_anchors = {}

        # Pending regulation attempts awaiting efficacy evaluation at the
        # next Consolidation pass (§4.5: "evaluated during Consolidation
        # only... over the ticks following a regulation attempt").
        self._pending_regulation = None

        # Hysteresis state for the regulation spike trigger (§4.1) -- same
        # banded pattern as fatigue's T1/T2, so a signal hovering right at
        # threshold doesn't fire regulation every other tick.
        self._regulating = False

        # Queue of external input waiting to be ingested this Learning
        # tick; when empty, self-study fires instead (§5.1).
        self._input_queue = []

        # Self-study saturation fix (this revision, found from production
        # data: node growth stalled ~104 nodes despite thousands of
        # Learning-state pulses, throughput ~0.1 edges/pulse). Root cause:
        # has_room()'s out_degree<3 cap is deliberate (prevents runaway
        # hub growth, see _select_self_study_target's docstring), but
        # self-study had no memory of which capped-out-of-room OR
        # zero-hyponym ("barren") nodes it had already tried. Once the
        # few productive, many-hyponym hub words hit the degree cap, a
        # growing fraction of random picks landed on WordNet leaf terms
        # (e.g. "brougham", "trolley coach" -- real hyponyms of "bus", but
        # themselves childless) that silently produce nothing, forever,
        # since the same dead ends kept getting re-picked. Tracked here so
        # a verified-empty target is never re-selected again.
        self._barren_self_study_targets = set()

        print("Prometheus Core Initialized with Fatigue Cycling")

    # ------------------------------------------------------------------
    # External input entry point (used by app.py / tests)
    # ------------------------------------------------------------------
    def queue_input(self, text: str, source: str = "user"):
        self._input_queue.append((text, source))

    def _ingest(self, text: str, source: str):
        """Runs one piece of text through sensory + association + chronos
        linking. Despite this docstring previously claiming to be "shared
        by both externally-queued input and self-study," it never
        actually was -- _self_study() has always called
        association.place_node() directly, bypassing this method
        entirely. Corrected here rather than left misleading."""
        self.sensory.ingest(text)
        basin_key = self.synthesizer.get_current_basin_key()
        felt_state = self.synthesizer.get_current_felt_state()
        anchor = None
        if felt_state != "Unformed":
            anchors = self.felt_state_anchors.get(basin_key, [])
            anchor = anchors[-1] if anchors else None

        result = self.association.place_node(text, definition="", source=source, context_node=anchor)
        node = result.get("term")
        if node:
            self.archivist.bump_activation(node)
        if anchor:
            self.archivist.bump_activation(anchor)

        # §2.1b item 4a: try to name any unnamed schemas tied to the felt
        # state active right now (schema naming trigger when user/dictionary
        # input provides a word while "in" that state).
        if node and source in ("user", "dictionary"):
            self.association.try_name_schemas(node, current_felt_state=felt_state)

        relations = self.sensory.detect_relational(text)
        if relations:
            self.association.link_relational(node, relations, source=source, felt_state=felt_state)

        if felt_state != "Unformed" and node:
            self.chronos.record_felt_state_link(basin_key, node)
            self.felt_state_anchors.setdefault(basin_key, []).append(node)

        # Explicit negation/correction (§3.4 mechanism 1): flag whatever
        # node was most recently active for gradual demotion at the next
        # Consolidation pass.
        text_lower = text.lower()
        negation_flagged = ("no, " in text_lower or "actually" in text_lower or "that's wrong" in text_lower)
        if negation_flagged and anchor:
            self.archivist.flag_negation(anchor)

        # Hormonal reaction to real input (new, this revision) -- only for
        # genuine externally-triggered input, not dictionary-sourced
        # self-study expansion text, which self-study's own (deliberately
        # smaller) dopamine bump already covers separately.
        if source == "user":
            self._react_to_input(text, relations, negation_flagged)

        return node

    def _react_to_input(self, text: str, relations: List[str], negation_flagged: bool):
        """Deterministic, rule-based hormonal reaction to real
        conversational input (§ Core Emergence Principle: this must stay
        rule-based, no sentiment-analysis/NLP model -- the same
        constraint that already governs sensory.py's negation/relational
        detection). Fixes the root cause behind "minimal emotional
        movement": previously nothing in _ingest() touched bio._hormones
        at all, so ordinary conversation produced zero somatic reaction --
        only self-study's faint trickle and manual Stimulus events ever
        moved the PAD landscape away from its decay-toward-baseline
        equilibrium.

        Signals used, all already computed elsewhere (no new inference):
          - message length, as a coarse intensity/engagement proxy (longer
            messages read as more arousing/engaging, not "understood" in
            any semantic sense -- just a deterministic magnitude signal).
          - detected relational edges (§2.1b, via sensory.detect_relational,
            already called by the caller): violates/responsible-for read
            as stress/guilt-adjacent (cortisol up); concerns-other reads
            as socially salient (mild arousal up); temporal-contrast reads
            as bittersweet/nostalgia-adjacent (small dopamine shift).
          - explicit negation/correction (§3.4): being corrected is mildly
            stressful (cortisol up).
        Every delta is small and clamped -- this is meant to restore
        *some* reactivity, not replace Stimulus's deliberate, larger
        manual events."""
        length_factor = min(len(text) / self.ENGAGEMENT_LENGTH_NORMALIZER, 1.0)

        with self.bio.lock:
            h = self.bio._hormones
            h["dopamine"] = min(1.0, h["dopamine"] + self.ENGAGEMENT_DOPAMINE_BUMP)
            h["adrenaline"] = min(1.0, h["adrenaline"] + self.ENGAGEMENT_AROUSAL_SCALE * length_factor)

            if "violates" in relations or "responsible-for" in relations:
                h["cortisol"] = min(1.0, h["cortisol"] + self.RELATIONAL_CORTISOL_BUMP)
            if "concerns-other" in relations:
                h["adrenaline"] = min(1.0, h["adrenaline"] + self.RELATIONAL_AROUSAL_BUMP)
            if "temporal-contrast" in relations:
                h["dopamine"] = min(1.0, h["dopamine"] + self.TEMPORAL_CONTRAST_DOPAMINE_DELTA)
            if negation_flagged:
                h["cortisol"] = min(1.0, h["cortisol"] + self.NEGATION_CORTISOL_BUMP)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def pulse(self):
        self.pulse_count += 1
        somatic = self.bio.step()

        # synthesizer must run first, before anything that conditions a
        # decision on its output (regulation trigger, executive bias) --
        # previously this ran *after* those checks, which meant they were
        # either reading last tick's synthesized state or (as fixed here)
        # reading raw hidden-layer data directly. Per the Core Emergence
        # Principle, prometheus.py and executive.py must only condition on
        # synthesizer.py's output, never on `somatic` directly.
        self.synthesizer.update_from_core(self.bio.get_raw_variables())
        intensity = self.synthesizer.get_current_intensity()

        bias = self.executive.bias_processing(intensity)

        # §4.1: hysteresis-banded spike detection on the synthesized
        # intensity signal, same pattern as fatigue's T1/T2 -- not a bare
        # threshold, and not somatic.urgency.
        if not self._regulating and intensity > self.REGULATION_SPIKE_THRESHOLD:
            self._regulating = True
        elif self._regulating and intensity < self.REGULATION_SPIKE_THRESHOLD - self.REGULATION_HYSTERESIS:
            self._regulating = False
        if self._regulating:
            self._apply_regulation(intensity)

        override = self.reflector.issue_directive(bias)
        if override != bias:
            bias = override

        if self.state == "Learning":
            if self._input_queue:
                text, source = self._input_queue.pop(0)
                self._ingest(text, source)
            else:
                self._self_study()

        self.chronos.record_pulse(
            somatic, bias,
            felt_state=self.synthesizer.get_current_felt_state(),
            avd=self.synthesizer.get_current_basin_key(),
        )

        self._update_fatigue()
        self._cycle_state()
        self.maybe_advance_epoch()

        results = self.archivist.retrieve("context")

        print(
            f"Pulse {self.pulse_count} | Epoch: {self.bio.epoch.value} | "
            f"State: {self.state} | Bias: {bias} | Fatigue: {self.fatigue:.2f} | "
            f"Felt: {self.synthesizer.get_current_felt_state()}"
        )
        return {
            "pulse": self.pulse_count,
            "bias": bias,
            "state": self.state,
            "epoch": self.bio.epoch.value,
            "felt_state": self.synthesizer.get_current_felt_state(),
        }

    # ------------------------------------------------------------------
    # §5.1 Autonomous idle expansion (self-study)
    # ------------------------------------------------------------------
    def _self_study(self):
        """During Learning, when no external input is queued, self-
        initiate dictionary expansion rather than sitting idle. Does NOT
        directly drain a fatigue counter -- it triggers a small hormonal
        reaction (dopamine bump) through the normal fast-layer pathway,
        and fatigue rises as a *consequence* of that, same as everything
        else (§5.1).

        Saturation fix (this revision): previously picked exactly one
        target and gave up silently if it had no WordNet hyponyms, with
        no memory of the attempt -- so once the graph's few productive,
        many-hyponym hub words hit the degree cap, an increasing fraction
        of ticks landed on WordNet leaf terms (real hyponyms with no
        hyponyms of their own) and produced nothing, forever, because the
        same dead ends kept getting re-picked. Now retries up to
        SELF_STUDY_MAX_ATTEMPTS different candidates per tick and
        memoizes any confirmed-barren target in
        self._barren_self_study_targets, permanently excluding it from
        future selection (see _select_self_study_target's has_room)."""
        target = None
        expansions = []
        for _ in range(self.SELF_STUDY_MAX_ATTEMPTS):
            target = self._select_self_study_target()
            if target is None:
                return
            expansions = self.sensory.lookup_expansion(target)
            if expansions:
                break
            # Verified dead end -- memoize so this specific node is never
            # wastefully re-picked again, freeing the random-selection
            # pool toward nodes that can actually still produce children.
            self._barren_self_study_targets.add(target)
            target = None

        if target is None or not expansions:
            return  # every attempt this tick hit a confirmed dead end

        # Anchor fix (this revision, found from production data after the
        # regulation-eligibility fix): felt_state_anchors was previously
        # only ever populated inside _ingest(), which only runs for
        # explicitly queued user/dictionary input -- despite _ingest's own
        # docstring claiming to be "shared by both externally-queued input
        # and self-study" (it never actually was). Under typical usage,
        # the overwhelming majority of Learning ticks are self-study, not
        # queued input (Run Batch queues nothing), so felt_state_anchors
        # stayed effectively empty. Once regulation was correctly scoped
        # to anchored nodes only (previous fix), this meant it almost
        # always found zero eligible candidates instead of the whole
        # graph -- regulatory efficacy sitting at the untouched 0.5
        # default for every node, never exercised at all, which is worse
        # than the original bug in practice even though more "correct."
        # Fixed by recording the same felt-state -> node anchor link
        # _ingest() does, for self-study's own placements.
        basin_key = self.synthesizer.get_current_basin_key()
        felt_state = self.synthesizer.get_current_felt_state()
        placed_children = []
        for child in expansions[:3]:
            definition = self.sensory.lookup_definition(child) or ""
            result = self.association.place_node(child, definition=definition, source="dictionary",
                                                   context_node=target)
            placed_children.append(result.get("term") or child)
        self.archivist.store(target, source="dictionary")  # reinforce parent's last_reinforced

        if felt_state != "Unformed":
            for child_node in placed_children:
                self.chronos.record_felt_state_link(basin_key, child_node)
                self.felt_state_anchors.setdefault(basin_key, []).append(child_node)
            # `target` recurs across multiple self-study ticks (until it
            # hits the degree cap), unlike each tick's freshly-created
            # children -- anchoring it too gives §6.1's naming-reliability
            # check and §4.2's regulation candidate pool a genuinely
            # consistent, repeatedly-reinforced node to work with, not
            # just a growing list of one-off terms.
            self.chronos.record_felt_state_link(basin_key, target)
            self.felt_state_anchors.setdefault(basin_key, []).append(target)

        # Activation touch (§11 pull-forward, this revision) -- smaller
        # than real input's default bump (archivist.ACTIVATION_BOOST),
        # same "gentler than externally-triggered" pattern already used
        # for the hormonal bump just below.
        self.archivist.bump_activation(target, self.ACTIVATION_BOOST_SELF_STUDY)
        for child_node in placed_children:
            self.archivist.bump_activation(child_node, self.ACTIVATION_BOOST_SELF_STUDY)

        # Small, scaled-down dopaminergic bump (§5.1, §9 risk 7) via the
        # same fast-layer pathway as any other event -- no bespoke
        # self-study fatigue tap.
        with self.bio.lock:
            self.bio._hormones["dopamine"] = min(
                1.0, self.bio._hormones["dopamine"] + self.SELF_STUDY_DOPAMINE_BUMP
            )

    def _select_self_study_target(self, hard_cap: int = 3):
        """(a) active/trusted nodes with few children, or (b) emotionally
        salient nodes weighted by *current* felt state (§5.1) -- historical
        emotional weighting stays inside Consolidation, not here.

        Fixes a real bug found by running the system: the out_degree<3 cap
        previously only applied to the primary (Working+ tier) candidate
        filter. Both fallback paths had no cap at all, so once that pool
        emptied out (which happens fast -- a dictionary-sourced node clears
        the Working threshold on its own base score alone, no corroboration
        needed), self-study fell through to fallbacks that could keep
        piling unlimited children onto an already-large hub -- producing
        exactly the runaway starburst clusters seen in testing.

        Second, related bug: requiring tier>=Working to even be a
        *candidate* structurally excluded every fresh/user-typed node,
        which starts Provisional. That's a chicken-and-egg deadlock --
        self-study is one of the main ways a node accumulates the
        corroboration needed to promote past Provisional, but a Provisional
        node could never be selected for self-study in the first place.
        This is why user input sat in a disconnected, unexpanded chain
        while dictionary hubs absorbed all self-study attention.

        Third fix, this revision: "room" is now counted on categorical
        out-edges only (is-a/part-of/associated-with), not relational
        (responsible-for/violates/etc.) or composed-of edges -- a node
        shouldn't be treated as "full" for hierarchy-branching purposes
        because it happens to carry unrelated relational/schema edges.
        Also excludes any node memoized in self._barren_self_study_targets
        (confirmed zero WordNet hyponyms, §5.1 saturation fix) so dead
        ends stop consuming picks from the random-selection pool.

        Fourth fix, this revision (§11 pull-forward, in response to "learn
        from a focused group" -- previously self-study picked uniformly at
        random within each eligible pool, which is closer to weighted-
        random than genuine attention/focus, per §11's own critique of
        itself). Selection within working_candidates/provisional_candidates
        is now weighted by each node's activation score (§ archivist.py's
        new activation layer) via _weighted_choice_by_activation() --
        nodes touched recently (real input, prior self-study, regulation
        use) are preferentially re-expanded, while an epsilon floor keeps
        untouched nodes from being permanently excluded (still
        exploration, not pure exploitation).

        `hard_cap` lets the (e) escape-valve fallback below retry with a
        looser ceiling once the strict cap has genuinely exhausted every
        productive node, rather than permanently halting growth.
        """
        graph = self.archivist.graph
        if graph.number_of_nodes() == 0:
            return None

        def categorical_out_degree(n):
            return sum(
                1 for _u, _v, edata in graph.out_edges(n, data=True)
                if edata.get("relation_type") in CATEGORICAL_EDGE_TYPES
            )

        def has_room(n, d):
            return (
                categorical_out_degree(n) < hard_cap
                and not d.get("is_schema")
                and n not in (SELF_NODE, OTHER_NODE)
                and n not in self._barren_self_study_targets
            )

        working_candidates = [
            n for n, d in graph.nodes(data=True)
            if d.get("tier", 0) >= TIER_WORKING and has_room(n, d)
        ]
        # (b) NEW: Provisional nodes with room, source-tagged non-self-generated
        # (i.e. real user/dictionary input, not the agent's own prior
        # self-study output) -- gives fresh input a genuine shot at
        # self-study attention instead of waiting on a tier it may never
        # reach without exactly this kind of reinforcement.
        provisional_candidates = [
            n for n, d in graph.nodes(data=True)
            if d.get("tier", 0) < TIER_WORKING and d.get("source") != "self_generated" and has_room(n, d)
        ]

        if working_candidates and provisional_candidates:
            # Weighted toward provisional: established hubs already got
            # their initial attention, fresh nodes need it more. Not a
            # tuned ratio (§10) -- worth a slider if this needs finer
            # control later.
            pool = provisional_candidates if random.random() < 0.6 else working_candidates
            return self._weighted_choice_by_activation(pool)
        if provisional_candidates:
            return self._weighted_choice_by_activation(provisional_candidates)
        if working_candidates:
            return self._weighted_choice_by_activation(working_candidates)

        # (c) fallback: whatever node is currently anchoring the felt
        # state, if any -- but only if it also still has room. Previously
        # uncapped, which was the main source of runaway single-node growth.
        key = self.synthesizer.get_current_basin_key()
        anchors = self.felt_state_anchors.get(key, [])
        if anchors:
            anchor = anchors[-1]
            if anchor in graph and has_room(anchor, graph.nodes[anchor]):
                return anchor

        # (d) last resort: any node with room, not just any node at all
        # (same fix -- this used to be truly uncapped).
        low_degree_any = [n for n, d in graph.nodes(data=True) if has_room(n, d)]
        if low_degree_any:
            return random.choice(low_degree_any)

        # (e) escape valve (this revision): (a)-(d) all failed, meaning
        # every non-barren node in the graph is already at hard_cap
        # categorical children. Rather than permanently halting growth
        # (the actual production symptom this fix addresses), retry once
        # with a softer ceiling -- still bounded, so this doesn't reopen
        # the unlimited-runaway-hub risk the strict cap exists to prevent,
        # it just means "everything productive is capped" isn't a
        # permanent dead end for the whole system.
        if hard_cap < self.SELF_STUDY_SOFT_CAP:
            return self._select_self_study_target(hard_cap=self.SELF_STUDY_SOFT_CAP)

        return None

    def _weighted_choice_by_activation(self, pool: List[str]) -> Optional[str]:
        """Activation-weighted random choice (§11 pull-forward, this
        revision) -- replaces uniform random.choice() for self-study
        target selection so recently-touched nodes are preferentially
        re-expanded, giving self-study something closer to genuine
        attention/focus. An epsilon floor (0.1) on every weight keeps
        untouched nodes selectable at nonzero probability -- this stays
        exploration-with-a-bias, not pure exploitation of whatever's
        already active, which would risk narrowing the graph's growth to
        an ever-smaller hot set over time."""
        if not pool:
            return None
        weights = [self.archivist.graph.nodes[n].get("activation", 0.0) + 0.1 for n in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Fatigue / state cycling
    # ------------------------------------------------------------------
    def _update_fatigue(self):
        """Fatigue growth (§5) previously read somatic.urgency directly --
        the raw SomaticReadout returned by bio.step(), i.e. hidden-layer
        output that bypasses synthesizer.py entirely. Every other
        decision point in this file (regulation §4.1, executive bias) was
        already careful to route only through
        synthesizer.get_current_intensity(); fatigue was the one
        exception, in real violation of the Core Emergence Principle
        despite this file's own docstring/comments elsewhere insisting on
        it. Fixed to use the same synthesized intensity signal (arousal
        component of the current basin key, §2.1a) as everything else --
        no raw hidden-layer read anywhere in this method now."""
        intensity = self.synthesizer.get_current_intensity()
        self.fatigue = min(1.0, self.fatigue + intensity * self.FATIGUE_GROWTH_RATE)

    def _cycle_state(self):
        """Hysteresis-banded state cycling (§5 stability requirement)."""
        if self.state == "Learning":
            if self.fatigue >= self.T1:
                self.state = "Consolidation"
        elif self.state == "Consolidation":
            if self.fatigue >= self.T2:
                self.state = "Pruning"
            elif self.fatigue < self.T1 - self.HYSTERESIS:
                self.state = "Learning"
        elif self.state == "Pruning":
            if self.fatigue < self.T2 - self.HYSTERESIS:
                self.state = "Consolidation"

        if self.state == "Consolidation":
            self._run_consolidation()
            # §5: "fatigue must have its own recovery curve (drops during
            # Consolidation) so the system self-cycles rather than
            # ratcheting into permanent Pruning." This was named as a
            # requirement in the design but never implemented -- Consolidation
            # applied zero recovery, while Pruning (the costlier, more
            # effortful state) recovered 30%. That inversion created a
            # stable Consolidation<->Pruning oscillation that never dropped
            # back below the Learning re-entry threshold, so Learning
            # (and therefore all graph growth) became unreachable after
            # the first few ticks. Consolidation should recover more than
            # Pruning, since it's the restorative state, not the costly one.
            # Rate is an undecided tuning placeholder (§10) -- exposing
            # this via a debug-tab slider is the planned next step rather
            # than guessing a "correct" number here.
            self.fatigue *= self.FATIGUE_RECOVERY_CONSOLIDATION
        elif self.state == "Pruning":
            pruned = self.archivist.prune()
            if pruned:
                print(f"Pruning: removed {pruned} stale Tier-0 node(s).")
            self.fatigue *= self.FATIGUE_RECOVERY_PRUNING  # Recovery (§5: "fatigue must have its own recovery curve")

    def _run_consolidation(self):
        """
        Everything the spec pins to the Consolidation clock, in one place
        (§5, §3.3, §2.3 mechanism 3, §4.5, §2.1b, §5's slow-layer baseline
        note) -- "one clock, not several" per the design's own governing
        principle (see conversation summary).
        """
        self.synthesizer.consolidate_basins()
        trust_summary = self.archivist.run_consolidation_pass()
        reparented = self.association.run_reparenting_pass()
        new_schemas = self.reflector.detect_schemas()
        self._evaluate_pending_regulation()
        # Activation decay (§11 pull-forward, this revision) -- same
        # Consolidation clock as basin/trust/schema/efficacy evaluation,
        # per the design's own "one clock, not several" principle.
        self.archivist.decay_activation()

        # §4C: the single checkpoint call for this pass -- everything
        # above mutates the graph and/or hormonal state without saving
        # individually (see the "No self.save() here" comments in
        # archivist.py, reflector.py, and hormonal.py's step()). This is
        # the one clock persistence is gated to.
        self.archivist.save()
        self.bio.save_state()
        self.synthesizer.save_state()

        if trust_summary.get("promotions") or trust_summary.get("demotions"):
            print(f"Consolidation trust pass: {trust_summary}")
        if reparented:
            print(f"Consolidation: re-parented {reparented} node(s).")
        if new_schemas:
            print(f"Consolidation: formed {len(new_schemas)} new Schema Node(s): {new_schemas}")

    # ------------------------------------------------------------------
    # §4 Regulation
    # ------------------------------------------------------------------
    def _apply_regulation(self, intensity: float):
        """
        Regulation per §4: accelerates core.py's fast-layer decay,
        restricted to Working/Trusted-tier nodes anchored to the current
        felt state (§4.2), capped and scaled by regulatory efficacy
        (§4.4/§4.5), costs fatigue (§4.6). Takes the synthesized intensity
        signal (synthesizer.get_current_intensity()), never raw somatic
        data -- see the Core Emergence Principle note on pulse().
        """
        key = self.synthesizer.get_current_basin_key()
        anchors = self.felt_state_anchors.get(key, [])
        # Bug fix (found from production data: every node's regulatory
        # efficacy sat at exactly the same value, 0.05 below the 0.5
        # default, across the entire eligible pool -- only possible if a
        # single event nudged literally everyone at once). `anchors or
        # None` treated an empty anchor list the same as "no restriction
        # requested," so whenever no felt-state anchor had been recorded
        # yet (common, especially before Childhood naming has happened
        # for a given basin), eligible_regulation_nodes(None) fell back to
        # *every* Working/Trusted-tier node in the graph -- not the
        # felt-state-scoped set §4.2 specifies. Passing `anchors` directly
        # (even when empty) means an empty anchor list correctly produces
        # zero eligible nodes, which hits the pre-existing "legitimate
        # state, nothing eligible yet" early-return below instead.
        regulating_nodes = self.archivist.eligible_regulation_nodes(anchors)

        if not regulating_nodes:
            # §4.2: legitimate state (nothing eligible yet), not an error.
            return

        # Activation touch (§11 pull-forward, this revision): a node
        # actually used for regulation is clearly currently relevant --
        # feeds back into self-study's activation-weighted targeting and
        # the Graph tab's focused rendering.
        for n in regulating_nodes:
            self.archivist.bump_activation(n)

        avg_efficacy = sum(
            self.archivist.graph.nodes[n].get("regulatory_efficacy", 0.5) for n in regulating_nodes
        ) / len(regulating_nodes)

        # §4.4: capped, not instant -- scaled by efficacy, never fully
        # flattens a spike in one tick.
        rate = self.REGULATION_DAMPENING_CAP * avg_efficacy
        self.bio.decay_fast(rate=rate)

        # §4.6: regulation costs fatigue, same economy as self-study.
        self.fatigue = min(1.0, self.fatigue + self.REGULATION_FATIGUE_COST)

        self._pending_regulation = {
            "nodes": regulating_nodes,
            "intensity_before": intensity,
            "pulse": self.pulse_count,
        }
        print(f"Regulation applied via {len(regulating_nodes)} eligible node(s), rate={rate:.3f}.")

    def _evaluate_pending_regulation(self):
        """§4.5: efficacy evaluated during Consolidation only -- check
        whether felt-state intensity dropped faster than baseline decay
        alone would predict, over the ticks following the attempt. Uses
        the same synthesized intensity signal regulation was triggered on,
        not raw somatic data."""
        pending = self._pending_regulation
        if not pending:
            return
        current_intensity = self.synthesizer.get_current_intensity()
        dropped = pending["intensity_before"] - current_intensity
        # Simple baseline-decay proxy: natural decay toward 0.5 baseline
        # over the elapsed ticks would account for some drop on its own;
        # only credit regulation if the drop clearly exceeds that.
        worked = dropped > 0.15
        for node in pending["nodes"]:
            self.archivist.update_regulatory_efficacy(node, worked)
        self._pending_regulation = None

    # ------------------------------------------------------------------
    # §6 Epoch transitions -- cross-layer, owned by prometheus.py only.
    # ------------------------------------------------------------------
    def maybe_advance_epoch(self):
        if self.bio.epoch == Epoch.CHILDHOOD:
            if self._childhood_gate_met():
                self._advance_to(Epoch.ADOLESCENCE)
        elif self.bio.epoch == Epoch.ADOLESCENCE:
            if self._adolescence_gate_met():
                self._advance_to(Epoch.MATURITY)

    def _childhood_gate_met(self) -> bool:
        """§6.1: a stabilized basin (§2.1a) that reliably/consistently
        links to the same knowledge node across repeated occurrences."""
        for key in self.chronos.all_linked_basins():
            if key not in self.synthesizer.stabilized_basins:
                continue  # basin stabilization is itself a precondition
            node, consistency, occurrences = self.chronos.naming_reliability(
                key, window=self.NAMING_WINDOW, min_occurrences=self.NAMING_MIN_OCCURRENCES
            )
            if node is not None and consistency >= self.NAMING_CONSISTENCY_THRESHOLD:
                return True
        return False

    def _adolescence_gate_met(self) -> bool:
        """§6.2: Schema Node formation (§2.1b), not a raw regulation-event
        count or variance placeholder."""
        return self.reflector.schema_count() >= self.SCHEMA_NODES_REQUIRED_FOR_MATURITY

    def _advance_to(self, epoch: Epoch):
        print(f"Epoch transition: {self.bio.epoch.value} -> {epoch.value}")
        self.bio.epoch = epoch
        # §5: "slow-layer hormonal baseline shifts happen here [at
        # Consolidation-adjacent points], not instantly at epoch
        # transition" -- modeled as a gentle nudge rather than a snap.
        if epoch == Epoch.ADOLESCENCE:
            self.bio.shift_slow_baseline({"testosterone": 0.05, "estrogen": 0.05})
        elif epoch == Epoch.MATURITY:
            self.bio.shift_slow_baseline({"serotonin": 0.05, "oxytocin": 0.05})
        self.bio.save_state()

    # ------------------------------------------------------------------
    def run(self, num_pulses=10):
        for _ in range(num_pulses):
            self.pulse()
        print("Run complete.")

    @staticmethod
    def reset_persistent_memory():
        """Deletes every module's on-disk checkpoint (§4C): the knowledge
        graph, chronos's rolling log, hormonal's slow-layer baseline +
        epoch, and the basin/schema landscape. Does NOT touch a live
        instance's in-memory state -- callers must also discard their
        current Prometheus() object and create a fresh one (e.g. clear
        st.session_state.prom in app.py) for a reset to actually take
        effect, since __init__ only loads from disk once, at creation.
        Safe to call even if some/all files don't exist yet. Returns the
        list of paths actually removed, for a confirmation message."""
        from .archivist import EPISTEMIC_GRAPH_PATH
        from .chronos import CHRONOS_LOG_PATH
        from .hormonal import BIOSYSTEM_STATE_PATH
        from .synthesizer import BASIN_STATE_PATH

        removed = []
        for path in (EPISTEMIC_GRAPH_PATH, CHRONOS_LOG_PATH, BIOSYSTEM_STATE_PATH, BASIN_STATE_PATH):
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)
        return removed
