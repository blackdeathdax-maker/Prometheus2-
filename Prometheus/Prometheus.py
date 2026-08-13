import os
import random
import logging
from collections import deque
from typing import List, Optional

from .hormonal import BioSystem, Epoch
from .archivist import ArchivistModule, TIER_WORKING, TIER_TRUSTED, SELF_NODE, OTHER_NODE
from .executive import ExecutiveModule
from .synthesizer import SynthesizerModule
from .reflector import ReflectorModule
from .chronos import ChronosModule
from .working_memory import WorkingMemoryModule
from .self_narrative import NarrativeModule
from .focus import FocusModule
from .somatic_topo import SomaticTopo
from .felt_anchors import FeltAnchorStore
from .modulators import FastModulators
from .long_term_interest import LongTermInterest
from .schema_felt import SchemaFeltBinder
from .sensory import SensoryModule
from .association import AssociationEngine
from .stimulus import SyntheticStimulusEngine


logger = logging.getLogger(__name__)


class Prometheus:
    """
    Orchestrator (§7). Owns cross-layer epoch transition checks and tick
    sequencing. Per the Core Emergence Principle, this class must only
    condition its decisions on visible-layer felt states (synthesizer's
    output) -- never on bio._hormones or bio.get_raw_variables() directly.
    """

    # Fatigue state-cycling thresholds with hysteresis margins (spec §5).
    # --- Sleep pressure / micro-day cycle (replaces pure T1/T2 sawtooth) ---
    # Soft threshold band (no urgency → sleep sooner). Hard max always forces sleep.
    SLEEP_SOFT_MIN = 0.35          # low-urgency ceiling (was near T1)
    SLEEP_HARD_MAX = 0.92          # mandatory sleep
    SLEEP_WAKE_BELOW = 0.22        # exit sleep climate when pressure under this (scaled by debt)
    HYSTERESIS = 0.05
    # Legacy aliases so Debug sliders / old docs still map
    T1 = 0.35
    T2 = 0.75

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
    FATIGUE_RECOVERY_SLEEP = 0.55          # per-pulse pressure drop in sleep climate
    FATIGUE_URGENCY_GROWTH_MULT = 1.75     # high urgency accelerates pressure
    MICRO_DAY_PULSES = 60                  # lab micro-day length (~hour/16 at 1s ticks)
    SLEEP_FRACTION_DEFAULT = 0.33          # target share of micro-day in sleep climate

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

    # Self-study arousal/dominance touch (new, this revision). Previously
    # self-study ONLY ever touched dopamine -- adrenaline, cortisol, and
    # testosterone (the hormones driving the arousal and dominance PAD
    # axes, §2.1a) never moved from autonomous activity at all. Confirmed
    # at production scale (2700+ nodes, almost entirely self-study-driven
    # growth): those hormones simply decayed toward the 0.5 baseline every
    # tick with nothing ever counteracting it, so only valence showed any
    # autonomous movement. Fixed with a single small adrenaline touch --
    # per hormonal.py's own raw-variable mapping (get_raw_variables),
    # adrenaline alone already feeds BOTH heart_rate (arousal) and
    # vascular_constriction (dominance), so one bump addresses both frozen
    # axes without a second hormone. Deliberately does NOT touch
    # testosterone: it's a slow-layer hormone (meant to represent
    # temperament, shifted only deliberately via shift_slow_baseline,
    # currently just at epoch transitions), not something that should
    # move live on every self-study tick -- same boundary already
    # respected when designing the parental-feedback mechanism (§13.2).
    # An order of magnitude smaller than the dopamine bump, so valence
    # stays the clearly dominant self-study signal (curiosity/
    # satisfaction, per §5.1's own framing) -- this just keeps arousal/
    # dominance from going completely flat during long batch-only runs,
    # it doesn't make self-study a real driver of felt-state exploration
    # the way genuine engagement is.
    SELF_STUDY_AROUSAL_BUMP = 0.025
    SELF_STUDY_CORTISOL_BUMP = 0.008   # mild load while working
    SELF_STUDY_SEROTONIN_BUMP = 0.006  # small settle on successful expand
    SELF_STUDY_TESTOSTERONE_BUMP = 0.004  # slow stance — rare micro-nudge
    FOCUS_PRED_DOPAMINE = 0.01         # prediction residual inject path
    FOCUS_STAGNANT_CORTISOL = 0.012    # stuck focus → stress load
    FOCUS_SWITCH_DOPAMINE = 0.008      # leaving a focus → small release
    OXTYOCIN_PARENTAL_EXTRA = 0.02     # warmth/approval affiliation

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

    # Parental emotional feedback (§13.2, new) -- "mirror neuron" style
    # implicit guidance. The reaction itself is JUST a small, deterministic
    # hormone nudge (same fast-layer-only pattern as ENGAGEMENT_* above --
    # slow-layer/temperament drift from repeated reactions is a plausible
    # future extension, deliberately not built here, to keep this change
    # the same size/shape as the proven _react_to_input pattern rather
    # than opening a second new mechanism at once). What makes this
    # *learning* rather than just mood noise is VALENCE_COLORING_STEP,
    # applied separately in give_parental_reaction() to whichever node(s)
    # are the CURRENT felt-state anchor -- see archivist.nudge_valence_
    # coloring()'s docstring. No word or category ever gets a hand-
    # assigned valence anywhere in this mechanism; coloring only
    # accumulates through repeated real co-occurrence.
    PARENTAL_APPROVAL_DOPAMINE = 0.06
    PARENTAL_DISAPPROVAL_CORTISOL = 0.06
    PARENTAL_WARMTH_DOPAMINE = 0.05
    PARENTAL_WARMTH_CORTISOL_RELIEF = 0.03   # warmth also mildly *reduces* cortisol (safety)
    PARENTAL_CONCERN_AROUSAL = 0.05
    PARENTAL_CONCERN_CORTISOL = 0.03
    VALENCE_COLORING_STEP = 0.15
    VALENCE_COLORING_CAP = 1.0

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

    # Bias-modulated self-study targeting (§13.1, new -- designed but
    # never built until this revision; executive.py's EXPLORE/STABILIZE/
    # NEUTRAL bias signal was computed every tick and logged, but nothing
    # ever consumed it, §10 item 23). Two independent effects, both
    # expressing the same "explore = novelty, stabilize = depth" idea:
    # (a) which tier-pool self-study draws from (below), and (b)
    # _weighted_choice_by_activation's weighting direction within that
    # pool (inverted under EXPLORE -- see that method's docstring).
    # SELF_STUDY_PROVISIONAL_PROB_NEUTRAL preserves the old hardcoded 0.6
    # default exactly, so NEUTRAL bias reproduces prior behavior bit for
    # bit. Same "not yet numerically tuned" placeholder status as
    # everything else (§10).
    SELF_STUDY_PROVISIONAL_PROB_EXPLORE = 0.8
    SELF_STUDY_PROVISIONAL_PROB_STABILIZE = 0.3
    SELF_STUDY_PROVISIONAL_PROB_NEUTRAL = 0.6

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
        self.working_memory = WorkingMemoryModule(self.archivist, self.synthesizer)
        self.self_narrative = NarrativeModule(self.archivist, self.synthesizer)
        try:
            self.self_narrative.load()
        except Exception:
            pass
        self.focus = FocusModule()
        self.working_memory.focus = self.focus  # §13.y WM consumer hook
        self.somatic_topo = SomaticTopo()
        self.felt_anchors = FeltAnchorStore()
        self.modulators = FastModulators()
        self.long_term_interest = LongTermInterest()
        self.schema_felt = SchemaFeltBinder(threshold=3)
        self.last_collapse_summary = {"collapsed": 0, "conflicts": 0, "candidates_considered": 0}
        self.last_focus_summary = {}

        # Barren self-study targets that fell out of dead-end detection's
        # proxy check (§14.6 item 2) need the same tracking self-study's
        # own barren set already uses -- reuse it directly rather than a
        # second structure, see _self_study()'s Childhood-gating block.

        self.pulse_count = 0
        self.fatigue = 0.0  # sleep pressure (continuous)
        self.state = "Learning"  # Learning | Consolidation | Sleep
        self.sleep_stage = "none"  # none | digest | reorganize | homeostatic | wake_prep
        self.sleep_debt = 0.0      # excess pressure carried into sleep (lengthens recovery)
        self.micro_day_pulse = 0
        self._last_urgency = 0.0
        self.load_extended_state()  # restore focus/felt/topo/runtime if present

        # Per-basin anchor nodes accumulated as input is ingested under a
        # given felt state (§4.2's "stable felt-state -> node anchor
        # established in Childhood"). {basin_key: deque(maxlen=ANCHOR_WINDOW_SIZE)}
        #
        # Bug fix (this revision): previously a plain, unbounded list,
        # appended to on every qualifying tick by both _ingest() and
        # _self_study() with no cap and no dedup. Over thousands of
        # pulses in a popular felt state, this list could grow into the
        # hundreds -- and since app.py's Graph tab and _apply_regulation
        # both pass the *entire* list as always_include /
        # eligible-candidate scoping, it eventually swamped both the
        # top-K activation filter (Graph tab rendering never actually
        # stayed focused at scale) and regulation's tier-restricted
        # candidate pool (drifting back toward "most of the graph," the
        # same class of problem the original anchor-scoping fix addressed
        # earlier this session). Every other rolling-history structure in
        # this design (chronos's log, basin dwell-time decay) is
        # deliberately bounded, not unbounded -- this one was an
        # oversight that never got the same treatment. Fixed with a
        # bounded deque per basin via _record_anchor().
        self.felt_state_anchors = {}
        self.ANCHOR_WINDOW_SIZE = 20  # same tuning-placeholder category as everything else (§10)

        # Bug fix (this session): felt_state_anchors is written by BOTH
        # _ingest() (real user/dictionary input) and _self_study() (lines
        # ~649/657), sharing one bounded deque per basin key. Under a Run
        # Batch with few real messages and many self-study ticks (the
        # reported case: 5 real seed words, 255 total pulses), self-
        # study's own volume of appends evicts real anchors out of the
        # window entirely within the first ANCHOR_WINDOW_SIZE (20)
        # self-study touches after they were typed -- long before pulse
        # 255. Two concrete, compounding consequences: (a)
        # get_current_working_memory() can no longer see the real seed
        # words as candidates at all (not just outranked -- evicted from
        # the candidate pool entirely), which is what the person's
        # screenshot showed (7 self-study-derived words, zero of the 5
        # real ones); (b) _select_self_study_target()'s own working-
        # memory-scoped candidate pool (§14, "colors in, Hundred Years'
        # War never comes up") loses its tether for the same reason --
        # self-study drifts away from exactly the topic it was designed
        # to stay anchored to, because its own churn evicted the anchor
        # meant to constrain it.
        #
        # Fixed with a second, small, separately-bounded deque per basin
        # key that ONLY _ingest() ever writes to (self-study never
        # touches it, by construction -- see _record_protected_anchor's
        # only call site). _get_unique_anchors() merges both, so every
        # existing consumer (working-memory display, self-study's own
        # scoping, regulation, parental feedback) keeps seeing real
        # conversational anchors regardless of how much self-study churn
        # happened in between -- without changing felt_state_anchors'
        # existing structure/behavior at all, since every other read site
        # (e.g. _ingest's own `anchors[-1]` immediate-context lookup)
        # still reads the original deque directly, unaffected.
        # Bug fix (this session, revised after testing against the actual
        # reported scenario): initially implemented as a second deque
        # PER BASIN KEY, mirroring felt_state_anchors' own structure.
        # That didn't actually solve the reported problem -- confirmed by
        # reproducing it: real input got correctly recorded, but under
        # whatever basin key was active AT THE MOMENT of typing, and by
        # 255 pulses later the system had drifted to a different key,
        # so the protected entry was invisible again anyway, just for a
        # different reason (wrong-bucket, not evicted). The person's own
        # framing makes the actual need clear: five words typed as
        # deliberate topic-setting input ("colors, emotions, food,
        # animals, plants") should keep the system engaged with those
        # topics regardless of which momentary mood/basin it's currently
        # in -- not be treated as mood-specific remarks that only matter
        # when that exact mood recurs. That's a different, coarser kind
        # of memory than felt_state_anchors was ever designed to hold
        # (basin-scoped association is correct for ITS purpose -- this is
        # a separate need). Implemented here as a single, basin-
        # INDEPENDENT bounded pool instead: real conversational touches
        # stay visible to working memory / self-study scoping across mood
        # changes, until they age out of this small window on their own
        # terms (still bounded -- not a route back to unbounded growth).
        self.PROTECTED_ANCHOR_WINDOW_SIZE = 15  # global pool now (see docstring above) -- bigger than the old per-basin-key allowance made sense to be, since it's the only place real topics persist across mood changes; still a §10-category tuning placeholder
        self._global_protected_anchors = deque(maxlen=self.PROTECTED_ANCHOR_WINDOW_SIZE)

        # Co-activation broadening (§13.3, new). Diagnosed from production
        # data (18 tracked pairs / 0 stabilized after 3833 pulses, 3355
        # nodes): the original co-activation sources (a self-study cycle's
        # target+children, an ingestion's node+anchor) each only ever fire
        # ONCE per target's effective lifetime, because self-study's
        # degree cap excludes a target from future selection almost
        # immediately after it's touched -- so most pairs got exactly one
        # chance to accumulate, never a second or third, and could never
        # cross CO_ACTIVATION_STABILIZATION_THRESHOLD (3) regardless of
        # how long the system ran. Fixed in _record_anchor(): every time
        # any node gets anchored to a felt state, it's ALSO paired with
        # the most recent few distinct anchors already in that felt
        # state's window -- since felt states get genuinely revisited
        # over a run (§2.1a's whole premise), this gives pairs many more
        # natural chances to recur, not just one. Deliberately bounded to
        # a SMALL recent window, not the full ANCHOR_WINDOW_SIZE history:
        # pairing a new touch with all 20 prior anchors would make
        # co-activation nearly synonymous with "ever anchored to the same
        # felt state," diluting a signal meant to capture genuine,
        # temporally-close recurrence into something closer to noise.
        self.CO_ACTIVATION_RECENCY_WINDOW = 3

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

    def _record_anchor(self, basin_key, node: str):
        """Bounded write path for felt_state_anchors (this revision's fix
        for the unbounded-growth bug -- see the field's own docstring at
        __init__). Every write to felt_state_anchors goes through here now,
        so there's exactly one place that could reintroduce unbounded
        growth, not three scattered call sites.

        Co-activation broadening (§13.3, new -- see CO_ACTIVATION_
        RECENCY_WINDOW's docstring at __init__ for the full diagnosis).
        Before appending, pairs the new node with the most recent
        CO_ACTIVATION_RECENCY_WINDOW *distinct* anchors already recorded
        for this felt state. Distinct, not just "last N raw entries" --
        the deque can (and does) contain repeats of the same node, and
        pairing against repeats of one node wastes the bounded window on
        redundant pairs instead of genuinely different recent context."""
        if basin_key not in self.felt_state_anchors:
            self.felt_state_anchors[basin_key] = deque(maxlen=self.ANCHOR_WINDOW_SIZE)

        existing = self.felt_state_anchors[basin_key]
        if existing:
            recent_distinct = []
            seen = set()
            for n in reversed(existing):
                if n not in seen:
                    seen.add(n)
                    recent_distinct.append(n)
                if len(recent_distinct) >= self.CO_ACTIVATION_RECENCY_WINDOW:
                    break
            self.archivist.record_co_activation(recent_distinct + [node])

        self.felt_state_anchors[basin_key].append(node)

    def _record_protected_anchor(self, node: str):
        """Bug fix (this session) -- see the global pool's docstring at
        __init__ for the full diagnosis, including why this ended up
        basin-key-independent after the first version (keyed the same
        way as felt_state_anchors) turned out not to solve the actual
        reported problem. Only ever called from _ingest() (real user/
        dictionary input); _self_study() must never call this, by
        construction -- that exclusion is the entire mechanism."""
        self._global_protected_anchors.append(node)

    def _get_unique_anchors(self, basin_key) -> List[str]:
        """Bug fix, this revision (found from production data): the
        anchor deque is a touch LOG, not a set -- it can and does contain
        the same node multiple times (self-study re-touches its target
        every cycle it's re-selected). Every consumer that applies a
        per-node side effect to "the current anchors" -- regulation's
        activation bump and regulatory-efficacy update, parental
        feedback's valence-coloring nudge -- was previously iterating the
        raw list directly, so a node appearing N times in the window got
        that side effect applied N times within a SINGLE event. Concrete
        symptom: a single Parental Feedback click reported "20 node(s)
        colored" but the Reflection tab showed only 3 nodes with any
        coloring at all, two of them already pinned at the cap -- because
        those 2-3 nodes each appeared many times in the 20-entry window,
        each repeat re-applying the same nudge in one click, while the
        toast counted raw entries, not distinct nodes. Undermines the
        entire "accumulates only through genuinely repeated, SEPARATE
        co-occurrence events over time" property this mechanism depends
        on (§13.2). Order-preserving dedup (most-recent-first callers
        don't currently rely on order here, but preserving it is free and
        avoids surprising anyone who later does).

        Merge fix (this session, revised -- see the global pool's
        docstring at __init__): folds in _global_protected_anchors,
        regardless of basin_key -- real conversational topics stay
        candidates for working memory / self-study scoping across mood
        changes, not just within whatever felt state was active at the
        moment they were typed. Appended after the general deque's own
        entries, not prepended, for the same reason as before: preserves
        every existing consumer's "most recent genuine touch" bias while
        still guaranteeing real topics remain valid candidates. dict.
        fromkeys() below already dedups, so a node present in both stays
        at its first (general-deque) position.

        Also folds in self_narrative.linked_nodes_above_floor() (§16.5.1/
        §16.5.2, new this session) -- narratively significant nodes stay
        candidates for both working memory display and (via
        eligible_regulation_nodes()'s tier filter downstream) regulation,
        the same way real conversational topics do. Appended last, after
        both the general deque and the protected pool, so it's the
        lowest-priority tiebreak among the three -- narrative significance
        nudges what's visible, it doesn't compete with genuinely live
        conversational recency for ordering."""
        general = list(self.felt_state_anchors.get(basin_key, []))
        protected = list(self._global_protected_anchors)
        narrative = self.self_narrative.linked_nodes_above_floor()
        return list(dict.fromkeys(general + protected + narrative))

    def _ingest(self, text: str, source: str):
        """Runs one piece of text through sensory + association + chronos
        linking. Despite this docstring previously claiming to be "shared
        by both externally-queued input and self-study," it never
        actually was -- _self_study() has always called
        association.place_node() directly, bypassing this method
        entirely. Corrected here rather than left misleading."""
        self.sensory.ingest(text)
        if source == "user" and hasattr(self, "modulators"):
            self.modulators.pulse("user_input", amount=0.08)
        basin_key = self.synthesizer.get_current_basin_key()
        felt_state = self.synthesizer.get_current_felt_state()
        anchor = None
        if felt_state != "Unformed":
            anchors = self.felt_state_anchors.get(basin_key, [])
            anchor = anchors[-1] if anchors else None

        # User-taught hierarchy/part-of via chat ("yellow is a color")
        taught = None
        if source == "user":
            try:
                taught = self.sensory.parse_explicit_relation(text)
            except Exception:
                taught = None
        if taught:
            child, parent, etype = taught
            taught_result = self.association.teach_relation(
                child, parent, relation_type=etype, source=source,
            )
            if taught_result:
                print(f"User edge: {child} —{etype}→ {parent}")
                result = {
                    "term": taught_result["child"],
                    "created": True,
                    "taught": taught_result,
                }
            else:
                result = self.association.place_node(text, definition="", source=source, context_node=anchor)
        else:
            self_attr = None
            if source == "user":
                try:
                    self_attr = self.sensory.parse_self_attribute(text)
                except Exception:
                    self_attr = None
            if self_attr:
                _kind, attr, edge_hint = self_attr
                linked = self.association.link_self_attribute(
                    attr, edge_type=edge_hint, source=source,
                )
                if linked:
                    print(f"Self attribute ({_kind}): SELF —{edge_hint}→ {attr}")
                    result = {
                        "term": linked["attribute"],
                        "created": True,
                        "self_attribute": linked,
                    }
                    if hasattr(self, "modulators"):
                        self.modulators.pulse("user_input", amount=0.06)
                else:
                    result = self.association.place_node(text, definition="", source=source, context_node=anchor)
            else:
                result = self.association.place_node(text, definition="", source=source, context_node=anchor)
        if not isinstance(result, dict):
            result = {"term": result}
        node = result.get("term")
        if node:
            self.archivist.bump_activation(node)
            self.focus.boost_residual(node)
            if isinstance(result, dict) and result.get("taught"):
                p = result["taught"].get("parent")
                if p:
                    self.archivist.bump_activation(p)
                    self.focus.boost_residual(p)
            # Felt-anchor naming: short lemma-like user words while in a basin
            if source == "user" and len(text.split()) <= 3:
                self.felt_anchors.try_name_current(text.strip())
        if anchor:
            self.archivist.bump_activation(anchor)
            self.focus.boost_residual(anchor)
        # Co-activation (§13.3, new): node and anchor were touched in the
        # same event -- the raw signal epistemic schema clustering
        # depends on. A no-op if anchor is None (fewer than 2 real nodes).
        if node and anchor:
            self.archivist.record_co_activation([node, anchor])

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
            self._record_anchor(basin_key, node)

        # Bug fix (this session), split out from the block above on
        # purpose: the `felt_state != "Unformed"` gate exists to protect
        # chronos.record_felt_state_link()'s role as the evidence log
        # §6.1's naming-reliability gate reads from -- recording a link
        # before a basin has a name would muddy what that mechanism is
        # measuring. But that rationale has nothing to do with protected-
        # anchor visibility (§14 working memory, self-study's own
        # candidate scoping) -- those only care about the numeric
        # (arousal, valence, dominance) key, which is always well-defined
        # every tick regardless of whether that region has stabilized
        # into a NAMED felt state yet (naming is a lookup label on top of
        # an always-valid key, not a precondition for the key's
        # existence). Conflating the two under one gate meant real input
        # typed before any felt state had stabilized -- the ordinary case
        # for a fresh session's first few messages, confirmed as the
        # actual root cause of the reported "working memory never shows
        # my seed words" symptom -- never got a protected anchor recorded
        # at all, permanently, since _ingest() only records at the moment
        # of typing and there's no later backfill once a name eventually
        # gets assigned to that region. _record_anchor's own gating is
        # deliberately left untouched here -- untangling its relationship
        # to naming-reliability/co-activation is a separate, larger
        # question this fix doesn't take on.
        if node and source in ("user", "dictionary"):
            self._record_protected_anchor(node)

        # Explicit negation/correction (§3.4 mechanism 1): flag whatever
        # node was most recently active for gradual demotion at the next
        # Consolidation pass.
        text_lower = text.lower()
        negation_flagged = ("no, " in text_lower or "actually" in text_lower or "that's wrong" in text_lower)
        if negation_flagged and anchor:
            self.archivist.flag_negation(anchor)
            # §16.6, new this session: if the negated node is covered by
            # an existing narrative element, that element takes an
            # immediate, larger-than-normal-decay cut rather than waiting
            # for the next ordinary decay pass -- a correction to
            # something narratively significant should land harder than
            # an ordinary fact getting demoted.
            self.self_narrative.apply_negation_penalty(anchor)

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

    # Fixed, small, closed vocabulary of reaction types -- deliberately
    # not free-text sentiment interpretation (that would need its own
    # deterministic keyword layer, same as sensory.py's existing
    # negation/relational detectors, and is a natural but separate
    # future extension, not built here). This is a UI-level vocabulary
    # (which button was clicked), not a knowledge-graph vocabulary -- it
    # never names or asserts anything about a concept, only nudges
    # somatic state and (separately) an existing node's coloring.
    _PARENTAL_REACTION_TYPES = ("approval", "disapproval", "warmth", "concern")

    def give_parental_reaction(self, reaction_type: str):
        """
        §13.2, new: implicit parental emotional guidance, "mirror neuron"
        style. Two independent effects, deliberately kept separate:

        1. A small, deterministic, fast-layer-only hormone nudge -- same
           shape and magnitude class as _react_to_input(), fires live
           (not queued/Consolidation-gated), matching how every other
           somatic reaction in this design works (Stimulus's manual
           trigger, self-study's dopamine bump, _react_to_input).

        2. A valence_coloring nudge (archivist.nudge_valence_coloring) on
           whichever node(s) are the CURRENT felt-state anchor -- i.e.
           whatever the system was just "thinking about" when the
           reaction arrived. This is the actual learning mechanism: no
           word or category is ever assigned a valence directly here or
           anywhere else in this design. A node's coloring only moves
           because it happened to be active at the same moment a reaction
           occurred, repeated over many interactions -- genuine earned
           association, consistent with every other "no predetermined
           categories" constraint in this spec (§2.1a, §2.1b, §13.3.1).

        This method does NOT create, name, or otherwise touch the
        knowledge graph's structure -- only existing nodes' coloring, and
        only if something is currently anchored. If nothing is anchored
        yet (e.g. very early in Childhood, before any basin has
        stabilized), effect (1) still fires but effect (2) is a no-op --
        a legitimate, non-error state, same as regulation's "nothing
        eligible yet" case (§4.2).
        """
        if reaction_type not in self._PARENTAL_REACTION_TYPES:
            raise ValueError(f"Unknown parental reaction type: {reaction_type!r}")

        with self.bio.lock:
            h = self.bio._hormones
            if reaction_type == "approval":
                h["dopamine"] = min(1.0, h["dopamine"] + self.PARENTAL_APPROVAL_DOPAMINE)
                h["oxytocin"] = min(1.0, h.get("oxytocin", 0.5) + self.OXTYOCIN_PARENTAL_EXTRA * 0.7)
            elif reaction_type == "disapproval":
                h["cortisol"] = min(1.0, h["cortisol"] + self.PARENTAL_DISAPPROVAL_CORTISOL)
            elif reaction_type == "warmth":
                h["dopamine"] = min(1.0, h["dopamine"] + self.PARENTAL_WARMTH_DOPAMINE)
                h["cortisol"] = max(0.0, h["cortisol"] - self.PARENTAL_WARMTH_CORTISOL_RELIEF)
                h["oxytocin"] = min(1.0, h["oxytocin"] + self.OXTYOCIN_PARENTAL_EXTRA)
            elif reaction_type == "concern":
                h["adrenaline"] = min(1.0, h["adrenaline"] + self.PARENTAL_CONCERN_AROUSAL)
                h["cortisol"] = min(1.0, h["cortisol"] + self.PARENTAL_CONCERN_CORTISOL)

        coloring_delta = {
            "approval": self.VALENCE_COLORING_STEP,
            "disapproval": -self.VALENCE_COLORING_STEP,
            "warmth": self.VALENCE_COLORING_STEP * 0.7,
            "concern": -self.VALENCE_COLORING_STEP * 0.5,
        }[reaction_type]

        # Narrow targets: parental signal is about *what was just in mind*,
        # not the entire basin history window (that painted dozens of nodes).
        if hasattr(self, "modulators"):
            self.modulators.pulse(reaction_type if reaction_type in ("approval", "disapproval") else (
                "approval" if reaction_type == "warmth" else "disapproval"
            ), amount=0.12)
        targets = self._parental_targets(max_n=1)
        for n in targets:
            self.archivist.nudge_valence_coloring(n, coloring_delta, cap=self.VALENCE_COLORING_CAP)
            # Stamp explicit parental tag so UI can show *why* it is colored
            data = self.archivist.graph.nodes.get(n)
            if data is not None:
                hist = data.setdefault("parental_history", [])
                hist.append({
                    "reaction": reaction_type,
                    "delta": coloring_delta,
                    "pulse": int(self.pulse_count),
                })
                # keep short
                if len(hist) > 8:
                    del hist[:-8]
                data["last_parental_reaction"] = reaction_type
            self.archivist.bump_activation(n)
            self.focus.boost_residual(n)
            if reaction_type in ("disapproval", "concern"):
                self.focus.add_parental_residual(n, 1.0)
            elif reaction_type in ("approval", "warmth"):
                self.focus.add_parental_residual(n, 0.3)

        self.last_parental_feedback = {
            "reaction": reaction_type,
            "anchors_colored": list(targets),
            "pulse": int(self.pulse_count),
        }
        return self.last_parental_feedback

    def _parental_targets(self, max_n: int = 1) -> list:
        """Parental signal hits what is in mind now — not a basin-wide paint.

        Priority: sticky focus → single most recent user-ingested node.
        Default max_n=1 so approval/disapproval is legible.
        """
        out = []
        seen = set()

        def add(n):
            if not n or n in seen:
                return
            if n in ("SELF", "OTHER"):
                return
            if n not in self.archivist.graph:
                return
            data = self.archivist.graph.nodes.get(n) or {}
            # Skip pure schema wrappers unless they are the focus
            seen.add(n)
            out.append(n)

        fid = getattr(self.focus, "focus_id", None)
        add(fid)
        if len(out) < max_n:
            try:
                for n in reversed(list(self._global_protected_anchors)):
                    add(n)
                    if len(out) >= max_n:
                        break
            except Exception:
                pass
        return out[:max_n]

    def parental_coloring_report(self, top_n: int = 25) -> dict:
        """What parental reactions have actually marked — for UI transparency."""
        rows = []
        for n, d in self.archivist.graph.nodes(data=True):
            hist = d.get("parental_history") or []
            if not hist and not d.get("last_parental_reaction"):
                vc = d.get("valence_coloring")
                if vc is None or abs(float(vc or 0)) < 1e-6:
                    continue
            rows.append({
                "id": n,
                "name": d.get("name") or n,
                "last_reaction": d.get("last_parental_reaction"),
                "valence_coloring": round(float(d.get("valence_coloring") or 0), 4),
                "history": list(hist)[-3:] if hist else [],
            })
        rows.sort(key=lambda r: -abs(r["valence_coloring"]))
        last = getattr(self, "last_parental_feedback", None)
        return {
            "last_feedback": last,
            "marked_nodes": len(rows),
            "top": rows[:top_n],
        }


    def pulse(self):
        self.pulse_count += 1
        somatic = self.bio.step()

        # Fast neuromodulators (necessities): decay, medium bias, body gusts
        if hasattr(self, "modulators"):
            self.modulators.decay_toward_baseline()
            try:
                self.modulators.apply_medium_bias(self.bio._hormones)
            except Exception:
                pass
            fast_delta = self.modulators.body_delta()
        else:
            fast_delta = None

        # Body surface = medium/slow hormones + fast gusts (still no chemical names)
        body = self.bio.get_raw_variables(fast_body_delta=fast_delta)

        # synthesizer must run first, before anything that conditions a
        # decision on its output (regulation trigger, executive bias).
        self.synthesizer.update_from_core(body)
        self.somatic_topo.record(self.synthesizer.get_current_basin_key())
        self.felt_anchors.observe(
            self.synthesizer.get_current_basin_key(),
            raw_body=body,
        )
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
        elif self.state == "Sleep":
            # Usability: never hard-mute — drain one queued input under sleep bias
            if self._input_queue:
                text, source = self._input_queue.pop(0)
                self._ingest(text, source)
            # no self_study during sleep climate
        elif self.state == "Consolidation":
            if self._input_queue:
                text, source = self._input_queue.pop(0)
                self._ingest(text, source)

        self.chronos.record_pulse(
            somatic, bias,
            felt_state=self.synthesizer.get_current_felt_state(),
            avd=self.synthesizer.get_current_basin_key(),
        )

        self._update_fatigue()
        self._cycle_state()
        self.maybe_advance_epoch()

        # §13.y: residual decay + prediction error + sticky focus selection
        key = self.synthesizer.get_current_basin_key()
        basin_anchors = set(self._get_unique_anchors(key))
        if hasattr(self, "modulators"):
            self.focus.switch_cost_mult = self.modulators.switch_cost_mult()
        self.last_focus_summary = self.focus.tick(
            self.archivist.graph,
            pulse=self.pulse_count,
            basin_anchor_set=basin_anchors,
        )

        results = self.archivist.retrieve("context")


        # Schema ↔ felt-anchor co-occurrence (implicit; no emotion taxonomy)
        try:
            active_schemas = []
            fid = getattr(self.focus, "focus_id", None)
            if fid and fid in self.archivist.graph:
                nd = self.archivist.graph.nodes[fid]
                if (
                    str(fid).startswith("epistemic_")
                    or str(fid).startswith("schema_")
                    or nd.get("is_schema")
                    or nd.get("somatic")
                    or nd.get("node_type") in ("schema", "epistemic_schema")
                ):
                    active_schemas.append(fid)
            if hasattr(self, "working_memory") and self.working_memory is not None:
                try:
                    wm = self.working_memory.get_current_working_memory()
                    for sid in (wm.get("slots") or []):
                        if sid in self.archivist.graph:
                            nt = self.archivist.graph.nodes[sid].get("node_type")
                            if nt in ("schema", "epistemic_schema") or str(sid).startswith("epistemic_"):
                                active_schemas.append(sid)
                except Exception:
                    pass
            cur = self.felt_anchors.current()
            if cur is not None:
                self.schema_felt.note(active_schemas, cur.anchor_id)
            # Phase B: if focus is a collapsed parent, rehydrate a little detail
            fid = getattr(self.focus, "focus_id", None)
            if fid and fid in self.archivist.graph:
                abs_list = self.archivist.graph.nodes[fid].get("absorbed") or []
                if abs_list:
                    n_reh = self.archivist.rehydrate_for_parent(fid, max_children=2)
                    if n_reh:
                        print(f"Rehydrate on focus {fid}: {n_reh} child(ren)")
            # Hormone: stagnant focus → cortisol; force switch → dopamine
            fs = self.last_focus_summary or {}
            with self.bio.lock:
                if fs.get("stagnation_escape"):
                    self.bio._hormones["cortisol"] = min(
                        1.0, self.bio._hormones["cortisol"] + self.FOCUS_STAGNANT_CORTISOL
                    )
                    self.bio._hormones["dopamine"] = min(
                        1.0, self.bio._hormones["dopamine"] + self.FOCUS_SWITCH_DOPAMINE
                    )
        except Exception:
            pass

        print(
            f"Pulse {self.pulse_count} | Epoch: {self.bio.epoch.value} | "
            f"State: {self.state}/{getattr(self,'sleep_stage','')} | Bias: {bias} | Pressure: {self.fatigue:.2f} | "
            f"Felt: {self.synthesizer.get_current_felt_state()} | "
            f"Focus: {self.focus.focus_id}"
        )
        return {
            "pulse": self.pulse_count,
            "bias": bias,
            "state": self.state,
            "epoch": self.bio.epoch.value,
            "felt_state": self.synthesizer.get_current_felt_state(),
            "focus_id": self.focus.focus_id,
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
            result = self.association.place_node(
                child, definition=definition, source="dictionary",
                context_node=target, max_parent_children=self.SELF_STUDY_SOFT_CAP,
            )
            placed_children.append(result.get("term") or child)
        self.archivist.store(target, source="dictionary")  # reinforce parent's last_reinforced

        # Co-activation (§13.3, new): target and its newly-placed children
        # were all touched in the same self-study cycle -- record every
        # pairwise combination among them. This is the primary source of
        # co-activation data in practice, since self-study runs far more
        # often than real input in typical usage.
        self.archivist.record_co_activation([target] + placed_children)

        if felt_state != "Unformed":
            for child_node in placed_children:
                self.chronos.record_felt_state_link(basin_key, child_node)
                self._record_anchor(basin_key, child_node)
            # `target` recurs across multiple self-study ticks (until it
            # hits the degree cap), unlike each tick's freshly-created
            # children -- anchoring it too gives §6.1's naming-reliability
            # check and §4.2's regulation candidate pool a genuinely
            # consistent, repeatedly-reinforced node to work with, not
            # just a growing list of one-off terms.
            self.chronos.record_felt_state_link(basin_key, target)
            self._record_anchor(basin_key, target)

        # Activation touch (§11 pull-forward, this revision) -- smaller
        # than real input's default bump (archivist.ACTIVATION_BOOST),
        # same "gentler than externally-triggered" pattern already used
        # for the hormonal bump just below.
        self.archivist.bump_activation(target, self.ACTIVATION_BOOST_SELF_STUDY)
        self.focus.boost_residual(target)
        for child_node in placed_children:
            self.archivist.bump_activation(child_node, self.ACTIVATION_BOOST_SELF_STUDY)
            self.focus.boost_residual(child_node)

        # Small, scaled-down dopaminergic bump (§5.1, §9 risk 7) via the
        # same fast-layer pathway as any other event -- no bespoke
        # self-study fatigue tap. Also a small adrenaline touch (new, this
        # revision) -- see SELF_STUDY_AROUSAL_BUMP's docstring for why:
        # without this, arousal and dominance never moved at all during
        # autonomous-only activity, only valence did. Adrenaline alone
        # covers both axes (heart_rate + vascular_constriction), so no
        # separate testosterone touch is needed or appropriate.
        with self.bio.lock:
            h = self.bio._hormones
            h["dopamine"] = min(1.0, h["dopamine"] + self.SELF_STUDY_DOPAMINE_BUMP)
            h["adrenaline"] = min(1.0, h["adrenaline"] + self.SELF_STUDY_AROUSAL_BUMP)
            h["cortisol"] = min(1.0, h["cortisol"] + self.SELF_STUDY_CORTISOL_BUMP)
            if placed_children:
                h["serotonin"] = min(1.0, h["serotonin"] + self.SELF_STUDY_SEROTONIN_BUMP)
            try:
                inten = float(self.synthesizer.get_current_intensity())
            except Exception:
                inten = 0.0
            if inten >= 0.55:
                h["testosterone"] = min(1.0, h["testosterone"] + self.SELF_STUDY_TESTOSTERONE_BUMP)

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

        §14, new: working-memory-aware targeting. In Childhood, self-study
        candidates are hard-restricted to whatever's currently reachable
        from working memory (SELF/basin/schema slots, §14.1) -- the direct
        mechanism behind "stored-schema expansion heavily suppressed,
        permitted only on clear dead-ends" (§14.2), and the fix for
        self-study drifting arbitrarily far from what the user actually
        taught (colors in, Hundred Years' War never comes up, because the
        small working-memory buffer stays close to real input this early).
        The dead-end check is is_dead_end()'s documented PROXY (§14.6 item
        2 -- genuinely unresolved as a full mechanism, not just untuned;
        see that method's docstring), not the fully-resolved design. In
        Adolescence/Maturity, working-memory contents are boosted in the
        weighted choice instead of hard-gating -- matching §14.2's
        "transitional"/"no longer monopolizing" framing.
        """
        graph = self.archivist.graph
        if graph.number_of_nodes() == 0:
            return None

        epoch_value = self.bio.epoch.value
        key = self.synthesizer.get_current_basin_key()
        basin_anchors = self._get_unique_anchors(key)
        wm = self.working_memory.get_working_memory(epoch_value, basin_anchors)
        in_scope_nodes = self.working_memory.reachable_nodes(wm)

        def has_room(n, d):
            return (
                self.archivist.categorical_out_degree(n) < hard_cap
                and not d.get("is_schema")
                and n not in (SELF_NODE, OTHER_NODE)
                and n not in self._barren_self_study_targets
            )

        # Curiosity narrowing: prefer focus neighborhood over whole graph
        local = self.focus_neighborhood_ids(cap=48)
        pin = None
        try:
            # session pin is app-side; residual-top is engine-side
            if self.focus.residuals:
                pin = max(self.focus.residuals.items(), key=lambda t: t[1])[0]
                local.add(pin)
        except Exception:
            pass

        working_candidates = [
            n for n, d in list(graph.nodes(data=True))
            if d.get("tier", 0) >= TIER_WORKING and has_room(n, d)
        ]
        # (b) NEW: Provisional nodes with room, source-tagged non-self-generated
        # (i.e. real user/dictionary input, not the agent's own prior
        # self-study output) -- gives fresh input a genuine shot at
        # self-study attention instead of waiting on a tier it may never
        # reach without exactly this kind of reinforcement.
        provisional_candidates = [
            n for n, d in list(graph.nodes(data=True))
            if d.get("tier", 0) < TIER_WORKING and d.get("source") != "self_generated" and has_room(n, d)
        ]

        # Soft gate: if focus neighborhood has room, study there first (less random)
        if local:
            loc_w = [n for n in working_candidates if n in local]
            loc_p = [n for n in provisional_candidates if n in local]
            if loc_w or loc_p:
                working_candidates = loc_w or working_candidates
                provisional_candidates = loc_p or provisional_candidates

        # §14: Childhood hard-gate, unless the dead-end proxy fires.
        if epoch_value == "Childhood":
            dead_end = self.working_memory.is_dead_end(
                wm, lambda n: has_room(n, graph.nodes.get(n, {})), self._barren_self_study_targets,
            )
            if not dead_end:
                working_candidates = [n for n in working_candidates if n in in_scope_nodes]
                provisional_candidates = [n for n in provisional_candidates if n in in_scope_nodes]

        # Bias-modulated pool selection (§13.1, new). NEUTRAL reproduces
        # the old hardcoded 0.6 exactly -- this is a strict extension, not
        # a behavior change, for anyone who hasn't touched the bias signal.
        bias = self.executive.current_bias
        provisional_prob = {
            "BIAS_EXPLORE": self.SELF_STUDY_PROVISIONAL_PROB_EXPLORE,
            "BIAS_STABILIZE": self.SELF_STUDY_PROVISIONAL_PROB_STABILIZE,
        }.get(bias, self.SELF_STUDY_PROVISIONAL_PROB_NEUTRAL)

        # §14: outside Childhood, working-memory contents are boosted in
        # the weighted choice rather than hard-gated -- "transitional"/
        # "no longer monopolizing" per §14.2. Childhood already hard-gated
        # the pools themselves above, so no additional boost is needed
        # there (and would be redundant, since everything left in-pool is
        # already in scope).
        boost_set = in_scope_nodes if epoch_value != "Childhood" else None

        if working_candidates and provisional_candidates:
            # Weighted toward provisional under NEUTRAL/default: established
            # hubs already got their initial attention, fresh nodes need it
            # more. Under EXPLORE/STABILIZE, the ratio shifts instead per
            # provisional_prob above. Not a tuned ratio (§10) either way.
            pool = provisional_candidates if random.random() < provisional_prob else working_candidates
            return self._weighted_choice_by_activation(pool, bias=bias, boost_set=boost_set)
        if provisional_candidates:
            return self._weighted_choice_by_activation(provisional_candidates, bias=bias, boost_set=boost_set)
        if working_candidates:
            return self._weighted_choice_by_activation(working_candidates, bias=bias, boost_set=boost_set)

        # (c) fallback: whatever node is currently anchoring the felt
        # state, if any -- but only if it also still has room. Previously
        # uncapped, which was the main source of runaway single-node growth.
        # Uses the RAW deque here (not basin_anchors, which is deduped and
        # doesn't preserve most-recent-last order) so `[-1]` still means
        # "most recently touched," matching original behavior exactly.
        raw_anchors = self.felt_state_anchors.get(key, [])
        if raw_anchors:
            anchor = raw_anchors[-1]
            if anchor in graph and has_room(anchor, graph.nodes[anchor]):
                return anchor

        # (d) last resort: any node with room, not just any node at all
        # (same fix -- this used to be truly uncapped).
        low_degree_any = [n for n, d in list(graph.nodes(data=True)) if has_room(n, d)]
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

    def _weighted_choice_by_activation(self, pool: List[str], bias: str = "BIAS_NEUTRAL",
                                        boost_set: Optional[set] = None) -> Optional[str]:
        """Activation-weighted random choice (§11 pull-forward) --
        replaces uniform random.choice() for self-study target selection
        so recently-touched nodes are preferentially re-expanded, giving
        self-study something closer to genuine attention/focus. An
        epsilon floor (0.1) on every weight keeps untouched nodes
        selectable at nonzero probability -- this stays exploration-with-
        a-bias, not pure exploitation of whatever's already active, which
        would risk narrowing the graph's growth to an ever-smaller hot
        set over time.

        `bias` (§13.1): under BIAS_EXPLORE, the weighting is inverted --
        low-activation (novel, rarely-touched) nodes are preferred
        instead, using ACTIVATION_CAP minus each node's activation as the
        weight, same epsilon floor for the opposite reason (keeps a
        saturated node selectable at nonzero probability rather than
        fully excluded). BIAS_STABILIZE and BIAS_NEUTRAL both keep the
        original high-activation-preferring weighting -- NEUTRAL
        reproduces prior behavior exactly; STABILIZE reads as "deepen
        what's already active," which is what the un-inverted weighting
        already does.

        `boost_set` (new, this revision -- §14): a multiplicative bonus
        for nodes reachable from current working memory, used outside
        Childhood (which hard-gates the candidate pools instead, so
        nothing needs boosting there -- see _select_self_study_target).
        Multiplicative rather than additive so it composes with whatever
        the bias weighting already computed instead of overriding it --
        a boosted node under BIAS_EXPLORE still respects the inverted
        preference, just scaled up within it."""
        if not pool:
            return None
        if bias == "BIAS_EXPLORE":
            cap = self.archivist.ACTIVATION_CAP
            weights = [
                (cap - self.archivist.graph.nodes[n].get("activation", 0.0)) + 0.1 for n in pool
            ]
        else:
            weights = [self.archivist.graph.nodes[n].get("activation", 0.0) + 0.1 for n in pool]
        if boost_set:
            weights = [w * 3.0 if n in boost_set else w for n, w in zip(pool, weights)]
        try:
            local = self.focus_neighborhood_ids(cap=48)
            weights = [w * 4.0 if n in local else w for n, w in zip(pool, weights)]
            fid = getattr(self.focus, "focus_id", None)
            if fid:
                weights = [w * 2.5 if n == fid else w for n, w in zip(pool, weights)]
        except Exception:
            pass
        try:
            if hasattr(self, "long_term_interest"):
                weights = [
                    w * self.long_term_interest.curiosity_multiplier(n)
                    for n, w in zip(pool, weights)
                ]
        except Exception:
            pass
        # §13.y: sticky focus / residual neighbourhood bias
        weights = [w * self.focus.self_study_weight(n) for n, w in zip(pool, weights)]
        return random.choices(pool, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Fatigue / state cycling
    # ------------------------------------------------------------------
    def _compute_urgency(self) -> float:
        """Organic urgency in [0,1]: pending input, focus residuals, prediction heat."""
        u = 0.0
        if getattr(self, "_input_queue", None):
            u = max(u, min(1.0, 0.25 * len(self._input_queue)))
        try:
            fid = getattr(self.focus, "focus_id", None)
            if fid:
                act = float(self.focus.residuals.get(fid, 0.0) or 0.0)
                pred = float(self.focus.r_pred.get(fid, 0.0) or 0.0)
                u = max(u, min(1.0, (act + pred) / 12.0))
            if self.focus.residuals:
                top = max(self.focus.residuals.values())
                u = max(u, min(1.0, float(top) / 15.0))
        except Exception:
            pass
        self._last_urgency = u
        return u

    def _sleep_threshold(self, urgency: float) -> float:
        """Low urgency → lower ceiling (sleep sooner). High urgency → can push higher."""
        soft = float(getattr(self, "SLEEP_SOFT_MIN", self.T1))
        hard = float(getattr(self, "SLEEP_HARD_MAX", 0.92))
        # T2 slider maps near "high push" band for lab control
        high = min(hard, max(soft + 0.05, float(self.T2)))
        return soft + (high - soft) * max(0.0, min(1.0, urgency))

    def _update_fatigue(self):
        """Continuous sleep pressure. Intensity + work raise it; urgency accelerates growth."""
        intensity = self.synthesizer.get_current_intensity()
        urgency = self._compute_urgency()
        growth = float(self.FATIGUE_GROWTH_RATE)
        if urgency > 0.35:
            growth *= 1.0 + (self.FATIGUE_URGENCY_GROWTH_MULT - 1.0) * urgency
            # Hidden body cost of pushing (mind only sees body later)
            try:
                with self.bio.lock:
                    self.bio._hormones["cortisol"] = min(
                        1.0, self.bio._hormones.get("cortisol", 0.4) + 0.01 * urgency
                    )
                    self.bio._hormones["adrenaline"] = min(
                        1.0, self.bio._hormones.get("adrenaline", 0.5) + 0.008 * urgency
                    )
            except Exception:
                pass
        # Micro-day baseline drift: slight pressure even when quiet (circadian-ish)
        self.micro_day_pulse = (int(self.micro_day_pulse) + 1) % max(8, int(self.MICRO_DAY_PULSES))
        baseline = 0.02 * (self.micro_day_pulse / max(1, self.MICRO_DAY_PULSES))
        if self.state != "Sleep":
            self.fatigue = min(1.0, self.fatigue + intensity * growth + baseline * 0.05)
        else:
            # Sleep recovers; debt lengthens effective recovery need
            rec = float(self.FATIGUE_RECOVERY_SLEEP)
            self.fatigue = max(0.0, self.fatigue * rec - 0.01 * (1.0 - min(1.0, self.sleep_debt)))

    def _enter_sleep(self):
        self.state = "Sleep"
        self.sleep_stage = "digest"
        self.sleep_debt = max(0.0, self.fatigue - float(getattr(self, "SLEEP_SOFT_MIN", self.T1)))
        if hasattr(self, "modulators"):
            self.modulators.pulse("sleep_enter", amount=0.15)
        print(f"Sleep climate enter (pressure={self.fatigue:.3f}, debt={self.sleep_debt:.3f}, urgency={self._last_urgency:.2f})")

    def _run_sleep_pulse(self):
        """Multi-step sleep: digest → reorganize → homeostatic → wake_prep. Not prune-only."""
        stage = self.sleep_stage or "digest"
        if stage == "digest":
            # Consolidation-class work: schemas, binds, checkpoint
            self._run_consolidation()
            self.fatigue *= float(self.FATIGUE_RECOVERY_CONSOLIDATION)
            self.sleep_stage = "reorganize"
        elif stage == "reorganize":
            # Collapse already runs inside consolidation; light extra prune of tier-0
            try:
                pruned = self.archivist.prune()
                if pruned:
                    print(f"Sleep reorganize: pruned {pruned} stale Tier-0 node(s).")
            except Exception as e:
                print(f"Sleep prune: {e}")
            self.fatigue *= float(self.FATIGUE_RECOVERY_PRUNING)
            self.sleep_stage = "homeostatic"
        elif stage == "homeostatic":
            # Downscale residuals / focus heat; body recovery nudge
            try:
                self.focus.decay_residuals(consolidation=True)
            except Exception:
                pass
            try:
                self.bio.decay_fast(rate=0.35)
            except Exception:
                pass
            self.fatigue = max(0.0, self.fatigue * float(self.FATIGUE_RECOVERY_SLEEP) - 0.03)
            self.sleep_stage = "wake_prep"
        else:
            # wake_prep: soft exit if pressure low enough
            self.fatigue = max(0.0, self.fatigue * float(self.FATIGUE_RECOVERY_SLEEP))
            wake_bar = float(getattr(self, "SLEEP_WAKE_BELOW", 0.22))
            # Higher debt → need lower pressure to wake (longer sleep)
            wake_bar = max(0.08, wake_bar - 0.1 * min(1.0, self.sleep_debt))
            if self.fatigue <= wake_bar + float(self.HYSTERESIS):
                self.state = "Learning"
                self.sleep_stage = "none"
                self.sleep_debt *= 0.5
                if hasattr(self, "modulators"):
                    self.modulators.pulse("sleep_exit", amount=0.1)
                print(f"Sleep climate exit → Learning (pressure={self.fatigue:.3f})")
            else:
                # Loop: another digest pass if still heavily indebted
                self.sleep_stage = "digest"

    def _cycle_state(self):
        """Sleep-pressure state machine: Learning ↔ Consolidation ↔ Sleep climate."""
        urgency = self._last_urgency if self._last_urgency else self._compute_urgency()
        thresh = self._sleep_threshold(urgency)
        soft = float(getattr(self, "SLEEP_SOFT_MIN", self.T1))
        hard = float(getattr(self, "SLEEP_HARD_MAX", 0.92))

        if self.state == "Learning":
            # Soft: prefer evening consolidation before full sleep
            if self.fatigue >= soft and self.fatigue < thresh:
                self.state = "Consolidation"
            elif self.fatigue >= thresh or self.fatigue >= hard:
                self._enter_sleep()
        elif self.state == "Consolidation":
            if self.fatigue >= thresh or self.fatigue >= hard:
                self._enter_sleep()
            elif self.fatigue < soft - float(self.HYSTERESIS):
                self.state = "Learning"
        elif self.state == "Sleep":
            pass  # handled below
        elif self.state == "Pruning":
            # Migrate legacy state name
            self._enter_sleep()

        if self.state == "Consolidation":
            self._run_consolidation()
            self.fatigue *= float(self.FATIGUE_RECOVERY_CONSOLIDATION)
        elif self.state == "Sleep":
            self._run_sleep_pulse()

    def get_current_working_memory(self) -> dict:
        """§14 convenience wrapper -- supplies the current epoch and basin
        anchors automatically, so app.py's diagnostic panel (and any
        other caller) doesn't need to reconstruct them each time. Safe to
        call every Reflection-tab render; get_working_memory() itself is
        a cheap on-demand computation, not maintained incremental state."""
        key = self.synthesizer.get_current_basin_key()
        basin_anchors = self._get_unique_anchors(key)
        # Self-heal duplicate schema names so UI cannot show triple Sweat
        try:
            self.reflector.merge_schemas_sharing_name()
        except Exception:
            pass
        return self.working_memory.get_working_memory(self.bio.epoch.value, basin_anchors)

    def get_narrative_report(self, top_n: int = 10) -> dict:
        """§16 convenience wrapper, matching get_current_working_memory()'s
        pattern -- app.py's Mind/Narrative tab reads this directly rather
        than reaching into self.self_narrative itself. Cheap on-demand
        read, not maintained incremental state (the underlying element
        set only changes at Consolidation; this just formats it)."""
        return self.self_narrative.report(top_n=top_n)

    def get_focus_report(self) -> dict:
        """§13.y debug/Reflection helper -- residual + sticky focus state."""
        return self.focus.report()

    def get_somatic_topo_report(self) -> dict:
        """Basin dwell + transition map (not raw hormone gauges)."""
        return self.somatic_topo.report()

    def get_felt_anchor_report(self) -> dict:
        """Linkable felt identities over PAD (coords stay internal)."""
        rep = self.felt_anchors.report()
        # Attach reverse binds for UI
        if isinstance(rep, dict) and "anchors" in rep:
            for row in rep["anchors"]:
                row["bound_schemas"] = self.schema_felt.schemas_for_anchor(row["id"])
        return rep

    def get_schema_felt_report(self) -> dict:
        """Schema ↔ felt co-occurrence / promotion diagnostic."""
        return self.schema_felt.report()


    def node_neighborhood(self, node_id: str, max_each: int = 20) -> dict:
        """Parents/children for search expand — list only, not full graph render."""
        graph = self.archivist.graph
        if not node_id or node_id not in graph:
            return {"id": node_id, "parents": [], "children": [], "related": []}
        parents, children, related = [], [], []
        # Out edges from node = often child→parent for is-a, or parent→child for composed-of
        for _, v, data in graph.out_edges(node_id, data=True):
            rel = data.get("relation_type") or "associated-with"
            row = {"id": str(v), "relation": rel, "name": graph.nodes[v].get("name")}
            if rel in ("is-a", "part-of"):
                parents.append(row)  # yellow is-a color → color is parent-ish endpoint
            elif rel in ("composed-of", "member-of"):
                children.append(row)
            else:
                related.append(row)
        for u, _, data in graph.in_edges(node_id, data=True):
            rel = data.get("relation_type") or "associated-with"
            row = {"id": str(u), "relation": rel, "name": graph.nodes[u].get("name")}
            if rel in ("is-a", "part-of"):
                # incoming is-a means u is-a node → u is child
                children.append(row)
            elif rel in ("composed-of",):
                parents.append(row)
            else:
                related.append(row)
        return {
            "id": node_id,
            "parents": parents[:max_each],
            "children": children[:max_each],
            "related": related[:max_each],
            "parent_count": len(parents),
            "child_count": len(children),
            "related_count": len(related),
        }

    def focus_neighborhood_ids(self, radius: int = 1, cap: int = 40) -> set:
        """Nodes near sticky focus — curiosity should live here first."""
        fid = getattr(self.focus, "focus_id", None)
        graph = self.archivist.graph
        out = set()
        if fid and fid in graph:
            out.add(fid)
            for _, v in graph.out_edges(fid):
                out.add(v)
            for u, _ in graph.in_edges(fid):
                out.add(u)
            # residual neighborhood
            try:
                for n, val in sorted(self.focus.residuals.items(), key=lambda t: -t[1])[:cap]:
                    if val >= getattr(self.focus, "RESIDUAL_FLOOR", 0.05):
                        out.add(n)
            except Exception:
                pass
        # Long-term themes orient neighborhood without global lottery
        if hasattr(self, "long_term_interest"):
            for tid in list(self.long_term_interest.theme_ids())[: max(8, cap // 4)]:
                if tid in graph:
                    out.add(tid)
        return out

    def search_graph(self, query: str, limit: int = 40) -> list:
        """Substring search over node ids, names, and definitions.
        Returns list of dicts for UI: id, name, kind, tier, activation.
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        graph = self.archivist.graph
        hits = []
        for n, d in list(graph.nodes(data=True)):
            nid = str(n)
            name = str(d.get("name") or "")
            definition = str(d.get("definition") or "")
            blob = f"{nid} {name} {definition}".lower()
            if q not in blob:
                continue
            is_schema = bool(
                d.get("is_schema")
                or d.get("node_type") in ("schema", "epistemic_schema")
                or nid.startswith("epistemic_")
                or nid.startswith("schema_")
            )
            hits.append({
                "id": nid,
                "name": name or None,
                "kind": "schema" if is_schema else "node",
                "tier": d.get("tier", 0),
                "activation": round(float(d.get("activation", 0.0) or 0.0), 3),
                "named": bool(d.get("named")),
            })
        hits.sort(key=lambda h: (-h["activation"], h["id"]))
        return hits[: max(1, int(limit))]


    def _mod_boost_residual(self, node_id: str, amount: float = None) -> None:
        if not hasattr(self, "focus") or self.focus is None:
            return
        amt = amount if amount is not None else getattr(self.focus, "RESIDUAL_BOOST", 1.0)
        if hasattr(self, "modulators"):
            amt = float(amt) * float(self.modulators.residual_gain())
        self.focus.boost_residual(node_id, amount=amt)

    def get_long_term_interest_report(self) -> dict:
        if hasattr(self, "long_term_interest"):
            return self.long_term_interest.report()
        return {}

    def get_modulator_report(self) -> dict:
        if hasattr(self, "modulators"):
            return self.modulators.report()
        return {}

    def pin_search_hit(self, node_id: str) -> bool:
        """Boost residual / activation so search can steer attention."""
        if not node_id or node_id not in self.archivist.graph:
            return False
        self.archivist.bump_activation(node_id)
        if hasattr(self, "focus") and self.focus is not None:
            self.focus.boost_residual(node_id, amount=2.0)
        return True

    def _run_consolidation(self):
        """
        Everything the spec pins to the Consolidation clock, in one place
        (§5, §3.3, §2.3 mechanism 3, §4.5, §2.1b, §5's slow-layer baseline
        note) -- "one clock, not several" per the design's own governing
        principle (see conversation summary).
        """
        self.synthesizer.consolidate_basins()
        topo_summary = self.somatic_topo.consolidate()
        print(f"Consolidation: somatic topo {topo_summary}")

        # Bug fix, this revision: stabilized_basins was previously only
        # ever a string mapping inside synthesizer.py -- no corresponding
        # node ever existed in archivist.graph. Every schema's "linked
        # back to its component basin" edge (reflector.detect_schemas)
        # was silently falling back to SELF_NODE instead, since the basin
        # string was never actually `in graph`. Sync real basin nodes here,
        # before run_consolidation_pass (so they're correctly exempted
        # from trust-tier evaluation, §6A) and before detect_schemas (so
        # any schema formed this same pass has a real node to link to).
        for basin_key, basin_id in self.synthesizer.stabilized_basins.items():
            self.archivist.ensure_basin_node(
                basin_id, pad_coordinates=basin_key,
                dwell_density=self.synthesizer.basin_grid.get(basin_key, 0.0),
            )

        trust_summary = self.archivist.run_consolidation_pass()
        reparented = self.association.run_reparenting_pass()
        new_schemas = self.reflector.detect_schemas()
        # Phase B: new somatic schemas start co-occurrence with current felt place
        if new_schemas:
            cur = self.felt_anchors.current() if hasattr(self, "felt_anchors") else None
            if cur is not None:
                self.schema_felt.note(new_schemas, cur.anchor_id)
                # Pre-credit so bind can form sooner for lived emotion structure
                for sid in new_schemas:
                    self.schema_felt.cooccur[sid][cur.anchor_id] = max(
                        self.schema_felt.cooccur[sid][cur.anchor_id],
                        self.schema_felt.threshold - 1,
                    )
        self._evaluate_pending_regulation()
        # Activation decay (§11 pull-forward, this revision) -- same
        # Consolidation clock as basin/trust/schema/efficacy evaluation,
        # per the design's own "one clock, not several" principle.
        self.archivist.decay_activation()

        # §13.3, new: epistemic (knowledge-cluster) schema formation, same
        # Consolidation clock as everything else. Detection runs BEFORE
        # decay -- same ordering already used for basin stabilization
        # (synthesizer.consolidate_basins() checks the stabilization
        # threshold first, decays after, in that order, within one pass).
        # An earlier version of this had the order backwards (decay first,
        # then detect), which meant decay_co_activation()'s multiply
        # could shrink a just-crossed-threshold count back below it
        # before detection ever saw it at full value -- caught in testing
        # before shipping. Naming scan runs last, after any new clusters
        # this same pass have been created, so they're immediately
        # eligible.
        new_epistemic_schemas = self.reflector.detect_epistemic_clusters()
        self.archivist.decay_co_activation()
        merged_epistemic = self.reflector.merge_duplicate_epistemic_schemas()
        named_epistemic = self.reflector.try_name_epistemic_schemas()
        merged_names = self.reflector.merge_schemas_sharing_name()
        print(f"Consolidation: merged same-name schemas {merged_names}")
        expired_epistemic = self.reflector.expire_unnamed_epistemic_schemas()
        bind_summary = self.schema_felt.promote(self.archivist.graph)
        print(f"Consolidation: schema-felt binds {bind_summary}")

        # §13.4, new this session: Graph Collapse & Abstraction Layer.
        # Runs after schema detection ("new schemas can claim members
        # before leaves are absorbed", §13.4.10), before Self-Narrative
        # (moved below this call, per §13.4.10's reconciled ordering --
        # so a Narrative Element referencing a node collapsed THIS pass
        # can absorb into its new parent immediately via self_narrative's
        # own §16.4 ancestor-walk, not a pass late).
        #
        # protected_nodes assembles §13.4.2's "protected nodes never
        # collapse" set from every source that currently exists: real
        # conversational topics (_global_protected_anchors, basin-
        # independent), narratively significant nodes (self_narrative.
        # linked_nodes_above_floor()), and whatever's currently in the
        # 7-slot working memory (get_current_working_memory()'s slots --
        # collapsing something the person is actively being shown as "in
        # mind right now" would be confusing regardless of its neglect
        # stats). SELF/OTHER are protected unconditionally inside
        # archivist.collapse_eligible() itself, not repeated here.
        # Active Thread's central_node/supporting_nodes and a Goal's
        # target_node (§13.4.14 item 1/2) will need to join this same
        # union once those modules exist -- not yet, since neither is
        # built.
        protected_nodes = set(self._global_protected_anchors) | set(self.self_narrative.linked_nodes_above_floor())
        try:
            protected_nodes |= set(self.get_current_working_memory().get("slots", []))
        except Exception as e:  # defensive -- working memory's own computation must never block Consolidation
            logger.warning("Could not compute working-memory protection set for collapse pass: %s", e)
        # §13.y: current sticky focus never collapses while active
        protected_nodes |= self.focus.protected_ids()

        # §13.y Piece D: update schema expected_families before collapse densifies parents
        n_exp = self.focus.update_expected_families(self.archivist.graph)
        if n_exp:
            print(f"Consolidation: updated expected_families on {n_exp} schema(s)")

        collapse_summary = self.archivist.run_collapse_pass(
            protected_nodes=protected_nodes, current_pulse=self.pulse_count,
        )
        self.last_collapse_summary = collapse_summary
        self.focus.decay_residuals(consolidation=True)

        # §16, new this session: Self-Narrative evaluation. Runs after
        # both schema-detection passes AND the collapse pass (moved here
        # this revision, §13.4.10) so this pass's newly-formed schemas
        # and any node collapse just resolved are both available as
        # trigger/absorption material the same cycle, not a pass late --
        # same ordering rationale already used for everything else in
        # this method.
        narrative_summary = self.self_narrative.evaluate(
            new_somatic_schema_ids=new_schemas,
            new_epistemic_schema_ids=new_epistemic_schemas,
            current_intensity=self.synthesizer.get_current_intensity(),
        )

        # §4C: the single checkpoint call for this pass -- everything
        # above mutates the graph and/or hormonal state without saving
        # individually (see the "No self.save() here" comments in
        # archivist.py, reflector.py, and hormonal.py's step()). This is
        # the one clock persistence is gated to.
        # Full checkpoint — only intentional reset should wipe the mind
        self.save_full_checkpoint()

        if trust_summary.get("promotions") or trust_summary.get("demotions"):
            print(f"Consolidation trust pass: {trust_summary}")
        if reparented:
            print(f"Consolidation: re-parented {reparented} node(s).")
        if new_schemas:
            print(f"Consolidation: formed {len(new_schemas)} new Schema Node(s): {new_schemas}")
        if new_epistemic_schemas:
            print(f"Consolidation: formed {len(new_epistemic_schemas)} new Epistemic Schema Node(s): {new_epistemic_schemas}")
        if merged_epistemic:
            print(f"Consolidation: merged {merged_epistemic} duplicate Epistemic Schema Node(s) into their named parent.")
        if named_epistemic:
            print(f"Consolidation: named {named_epistemic} epistemic schema(s) (delayed naming).")
        if expired_epistemic:
            print(f"Consolidation: dissolved {expired_epistemic} stagnant unnamed epistemic schema(s).")
        # Always print (including zeros) so Streamlit logs show the pass ran
        print(f"Consolidation: §13.4 collapse {collapse_summary}")
        if narrative_summary.get("created") or narrative_summary.get("absorbed") or narrative_summary.get("pruned"):
            print(f"Consolidation: Self-Narrative {narrative_summary}")

        # Long-term interest: promote recurring focus/narrative/parental into themes
        if hasattr(self, "long_term_interest"):
            residual_totals = {}
            try:
                residual_totals = {k: float(v) for k, v in self.focus.top_residuals(30)}
            except Exception:
                residual_totals = {}
            parental_nodes = []
            try:
                for n, d in self.archivist.graph.nodes(data=True):
                    if d.get("last_parental_reaction"):
                        parental_nodes.append(n)
            except Exception:
                pass
            felt_bound = []
            try:
                for n, d in self.archivist.graph.nodes(data=True):
                    if d.get("primary_felt_anchor"):
                        felt_bound.append(n)
            except Exception:
                pass
            lti_summary = self.long_term_interest.promote(
                focus_id=getattr(self.focus, "focus_id", None),
                residual_totals=residual_totals,
                narrative_elements=getattr(self.self_narrative, "elements", {}),
                parental_nodes=parental_nodes[:20],
                felt_bound_schemas=felt_bound[:30],
            )
            print(f"Consolidation: long-term interest {lti_summary}")
            try:
                self.long_term_interest.save()
            except Exception as e:
                print(f"LTI save failed: {e}")

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
        # Deduplicated (this revision): felt_state_anchors is a touch LOG,
        # not a set -- it can contain the same node many times if it's
        # been repeatedly re-touched (e.g. a favored self-study target).
        # Using the raw list here meant a node appearing N times in the
        # window got bumped/efficacy-updated N times from a SINGLE
        # regulation event, not once -- the same class of bug found and
        # fixed in give_parental_reaction() (§13.2), just affecting
        # regulation instead. eligible_regulation_nodes()'s own tier
        # filter still applies on top of this.
        unique_anchors = self._get_unique_anchors(key)
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
        regulating_nodes = self.archivist.eligible_regulation_nodes(unique_anchors)

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


    # Phase persistence: full mind checkpoint (wipe only via reset_persistent_memory)
    @staticmethod
    def _data_dir() -> str:
        import os
        return os.environ.get(
            "PROMETHEUS_DATA_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        )

    def _runtime_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "runtime_state.json")

    def _focus_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "focus_state.json")

    def _felt_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "felt_anchors.json")

    def _schema_felt_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "schema_felt.json")

    def _topo_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "somatic_topo.json")

    def save_runtime_state(self) -> None:
        """pulse_count / fatigue / mode — so restart continues the clock."""
        import json, os
        try:
            os.makedirs(self._data_dir(), exist_ok=True)
            data = {
                "pulse_count": int(self.pulse_count),
                "fatigue": float(self.fatigue),
                "state": self.state,
                "sleep_stage": getattr(self, "sleep_stage", "none"),
                "sleep_debt": float(getattr(self, "sleep_debt", 0.0)),
                "micro_day_pulse": int(getattr(self, "micro_day_pulse", 0)),
            }
            with open(self._runtime_path(), "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"save_runtime_state failed: {e}")

    def load_runtime_state(self) -> None:
        import json, os
        path = self._runtime_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.pulse_count = int(data.get("pulse_count") or 0)
            self.fatigue = float(data.get("fatigue") or 0.0)
            st = data.get("state") or "Learning"
            if st == "Pruning":
                st = "Sleep"
            if st in ("Learning", "Consolidation", "Sleep"):
                self.state = st
            self.sleep_stage = data.get("sleep_stage") or "none"
            self.sleep_debt = float(data.get("sleep_debt") or 0.0)
            self.micro_day_pulse = int(data.get("micro_day_pulse") or 0)
        except Exception as e:
            print(f"load_runtime_state failed: {e}")

    def save_full_checkpoint(self) -> None:
        """Single intentional-surviving checkpoint: graph + body + attention + felt."""
        self.archivist.save()
        self.bio.save_state()
        self.synthesizer.save_state()
        try:
            self.self_narrative.save()
        except Exception:
            pass
        try:
            self.focus.save_state(self._focus_path())
        except Exception as e:
            print(f"focus save failed: {e}")
        try:
            self.felt_anchors.save_state(self._felt_path())
        except Exception as e:
            print(f"felt save failed: {e}")
        try:
            self.schema_felt.save_state(self._schema_felt_path())
        except Exception as e:
            print(f"schema_felt save failed: {e}")
        try:
            self.somatic_topo.save_state(self._topo_path())
        except Exception as e:
            print(f"topo save failed: {e}")
        if hasattr(self, "modulators"):
            try:
                self.modulators.save_state()
            except Exception as e:
                print(f"modulators save failed: {e}")
        if hasattr(self, "long_term_interest"):
            try:
                self.long_term_interest.save()
            except Exception as e:
                print(f"LTI save failed: {e}")
        self.save_runtime_state()

    def load_extended_state(self) -> None:
        """Called from __init__ after modules exist — restore non-graph mind state."""
        try:
            self.focus.load_state(self._focus_path())
        except Exception as e:
            print(f"focus load failed: {e}")
        try:
            self.felt_anchors.load_state(self._felt_path())
        except Exception as e:
            print(f"felt load failed: {e}")
        try:
            self.schema_felt.load_state(self._schema_felt_path())
        except Exception as e:
            print(f"schema_felt load failed: {e}")
        try:
            self.somatic_topo.load_state(self._topo_path())
        except Exception as e:
            print(f"topo load failed: {e}")
        if hasattr(self, "modulators"):
            try:
                self.modulators.load_state()
            except Exception as e:
                print(f"modulators load failed: {e}")
        self.load_runtime_state()


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
        import os
        from .archivist import EPISTEMIC_GRAPH_PATH, CO_ACTIVATION_PATH
        from .chronos import CHRONOS_LOG_PATH
        from .hormonal import BIOSYSTEM_STATE_PATH
        from .synthesizer import BASIN_STATE_PATH
        from .self_narrative import NARRATIVE_STATE_PATH

        data_dir = Prometheus._data_dir()
        extra = [
            os.path.join(data_dir, "runtime_state.json"),
            os.path.join(data_dir, "focus_state.json"),
            os.path.join(data_dir, "felt_anchors.json"),
            os.path.join(data_dir, "schema_felt.json"),
            os.path.join(data_dir, "somatic_topo.json"),
            os.path.join(data_dir, "modulators_state.json"),
            os.path.join(data_dir, "long_term_interest.json"),
            CO_ACTIVATION_PATH,
        ]
        removed = []
        for path in (EPISTEMIC_GRAPH_PATH, CHRONOS_LOG_PATH, BIOSYSTEM_STATE_PATH, BASIN_STATE_PATH,
                     NARRATIVE_STATE_PATH, *extra):
            if path and os.path.exists(path):
                os.remove(path)
                removed.append(path)
        return removed
