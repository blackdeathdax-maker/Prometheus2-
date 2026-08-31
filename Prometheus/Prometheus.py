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

try:
    from .others import OthersRegistry
except ImportError:
    OthersRegistry = None  # optional until others.py is deployed
try:
    from .goals import GoalModule
except ImportError:
    GoalModule = None
try:
    from .operators import OperatorModule
except ImportError:
    OperatorModule = None
try:
    from .active_thread import ActiveThreadModule
except ImportError:
    ActiveThreadModule = None
try:
    from .allostasis import AllostasisModule
except ImportError:
    AllostasisModule = None


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
    SLEEP_SOFT_MIN = 0.55          # longer Learning day before evening
    SLEEP_HARD_MAX = 0.92          # mandatory sleep
    MIN_LEARNING_PULSES_BETWEEN_SLEEP = 45  # schema fuel needs a real Learning stretch
    SLEEP_WAKE_BELOW = 0.22        # exit sleep climate when pressure under this (scaled by debt)
    HYSTERESIS = 0.05
    # Legacy aliases so Debug sliders / old docs still map
    T1 = 0.55
    T2 = 0.85

    # Fatigue growth (per tick, scaled by urgency) and per-state recovery
    # rates. Consolidation recovers more than Pruning -- it's the
    # restorative state, Pruning is the costly one (fixed bug: Consolidation
    # previously applied zero recovery at all, trapping the system in a
    # permanent Consolidation<->Pruning oscillation that made Learning, and
    # therefore all graph growth, unreachable after the first few ticks).
    # All three remain undecided tuning placeholders (§10) -- named here
    # specifically so the Debug tab's sliders can adjust them live.
    FATIGUE_GROWTH_RATE = 0.12
    FATIGUE_RECOVERY_CONSOLIDATION = 0.88  # mild only — must not erase sleep pressure
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
    SELF_STUDY_MAX_ATTEMPTS = 8
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
        self.others = OthersRegistry(self.archivist) if OthersRegistry is not None else None
        self.goals = GoalModule() if GoalModule is not None else None
        self.operators = OperatorModule() if OperatorModule is not None else None
        self.active_thread = ActiveThreadModule() if ActiveThreadModule is not None else None
        self.allostasis = AllostasisModule() if AllostasisModule is not None else None
        # Package A: tiny world stub for action→consequence
        try:
            from .world_stub import WorldStub
            self.world_stub = WorldStub()
        except Exception:
            try:
                from world_stub import WorldStub
                self.world_stub = WorldStub()
            except Exception:
                self.world_stub = None
        self._pre_act_body_snap: dict = {}
        self._pre_act_world_snap: dict = {}
        # Package D: compositional plans over causal traces
        try:
            from .plan import PlanModule
            self.planner = PlanModule()
        except Exception:
            try:
                from plan import PlanModule
                self.planner = PlanModule()
            except Exception:
                self.planner = None

        # --- Learning policy (phase / inhibition / valence / lookup budgets) ---
        self.DICT_LOOKUPS_PER_PULSE = 6
        self.DICT_LOOKUPS_PER_WAKE = 300
        self._dict_lookups_this_pulse = 0
        self._dict_lookups_this_wake = 0
        self.PEDAGOGICAL_COACT_WEIGHT = 2.5
        self.OFF_BASIN_COACT_WEIGHT = 0.25
        self.CLOSED_PARENT_COACT_WEIGHT = 0.35
        self.LATERAL_INHIBITION_SCALE = 0.55
        self.SIBLING_INHIBITION_TOP_K = 6

        if getattr(self, "association", None) is not None:
            self.association.lookup_gate = self._may_dictionary_lookup
        if self.goals is not None:
            def _goal_narr(event, target, detail="", pulse=0):
                try:
                    self.self_narrative.record_goal_event(event, target, detail=detail, pulse=pulse)
                except Exception:
                    pass
            self.goals._on_event = _goal_narr
        self.last_collapse_summary = {"collapsed": 0, "conflicts": 0, "candidates_considered": 0}
        self.last_focus_summary = {}
        self.last_hierarchy_summary = {}
        self.last_tier2_created = []

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
        self.pulses_since_sleep = 0
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
        # per-target expansion phase: 0=hyponym, 1=hypernym, 2=synonym, 3=done
        self._self_study_phase = {}

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
        # Physical language → fixed body channels (parts of focus schema, not children)
        if source == "user":
            try:
                self._note_body_from_text(text)
            except Exception as e:
                logger.warning("_note_body_from_text: %s", e)

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
        # Self-relevance: ingested content co-activates with SELF so
        # epistemic/somatic schemas and narrative can bind self-relevantly.
        if node:
            try:
                from .archivist import SELF_NODE
                self.archivist.bump_activation(SELF_NODE)
                self.archivist.record_co_activation([node, SELF_NODE])
            except Exception:
                pass

        # §2.1b item 4a: try to name any unnamed schemas tied to the felt
        # state active right now (schema naming trigger when user/dictionary
        # input provides a word while "in" that state).
        if node and source in ("user", "dictionary"):
            self.association.try_name_schemas(node, current_felt_state=felt_state)

        # Named others (multi-entity social layer)
        other_ids = []
        if hasattr(self, "others") and source in ("user", "dictionary"):
            try:
                other_ids = self.others.process_text(text, pulse=self.pulse_count)
            except Exception as e:
                logger.warning("others.process_text failed: %s", e)

        relations = self.sensory.detect_relational(text)
        if relations and node:
            # other_ids only if association.link_relational supports it
            try:
                import inspect
                sig = inspect.signature(self.association.link_relational)
                if "other_ids" in sig.parameters:
                    self.association.link_relational(
                        node, relations, source=source, felt_state=felt_state,
                        other_ids=other_ids or None,
                    )
                else:
                    self.association.link_relational(
                        node, relations, source=source, felt_state=felt_state,
                    )
            except TypeError:
                # Fallback for unexpected signature differences
                self.association.link_relational(
                    node, relations, source=source, felt_state=felt_state,
                )
        # Bind named others into the same co-activation cluster as SELF + event
        if node and other_ids:
            try:
                from .archivist import SELF_NODE
                cluster = [node, SELF_NODE] + [o for o in other_ids if o][:4]
                self._record_co_activation_gated(cluster)
                for oid in other_ids[:4]:
                    self.archivist.bump_activation(oid)
            except Exception:
                pass

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
        if node and source == "user":
            try:
                self._mark_pedagogical([node])
                # Multi-word lessons: also mark individual content tokens already in graph
                for w in str(text).replace(",", " ").split():
                    tok = "".join(ch for ch in w if ch.isalnum())
                    if len(tok) >= 3 and tok in self.archivist.graph:
                        self._mark_pedagogical([tok])
            except Exception:
                pass

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



    def get_identity_hub_report(self) -> dict:
        """Unified SELF: body parts, felt places, narrative parts, agency."""
        from .archivist import SELF_NODE
        from .edge_types import is_body_channel_node, BODY_CHANNEL_NODE_IDS, EDGE_COMPOSED_OF, EDGE_PART_OF, EDGE_ASSOCIATED_WITH
        g = self.archivist.graph
        out = {
            "self_present": SELF_NODE in g,
            "felt": {},
            "body": [],
            "narrative_parts": [],
            "agency": [],
            "self_edges": [],
        }
        if SELF_NODE not in g:
            return out
        nd = g.nodes[SELF_NODE]
        out["felt"] = {
            "arousal": nd.get("felt_arousal"),
            "valence": nd.get("felt_valence"),
            "dominance": nd.get("felt_dominance"),
            "label": nd.get("last_felt_label"),
            "key": nd.get("last_felt_key"),
            "valence_coloring": nd.get("valence_coloring"),
        }
        for _u, v, ed in g.out_edges(SELF_NODE, data=True):
            rel = ed.get("relation_type")
            item = {"to": v, "relation": rel, "dwell": ed.get("dwell"), "source": ed.get("source")}
            out["self_edges"].append(item)
            if is_body_channel_node(v) or str(v).startswith("body:"):
                out["body"].append({
                    "node": v,
                    "value": g.nodes.get(v, {}).get("body_value"),
                    "relation": rel,
                })
            elif str(v).startswith("narr:") or g.nodes.get(v, {}).get("is_narrative_element"):
                out["narrative_parts"].append({
                    "node": v,
                    "weight": g.nodes.get(v, {}).get("narrative_weight"),
                    "type": g.nodes.get(v, {}).get("element_type"),
                    "linked_nodes": g.nodes.get(v, {}).get("linked_nodes"),
                })
            elif str(v).startswith("felt:"):
                out.setdefault("felt_places", []).append(v)
            elif rel in (EDGE_ASSOCIATED_WITH, "associated-with") and g.nodes.get(v, {}).get("is_schema"):
                out["agency"].append(v)
            elif g.nodes.get(v, {}).get("is_process_tag"):
                out.setdefault("process_tags", []).append({
                    "node": v,
                    "relation": rel,
                    "op": g.nodes.get(v, {}).get("process_op"),
                })
        # Active thread snapshot (identity surface)
        try:
            if getattr(self, "active_thread", None) is not None:
                out["active_thread"] = self.active_thread.report()
        except Exception:
            pass
        # Package A: world stub + act outcomes
        try:
            if getattr(self, "world_stub", None) is not None:
                out["world"] = self.world_stub.report()
        except Exception:
            pass
        try:
            if getattr(self, "operators", None) is not None:
                rep = self.operators.report()
                out["act"] = {
                    "last_op": rep.get("last_act_op") or rep.get("last_operator"),
                    "body_deltas": rep.get("last_act_body_deltas"),
                    "world_deltas": rep.get("last_act_world_deltas"),
                    "pending_body": rep.get("pending_body_delta"),
                    "confidence_top": rep.get("outcome_confidence_top"),
                }
        except Exception:
            pass
        return out


    def _note_body_from_text(self, text: str) -> list:
        """Map plain physical language onto fixed body-channel nodes.

        User may teach 'heart racing', 'sweating', etc. Those reinforce
        anatomy nodes and, if focus/schema is active, part-link into it.
        Does not create new channel types.
        """
        if not text:
            return []
        from .edge_types import (
            BODY_CHANNELS, body_channel_node_id, EDGE_COMPOSED_OF, EDGE_PART_OF,
            EDGE_ASSOCIATED_WITH, is_body_channel_node,
        )
        from .archivist import SELF_NODE, TIER_WORKING

        t = text.lower()
        aliases = {
            "heart_rate": ["heart", "heartbeat", "heart rate", "pulse", "racing heart", "heart racing"],
            "breath": ["breath", "breathing", "breathe", "respiration", "short of breath"],
            "muscle_tension": ["tension", "tense", "muscle", "clench", "tight muscles", "tightness"],
            "sweat_skin": ["sweat", "sweating", "sweaty", "perspir", "clammy"],
            "gut": ["gut", "stomach", "nausea", "butterflies", "belly"],
            "energy": ["energy", "energetic", "tired", "fatigue", "exhausted", "wired"],
            "warmth": ["warm", "warmth", "hot", "cold", "chill", "flush"],
        }
        hit = []
        for ch, words in aliases.items():
            if any(w in t for w in words):
                hit.append(ch)
        if not hit:
            return []

        if hasattr(self.archivist, "_seed_body_channels"):
            self.archivist._seed_body_channels()
        g = self.archivist.graph
        felt = None
        try:
            felt = self.synthesizer.get_current_felt_state()
            if felt in ("Unformed", "None"):
                felt = None
        except Exception:
            pass

        # Focus / recent schema as whole for part links
        wholes = []
        try:
            fid = None
            if hasattr(self, "focus") and hasattr(self.focus, "get_focus"):
                fid = self.focus.get_focus()
            elif hasattr(self, "focus") and hasattr(self.focus, "current"):
                fid = self.focus.current
            if fid and fid in g and not is_body_channel_node(fid):
                wholes.append(fid)
        except Exception:
            pass
        try:
            for n, d in g.nodes(data=True):
                if d and (d.get("is_schema") or "epistemic" in str(d.get("node_type") or "")):
                    if float(d.get("activation") or 0) >= 0.2 and not is_body_channel_node(n):
                        wholes.append(n)
            wholes = list(dict.fromkeys(wholes))[:5]
        except Exception:
            pass

        for ch in hit:
            nid = body_channel_node_id(ch)
            if nid not in g:
                continue
            # bump as "noticed"
            g.nodes[nid]["body_value"] = max(float(g.nodes[nid].get("body_value") or 0.5), 0.72)
            g.nodes[nid]["last_reinforced"] = __import__("datetime").datetime.now()
            try:
                self.archivist.link(SELF_NODE, nid, EDGE_ASSOCIATED_WITH, source="user", placement="self_body_user")
                self.archivist.bump_activation(nid)
            except Exception:
                pass
            for whole in wholes:
                try:
                    self.archivist.link(whole, nid, EDGE_COMPOSED_OF, source="user", placement="body_part_of_schema", felt_state=felt)
                    self.archivist.link(nid, whole, EDGE_PART_OF, source="user", placement="body_part_of_schema", felt_state=felt)
                except Exception:
                    pass
        return hit

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
            # Ongoing micro-motion (stand-in until live head / speech drive felt)
            try:
                self.modulators.ambient_tick(self.pulse_count)
            except Exception:
                pass
            # Bipolar climate shove: without live head, force hormones to
            # visit high-arousal and low-arousal poles so basins can form.
            try:
                import math
                h = self.bio._hormones
                p = self.pulse_count
                # slow carrier ~80 pulses: adrenaline/cortisol vs serotonin
                phase = math.sin(p * 0.0785)  # ~80-pulse period
                phase2 = math.sin(p * 0.041)
                # amplitude large enough to move body map across bins
                h["adrenaline"] = max(0.05, min(0.95, float(h.get("adrenaline", 0.5)) + 0.012 * phase))
                h["cortisol"] = max(0.05, min(0.95, float(h.get("cortisol", 0.5)) + 0.010 * phase))
                h["serotonin"] = max(0.05, min(0.95, float(h.get("serotonin", 0.5)) + 0.012 * (-phase)))
                h["dopamine"] = max(0.05, min(0.95, float(h.get("dopamine", 0.5)) + 0.008 * phase2))
                h["oxytocin"] = max(0.05, min(0.95, float(h.get("oxytocin", 0.5)) + 0.006 * (-phase2)))
            except Exception:
                pass
            # Optional external / synthetic stimulus engine
            try:
                if hasattr(self, "stimulus") and self.stimulus is not None:
                    for meth in ("tick", "step", "pulse"):
                        if hasattr(self.stimulus, meth):
                            getattr(self.stimulus, meth)()
                            break
            except Exception as e:
                logger.warning("stimulus tick failed: %s", e)
            fast_delta = self.modulators.body_delta()
        else:
            fast_delta = None

        # Package A: merge pending operator→body outcomes from prior pulse
        try:
            if self.operators is not None:
                act_delta = self.operators.consume_pending_body_delta()
                if act_delta:
                    if fast_delta is None:
                        fast_delta = {}
                    for ch, dv in act_delta.items():
                        fast_delta[ch] = float(fast_delta.get(ch, 0.0)) + float(dv)
        except Exception as e:
            logger.debug("consume act body delta: %s", e)

        # Body surface = medium/slow hormones + fast gusts (still no chemical names)
        body = self.bio.get_raw_variables(fast_body_delta=fast_delta)

        # Thin C: update outcome confidence from last act (before vs after sense)
        try:
            if self.operators is not None and self.operators._last_act_op:
                op0 = self.operators._last_act_op
                pred_b = dict(self.operators._last_act_body_deltas or {})
                before_b = dict(self._pre_act_body_snap or {})
                if pred_b and before_b:
                    self.operators.update_outcome_confidence(
                        op0, pred_b, before_b, body, prefix="",
                    )
                pred_w = dict(self.operators._last_act_world_deltas or {})
                before_w = dict(self._pre_act_world_snap or {})
                after_w = {}
                if getattr(self, "world_stub", None) is not None:
                    after_w = self.world_stub.observe()
                if pred_w and before_w:
                    self.operators.update_outcome_confidence(
                        op0, pred_w, before_w, after_w, prefix="world:",
                    )
        except Exception as e:
            logger.debug("outcome confidence update: %s", e)

        # synthesizer must run first, before anything that conditions a
        # decision on its output (regulation trigger, executive bias).
        self.synthesizer.update_from_core(body)
        # Mixed-affect conflict → fast modulators (alert up, settle down)
        if hasattr(self, "modulators") and hasattr(self.synthesizer, "get_conflict_score"):
            try:
                self.modulators.apply_conflict(self.synthesizer.get_conflict_score())
            except Exception as e:
                logger.warning("apply_conflict failed: %s", e)
        self.somatic_topo.record(self.synthesizer.get_current_basin_key())
        self.felt_anchors.observe(
            self.synthesizer.get_current_basin_key(),
            raw_body=body,
        )
        # Allostasis & Affect: adaptive set-points + pain/pleasure caps
        try:
            if getattr(self, "allostasis", None) is not None:
                _intent = ""
                _goals_on = False
                try:
                    if self.active_thread is not None:
                        _intent = str(self.active_thread.thread.intent or "")
                        _goals_on = bool(self.active_thread.thread.goal_ids)
                except Exception:
                    pass
                if not _goals_on and getattr(self, "goals", None) is not None:
                    try:
                        _goals_on = bool(self.goals.active_target_ids())
                    except Exception:
                        pass
                ep = ""
                try:
                    ep = str(self.bio.epoch.value)
                except Exception:
                    pass
                fat = float(getattr(self, "fatigue", 0) or 0)
                self.allostasis.compute_setpoints(
                    intent=_intent, has_goals=_goals_on, epoch=ep, fatigue=fat,
                )
                _bmax = 0.0
                try:
                    if self.active_thread is not None:
                        _bmax = float(self.active_thread.thread.max_abs_body_error or 0)
                except Exception:
                    pass
                _conflict = 0.0
                try:
                    if hasattr(self.synthesizer, "get_conflict_score"):
                        _conflict = float(self.synthesizer.get_conflict_score() or 0)
                except Exception:
                    pass
                _barren = False
                try:
                    fid0 = getattr(self.focus, "focus_id", None)
                    if fid0 and fid0 in getattr(self, "_barren_self_study_targets", set()):
                        _barren = True
                except Exception:
                    pass
                _last_op = ""
                try:
                    dec0 = getattr(self.operators, "last_decision", None)
                    if dec0 is not None:
                        _last_op = str(getattr(dec0, "operator", "") or "")
                except Exception:
                    pass
                self.allostasis.update_affect(
                    body,
                    body_error_max=_bmax,
                    barren_focus=_barren,
                    conflict=_conflict,
                    last_op=_last_op,
                )
                self._last_body_surface = dict(body)
        except Exception as e:
            logger.debug("allostasis update failed: %s", e)
        # Pre-linguistic self-awareness: SELF is always bound to how the
        # body feels right now (infants have this without words).
        try:
            self._sync_self_felt()
        except Exception as e:
            logger.warning("_sync_self_felt failed: %s", e)
        try:
            if hasattr(self.archivist, "repair_identity_edges") and self.pulse_count % 10 == 1:
                self.archivist.repair_identity_edges()
            if hasattr(self, "reflector") and self.pulse_count % 10 == 1:
                try:
                    self.reflector.scrub_invalid_schema_names()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("repair_identity_edges: %s", e)
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
                # Bias focus toward goals + current WM (anti-drift)
        try:
            prefer = set()
            if getattr(self, "goals", None) is not None:
                prefer |= set(self.goals.protected_ids(self.archivist.graph))
            wm_ids = set()
            try:
                wm_ids = set(self.get_current_working_memory().get("slots") or [])
            except Exception:
                pass
            prefer |= wm_ids
            self.focus.set_attention_context(prefer_ids=prefer, wm_ids=wm_ids)
        except Exception as e:
            logger.debug("set_attention_context failed: %s", e)
        self.last_focus_summary = self.focus.tick(
            self.archivist.graph,
            pulse=self.pulse_count,
            basin_anchor_set=basin_anchors,
        )
        try:
            self.focus.refresh_closure_cache(self.archivist.graph)
            # Schema-guided residual: keep members "warm" while schema is focus
            for nid in list(getattr(self.focus, "_closure_cache", set()) or [])[:40]:
                self.focus.boost_residual(nid, amount=0.08)
        except Exception as e:
            logger.debug("schema-guided focus warm failed: %s", e)

        # Explicit commitments (contentful substrate)
        if getattr(self, "goals", None) is not None:
            try:
                fid = self.focus.focus_id
                self.goals.observe_focus(fid, self.pulse_count, graph=self.archivist.graph)
                fs = self.last_focus_summary or {}
                # Thread intent / body_error may lag one pulse (operators run later);
                # still pass best-known values for REGULATE soft-delay.
                _thr = getattr(self, "active_thread", None)
                _tr = _thr.thread if _thr is not None else None
                summary = self.goals.tick(
                    pulse=self.pulse_count,
                    focus_id=fid,
                    residual_fn=lambda n: self.focus.total_residual(n),
                    stagnation=bool(fs.get("stagnation_escape")),
                    force_switch=bool(fs.get("force_switch") or fs.get("hard_age_escape")),
                    graph=self.archivist.graph,
                    thread_intent=getattr(_tr, "intent", "") if _tr else "",
                    max_body_error=float(getattr(_tr, "max_abs_body_error", 0) or 0) if _tr else 0.0,
                )
                # Inject commitment residual into focus store (target + schema closure)
                graph = self.archivist.graph
                boosted = set()
                for tid in self.goals.active_target_ids():
                    boost = self.goals.commitment_boost(tid, graph=graph)
                    # Ensure open goals stay in the attention contest
                    try:
                        self.focus.boost_residual(tid, amount=max(0.25, float(boost or 0) * 0.5))
                    except Exception:
                        pass
                    if boost > 0 and tid not in boosted:
                        self.focus.boost_residual(tid, amount=min(1.2, boost * 0.35))
                        boosted.add(tid)
                    # Warm schema members under the goal
                    try:
                        for mid in list(self.goals.schema_closure_ids(graph, tid))[:32]:
                            if mid in boosted:
                                continue
                            b = self.goals.commitment_boost(mid, graph=graph)
                            if b > 0:
                                self.focus.boost_residual(mid, amount=min(0.8, b * 0.25))
                                boosted.add(mid)
                    except Exception:
                        pass
                if summary.get("satisfied_this_tick") or summary.get("failed_this_tick"):
                    print(f"Goals: {summary}")
                # Package D: rebuild plan from causal traces for primary goal
                try:
                    if getattr(self, "planner", None) is not None:
                        gids = list(self.goals.active_target_ids() or [])
                        def _closure(gid):
                            try:
                                return self.goals.schema_closure_ids(graph, gid)
                            except Exception:
                                return set()
                        plan = self.planner.tick(
                            pulse=self.pulse_count,
                            graph=graph,
                            goal_ids=gids,
                            schema_closure_fn=_closure,
                        )
                        means = self.planner.next_means_id()
                        if means:
                            try:
                                self.focus.boost_residual(means, amount=0.55)
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("planner tick: %s", e)
                # Allostatic safety: extreme pain can release weakest off-focus goal
                try:
                    if getattr(self, "allostasis", None) is not None and self.goals is not None:
                        nesc = self.goals.allostatic_escape_close(
                            self.pulse_count,
                            pain=float(self.allostasis.state.pain or 0),
                        )
                        if nesc:
                            print(f"Goals: allostatic_escape closed {nesc}")
                except Exception:
                    pass
                # Pleasure pulse on growth satisfaction
                try:
                    if (
                        getattr(self, "allostasis", None) is not None
                        and summary.get("satisfied_this_tick")
                    ):
                        self.allostasis.update_affect(
                            getattr(self, "_last_body_surface", {}) or {"pain": self.allostasis.state.pain, "pleasure": self.allostasis.state.pleasure},
                            goal_growth=True,
                            last_op="",
                        )
                except Exception:
                    pass
                # Narrative beats for newly closed goals
                try:
                    for h in (self.goals.history or [])[-3:]:
                        pulse_h = h.satisfied_pulse or h.failed_pulse
                        if pulse_h == self.pulse_count:
                            self.self_narrative.record_goal_event(
                                h.status, h.target_id,
                                detail=h.success_reason or h.fail_reason,
                                pulse=self.pulse_count,
                            )
                except Exception:
                    pass
            except Exception as e:
                logger.warning("goals tick failed: %s", e)

        # Stream of consciousness (pulse-time trace)
        try:
            fid = self.focus.focus_id if self.focus else None
            wm_slots = []
            try:
                wm_slots = list(self.get_current_working_memory().get("slots") or [])
            except Exception:
                pass
            goals = []
            try:
                if getattr(self, "goals", None) is not None:
                    goals = list(self.goals.active_target_ids() or [])
            except Exception:
                pass
            residual_top = []
            try:
                # top residual keys
                items = sorted(
                    (self.focus.residuals or {}).items(),
                    key=lambda kv: -float(kv[1] or 0),
                )[:5]
                residual_top = [k for k, _v in items]
            except Exception:
                pass
            # Pulse-time narrative elements (not only Consolidation)
            try:
                last_op = ""
                expand_placed = 0
                try:
                    dec = getattr(getattr(self, "operators", None), "last_decision", None)
                    if dec is not None:
                        last_op = str(getattr(dec, "operator", "") or "")
                    # last expand placement from episode if any
                    eps = getattr(getattr(self, "operators", None), "episodes", None) or []
                    if eps:
                        last_ep = eps[-1]
                        expand_placed = int(last_ep.get("placed") or last_ep.get("nodes_placed") or 0)
                except Exception:
                    pass
                if hasattr(self.self_narrative, "observe_live"):
                    self.self_narrative.observe_live(
                        pulse=self.pulse_count,
                        focus_id=fid,
                        goal_targets=goals,
                        wm_slots=wm_slots,
                        operator=last_op,
                        expand_placed=expand_placed,
                    )
            except Exception:
                pass
            hub_line = self._hub_stream_line()
            self.self_narrative.record_stream_beat(
                pulse=self.pulse_count,
                focus_id=fid,
                felt_state=str(self.synthesizer.get_current_felt_state() or ""),
                basin_key=str(self.synthesizer.get_current_basin_key() or ""),
                wm_slots=wm_slots,
                goal_targets=goals,
                hub_line=hub_line,
                bias=str(getattr(self.executive, "current_bias", "") or ""),
                state=str(self.state or ""),
                residual_top=residual_top,
            )
        except Exception as e:
            logger.debug("stream beat failed: %s", e)

        try:
            op_summary = self._run_cognition_operators()
            if op_summary and op_summary.get("operator") not in (None, "HOLD"):
                print(f"Operator: {op_summary}")
        except Exception as e:
            logger.warning("cognition operators failed: %s", e)

        # hierarchy micro-repair when focused on a kind hub
        try:
            fid = self.focus.focus_id if self.focus else None
            hubs = set()
            try:
                hubs = self._kind_hub_ids()
            except Exception:
                pass
            if fid and (fid in hubs or str(fid).casefold() in {str(h).casefold() for h in hubs}) and self.pulse_count % 5 == 0:
                r = self._repair_hierarchy_edges()
                if r and any(r.values()):
                    print(f"Micro-repair: {r}")
                g = self._grow_kind_schema_membership()
                if g and g.get("members_linked"):
                    print(f"Micro-grow: {g}")
        except Exception as e:
            logger.debug("micro-repair failed: %s", e)




        # Working-memory co-presence → co-activation (§14 / kind schemas).
        # Nodes held together "in mind" (e.g. Emotion, Sad, Happy) should
        # accumulate pair evidence even without self-study co-touch.
        try:
            wm = self.get_current_working_memory()
            slots = [s for s in (wm.get("slots") or []) if s and s in self.archivist.graph]
            concepts = []
            for s in slots:
                if s in ("SELF", "OTHER"):
                    continue
                nd = self.archivist.graph.nodes.get(s, {})
                if nd.get("node_type") in ("basin",):
                    continue
                concepts.append(s)
            if len(concepts) >= 2:
                self.archivist.record_co_activation(concepts)
                lower_map = {c.lower(): c for c in concepts}
                for parent_key, children in (
                    ("emotion", (
                        "sad", "happy", "angry", "fear", "afraid", "joy", "disgust",
                        "surprise", "anger", "sadness", "happiness", "dismay",
                    )),
                    ("color", (
                        "red", "blue", "green", "yellow", "black", "white", "orange",
                        "purple", "pink", "brown", "gray", "grey",
                    )),
                ):
                    parent = lower_map.get(parent_key)
                    if not parent:
                        continue
                    for child_key in children:
                        child = lower_map.get(child_key)
                        if child and child != parent:
                            try:
                                self.archivist.link(
                                    child, parent, "is-a",
                                    source="wm_inference", placement="explicit",
                                )
                            except Exception:
                                pass
        except Exception as e:
            logger.warning("WM co-activation failed: %s", e)

        results = self.archivist.retrieve("context")


        # Schema ↔ felt-anchor co-occurrence (implicit; no emotion taxonomy)
        try:
            active_schemas = []
            graph = self.archivist.graph

            def _maybe_schema(nid):
                if not nid or nid not in graph:
                    return
                nd = graph.nodes.get(nid) or {}
                nt = nd.get("node_type")
                if (
                    str(nid).startswith("epistemic_")
                    or str(nid).startswith("schema_")
                    or nd.get("is_schema")
                    or nt in ("schema", "epistemic_schema")
                ):
                    active_schemas.append(nid)
                    return
                # Knowledge lemmas (Anger, Color) count when focused —
                # binding should not require only formal schema nodes.
                if nt in ("lemma", "concept", "entity", None) or nd.get("name"):
                    # Skip pure body/felt/narr infrastructure
                    s = str(nid)
                    if s.startswith(("body:", "felt_", "felt:", "narr:", "SELF")):
                        return
                    if nd.get("is_felt_place") or nd.get("body_channel"):
                        return
                    active_schemas.append(nid)

            fid = getattr(self.focus, "focus_id", None)
            _maybe_schema(fid)

            # Active goals
            try:
                if hasattr(self, "goals") and self.goals is not None:
                    for g in (self.goals.active_list() if hasattr(self.goals, "active_list") else []) or []:
                        tid = g.get("target") if isinstance(g, dict) else getattr(g, "target", None)
                        _maybe_schema(tid)
                    # common alternate APIs
                    for attr in ("active_targets", "targets", "list_active"):
                        if hasattr(self.goals, attr):
                            vals = getattr(self.goals, attr)
                            vals = vals() if callable(vals) else vals
                            for tid in (vals or []):
                                if isinstance(tid, dict):
                                    tid = tid.get("target") or tid.get("id")
                                _maybe_schema(tid)
            except Exception:
                pass

            if hasattr(self, "working_memory") and self.working_memory is not None:
                try:
                    wm = self.working_memory.get_current_working_memory()
                    for sid in (wm.get("slots") or []):
                        _maybe_schema(sid)
                except Exception:
                    pass

            # Dedupe preserve order
            seen = set()
            deduped = []
            for s in active_schemas:
                if s not in seen:
                    seen.add(s)
                    deduped.append(s)
            active_schemas = deduped

            cur = self.felt_anchors.current()
            if cur is not None and active_schemas:
                self.schema_felt.note(active_schemas, cur.anchor_id)
                # Live promote when any pair crosses threshold so UI fills
                # without waiting only for sleep consolidation
                try:
                    for sid in active_schemas:
                        c = int(self.schema_felt.cooccur.get(sid, {}).get(cur.anchor_id, 0))
                        if c >= self.schema_felt.threshold and cur.anchor_id not in self.schema_felt.binds.get(sid, ()):
                            self.schema_felt.promote(self.archivist.graph)
                            break
                except Exception:
                    pass
            # Schema–schema co-activation: schemas sharing WM/focus get paired
            # so Tier-2 stacking has stabilized fuel (not only leaf co-touch).
            if len(active_schemas) >= 2 and hasattr(self.archivist, "record_schema_co_activation"):
                try:
                    self.archivist.record_schema_co_activation(
                        list(dict.fromkeys(active_schemas)), amount=1.0
                    )
                except Exception as e:
                    logger.warning("schema co-activation (pulse) failed: %s", e)
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


    # ==================================================================
    # Learning policy: reason-gated lookup, phase, inhibition, valence
    # ==================================================================
    def _reset_pulse_lookup_budget(self) -> None:
        self._dict_lookups_this_pulse = 0

    def _note_wake_boundary(self) -> None:
        """Call when leaving Sleep → Learning."""
        self._dict_lookups_this_wake = 0

    def _open_parent_ids(self) -> set:
        """Parents currently 'open' (phase window): focus closure, WM, goals.

        Lemma and kind-schema (Color ↔ epistemic_of_Color) share one window.
        """
        open_ids = set()
        try:
            fid = self.focus.focus_id if self.focus else None
            if fid:
                open_ids.add(fid)
                open_ids |= set(self.focus.schema_closure(self.archivist.graph, fid) or [])
        except Exception:
            pass
        try:
            wm = self.get_current_working_memory()
            open_ids |= set(wm.get("slots") or [])
        except Exception:
            pass
        try:
            if getattr(self, "goals", None) is not None:
                open_ids |= set(self.goals.protected_ids(self.archivist.graph) or [])
        except Exception:
            pass
        # Expand each id to its kind family (lemma ↔ schema)
        expanded = set(open_ids)
        for nid in list(open_ids):
            try:
                expanded |= set(self.archivist.kind_family(nid) or [])
            except Exception:
                pass
        return expanded

    def _may_dictionary_lookup(self, term: str, source: str = "dictionary",
                               context_node=None) -> bool:
        """WordNet shelf: allow expand-from-known; budget new free-floating nodes.

        Self-study always passes context_node=target (known node) → allow.
        Budget only limits brand-new terms without a graph parent context.
        """
        if source != "dictionary":
            return True
        g = self.archivist.graph
        # Already present (any case) — free reinforce
        if term in g:
            return True
        if term:
            low = str(term).casefold()
            for n in g.nodes:
                if str(n).casefold() == low:
                    return True

        pulse_n = int(getattr(self, "_dict_lookups_this_pulse", 0) or 0)
        wake_n = int(getattr(self, "_dict_lookups_this_wake", 0) or 0)
        per_pulse = int(getattr(self, "DICT_LOOKUPS_PER_PULSE", 6) or 6)
        per_wake = int(getattr(self, "DICT_LOOKUPS_PER_WAKE", 300) or 300)
        if pulse_n >= per_pulse or wake_n >= per_wake:
            return False

        # Expand from a known context/target → always a valid reason
        if context_node:
            if context_node in g:
                self._dict_lookups_this_pulse = pulse_n + 1
                self._dict_lookups_this_wake = wake_n + 1
                return True
            # case-insensitive context match
            cl = str(context_node).casefold()
            for n in g.nodes:
                if str(n).casefold() == cl:
                    self._dict_lookups_this_pulse = pulse_n + 1
                    self._dict_lookups_this_wake = wake_n + 1
                    return True

        open_ids = self._open_parent_ids()
        if open_ids:
            self._dict_lookups_this_pulse = pulse_n + 1
            self._dict_lookups_this_wake = wake_n + 1
            return True
        return False

    def _current_basin_key(self) -> str:
        try:
            return str(self.synthesizer.get_current_basin_key() or "")
        except Exception:
            return ""

    def _node_basin_tag(self, node_id: str) -> str:
        d = self.archivist.graph.nodes.get(node_id, {})
        return str(d.get("basin_tag") or d.get("felt_state_at_creation") or "")

    def _stamp_basin_tag(self, node_id: str) -> None:
        if node_id not in self.archivist.graph:
            return
        key = self._current_basin_key()
        if not key:
            return
        d = self.archivist.graph.nodes[node_id]
        if not d.get("basin_tag"):
            d["basin_tag"] = key

    def _coact_weight_for_nodes(self, nodes) -> float:
        """Phase + valence + pedagogical + goal-family gating for kind evidence."""
        nodes = [n for n in nodes if n in self.archivist.graph]
        if len(nodes) < 2:
            return 0.0
        w = 1.0
        open_ids = self._open_parent_ids()
        if open_ids and not any(n in open_ids for n in nodes):
            w *= self.CLOSED_PARENT_COACT_WEIGHT
        cur = self._current_basin_key()
        if cur:
            tags = [self._node_basin_tag(n) for n in nodes]
            tagged = [t for t in tags if t]
            if tagged and not any(t == cur for t in tagged):
                w *= self.OFF_BASIN_COACT_WEIGHT
        ped = 0
        for n in nodes:
            if self.archivist.graph.nodes.get(n, {}).get("pedagogical"):
                ped += 1
        if ped >= 2:
            w *= self.PEDAGOGICAL_COACT_WEIGHT
        elif ped == 1:
            w *= 1.4

        # Active goal family: strongly suppress off-family pair evidence
        # (stops Color+baby+child mud clusters while Color goal is open)
        try:
            if getattr(self, "goals", None) is not None:
                gts = list(self.goals.active_target_ids() or [])
                if gts:
                    fam = set()
                    for g in gts[:3]:
                        fam |= set(self.archivist.kind_family(g) or [])
                        try:
                            fam |= set(self.focus.schema_closure(self.archivist.graph, g) or [])
                        except Exception:
                            pass
                    if fam:
                        on = sum(1 for n in nodes if n in fam)
                        if on == 0:
                            w *= 0.05  # pure off-family
                        elif on < len(nodes):
                            w *= 0.25  # mixed pair
                        else:
                            w *= 1.6   # pure on-family
        except Exception:
            pass
        return w

    def _record_co_activation_gated(self, nodes) -> None:
        nodes = [n for n in nodes if n and n in self.archivist.graph]
        if len(nodes) < 2:
            return
        w = self._coact_weight_for_nodes(nodes)
        if w <= 0:
            return
        self.archivist.record_co_activation(nodes, weight=w)

    def _apply_lateral_inhibition(self) -> int:
        """When focus wins among siblings under same is-a parent, damp rivals."""
        fid = self.focus.focus_id if self.focus else None
        if not fid or fid not in self.archivist.graph:
            return 0
        graph = self.archivist.graph
        # Parents of focus via is-a out-edges
        parents = []
        for _u, v, ed in graph.out_edges(fid, data=True):
            if ed.get("relation_type") == "is-a":
                parents.append(v)
        if not parents:
            return 0
        damp_n = 0
        scale = self.LATERAL_INHIBITION_SCALE
        for parent in parents:
            siblings = []
            for u, _v, ed in graph.in_edges(parent, data=True):
                if ed.get("relation_type") != "is-a":
                    continue
                if u == fid:
                    continue
                # only same-level standard nodes
                if graph.nodes.get(u, {}).get("node_type") in (
                    "epistemic_schema", "schema", "basin"
                ):
                    continue
                siblings.append(u)
            # also children of parent via reverse if stored parent→child (skip)
            siblings = siblings[: self.SIBLING_INHIBITION_TOP_K * 3]
            # damp residual on siblings
            for s in siblings[: self.SIBLING_INHIBITION_TOP_K]:
                try:
                    r = self.focus.residuals.get(s, 0.0)
                    if r > 0:
                        self.focus.residuals[s] = r * scale
                        damp_n += 1
                    # activation damp
                    if s in graph:
                        act = float(graph.nodes[s].get("activation", 0) or 0)
                        if act > 0:
                            graph.nodes[s]["activation"] = act * scale
                except Exception:
                    pass
            # slight vertical boost to parent
            try:
                self.focus.boost_residual(parent, amount=0.05)
            except Exception:
                pass
        return damp_n

    def _mark_pedagogical(self, node_ids) -> None:
        for n in node_ids:
            if n in self.archivist.graph:
                self.archivist.graph.nodes[n]["pedagogical"] = True
                self.archivist.graph.nodes[n]["user_linked"] = True
                self._stamp_basin_tag(n)


    # Dict hygiene under emotion / affective focus (not an emotion ontology)
    DICT_NOISE_LEAVES = frozenset({
        "akvavit", "aquavit", "arrack", "arak", "bitters", "absinthe",
        "brandy", "whiskey", "whisky", "vodka", "gin", "rum", "tequila",
        "liqueur", "schnapps", "ouzo", "sambuca", "vermouth", "cognac",
        "bourbon", "scotch", "moonshine", "hooch", "grog", "mead",
        "cordial", "aperitif", "digestif", "kirsch", "slivovitz",
    })
    AFFECT_LEMMA_HINTS = frozenset({
        "emotion", "feeling", "mood", "affect", "joy", "fear", "anger",
        "sadness", "disgust", "surprise", "anxiety", "calm", "pride",
        "shame", "guilt", "love", "hate", "hope", "dread", "elation",
        "grief", "rage", "terror", "delight", "sorrow", "pleasure", "pain",
    })

    def _focus_is_affective(self, target: str = None) -> bool:
        """True when self-study is under emotion/taught affect neighborhood."""
        try:
            g = self.archivist.graph
            ids = set()
            fid = getattr(self.focus, "focus_id", None)
            if fid:
                ids.add(str(fid))
            if target:
                ids.add(str(target))
            try:
                ids |= set(self.focus_neighborhood_ids(cap=24) or [])
            except Exception:
                pass
            for nid in list(ids)[:40]:
                low = str(nid).lower().replace("epistemic_of_", "").replace("_", " ")
                if any(h in low for h in self.AFFECT_LEMMA_HINTS):
                    return True
                nd = g.nodes.get(nid, {}) or {}
                name = str(nd.get("name") or "").lower()
                if any(h in name for h in self.AFFECT_LEMMA_HINTS):
                    return True
                # is-a emotion / member of emotion schema
                if nid in g:
                    for _u, v, ed in g.out_edges(nid, data=True):
                        if (ed or {}).get("relation_type") == "is-a":
                            vl = str(v).lower()
                            if "emotion" in vl or "feeling" in vl or "mood" in vl:
                                return True
                    for u, _v, ed in g.in_edges(nid, data=True):
                        if (ed or {}).get("relation_type") in ("composed-of", "member-of"):
                            ul = str(u).lower()
                            if "emotion" in ul or "epistemic_of_emotion" in ul:
                                return True
        except Exception:
            pass
        return False

    def _is_noise_dict_leaf(self, term: str, affective: bool = False) -> bool:
        """Skip liquor / multi-word gloss leaves that pollute emotion soaks."""
        if not term or not isinstance(term, str):
            return True
        t = term.strip()
        low = t.lower()
        if low in self.DICT_NOISE_LEAVES:
            return True
        # multi-word dictionary glosses (not short lemmas)
        words = [w for w in low.replace("-", " ").split() if w]
        if len(words) > 2 or len(t) > 36:
            return True
        if affective and low in self.DICT_NOISE_LEAVES:
            return True
        # "spirits" liquor sense often expands to drink names
        if affective and low.endswith(" spirits"):
            return True
        return False

    def _ordered_self_study_expansions(self, target: str):
        """Phase-ordered expansion for a target (esp. WM/user nodes):

        phase 0: hyponyms (kind children) until exhausted / soft-cap
        phase 1: hypernym (parent)
        phase 2: synonyms
        phase 3: done → barren

        Skips terms already linked under the target so we advance phases
        instead of re-placing the same leaves forever.
        Under affective focus: tighter synonym cap + noise filter.
        """
        graph = self.archivist.graph
        phase = int(self._self_study_phase.get(target, 0))
        plan = []
        seen = set()
        affective = self._focus_is_affective(target)
        syn_cap = 1 if affective else 3
        hypo_cap = 4 if affective else 12

        # Already children of target (any categorical edge)
        already = set()
        if target in graph:
            for _u, v, ed in graph.out_edges(target, data=True):
                if ed.get("relation_type") in ("is-a", "part-of", "associated-with", "composed-of"):
                    already.add(v)
            for u, _v, ed in graph.in_edges(target, data=True):
                if ed.get("relation_type") in ("is-a", "part-of", "associated-with"):
                    already.add(u)

        def hypos(label):
            try:
                return list(self.sensory.lookup_expansion(label) or [])
            except Exception:
                return []

        def hyper(label):
            try:
                return self.sensory.lookup_hypernym(label)
            except Exception:
                return None

        def syns(label):
            try:
                return list(self.sensory.lookup_synonyms(label) or [])[:syn_cap]
            except Exception:
                return []

        labels = [target]
        if isinstance(target, str) and " " in target:
            stop = {"a","an","the","of","or","and","to","in","on","for","with","is","are","was","were"}
            for word in target.replace("-", " ").split():
                w = "".join(ch for ch in word if ch.isalnum()).lower()
                if len(w) >= 3 and w not in stop:
                    labels.append(w)

        def _ok(term):
            if not term or term == target or term in already or term in seen:
                return False
            if self._is_noise_dict_leaf(term, affective=affective):
                return False
            return True

        # --- phase 0: hyponyms ---
        if phase <= 0:
            for lab in labels:
                for h in hypos(lab):
                    if _ok(h):
                        seen.add(h)
                        plan.append(("hyponym", h))
                        if len(plan) >= hypo_cap:
                            break
                if len(plan) >= hypo_cap:
                    break
            if plan:
                return plan
            # exhausted hyponyms → advance
            self._self_study_phase[target] = 1
            phase = 1

        # --- phase 1: hypernym ---
        if phase == 1:
            for lab in labels:
                h = hyper(lab)
                if _ok(h):
                    seen.add(h)
                    plan.append(("hypernym", h))
            if plan:
                return plan
            self._self_study_phase[target] = 2
            phase = 2

        # --- phase 2: synonyms ---
        if phase == 2:
            for lab in labels:
                for s in syns(lab):
                    if _ok(s):
                        seen.add(s)
                        plan.append(("synonym", s))
            if plan:
                return plan
            self._self_study_phase[target] = 3

        return plan


    # --- Package B: causal co-occurrence from focus + act outcomes ---
    CAUSAL_DELTA_MIN = 0.02          # min |delta| to mint/reinforce a cause edge
    CAUSAL_WEIGHT_INC = 0.08
    CAUSAL_WEIGHT_CAP = 1.0
    CAUSAL_CONF_START = 0.30

    def _ensure_world_slot_node(self, slot: str) -> str:
        """Ensure world:<slot> exists as non-taxonomic infrastructure."""
        nid = f"world:{slot}"
        g = self.archivist.graph
        if nid not in g:
            g.add_node(
                nid,
                node_type="world_slot",
                is_world_slot=True,
                growable=False,
                source="world_stub",
                activation=0.0,
                body_value=None,
                world_value=float((self.world_stub.slots if self.world_stub else {}).get(slot, 0.0)),
            )
        elif self.world_stub is not None:
            g.nodes[nid]["world_value"] = float(self.world_stub.slots.get(slot, 0.0))
            g.nodes[nid]["is_world_slot"] = True
            g.nodes[nid]["growable"] = False
        return nid

    def _record_causal_cooccurrence(
        self,
        focus_id: Optional[str],
        op_name: str,
        body_deltas: dict,
        world_deltas: dict,
    ) -> int:
        """Mint/reinforce causes/enables/prevents from focus when act moves targets.

        Co-occurrence rule (user default for B):
          focus active + operator applied + |delta| >= CAUSAL_DELTA_MIN
          → focus --causes--> body:channel   (body change)
          → focus --enables--> world:slot    (positive world change)
          → focus --prevents--> world:slot   (negative world change on obstacle-like)

        Never uses is-a on body/world. Weight + confidence on edge for thin C continuity.
        """
        if not focus_id or focus_id not in self.archivist.graph:
            return 0
        try:
            from .edge_types import (
                EDGE_CAUSES, EDGE_ENABLES, EDGE_PREVENTS,
                body_channel_node_id, is_body_channel_node, is_felt_place_node,
                is_forbidden_epistemic_parent,
            )
        except Exception:
            return 0

        # Focus must be a knowledge lemma (not body/felt/SELF/schema shell)
        fl = str(focus_id)
        if fl in ("SELF", "OTHER") or fl.startswith(("body:", "felt:", "basin_", "narr:", "world:", "epistemic_")):
            return 0
        nd = self.archivist.graph.nodes.get(focus_id, {}) or {}
        if nd.get("is_schema") or nd.get("node_type") in ("schema", "epistemic_schema"):
            return 0

        made = 0
        conf = self.CAUSAL_CONF_START
        if self.operators is not None:
            # use mean confidence of this op's body outcomes if any
            try:
                confs = [
                    self.operators.get_confidence(op_name, ch)
                    for ch in (body_deltas or {})
                ]
                if confs:
                    conf = sum(confs) / len(confs)
            except Exception:
                pass

        # Body consequences: focus causes body:channel
        for ch, dv in (body_deltas or {}).items():
            if abs(float(dv)) < self.CAUSAL_DELTA_MIN:
                continue
            bnode = body_channel_node_id(ch)
            if bnode not in self.archivist.graph:
                continue
            w_inc = min(self.CAUSAL_WEIGHT_CAP, self.CAUSAL_WEIGHT_INC * (abs(float(dv)) / 0.05))
            try:
                self.archivist.link(
                    focus_id, bnode, EDGE_CAUSES,
                    source="causal_cooccur", placement="act_cooccur",
                    weight=w_inc, confidence=conf,
                )
                made += 1
            except Exception:
                pass

        # World consequences
        for slot, dv in (world_deltas or {}).items():
            if abs(float(dv)) < self.CAUSAL_DELTA_MIN:
                continue
            wnode = self._ensure_world_slot_node(slot)
            rel = EDGE_ENABLES if float(dv) > 0 else EDGE_PREVENTS
            # obstacle decrease is more "prevents obstacle" / enables progress
            if slot == "obstacle" and float(dv) < 0:
                rel = EDGE_PREVENTS
            elif float(dv) > 0:
                rel = EDGE_ENABLES
            else:
                rel = EDGE_CAUSES
            w_inc = min(self.CAUSAL_WEIGHT_CAP, self.CAUSAL_WEIGHT_INC * (abs(float(dv)) / 0.05))
            try:
                self.archivist.link(
                    focus_id, wnode, rel,
                    source="causal_cooccur", placement="act_cooccur",
                    weight=w_inc, confidence=conf,
                )
                made += 1
            except Exception:
                pass
        return made


    # --- body ↔ felt co-occurrence weights (option A) ---
    BODY_FELT_GATE = 0.35       # channel must be up to credit this pulse
    BODY_FELT_INC = 0.08        # increment while co-active
    BODY_FELT_DECAY = 0.995     # per-pulse multiply on all body_felt edges
    BODY_FELT_CAP = 1.0
    BODY_FELT_SHOW = 0.12       # below this = residual, not peer
    BODY_FELT_DROP = 0.02       # remove edge under this after decay

    def _update_body_felt_weights(self, g, basin_id: str = "", stamp=None):
        """Decay all body↔felt co-occur weights; reinforce current basin.

        Edges use associated-with + placement=body_felt_cooccur + float weight.
        Only the *current* basin is incremented this pulse (temporal locality).
        """
        from .edge_types import (
            EDGE_ASSOCIATED_WITH,
            BODY_CHANNELS,
            body_channel_node_id,
            is_body_channel_node,
        )
        from .archivist import SELF_NODE

        # 1) Decay every body↔felt associated-with edge (new + legacy binary)
        to_drop = []
        try:
            for u, v, k, attr in list(g.edges(keys=True, data=True)):
                if not isinstance(attr, dict):
                    continue
                if attr.get("relation_type") != EDGE_ASSOCIATED_WITH:
                    continue
                u_b = str(u).startswith("body:")
                v_b = str(v).startswith("body:")
                u_f = str(u).startswith(("felt:", "basin_"))
                v_f = str(v).startswith(("felt:", "basin_"))
                is_bf = (u_b and v_f) or (u_f and v_b)
                if not is_bf and attr.get("placement") != "body_felt_cooccur":
                    continue
                if not is_bf:
                    continue
                # Migrate legacy edges onto the weighted placement
                if attr.get("placement") in (None, "", "self_felt"):
                    attr["placement"] = "body_felt_cooccur"
                w = float(attr.get("weight") if attr.get("weight") is not None else 0.5)
                w = max(0.0, w * self.BODY_FELT_DECAY)
                attr["weight"] = w
                if w < self.BODY_FELT_DROP:
                    to_drop.append((u, v, k))
            for u, v, k in to_drop:
                try:
                    g.remove_edge(u, v, key=k)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("body↔felt decay failed: %s", e)

        if not basin_id or basin_id not in g:
            return

        # 2) Increment body channels that are currently elevated toward this basin
        for ch in BODY_CHANNELS:
            nid = body_channel_node_id(ch)
            if nid not in g:
                continue
            try:
                val = float(g.nodes[nid].get("body_value") or 0.0)
            except (TypeError, ValueError):
                val = 0.0
            if val < self.BODY_FELT_GATE:
                continue
            # Ensure bidirectional edges exist, then bump weight
            for a, b in ((nid, basin_id), (basin_id, nid)):
                try:
                    self.archivist.link(
                        a, b, EDGE_ASSOCIATED_WITH,
                        source="somatic", placement="body_felt_cooccur",
                        felt_state=stamp,
                    )
                except Exception:
                    pass
                ed = g.get_edge_data(a, b) or {}
                for _k, attr in ed.items():
                    if not isinstance(attr, dict):
                        continue
                    if attr.get("relation_type") != EDGE_ASSOCIATED_WITH:
                        continue
                    # Prefer placement match; fall back to any associated-with
                    if attr.get("placement") and attr.get("placement") != "body_felt_cooccur":
                        continue
                    attr["placement"] = "body_felt_cooccur"
                    prev = float(attr.get("weight") if attr.get("weight") is not None else 0.0)
                    attr["weight"] = min(self.BODY_FELT_CAP, prev + self.BODY_FELT_INC)
                    attr["dwell"] = float(attr.get("dwell") or 0.0) + 1.0
                    attr["last_value"] = val
                    attr["source"] = attr.get("source") or "somatic"
                    break
            if val >= 0.55:
                try:
                    self.archivist.bump_activation(nid)
                    self.archivist.bump_activation(basin_id)
                    self.archivist.record_co_activation([nid, basin_id, SELF_NODE])
                except Exception:
                    pass

    def _sync_self_felt(self):
        """Bind SELF to the current somatic felt place every pulse.

        Language is optional. The primary link is to a discrete PAD basin
        node derived from the synthesizer key (already binned). When a
        basin has stabilized into a named felt state, also link that
        label so later narrative can talk about it. Updates SELF node
        attributes with the live vector so identity is never an empty
        unconnected pin.
        """
        from .archivist import SELF_NODE, NODE_BASIN, TIER_TRUSTED, TIER_WORKING
        from .edge_types import EDGE_ASSOCIATED_WITH

        g = self.archivist.graph
        if SELF_NODE not in g:
            try:
                self.archivist._seed_self_node()
            except Exception:
                return
        if SELF_NODE not in g:
            return

        key = self.synthesizer.get_current_basin_key()
        if not key:
            return
        try:
            arousal = float(key[0])
            valence = float(key[1])
            dominance = float(key[2])
        except (TypeError, ValueError, IndexError):
            return

        felt_name = ""
        try:
            felt_name = (self.synthesizer.get_current_felt_state() or "").strip()
        except Exception:
            felt_name = ""

        nd = g.nodes[SELF_NODE]
        nd["last_felt_key"] = (arousal, valence, dominance)
        nd["felt_arousal"] = arousal
        nd["felt_valence"] = valence
        nd["felt_dominance"] = dominance
        nd["last_felt_label"] = felt_name or "Unformed"
        # Small continuous coloring toward live valence (identity carries mood)
        try:
            self.archivist.nudge_valence_coloring(SELF_NODE, valence * 0.04)
        except Exception:
            pass
        try:
            self.archivist.bump_activation(SELF_NODE)
        except Exception:
            pass

        # Discrete pre-verbal felt place (finite because key is already binned)
        basin_id = f"felt:{arousal:.2f}_{valence:.2f}_{dominance:.2f}"
        if basin_id not in g:
            self.archivist.store(
                basin_id,
                source="somatic",
                tier=TIER_WORKING,
            )
            if basin_id in g:
                g.nodes[basin_id]["node_type"] = NODE_BASIN
                g.nodes[basin_id]["is_felt_place"] = True
                g.nodes[basin_id]["basin_key"] = (arousal, valence, dominance)
        else:
            g.nodes[basin_id]["last_reinforced"] = __import__(
                "datetime"
            ).datetime.now()
            g.nodes[basin_id]["is_felt_place"] = True

        stamp = felt_name if felt_name and felt_name not in ("Unformed", "None") else None
        try:
            self.archivist.link(
                SELF_NODE,
                basin_id,
                EDGE_ASSOCIATED_WITH,
                source="somatic",
                placement="self_felt",
                felt_state=stamp,
            )
        except Exception as e:
            logger.warning("SELF↔basin link failed: %s", e)
            return

        # Reinforce dwell on the edge if present (pre-linguistic habit strength)
        try:
            edata = g.get_edge_data(SELF_NODE, basin_id) or {}
            for _k, attr in edata.items():
                if isinstance(attr, dict) and attr.get("relation_type") == EDGE_ASSOCIATED_WITH:
                    attr["dwell"] = float(attr.get("dwell") or 0.0) + 1.0
                    attr["source"] = attr.get("source") or "somatic"
                    break
        except Exception:
            pass

        # Named felt quality once the body has a stable word for this place
        if stamp:
            label = stamp.lower().replace(" ", "_")
            if label and label not in ("unformed", "none"):
                if label not in g:
                    self.archivist.store(label, source="somatic", tier=TIER_WORKING)
                    if label in g:
                        g.nodes[label]["is_felt_quality"] = True
                try:
                    self.archivist.link(
                        SELF_NODE,
                        label,
                        EDGE_ASSOCIATED_WITH,
                        source="somatic",
                        placement="self_felt_named",
                        felt_state=stamp,
                    )
                except Exception:
                    pass
                try:
                    edata = g.get_edge_data(SELF_NODE, label) or {}
                    for _k, attr in edata.items():
                        if isinstance(attr, dict) and attr.get("relation_type") == EDGE_ASSOCIATED_WITH:
                            attr["dwell"] = float(attr.get("dwell") or 0.0) + 1.0
                            break
                except Exception:
                    pass


        # --- Physical body surface (heart, sweat, …) ---
        # Fixed channels: linkable into epistemic graph as PARTS only,
        # never grown, never is-a. SELF owns the live readings; salient
        # channels can become composed-of parts of active epistemic hubs.
        try:
            from .edge_types import (
                BODY_CHANNELS, body_channel_node_id, EDGE_COMPOSED_OF,
                EDGE_PART_OF, EDGE_ASSOCIATED_WITH, is_body_channel_node,
            )
            body = {}
            if hasattr(self, "bio") and hasattr(self.bio, "get_raw_variables"):
                body = self.bio.get_raw_variables() or {}
            # Ensure seeds exist
            if hasattr(self.archivist, "_seed_body_channels"):
                self.archivist._seed_body_channels()

            salient = []
            for ch in BODY_CHANNELS:
                raw = body.get(ch, body.get(ch.replace("_", ""), 0.5))
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = 0.5
                nid = body_channel_node_id(ch)
                if nid not in g:
                    continue
                g.nodes[nid]["body_value"] = val
                g.nodes[nid]["last_reinforced"] = __import__("datetime").datetime.now()
                # SELF always senses the body (associated-with, not hierarchy)
                try:
                    self.archivist.link(
                        SELF_NODE, nid, EDGE_ASSOCIATED_WITH,
                        source="somatic", placement="self_body",
                        felt_state=stamp,
                    )
                except Exception:
                    pass
                try:
                    edata = g.get_edge_data(SELF_NODE, nid) or {}
                    for _k, attr in edata.items():
                        if isinstance(attr, dict) and attr.get("relation_type") == EDGE_ASSOCIATED_WITH:
                            attr["dwell"] = float(attr.get("dwell") or 0.0) + 1.0
                            attr["last_value"] = val
                            break
                except Exception:
                    pass
                # Salience: extreme or mid-high activation (not quiet baseline)
                if val >= 0.62 or val <= 0.28:
                    salient.append((ch, nid, val))

            # --- Weighted body ↔ felt co-occurrence (option A) ---
            # Binary links made a static hairball. Weight + temporal locality
            # + soft decay so unused body↔basin edges fade and differential
            # soaks can separate signatures. Uses body_value already written above.
            try:
                self._update_body_felt_weights(g, basin_id=basin_id, stamp=stamp)
            except Exception as e:
                logger.warning("body↔felt weight update failed: %s", e)

            # Link salient channels as PARTS of active epistemic schemas
            # anger --composed-of--> body:heart_rate  (part, not child)
            active_schemas = []
            try:
                for n, d in g.nodes(data=True):
                    if not d:
                        continue
                    if d.get("is_schema") or d.get("node_type") in (
                        "schema", "epistemic_schema", "NODE_SCHEMA",
                    ):
                        # only currently relevant
                        act = float(d.get("activation") or 0.0)
                        if act >= 0.15 or d.get("user_linked"):
                            active_schemas.append(n)
                    nt = str(d.get("node_type") or "")
                    if "epistemic" in nt.lower() or nt in ("schema",):
                        if n not in active_schemas:
                            active_schemas.append(n)
            except Exception:
                active_schemas = []
            # Cap: focus neighborhood schemas preferred
            try:
                local = set(self.focus_neighborhood_ids(cap=24) or [])
                active_schemas = [s for s in active_schemas if s in local] or active_schemas[:6]
            except Exception:
                active_schemas = active_schemas[:6]

            for ch, nid, val in salient[:4]:
                for sid in active_schemas[:4]:
                    if is_body_channel_node(sid):
                        continue
                    try:
                        # whole --composed-of--> body part
                        self.archivist.link(
                            sid, nid, EDGE_COMPOSED_OF,
                            source="somatic", placement="body_part_of_schema",
                            felt_state=stamp,
                        )
                        # body --part-of--> whole (symmetric reading)
                        self.archivist.link(
                            nid, sid, EDGE_PART_OF,
                            source="somatic", placement="body_part_of_schema",
                            felt_state=stamp,
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("body-channel self sync failed: %s", e)

        # Agency residue: open goals are part of "what I am doing" (weak)
        try:
            if hasattr(self, "goals") and self.goals is not None:
                open_goals = []
                if hasattr(self.goals, "open_goals"):
                    open_goals = list(self.goals.open_goals() or [])
                elif hasattr(self.goals, "active"):
                    open_goals = list(getattr(self.goals, "active", []) or [])
                for gid in open_goals[:3]:
                    if not gid or gid not in g:
                        continue
                    self.archivist.link(
                        SELF_NODE,
                        gid,
                        EDGE_ASSOCIATED_WITH,
                        source="self",
                        placement="self_agency",
                        felt_state=stamp,
                    )
        except Exception:
            pass

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
        expansion_plan = []  # ordered (kind, term) hyponym → hypernym → …
        for _ in range(self.SELF_STUDY_MAX_ATTEMPTS):
            target = self._select_self_study_target()
            if target is None:
                return
            try:
                expansion_plan = self._ordered_self_study_expansions(target)
                expansions = [t for _kind, t in expansion_plan]
            except Exception as e:
                logger.warning("ordered expansion(%r) failed: %s", target, e)
                expansions = []
                expansion_plan = []
            if expansions:
                break
            if int(getattr(self, "_self_study_phase", {}).get(target, 0)) >= 3:
                self._barren_self_study_targets.add(target)
            else:
                self._self_study_phase[target] = int(self._self_study_phase.get(target, 0)) + 1
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
        # Prefer hyponyms first (kind structure), then hypernyms.
        # Affective focus: at most 1 new dict leaf per tick (hygiene).
        place_cap = 1 if self._focus_is_affective(target) else 3
        for kind, child in (expansion_plan or [("+", c) for c in expansions])[:place_cap]:
            if self._is_noise_dict_leaf(str(child), affective=self._focus_is_affective(target)):
                continue
            definition = self.sensory.lookup_definition(child) or ""
            result = self.association.place_node(
                child, definition=definition, source="dictionary",
                context_node=target, max_parent_children=self.SELF_STUDY_SOFT_CAP,
                force_lookup=True,
            )
            term_id = result.get("term") if isinstance(result, dict) else result
            if term_id:
                placed_children.append(term_id)
                # Hyponyms get is-a toward target when missing (kind schema fuel)
                if kind == "hyponym" and term_id != target:
                    try:
                        self.archivist.link(
                            term_id, target, "is-a",
                            source="dictionary", placement="explicit",
                        )
                    except Exception:
                        pass
        self.archivist.store(target, source="dictionary")  # reinforce parent's last_reinforced

        if not placed_children:
            try:
                hyper = self.sensory.lookup_hypernym(target)
                if hyper and hyper != target:
                    result = self.association.place_node(
                        hyper,
                        definition=self.sensory.lookup_definition(hyper) or "",
                        source="dictionary",
                        context_node=target,
                        max_parent_children=self.SELF_STUDY_SOFT_CAP,
                        force_lookup=True,
                    )
                    term_id = result.get("term") if isinstance(result, dict) else result
                    if term_id:
                        placed_children.append(term_id)
            except Exception as e:
                logger.warning("hypernym fallback failed: %s", e)
        if not placed_children:
            print(f"Self-study: no children placed for target={target!r} plan={expansion_plan[:5]!r}")
            return

        # Co-activation (§13.3): target + newly-placed children this cycle
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

    def _select_self_study_target(self, hard_cap: int = 8):
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
            # Body channels are fixed anatomy: never self-study growth targets
            try:
                from .edge_types import is_body_channel_node
                if is_body_channel_node(n) or d.get("is_body_channel") or d.get("growable") is False:
                    return False
                if d.get("is_narrative_element") or str(n).startswith("narr:"):
                    return False
            except Exception:
                if d.get("is_body_channel") or d.get("growable") is False:
                    return False
                if d.get("is_narrative_element") or str(n).startswith("narr:"):
                    return False
            return (
                self.archivist.categorical_out_degree(n) < hard_cap
                and not d.get("is_schema")
                and n not in (SELF_NODE, OTHER_NODE)
                and n not in self._barren_self_study_targets
            )

        # Curiosity narrowing: prefer focus neighborhood over whole graph
        local = self.focus_neighborhood_ids(cap=48)
        # Prefer members of existing epistemic schemas (deepen kinds)
        try:
            g = self.archivist.graph
            for n, d in g.nodes(data=True):
                if d.get("node_type") in ("epistemic_schema", "schema") or d.get("is_schema"):
                    for _u, v, ed in g.out_edges(n, data=True):
                        if ed.get("relation_type") == "composed-of":
                            local.add(v)
                    local.add(n)
        except Exception:
            pass
        # Working-memory slots always preferred
        try:
            for sid in (wm.get("slots") or []):
                local.add(sid)
        except Exception:
            pass
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

        # Soft preference only (do not hard-restrict to local barren leaves)

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

        # Soft priority (~72%): expand WM / user nodes first (hyponym order
        # runs on the chosen target). Not absolute — still explores rest.
        try:
            wm_now = self.get_current_working_memory()
            wm_slots = set(wm_now.get("slots") or [])
        except Exception:
            wm_slots = set()

        def wm_prefer(cands):
            if not cands:
                return cands
            preferred = [n for n in cands if n in wm_slots]
            userish = [
                n for n in cands
                if n not in preferred
                and self.archivist.graph.nodes.get(n, {}).get("source") == "user"
            ]
            top = preferred + userish
            if top and random.random() < 0.72:
                return top
            return cands

        working_candidates = wm_prefer(working_candidates)
        provisional_candidates = wm_prefer(provisional_candidates)

        if working_candidates and provisional_candidates:
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

    def _self_study_importance_sets(self) -> dict:
        """Collect the sets that define 'what actually matters' for self-study.

        Priority floors (P1): user-linked + active goals + narrative ≫ distant WordNet.
        Built once per selection so weighted choice stays cheap.
        """
        graph = self.archivist.graph
        out = {
            "wm": set(),
            "user": set(),
            "goal": set(),
            "narrative": set(),
            "focus_local": set(),
            "focus_id": None,
        }
        try:
            wm = self.get_current_working_memory()
            out["wm"] = set(wm.get("slots") or [])
        except Exception:
            pass
        try:
            for n, d in list(graph.nodes(data=True)):
                if d.get("source") == "user" or d.get("user_linked") or d.get("pedagogical"):
                    out["user"].add(n)
        except Exception:
            pass
        try:
            if getattr(self, "goals", None) is not None:
                for tid in list(self.goals.active_target_ids() or []):
                    out["goal"].add(tid)
                    if hasattr(self.goals, "schema_closure_ids"):
                        out["goal"] |= set(self.goals.schema_closure_ids(graph, tid) or [])
                    if hasattr(self.goals, "protected_ids"):
                        out["goal"] |= set(self.goals.protected_ids(graph) or [])
        except Exception:
            pass
        try:
            if getattr(self, "self_narrative", None) is not None:
                if hasattr(self.self_narrative, "linked_nodes_above_floor"):
                    out["narrative"] |= set(self.self_narrative.linked_nodes_above_floor() or [])
                # narr:* element nodes and anything they point at
                for n, d in list(graph.nodes(data=True)):
                    if str(n).startswith("narr:") or d.get("is_narrative_element"):
                        out["narrative"].add(n)
                        for linked in (d.get("linked_nodes") or []):
                            out["narrative"].add(linked)
        except Exception:
            pass
        try:
            out["focus_local"] = set(self.focus_neighborhood_ids(cap=48) or [])
            out["focus_id"] = getattr(self.focus, "focus_id", None)
            if out["focus_id"]:
                out["focus_local"].add(out["focus_id"])
        except Exception:
            pass
        return out

    def _self_study_importance(self, node_id: str, sets: dict, bias: str = "BIAS_NEUTRAL") -> float:
        """Scalar importance for self-study target selection.

        Floors (multiplicative, ordered by intent):
          goal / schema-closure   ×12
          narrative-linked        ×8
          user-linked / taught    ×6
          working-memory slot     ×5
          focus neighborhood      ×4
          current focus id        ×2.5
          activation (or explore invert) + epsilon
          long-term curiosity, sticky residual, guided neigh

        Distant pure-dictionary leaves with no tie to the floors stay near
        the epsilon floor so they remain reachable but rarely win.
        """
        graph = self.archivist.graph
        nd = graph.nodes.get(node_id, {}) or {}
        act = float(nd.get("activation", 0.0) or 0.0)
        if bias == "BIAS_EXPLORE":
            cap = float(getattr(self.archivist, "ACTIVATION_CAP", 5.0) or 5.0)
            w = (cap - act) + 0.1
        else:
            w = act + 0.1

        if node_id in sets.get("goal", ()):
            w *= 12.0
        if node_id in sets.get("narrative", ()):
            w *= 8.0
        if node_id in sets.get("user", ()):
            w *= 6.0
        elif nd.get("source") == "user" or nd.get("user_linked") or nd.get("pedagogical"):
            w *= 6.0
        if node_id in sets.get("wm", ()):
            w *= 5.0
        if node_id in sets.get("focus_local", ()):
            w *= 4.0
        if sets.get("focus_id") and node_id == sets["focus_id"]:
            w *= 2.5

        # Pure dictionary leaf with no importance floor: soft demotion;
        # hard-skip weight in Childhood (pedagogy must not roam WordNet).
        # Under affective focus, demote pure dict even harder so emotion
        # soaks deepen user/schema structure instead of liquor synonyms.
        src = nd.get("source", "")
        pure_dict = (
            src == "dictionary"
            and node_id not in sets.get("goal", ())
            and node_id not in sets.get("narrative", ())
            and node_id not in sets.get("user", ())
            and node_id not in sets.get("wm", ())
            and node_id not in sets.get("focus_local", ())
        )
        if pure_dict:
            try:
                epoch = str(getattr(self.bio.epoch, "value", "") or "")
            except Exception:
                epoch = ""
            if epoch == "Childhood":
                w *= 0.02  # hard skip in practice
            else:
                w *= 0.35
            try:
                if self._focus_is_affective(node_id):
                    w *= 0.15
            except Exception:
                pass
            try:
                if self._is_noise_dict_leaf(str(node_id), affective=True):
                    w *= 0.05
            except Exception:
                pass

        try:
            if hasattr(self, "long_term_interest"):
                w *= float(self.long_term_interest.curiosity_multiplier(node_id) or 1.0)
        except Exception:
            pass
        try:
            w *= float(self.focus.self_study_weight(node_id) or 1.0)
        except Exception:
            pass
        try:
            guided = self.focus.neighbourhood_boost_ids(graph)
            if guided and node_id in guided:
                w *= 3.5
        except Exception:
            pass
        return max(0.05, float(w))

    def _weighted_choice_by_activation(self, pool: List[str], bias: str = "BIAS_NEUTRAL",
                                        boost_set: Optional[set] = None) -> Optional[str]:
        """Importance-weighted self-study target choice.

        P1 scoring: user-linked + goal + narrative floor ≫ distant WordNet.
        Activation / explore invert remains the base signal; importance
        floors are multiplicative so a cold but goal-linked node still
        outranks a hot unrelated dictionary leaf.
        """
        if not pool:
            return None
        sets = self._self_study_importance_sets()
        if boost_set:
            # fold legacy boost_set into WM-like preference
            sets["wm"] = set(sets.get("wm") or ()) | set(boost_set)
        weights = [self._self_study_importance(n, sets, bias=bias) for n in pool]
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
        """Continuous sleep pressure.

        Three additive terms while awake:
          1) MIN_PRESSURE_TICK — guaranteed floor every pulse (slider-proof)
          2) work — intensity × growth rate (Debug slider still matters)
          3) baseline — micro-day / circadian ramp

        Sleep is the only place pressure falls hard. This prevents the
        "stuck under 0.3" failure when intensity is low or growth slider
        is near zero.
        """
        try:
            intensity = float(self.synthesizer.get_current_intensity() or 0.0)
        except Exception:
            intensity = 0.3
        if intensity != intensity:  # NaN
            intensity = 0.3

        urgency = 0.0
        try:
            urgency = float(self._compute_urgency() or 0.0)
        except Exception:
            urgency = 0.0
        self._last_urgency = urgency

        growth = max(0.0, float(getattr(self, "FATIGUE_GROWTH_RATE", 0.2) or 0.0))
        if urgency > 0.35:
            growth *= 1.0 + (self.FATIGUE_URGENCY_GROWTH_MULT - 1.0) * urgency
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

        self.micro_day_pulse = (int(self.micro_day_pulse) + 1) % max(8, int(self.MICRO_DAY_PULSES))
        day_frac = self.micro_day_pulse / max(1.0, float(self.MICRO_DAY_PULSES))
        baseline = 0.004 + 0.010 * day_frac  # gentle circadian, not a rush to sleep

        # Guaranteed awake tick so soft threshold is reachable in ~10–20 pulses
        # even if growth slider is 0 and intensity is 0.
        MIN_PRESSURE_TICK = 0.008

        if self.state != "Sleep":
            work = max(0.0, intensity) * growth
            delta = MIN_PRESSURE_TICK + work + baseline
            self.fatigue = min(1.0, float(self.fatigue) + delta)
            self.pulses_since_sleep = int(getattr(self, "pulses_since_sleep", 0)) + 1
        else:
            rec = float(self.FATIGUE_RECOVERY_SLEEP)
            self.fatigue = max(0.0, float(self.fatigue) * rec - 0.01 * (1.0 - min(1.0, self.sleep_debt)))

    def _enter_sleep(self):
        self.state = "Sleep"
        self.sleep_stage = "digest"
        self.pulses_since_sleep = 0
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
            # Prune only when cortex has some Working mass; early graphs
            # need Tier-0 co-activation fuel for schemas.
            try:
                g = self.archivist.graph
                n_working = sum(
                    1 for _, d in g.nodes(data=True)
                    if d.get("tier", 0) >= 1
                )
                n_schema = sum(
                    1 for _, d in g.nodes(data=True)
                    if d.get("node_type") in ("epistemic_schema", "schema")
                    or d.get("is_schema")
                )
                if n_working >= 8 or n_schema >= 1:
                    pruned = self.archivist.prune()
                    if pruned:
                        print(f"Sleep reorganize: pruned {pruned} stale Tier-0 node(s).")
                else:
                    print(
                        f"Sleep reorganize: skip prune "
                        f"(working={n_working}, schemas={n_schema} — protecting fuel)"
                    )
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
                self._note_wake_boundary()
            else:
                # Loop: another digest pass if still heavily indebted
                self.sleep_stage = "digest"

    def _cycle_state(self):
        """Learning (long) → occasional Consolidation → Sleep only when due.

        Schema formation needs many Learning pulses + several Consolidation
        passes. Sleep must not fire every ~15 pulses and prune Tier-0 fuel.
        """
        urgency = self._last_urgency if self._last_urgency else self._compute_urgency()
        try:
            self.SLEEP_SOFT_MIN = float(getattr(self, "T1", self.SLEEP_SOFT_MIN))
        except Exception:
            pass
        try:
            self.SLEEP_HARD_MAX = max(
                float(getattr(self, "SLEEP_SOFT_MIN", 0.55)) + 0.05,
                float(getattr(self, "T2", self.SLEEP_HARD_MAX)),
            )
        except Exception:
            pass
        soft = float(getattr(self, "SLEEP_SOFT_MIN", self.T1))
        hard = float(getattr(self, "SLEEP_HARD_MAX", 0.92))
        min_learn = int(getattr(self, "MIN_LEARNING_PULSES_BETWEEN_SLEEP", 45))
        since = int(getattr(self, "pulses_since_sleep", 0))
        day_ok = since >= min_learn

        if self.state == "Learning":
            if self.fatigue >= hard and day_ok:
                self.state = "Consolidation"
            elif self.fatigue >= soft and day_ok:
                self.state = "Consolidation"
            elif self.fatigue >= hard and not day_ok:
                # Too early for sleep: still consolidate to form schemas, then Learning
                self.state = "Consolidation"
        elif self.state == "Pruning":
            self._enter_sleep()

        if self.state == "Consolidation":
            self._run_consolidation()
            self.fatigue *= float(self.FATIGUE_RECOVERY_CONSOLIDATION)
            # Sleep only after a real Learning stretch; otherwise return to Learning
            if day_ok and self.fatigue >= soft * 0.7:
                self._enter_sleep()
            else:
                # Partial day: keep Learning so co-activation + tiers can mature
                self.fatigue = min(soft * 0.9, max(0.15, self.fatigue))
                self.state = "Learning"
                print(
                    f"Consolidation done → Learning "
                    f"(since_sleep={since}/{min_learn}, pressure={self.fatigue:.3f})"
                )
        elif self.state == "Sleep":
            self._run_sleep_pulse()








    def _kind_hub_ids(self) -> set:
        """Structural kind hubs — no hardcoded ontology.

        A hub is any non-schema node that is:
          - current focus / open parent / active goal family, or
          - user / pedagogical, or
          - already has ≥2 is-a children (emergent kind).
        """
        graph = self.archivist.graph
        hubs = set()

        def is_schema(nid):
            d = graph.nodes.get(nid, {}) or {}
            return d.get("node_type") in ("schema", "epistemic_schema") or bool(d.get("is_schema"))

        try:
            for hid in (self._open_parent_ids() or []):
                if hid in graph and not is_schema(hid):
                    hubs.add(hid)
        except Exception:
            pass
        try:
            fid = getattr(self.focus, "focus_id", None) if self.focus else None
            if fid and fid in graph and not is_schema(fid):
                hubs.add(fid)
                try:
                    hubs |= set(self.archivist.kind_family(fid) or [])
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self.goals:
                for g in (self.goals.open_goals() if hasattr(self.goals, "open_goals") else []):
                    nid = getattr(g, "node_id", None) or (g.get("node_id") if isinstance(g, dict) else None)
                    if nid and nid in graph and not is_schema(nid):
                        hubs.add(nid)
        except Exception:
            pass
        for n, d in list(graph.nodes(data=True)):
            if is_schema(n):
                continue
            if d.get("source") == "user" or d.get("pedagogical") or d.get("user_linked"):
                hubs.add(n)
            # emergent: 2+ inbound is-a
            try:
                isa_in = 0
                for u, _, ed in list(graph.in_edges(n, data=True)):
                    if (ed or {}).get("relation_type") == "is-a":
                        isa_in += 1
                        if isa_in >= 2:
                            hubs.add(n)
                            break
            except Exception:
                pass
        return {h for h in hubs if h in graph and not is_schema(h)}

    def _repair_hierarchy_edges(self) -> dict:
        """Generic hierarchy hygiene — no domain word lists.

        Rules (structural only):
          1. Break is-a 2-cycles; prefer higher-act / higher in-degree endpoint as parent.
          2. Never lemma ↔ schema is-a (membership is composed-of).
          3. Under kind hubs, drop weak dictionary is-a children (not user/pedagogical,
             low activation, dictionary/self_study source) — keep strong/user ones.
          4. Drop composed-of from schemas that are *children* of a hub onto that hub.
        """
        graph = self.archivist.graph
        summary = {
            "isa_flipped": 0, "isa_dropped": 0, "composed_dropped": 0, "noise_pruned": 0,
        }

        def is_schema(nid):
            d = graph.nodes.get(nid, {}) or {}
            return d.get("node_type") in ("schema", "epistemic_schema") or bool(d.get("is_schema"))

        def low(nid):
            d = graph.nodes.get(nid, {}) or {}
            return str(d.get("name") or nid).casefold().strip()

        def act(nid):
            return float((graph.nodes.get(nid) or {}).get("activation") or 0)

        def isa_in_degree(nid):
            c = 0
            try:
                for _, _, ed in list(graph.in_edges(nid, data=True)):
                    if (ed or {}).get("relation_type") == "is-a":
                        c += 1
            except Exception:
                pass
            return c

        def remove_rel(u, v, rel):
            if u not in graph or v not in graph:
                return False
            removed = False
            try:
                data = graph.get_edge_data(u, v)
                if data is None:
                    return False
                if graph.is_multigraph():
                    for key in list(data.keys()):
                        attr = data[key] or {}
                        if attr.get("relation_type") == rel:
                            graph.remove_edge(u, v, key)
                            removed = True
                else:
                    if (data or {}).get("relation_type") == rel:
                        graph.remove_edge(u, v)
                        removed = True
            except Exception:
                return False
            return removed

        hubs = self._kind_hub_ids()

        pairs = []
        try:
            if graph.is_multigraph():
                for u, v, key, ed in list(graph.edges(keys=True, data=True)):
                    if (ed or {}).get("relation_type") == "is-a":
                        pairs.append((u, v))
            else:
                for u, v, ed in list(graph.edges(data=True)):
                    if (ed or {}).get("relation_type") == "is-a":
                        pairs.append((u, v))
        except Exception:
            pairs = []
        pair_set = set(pairs)

        # 2-cycles
        seen = set()
        for u, v in pairs:
            if (u, v) in seen or (v, u) not in pair_set:
                continue
            seen.add((u, v)); seen.add((v, u))
            if is_schema(u) or is_schema(v):
                remove_rel(u, v, "is-a"); remove_rel(v, u, "is-a")
                summary["isa_dropped"] += 2
                continue
            # Prefer parent = higher hub-ness / in-degree / activation
            score_u = (1 if u in hubs else 0) * 10 + isa_in_degree(u) + act(u)
            score_v = (1 if v in hubs else 0) * 10 + isa_in_degree(v) + act(v)
            # keep lower→higher (child → parent)
            if score_v >= score_u:
                remove_rel(v, u, "is-a")  # drop reverse
                summary["isa_dropped"] += 1
            else:
                remove_rel(u, v, "is-a")
                summary["isa_dropped"] += 1

        # One-way cleanups
        for u, v in list(pairs):
            if u not in graph or v not in graph:
                continue
            # lemma is-a schema / schema is-a lemma → drop
            if is_schema(u) ^ is_schema(v):  # xor: one is schema
                if remove_rel(u, v, "is-a"):
                    summary["isa_dropped"] += 1
                continue
            # weak child under hub
            if v in hubs and not is_schema(u):
                d = graph.nodes.get(u, {}) or {}
                strong = (
                    d.get("source") == "user"
                    or d.get("pedagogical")
                    or d.get("user_linked")
                    or act(u) >= 1.0
                    or u in hubs
                )
                weak = d.get("source") in ("dictionary", "schema", "self_study", None, "repair")
                if weak and not strong:
                    if remove_rel(u, v, "is-a"):
                        summary["noise_pruned"] += 1

        # composed-of: schema that is itself an is-a child of hub must not claim hub
        try:
            edge_iter = (
                list(graph.edges(keys=True, data=True))
                if graph.is_multigraph()
                else [(u, v, None, ed) for u, v, ed in graph.edges(data=True)]
            )
            for u, v, key, ed in edge_iter:
                if (ed or {}).get("relation_type") != "composed-of":
                    continue
                if not is_schema(u) or v not in hubs:
                    continue
                # if schema's lemma-name is-a the hub, or schema has is-a to hub, drop
                schema_child_of_hub = False
                try:
                    for _s, p, e2 in list(graph.out_edges(u, data=True)):
                        if e2.get("relation_type") == "is-a" and p == v:
                            schema_child_of_hub = True
                            break
                except Exception:
                    pass
                # also: name of schema looks like child of hub (epistemic_of_X where X is-a hub)
                tail = low(u)
                if tail.startswith("epistemic_of_"):
                    tail = tail[len("epistemic_of_"):].replace("_", " ")
                for n in list(graph.nodes):
                    if low(n) == tail and not is_schema(n):
                        try:
                            for _a, p, e2 in list(graph.out_edges(n, data=True)):
                                if e2.get("relation_type") == "is-a" and p == v:
                                    schema_child_of_hub = True
                                    break
                        except Exception:
                            pass
                if schema_child_of_hub:
                    try:
                        if key is not None:
                            graph.remove_edge(u, v, key)
                        else:
                            graph.remove_edge(u, v)
                        summary["composed_dropped"] += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return summary

    def _grow_kind_schema_membership(self) -> dict:
        """For each structural kind hub, ensure a kind schema owns its is-a children.

        No domain lists: hub set from _kind_hub_ids(); members = current is-a children.
        """
        graph = self.archivist.graph
        summary = {"hubs": 0, "members_linked": 0, "isa_cleared": 0, "schemas": []}

        def is_schema(nid):
            d = graph.nodes.get(nid, {}) or {}
            return d.get("node_type") in ("schema", "epistemic_schema") or bool(d.get("is_schema"))

        def low(nid):
            d = graph.nodes.get(nid, {}) or {}
            return str(d.get("name") or nid).casefold().strip()

        hubs = self._kind_hub_ids()
        if not hubs:
            return summary

        for lemma in list(hubs):
            if lemma not in graph or is_schema(lemma):
                continue
            # find or create epistemic schema for this lemma
            schema_id = None
            prefer = f"epistemic_of_{str(lemma).casefold().replace(' ', '_')}"
            for cand in (prefer, f"epistemic_of_{lemma}"):
                if cand in graph and is_schema(cand):
                    schema_id = cand
                    break
            if schema_id is None:
                for n, d in list(graph.nodes(data=True)):
                    if not is_schema(n):
                        continue
                    if d.get("kind_of") == lemma:
                        schema_id = n
                        break
                    nm = low(n).replace("epistemic_of_", "").replace("_", " ")
                    if nm == low(lemma):
                        schema_id = n
                        break
            if schema_id is None:
                schema_id = prefer
                try:
                    if schema_id not in graph:
                        self.archivist.store(schema_id, source="schema")
                    if schema_id in graph:
                        graph.nodes[schema_id]["node_type"] = "epistemic_schema"
                        graph.nodes[schema_id]["is_schema"] = True
                        graph.nodes[schema_id]["name"] = low(lemma)
                        graph.nodes[schema_id]["kind_of"] = lemma
                except Exception:
                    continue
            summary["hubs"] += 1
            summary["schemas"].append(schema_id)

            # lemma as member
            try:
                has_mem = any(
                    ed.get("relation_type") == "composed-of" and v == lemma
                    for _, v, ed in list(graph.out_edges(schema_id, data=True))
                )
                if not has_mem:
                    self.archivist.link(
                        schema_id, lemma, "composed-of",
                        source="schema", placement="kind_schema",
                    )
                    summary["members_linked"] += 1
            except Exception:
                pass

            # is-a children of lemma → schema members
            children = []
            try:
                for u, _, ed in list(graph.in_edges(lemma, data=True)):
                    if (ed or {}).get("relation_type") == "is-a" and not is_schema(u):
                        children.append(u)
            except Exception:
                pass
            for child in children:
                try:
                    has_c = any(
                        ed.get("relation_type") == "composed-of" and v == child
                        for _, v, ed in list(graph.out_edges(schema_id, data=True))
                    )
                    if not has_c:
                        self.archivist.link(
                            schema_id, child, "composed-of",
                            source="schema", placement="kind_member",
                        )
                        summary["members_linked"] += 1
                except Exception:
                    pass

            # clear schema ↔ lemma is-a
            try:
                for u, v, ed in list(graph.out_edges(schema_id, data=True)):
                    if ed.get("relation_type") != "is-a":
                        continue
                    data = graph.get_edge_data(schema_id, v)
                    if graph.is_multigraph() and data:
                        for key in list(data.keys()):
                            if (data[key] or {}).get("relation_type") == "is-a":
                                graph.remove_edge(schema_id, v, key)
                                summary["isa_cleared"] += 1
                    elif data:
                        graph.remove_edge(schema_id, v)
                        summary["isa_cleared"] += 1
                for u, v, ed in list(graph.in_edges(schema_id, data=True)):
                    if ed.get("relation_type") != "is-a":
                        continue
                    data = graph.get_edge_data(u, schema_id)
                    if graph.is_multigraph() and data:
                        for key in list(data.keys()):
                            if (data[key] or {}).get("relation_type") == "is-a":
                                graph.remove_edge(u, schema_id, key)
                                summary["isa_cleared"] += 1
                    elif data:
                        graph.remove_edge(u, schema_id)
                        summary["isa_cleared"] += 1
            except Exception:
                pass
            if schema_id in graph:
                graph.nodes[schema_id]["kind_of"] = lemma
                graph.nodes[schema_id]["is_schema"] = True
        return summary

    def _promote_kind_associations(self) -> int:
        """Under open/user kind hubs (Color), turn short associated-with
        neighbors into is-a children so kind-schemas can form.

        Fixes the failure mode where blue/yellow stay associated-with Color
        forever and never join epistemic_of_Color.
        """
        graph = self.archivist.graph
        hubs = set()
        # Active focus/goal family
        try:
            hubs |= set(self._open_parent_ids() or [])
        except Exception:
            pass
        # Explicit color-like / pedagogical hubs
        for n, d in list(graph.nodes(data=True)):
            if d.get("node_type") in ("schema", "epistemic_schema"):
                continue
            if d.get("source") == "user" or d.get("pedagogical"):
                hubs.add(n)
        # Normalize via kind_family
        expanded = set()
        for h in list(hubs):
            try:
                expanded |= set(self.archivist.kind_family(h) or [h])
            except Exception:
                expanded.add(h)
        hubs = {h for h in expanded if h in graph and graph.nodes[h].get("node_type") not in ("schema", "epistemic_schema")}
        promoted = 0
        skip_words = {
            "property", "attribute", "thing", "entity", "object", "stuff",
            "something", "anything", "nothing",
        }
        for hub in hubs:
            # associated-with either direction
            nbrs = []
            try:
                for _u, v, ed in list(graph.out_edges(hub, data=True)):
                    if ed.get("relation_type") in ("associated-with", "related-to"):
                        nbrs.append(v)
                for u, _v, ed in list(graph.in_edges(hub, data=True)):
                    if ed.get("relation_type") in ("associated-with", "related-to"):
                        nbrs.append(u)
            except Exception:
                continue
            for child in nbrs:
                if child not in graph or child == hub:
                    continue
                cd = graph.nodes.get(child, {})
                if cd.get("node_type") in ("schema", "epistemic_schema"):
                    continue
                lab = str(cd.get("name") or child).strip()
                low = lab.casefold()
                if low in skip_words or low == str(hub).casefold():
                    continue
                # short lexical children only (not gloss phrases)
                if len(lab.split()) > 2 or len(lab) > 24:
                    continue
                # already is-a to hub?
                already = False
                try:
                    for _u, p, ed in list(graph.out_edges(child, data=True)):
                        if ed.get("relation_type") == "is-a" and p == hub:
                            already = True
                            break
                except Exception:
                    pass
                if already:
                    continue
                try:
                    self.archivist.link(
                        child, hub, "is-a",
                        source="promote_kind", placement="assoc_to_isa",
                    )
                    promoted += 1
                except Exception:
                    pass
            # Ensure epistemic_of_{hub} composes children (never body/felt hubs)
            try:
                from .edge_types import is_forbidden_epistemic_parent as _forbid_ep
            except Exception:
                def _forbid_ep(h):
                    return str(h).startswith(("body:", "felt:", "basin_", "narr:", "epistemic_of_body", "epistemic_of_felt"))
            if _forbid_ep(hub):
                continue
            kind_id = f"epistemic_of_{hub}" if hub in graph else None
            for kid in (f"epistemic_of_{hub}", f"epistemic_of_{str(hub).casefold()}"):
                if kid in graph:
                    kind_id = kid
                    break
            if kind_id and _forbid_ep(kind_id):
                kind_id = None
            if kind_id and kind_id in graph:
                for child in nbrs:
                    if child not in graph:
                        continue
                    lab = str(graph.nodes.get(child, {}).get("name") or child)
                    if len(lab.split()) > 2:
                        continue
                    try:
                        has = any(
                            v == child and ed.get("relation_type") == "composed-of"
                            for _u, v, ed in graph.out_edges(kind_id, data=True)
                        )
                        if not has and graph.nodes.get(child, {}).get("node_type") not in ("schema", "epistemic_schema"):
                            graph.add_edge(
                                kind_id, child, relation_type="composed-of",
                                source="promote_kind",
                            )
                    except Exception:
                        pass
        return promoted

    def _expand_under_parent(self, parent: str, max_new: int = 3) -> int:
        """Directed lookup under a known parent (Color), not random self-study.

        Ignores the usual self-study target picker so EXPAND actually grows
        the focused family. Soft-cap per parent still applies.
        """
        if not parent:
            return 0
        # Prefer merged lemma if case twin exists
        try:
            self.archivist.merge_case_variant_lemmas()
        except Exception:
            pass
        if parent not in self.archivist.graph:
            # try casefold match
            low = str(parent).casefold()
            for n in list(self.archivist.graph.nodes):
                if str(n).casefold() == low:
                    parent = n
                    break
        if parent not in self.archivist.graph:
            return 0
        try:
            self._promote_kind_associations()
        except Exception:
            pass
        try:
            self._repair_hierarchy_edges()
        except Exception:
            pass
        try:
            self._grow_kind_schema_membership()
        except Exception:
            pass
        # Un-barren the parent for this directed expand
        try:
            self._barren_self_study_targets.discard(parent)
        except Exception:
            pass
        placed = 0
        try:
            plan = self._ordered_self_study_expansions(parent)
        except Exception as e:
            logger.warning("expand plan failed for %r: %s", parent, e)
            plan = []
        if not plan:
            # Fallback: sensory hyponyms directly
            try:
                hypos = []
                if hasattr(self.sensory, "lookup_hyponyms"):
                    hypos = list(self.sensory.lookup_hyponyms(parent) or [])[:6]
                elif hasattr(self.sensory, "hyponyms_for"):
                    hypos = list(self.sensory.hyponyms_for(parent) or [])[:6]
                plan = [("hyponym", h) for h in hypos if h]
            except Exception:
                plan = []
        # Structural EXPAND filter under kind hubs: drop obvious noise sources,
        # prefer short lemmas / already-in-graph / user-linked — no domain lists.
        hubs = set()
        try:
            hubs = self._kind_hub_ids()
        except Exception:
            pass
        if parent in hubs or str(parent).casefold() in {str(h).casefold() for h in hubs}:
            filtered, seen = [], set()
            for kind, child in plan:
                c = str(child).casefold().strip()
                if not c or c in seen or c == str(parent).casefold():
                    continue
                # skip multiword dictionary sludge and very long names
                if len(c) > 24 or c.count(" ") >= 2:
                    continue
                if c.startswith("epistemic_"):
                    continue
                seen.add(c)
                filtered.append((kind, child))
            # prefer children already related in graph
            related = []
            rest = []
            for kind, child in filtered:
                ch = child if child in self.archivist.graph else None
                if ch is None:
                    for n in self.archivist.graph.nodes:
                        if str(n).casefold() == str(child).casefold():
                            ch = n
                            break
                if ch is not None and (self.archivist.graph.has_edge(ch, parent) or self.archivist.graph.has_edge(parent, ch)):
                    related.append((kind, child))
                else:
                    rest.append((kind, child))
            plan = (related + rest)[:12]

        soft = int(getattr(self, "SELF_STUDY_SOFT_CAP", 8) or 8)
        for kind, child in plan[: max_new + 2]:
            if placed >= max_new:
                break
            if not child or child == parent:
                continue
            if child in self.archivist.graph:
                # Reinforce is-a toward parent if missing
                if kind == "hyponym":
                    try:
                        self.archivist.link(
                            child, parent, "is-a",
                            source="dictionary", placement="expand_under",
                        )
                    except Exception:
                        pass
                continue
            try:
                definition = ""
                try:
                    definition = self.sensory.lookup_definition(child) or ""
                except Exception:
                    pass
                result = self.association.place_node(
                    child,
                    definition=definition,
                    source="dictionary",
                    context_node=parent,
                    max_parent_children=soft,
                    force_lookup=True,
                )
                term_id = result.get("term") if isinstance(result, dict) else result
                if not term_id:
                    continue
                placed += 1
                if kind == "hyponym" and term_id != parent:
                    try:
                        self.archivist.link(
                            term_id, parent, "is-a",
                            source="dictionary", placement="expand_under",
                        )
                    except Exception:
                        pass
                try:
                    self.archivist.bump_activation(term_id, getattr(self, "ACTIVATION_BOOST_SELF_STUDY", 0.4))
                except Exception:
                    pass
            except Exception as e:
                logger.debug("expand place %r failed: %s", child, e)
        if placed:
            try:
                self.archivist.bump_activation(parent, getattr(self, "ACTIVATION_BOOST_SELF_STUDY", 0.4))
                self._record_co_activation_gated(
                    [parent] + [
                        c for _k, c in plan[:placed]
                        if c in self.archivist.graph
                    ]
                )
            except Exception:
                pass
        return placed

    def _run_cognition_operators(self) -> dict:
        """Prediction check + choose/apply one internal operator."""
        if getattr(self, "operators", None) is None:
            return {}
        graph = self.archivist.graph
        fid = self.focus.focus_id if self.focus else None
        goals = []
        goal_strength = 1.0
        try:
            if self.goals is not None:
                goals = list(self.goals.active_target_ids() or [])
                if goals and self.goals.active:
                    gobj = self.goals.active.get(self.goals._gid(goals[0]))
                    if gobj:
                        goal_strength = float(gobj.strength or 1.0)
        except Exception:
            pass
        wm_slots = []
        try:
            wm_slots = list(self.get_current_working_memory().get("slots") or [])
        except Exception:
            pass
        residual_top = []
        try:
            items = sorted(
                (self.focus.residuals or {}).items(),
                key=lambda kv: -float(kv[1] or 0),
            )[:6]
            residual_top = [k for k, _ in items]
        except Exception:
            pass
        bias = str(getattr(self.executive, "current_bias", "") or "")
        fatigue = float(getattr(self, "fatigue", 0) or 0)
        stagnation = False
        try:
            fs = self.last_focus_summary or {}
            stagnation = bool(fs.get("stagnation_escape") or fs.get("force_switch"))
        except Exception:
            pass
        open_ids = set()
        try:
            open_ids = set(self._open_parent_ids() or [])
        except Exception:
            pass
        parent_open = bool(open_ids) or bool(fid)
        lookup_ok = True
        try:
            lookup_ok = int(getattr(self, "_dict_lookups_this_pulse", 0) or 0) < int(
                getattr(self, "DICT_LOOKUPS_PER_PULSE", 6) or 6
            )
        except Exception:
            pass

        body_for_pred = {}
        try:
            body_for_pred = dict(self.bio.get_raw_variables() or {})
        except Exception:
            body_for_pred = {}
        barren_focus = False
        try:
            if fid and fid in getattr(self, "_barren_self_study_targets", set()):
                barren_focus = True
            # Escape: WM / goal / pin override barren
            if barren_focus:
                if fid in wm_slots or fid in goals:
                    barren_focus = False
                pin = None
                try:
                    pin = getattr(self.focus, "focus_id", None)
                except Exception:
                    pass
                # user-linked nodes may retry
                nd = graph.nodes.get(fid, {}) or {}
                if nd.get("source") == "user" or nd.get("user_linked") or nd.get("pedagogical"):
                    barren_focus = False
        except Exception:
            barren_focus = False
        thread_intent = ""
        try:
            if getattr(self, "active_thread", None) is not None:
                thread_intent = str(self.active_thread.thread.intent or "")
        except Exception:
            pass
        _pain = 0.0
        _pleasure = 0.0
        _escape = ""
        try:
            if getattr(self, "allostasis", None) is not None:
                _pain = float(self.allostasis.state.pain or 0)
                _pleasure = float(self.allostasis.state.pleasure or 0)
                _escape = str(self.allostasis.state.last_escape or "")
                body_for_pred["pain"] = _pain
                body_for_pred["pleasure"] = _pleasure
        except Exception:
            pass
        _plan_op = ""
        try:
            if getattr(self, "planner", None) is not None:
                _plan_op = str(self.planner.suggested_operator() or "")
        except Exception:
            _plan_op = ""
        dec = self.operators.choose(
            graph=graph,
            focus_id=fid,
            goal_targets=goals,
            wm_slots=wm_slots,
            residual_top=residual_top,
            bias=bias,
            fatigue=fatigue,
            stagnation=stagnation,
            lookup_budget_ok=lookup_ok,
            parent_open=parent_open,
            goal_strength=goal_strength,
            body=body_for_pred,
            thread_intent=thread_intent,
            barren_focus=barren_focus,
            pain=_pain,
            pleasure=_pleasure,
            allostatic_escape=_escape,
            plan_suggested_op=_plan_op,
        )
        # Package A: apply act → body (queued) + world stub (immediate)
        try:
            op_name = str(dec.operator or "HOLD")
            # snapshot for next-pulse confidence
            self._pre_act_body_snap = dict(body_for_pred or {})
            if getattr(self, "world_stub", None) is not None:
                self._pre_act_world_snap = self.world_stub.observe()
            # body effect lands next sense
            body_d = self.operators.queue_body_delta(op_name)
            # world effect now
            world_d = {}
            if getattr(self, "world_stub", None) is not None:
                before_w = dict(self._pre_act_world_snap)
                self.world_stub.apply_operator(op_name, pulse=self.pulse_count)
                after_w = self.world_stub.observe()
                world_d = {
                    k: float(after_w.get(k, 0)) - float(before_w.get(k, 0))
                    for k in after_w
                    if abs(float(after_w.get(k, 0)) - float(before_w.get(k, 0))) > 1e-9
                }
                self.operators._last_act_world_deltas = world_d
            # sync body_value on graph for elevated channels (visibility)
            try:
                from .edge_types import body_channel_node_id
                g = self.archivist.graph
                for ch, dv in (body_d or {}).items():
                    nid = body_channel_node_id(ch)
                    if nid in g:
                        cur = float(g.nodes[nid].get("body_value") or 0.5)
                        g.nodes[nid]["body_value"] = max(0.0, min(1.0, cur + float(dv)))
            except Exception:
                pass
            # Package B: causal co-occurrence (focus + act → changed targets)
            try:
                self._record_causal_cooccurrence(
                    focus_id=fid,
                    op_name=op_name,
                    body_deltas=body_d or {},
                    world_deltas=world_d or {},
                )
            except Exception as e:
                logger.debug("causal cooccur: %s", e)
        except Exception as e:
            logger.debug("act apply failed: %s", e)
        # Refresh Active Thread from this decision's prediction
        try:
            if getattr(self, "active_thread", None) is not None:
                pred = getattr(dec, "predict", None)
                body_err = {}
                body_exp = {}
                if pred is not None:
                    body_exp = dict(getattr(pred, "expected_body", None) or {})
                    obs = dict(getattr(pred, "observed_body", None) or {})
                    # signed error per channel when both present
                    for ch, exp_v in body_exp.items():
                        if ch in obs:
                            body_err[ch] = float(obs[ch]) - float(exp_v)
                    if not body_err and getattr(pred, "signed_body_error", None) is not None:
                        body_err["_mean"] = float(pred.signed_body_error)
                if getattr(self, "allostasis", None) is not None:
                    body_exp = dict(body_exp)
                    body_exp["_pain_level"] = float(self.allostasis.state.pain or 0)
                    # Merge allostatic channel errors when present
                    for ch, ev in (self.allostasis.state.error or {}).items():
                        if ch not in body_err:
                            body_err[ch] = float(ev)
                self.active_thread.update(
                    pulse=self.pulse_count,
                    focus_id=fid,
                    goal_ids=goals,
                    body_expect=body_exp,
                    body_error=body_err,
                    last_op=str(dec.operator or ""),
                    barren_focus=barren_focus,
                    bias=bias,
                    lookup_budget_ok=lookup_ok,
                )
        except Exception as e:
            logger.debug("active_thread update failed: %s", e)
        # Process attributes on SELF (rate-limited streaks)
        try:
            self._maybe_attach_process_tag(str(dec.operator or ""))
        except Exception as e:
            logger.debug("process tag failed: %s", e)
        detail = ""
        op = dec.operator

        def _lemma_of(nid):
            if not nid:
                return None
            try:
                fam = list(self.archivist.kind_family(nid) or [nid])
            except Exception:
                fam = [nid]
            for n in fam:
                nd = graph.nodes.get(n, {}) or {}
                if nd.get("node_type") not in ("schema", "epistemic_schema") and not nd.get("is_schema"):
                    return n
            # strip epistemic_of_
            s = str(nid)
            if s.startswith("epistemic_of_"):
                tail = s[len("epistemic_of_"):].replace("_", " ")
                for n in graph.nodes:
                    if str(n).casefold() == tail.casefold():
                        nd = graph.nodes.get(n, {}) or {}
                        if nd.get("node_type") not in ("schema", "epistemic_schema"):
                            return n
            return nid

        parent = _lemma_of(fid) or _lemma_of(goals[0] if goals else None) or fid

        if op == "RETURN" and goals:
            target = _lemma_of(goals[0]) or goals[0]
            try:
                # Mild pull — avoid fighting focus every pulse
                self.focus.boost_residual(target, amount=0.55)
                for mid in list(self.archivist.kind_family(target) or [])[:8]:
                    self.focus.boost_residual(mid, amount=0.15)
                detail = "return->" + str(target)
            except Exception as e:
                detail = "return_failed"
            try:
                self.self_narrative.stream_interrupt(
                    "return", target_id=target, detail=detail, pulse=self.pulse_count,
                )
            except Exception:
                pass

        elif op == "EXPAND":
            # Prefer expanding the goal hub when focus is a side-schema
            expand_target = parent
            if goals:
                g_lemma = _lemma_of(goals[0])
                if g_lemma:
                    expand_target = g_lemma
            if expand_target:
                try:
                    self.focus.boost_residual(expand_target, amount=0.5)
                    detail = "expand_under=" + str(expand_target)
                    if self.state == "Learning":
                        # Skip if recent expands under same target placed nothing
                        barren_streak = 0
                        try:
                            for e in reversed(self.operators.episodes[-5:]):
                                if e.get("operator") != "EXPAND":
                                    break
                                d = str(e.get("detail") or "")
                                if "placed=0" in d or "nodes+0" in d:
                                    barren_streak += 1
                                else:
                                    break
                        except Exception:
                            pass
                        if barren_streak >= 2:
                            detail += " skip_barren"
                            # convert to HOLD credit
                            try:
                                self.focus.boost_residual(expand_target, amount=0.2)
                            except Exception:
                                pass
                        else:
                            before = self.archivist.graph.number_of_nodes()
                            placed = self._expand_under_parent(expand_target)
                            after = self.archivist.graph.number_of_nodes()
                            detail += " placed=" + str(placed) + " nodes+" + str(after - before)
                            try:
                                if self.goals is not None and placed:
                                    self.goals.note_growth(
                                        expand_target, placed=placed, graph=self.archivist.graph,
                                    )
                            except Exception:
                                pass
                except Exception as e:
                    detail = "expand_failed:" + str(e)[:60]
            try:
                self.self_narrative.stream_interrupt(
                    "expand", target_id=expand_target or parent or "", detail=detail, pulse=self.pulse_count,
                )
            except Exception:
                pass

        elif op == "RELEASE":
            try:
                if fid:
                    self.focus.residuals[fid] = float(self.focus.residuals.get(fid, 0) or 0) * 0.3
                detail = "release_focus=" + str(fid)
            except Exception:
                pass
            try:
                self.self_narrative.stream_interrupt(
                    "release", target_id=fid or "", detail=detail, pulse=self.pulse_count,
                )
            except Exception:
                pass

        elif op == "SETTLE":
            try:
                if hasattr(self.executive, "current_bias"):
                    self.executive.current_bias = "BIAS_STABILIZE"
                detail = "bias=STABILIZE"
            except Exception:
                detail = "settle"
            try:
                self.self_narrative.stream_interrupt(
                    "settle", target_id=fid or "", detail=detail, pulse=self.pulse_count,
                )
            except Exception:
                pass

        else:
            try:
                if fid:
                    self.focus.boost_residual(fid, amount=0.15)
                detail = "hold=" + str(fid)
            except Exception:
                pass

        if dec.predict and not dec.predict.match:
            try:
                self.self_narrative.stream_interrupt(
                    "predict_violate",
                    target_id=dec.predict.expected_family,
                    detail=dec.predict.reason,
                    pulse=self.pulse_count,
                )
            except Exception:
                pass
        # Felt motion from prediction: family and/or body-part mismatch
        try:
            pred = dec.predict
            if pred is not None and hasattr(self, "modulators"):
                fam_bad = (not pred.match) or bool(getattr(pred, "off_family_focus", False))
                berr = float(getattr(pred, "body_error", 0.0) or 0.0)
                body_bad = not bool(getattr(pred, "body_match", True))
                if fam_bad or body_bad or berr > 0.12:
                    self.modulators.apply_prediction_error(
                        family_mismatch=fam_bad, body_error=berr if body_bad else 0.0
                    )
                # Mild medium-layer nudge (opaque hormones → body next ticks)
                if hasattr(self, "bio") and hasattr(self.bio, "_hormones"):
                    h = self.bio._hormones
                    if fam_bad:
                        h["adrenaline"] = min(1.0, float(h.get("adrenaline", 0.5)) + 0.035)
                        h["cortisol"] = min(1.0, float(h.get("cortisol", 0.5)) + 0.025)
                        h["serotonin"] = max(0.0, float(h.get("serotonin", 0.5)) - 0.02)
                    if body_bad and berr > 0.12:
                        h["adrenaline"] = min(1.0, float(h.get("adrenaline", 0.5)) + min(0.05, berr * 0.12))
                        h["cortisol"] = min(1.0, float(h.get("cortisol", 0.5)) + min(0.04, berr * 0.08))
                    if pred.match and getattr(pred, "overall_match", True):
                        h["serotonin"] = min(1.0, float(h.get("serotonin", 0.5)) + 0.015)
                        h["adrenaline"] = max(0.0, float(h.get("adrenaline", 0.5)) - 0.01)
        except Exception as e:
            logger.warning("predict→felt coupling failed: %s", e)

        # Operator → felt texture (speech/live-head will add more later)
        try:
            if hasattr(self, "modulators"):
                if op == "RETURN":
                    self.modulators.pulse("novelty", amount=0.07)
                elif op == "EXPAND":
                    self.modulators.pulse("self_study_hit", amount=0.06)
                elif op == "RELEASE":
                    self.modulators.pulse("focus_stagnant", amount=0.05)
                elif op == "SETTLE":
                    self.modulators.pulse("sleep_enter", amount=0.04)
        except Exception:
            pass

        self.operators.record_episode(
            pulse=self.pulse_count,
            decision=dec,
            focus_id=fid,
            goal=goals[0] if goals else None,
            detail=detail,
        )
        return {
            "operator": op,
            "note": dec.note,
            "detail": detail,
            "predict_match": dec.predict.match if dec.predict else None,
        }

    def get_operators_report(self) -> dict:
        if getattr(self, "operators", None) is None:
            return {}
        return self.operators.report()


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

    def get_stream_report(self, last_n: int = 12) -> dict:
        """Pulse-time stream of consciousness (rolling)."""
        return self.self_narrative.stream_report(last_n=last_n)

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
        """Parents/children/parts for search expand — list only, not full graph render.

        Taxonomy convention (must stay consistent with edge_types + archivist.link):
          is-a:        child → parent          (hyponym → hypernym)
          part-of:     part  → whole
          composed-of: whole → part

        Body / felt / narr are anatomy/identity infrastructure:
          they may only appear under part-of / composed-of / associated-with,
          never under is-a.
        """
        graph = self.archivist.graph
        empty = {
            "id": node_id,
            "parents": [],
            "children": [],
            "part_of": [],
            "has_parts": [],
            "member_of_schemas": [],
            "schema_members": [],
            "related": [],
            "parent_count": 0,
            "child_count": 0,
            "related_count": 0,
        }
        if not node_id or node_id not in graph:
            return empty

        try:
            from .edge_types import (
                is_body_channel_node,
                is_felt_place_node,
                is_narrative_graph_node,
                is_somatic_infrastructure,
            )
        except Exception:
            is_body_channel_node = lambda n: str(n).startswith("body:")
            is_felt_place_node = lambda n: str(n).startswith(("felt:", "basin_"))
            is_narrative_graph_node = lambda n: str(n).startswith("narr:")
            is_somatic_infrastructure = lambda n: (
                is_body_channel_node(n)
                or is_felt_place_node(n)
                or is_narrative_graph_node(n)
            )

        parents, children = [], []
        part_of, has_parts = [], []
        member_of_schemas, schema_members = [], []
        related = []
        show_floor = float(getattr(self, "BODY_FELT_SHOW", 0.12) or 0.12)

        def _row(nid, rel, data=None):
            data = data or {}
            w = data.get("weight")
            try:
                w = float(w) if w is not None else None
            except (TypeError, ValueError):
                w = None
            return {
                "id": str(nid),
                "relation": rel,
                "name": (graph.nodes.get(nid) or {}).get("name"),
                "weight": w,
                "dwell": data.get("dwell"),
                "placement": data.get("placement"),
            }

        def _keep_related(row, data):
            """Drop residual body↔felt edges below show floor so UI isn't egalitarian."""
            if (data or {}).get("placement") != "body_felt_cooccur":
                return True
            w = row.get("weight")
            if w is None:
                return True
            return float(w) >= show_floor

        # Out-edges: this → other
        for _, v, data in list(graph.out_edges(node_id, data=True)):
            rel = (data or {}).get("relation_type") or "associated-with"
            row = _row(v, rel, data)
            if rel == "is-a":
                if not is_somatic_infrastructure(node_id) and not is_somatic_infrastructure(v):
                    parents.append(row)
                elif _keep_related(row, data):
                    related.append(row)
            elif rel == "part-of":
                part_of.append(row)
            elif rel == "composed-of":
                has_parts.append(row)
                schema_members.append(row)
            else:
                if _keep_related(row, data):
                    related.append(row)

        # In-edges: other → this
        for u, _, data in list(graph.in_edges(node_id, data=True)):
            rel = (data or {}).get("relation_type") or "associated-with"
            row = _row(u, rel, data)
            if rel == "is-a":
                if not is_somatic_infrastructure(node_id) and not is_somatic_infrastructure(u):
                    children.append(row)
                elif _keep_related(row, data):
                    related.append(row)
            elif rel == "part-of":
                has_parts.append(row)
            elif rel == "composed-of":
                member_of_schemas.append(row)
                part_of.append(row)
            else:
                if _keep_related(row, data):
                    related.append(row)

        # Dedupe related by id (bidirectional edges otherwise list twice)
        seen_rel = {}
        for r in related:
            rid = r.get("id")
            if rid not in seen_rel:
                seen_rel[rid] = r
            else:
                # keep higher weight if present
                prev = seen_rel[rid]
                pw = prev.get("weight")
                nw = r.get("weight")
                try:
                    if nw is not None and (pw is None or float(nw) > float(pw)):
                        seen_rel[rid] = r
                except (TypeError, ValueError):
                    pass
        related = list(seen_rel.values())

        # Prefer stronger co-occurrence when weights present
        def _rel_key(r):
            w = r.get("weight")
            return -(float(w) if w is not None else 0.0)

        related.sort(key=_rel_key)

        return {
            "id": node_id,
            "parents": parents[:max_each],
            "children": children[:max_each],
            "part_of": part_of[:max_each],
            "has_parts": has_parts[:max_each],
            "member_of_schemas": member_of_schemas[:max_each],
            "schema_members": schema_members[:max_each],
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

    def get_goals_report(self) -> dict:
        if getattr(self, "goals", None) is not None:
            return self.goals.report()
        return {}

    def get_active_thread_report(self) -> dict:
        if getattr(self, "active_thread", None) is not None:
            return self.active_thread.report()
        return {}

    def get_allostasis_report(self) -> dict:
        if getattr(self, "allostasis", None) is not None:
            return self.allostasis.report()
        return {}

    def get_world_report(self) -> dict:
        """Package A world stub slots + last act deltas."""
        if getattr(self, "world_stub", None) is not None:
            return self.world_stub.report()
        return {}

    def get_act_report(self) -> dict:
        """Package A+thin C: last act body/world deltas + confidence top."""
        if getattr(self, "operators", None) is None:
            return {}
        rep = self.operators.report()
        return {
            "last_op": rep.get("last_act_op") or rep.get("last_operator"),
            "body_deltas": rep.get("last_act_body_deltas") or {},
            "world_deltas": rep.get("last_act_world_deltas") or {},
            "pending_body": rep.get("pending_body_delta") or {},
            "confidence_top": rep.get("outcome_confidence_top") or [],
        }

    def get_plan_report(self) -> dict:
        """Package D: compositional plan over causal traces."""
        if getattr(self, "planner", None) is not None:
            return self.planner.report()
        return {"active": False}

    def get_module_health_report(self) -> dict:
        """Which optional modules imported cleanly (Identity & Hygiene)."""
        names = (
            "modulators", "felt_anchors", "schema_felt", "executive",
            "stimulus", "goals", "operators", "active_thread", "self_narrative",
            "somatic_topo", "long_term_interest", "world_stub", "allostasis",
        )
        out = {}
        for n in names:
            out[n] = getattr(self, n, None) is not None
        return out

    def get_consolidation_report(self) -> dict:
        return dict(getattr(self, "_last_consolidation_report", {}) or {})

    def run_soak_checks(self) -> dict:
        try:
            from .soak_checks import run_soak_checks
            return run_soak_checks(self)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _hub_stream_line(self) -> Optional[str]:
        """At most one template clause from SELF hub (body / goal / process)."""
        try:
            thr = self.active_thread.thread if getattr(self, "active_thread", None) else None
            if thr and thr.goal_ids:
                g0 = thr.goal_ids[0]
                return f"what matters stays with {g0}"
            tag = getattr(self, "_last_process_tag", None)
            if tag:
                nice = str(tag).replace("process_", "").replace("_", " ")
                return f"I notice I {nice}"
            # Elevated body channel if present on report
            rep = self.get_identity_hub_report()
            body = rep.get("body") or []
            best = None
            best_v = 0.0
            for b in body:
                v = float(b.get("value") or 0)
                if v > best_v:
                    best_v = v
                    best = b.get("node")
            if best and best_v >= 0.65:
                return f"the body leans high on {best}"
        except Exception:
            pass
        return None

    def _maybe_attach_process_tag(self, operator: str) -> None:
        """Rate-limited SELF —associated-with→ process tags after streaks."""
        from .archivist import SELF_NODE
        op = (operator or "").upper()
        if op not in ("HOLD", "EXPAND", "SETTLE", "RELEASE"):
            return
        ring = []
        try:
            ring = list(getattr(self.operators, "_op_ring", []) or [])[-12:]
        except Exception:
            return
        if not ring:
            return
        streak = 0
        for o in reversed(ring):
            if str(o).upper() == op:
                streak += 1
            else:
                break
        need = {"HOLD": 5, "EXPAND": 2, "SETTLE": 2, "RELEASE": 2}.get(op, 99)
        if streak < need:
            return
        # Rate limit: one tag per op family per 40 pulses
        last = getattr(self, "_last_process_tag_pulse", {}) or {}
        if self.pulse_count - int(last.get(op, -999)) < 40:
            return
        tag_id = {
            "HOLD": "process_held_focus",
            "EXPAND": "process_explored",
            "SETTLE": "process_settled",
            "RELEASE": "process_released",
        }.get(op)
        if not tag_id:
            return
        g = self.archivist.graph
        if tag_id not in g:
            g.add_node(
                tag_id,
                source="system",
                is_process_tag=True,
                process_op=op,
                growable=False,
                activation=0.3,
            )
        else:
            g.nodes[tag_id]["is_process_tag"] = True
            g.nodes[tag_id]["process_op"] = op
        try:
            self.archivist.link(
                SELF_NODE, tag_id, "associated-with",
                source="system", placement="explicit",
            )
        except Exception:
            pass
        last[op] = self.pulse_count
        self._last_process_tag_pulse = last
        self._last_process_tag = tag_id

    def get_others_report(self) -> dict:

        if hasattr(self, "others"):
            return self.others.report()
        return {}

    def get_hierarchy_report(self) -> dict:
        if hasattr(self, "last_hierarchy_summary") and self.last_hierarchy_summary:
            return self.last_hierarchy_summary
        if hasattr(self.reflector, "hierarchy_report"):
            try:
                return self.reflector.hierarchy_report()
            except Exception:
                pass
        return {}

    def get_conflict_report(self) -> dict:
        out = {"conflict_score": 0.0, "secondary_felt": "None"}
        try:
            if hasattr(self.synthesizer, "get_conflict_score"):
                out["conflict_score"] = round(float(self.synthesizer.get_conflict_score()), 3)
            if hasattr(self.synthesizer, "get_secondary_felt_state"):
                out["secondary_felt"] = self.synthesizer.get_secondary_felt_state()
            if hasattr(self, "modulators") and hasattr(self.modulators, "report"):
                out["modulators"] = self.modulators.report()
        except Exception:
            pass
        return out

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
        # Soft recovery of lookup budget so multi-hour Learning soaks still expand
        self._dict_lookups_this_wake = max(0, int(getattr(self, '_dict_lookups_this_wake', 0) or 0) - 30)

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
        try:
            if hasattr(self.reflector, "promote_dictionary_nodes_to_working"):
                n_dict = self.reflector.promote_dictionary_nodes_to_working()
                if n_dict:
                    print(f"Consolidation: promoted {n_dict} dictionary node(s) to Working")
        except Exception as e:
            logger.warning("promote_dictionary_nodes_to_working failed: %s", e)
        try:
            n_dup = self.archivist.dedupe_parallel_edges()
            if n_dup:
                print(f"Consolidation: removed {n_dup} duplicate parallel edge(s)")
        except Exception as e:
            logger.warning("dedupe_parallel_edges failed: %s", e)
        new_epistemic_schemas = self.reflector.detect_epistemic_clusters()
        self.archivist.decay_co_activation()
        merged_epistemic = self.reflector.merge_duplicate_epistemic_schemas()
        named_epistemic = self.reflector.try_name_epistemic_schemas()
        merged_names = self.reflector.merge_schemas_sharing_name()
        print(f"Consolidation: merged same-name schemas {merged_names}")
        try:
            merged_lemmas = self.archivist.merge_case_variant_lemmas()
            if merged_lemmas:
                print(f"Consolidation: merged case-variant lemmas {merged_lemmas}")
        except Exception as e:
            logger.warning("lemma case-merge failed: %s", e)
        try:
            merged_case = self.reflector.merge_case_variant_kind_schemas()
            if merged_case:
                print(f"Consolidation: merged case-variant kind schemas {merged_case}")
        except Exception as e:
            logger.warning("case-variant merge failed: %s", e)
        try:
            promoted = self._promote_kind_associations()
            if promoted:
                print(f"Consolidation: promoted associated-with → is-a under kinds {promoted}")
        except Exception as e:
            logger.warning("kind promote failed: %s", e)
        try:
            repaired = self._repair_hierarchy_edges()
            if repaired and any(repaired.values()):
                print(f"Consolidation: hierarchy repair {repaired}")
        except Exception as e:
            logger.warning("hierarchy repair failed: %s", e)
        try:
            grown = self._grow_kind_schema_membership()
            if grown and grown.get("members_linked"):
                print(f"Consolidation: kind schema growth {grown}")
        except Exception as e:
            logger.warning("kind schema growth failed: %s", e)
        try:
            absorbed = self.reflector.absorb_hash_schemas_into_kind_parents()
            if absorbed:
                print(f"Consolidation: absorbed hash schemas into kind parents {absorbed}")
        except Exception as e:
            logger.warning("hash schema absorb failed: %s", e)
        try:
            hollow = self.reflector.prune_hollow_meta_schemas()
            if hollow:
                print(f"Consolidation: pruned hollow meta-schemas {hollow}")
        except Exception as e:
            logger.warning("hollow meta prune failed: %s", e)
        expired_epistemic = self.reflector.expire_unnamed_epistemic_schemas()
        garbage_epistemic = 0
        try:
            if hasattr(self.reflector, "prune_garbage_epistemic_schemas"):
                garbage_epistemic = self.reflector.prune_garbage_epistemic_schemas()
                if garbage_epistemic:
                    print(f"Consolidation: pruned {garbage_epistemic} garbage epistemic schema(s)")
        except Exception as e:
            logger.warning("prune_garbage_epistemic_schemas failed: %s", e)

        # Schema–schema fuel before Tier-2: schemas that share residual heat
        # or current focus neighbourhood get a consolidation co-activation bump.
        try:
            if hasattr(self.archivist, "record_schema_co_activation"):
                heat_nodes = []
                try:
                    heat_nodes = [k for k, _v in self.focus.top_residuals(20)]
                except Exception:
                    heat_nodes = []
                fid = None
                try:
                    fid = self.focus.focus_id
                except Exception:
                    pass
                if fid:
                    heat_nodes.append(fid)
                for t in getattr(self.focus, "stack", []) or []:
                    heat_nodes.append(t.target_id)
                n_pairs = self.archivist.record_schema_co_activation(
                    list(dict.fromkeys(heat_nodes)), amount=1.5
                )
                if n_pairs:
                    print(f"Consolidation: schema–schema co-activation pairs bumped {n_pairs}")
        except Exception as e:
            logger.warning("schema co-activation (consolidation) failed: %s", e)

        # Hierarchical stacking (Tier 2+) + promotion to Trusted
        new_tier2 = []
        promo = {}
        try:
            if hasattr(self.reflector, "detect_epistemic_tier2"):
                new_tier2 = self.reflector.detect_epistemic_tier2()
                self.last_tier2_created = list(new_tier2)
                if new_tier2:
                    print(f"Consolidation: tier-2+ schemas created {len(new_tier2)} {new_tier2[:5]}")
            if hasattr(self.reflector, "promote_stable_schemas"):
                promo = self.reflector.promote_stable_schemas()
                print(f"Consolidation: schema promotion {promo}")
            if hasattr(self.reflector, "hierarchy_report"):
                self.last_hierarchy_summary = self.reflector.hierarchy_report(top_n=8)
        except Exception as e:
            logger.warning("Tier-2 / promote_stable_schemas failed: %s", e)

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
        if getattr(self, "goals", None) is not None:
            try:
                protected_nodes |= set(self.goals.protected_ids(self.archivist.graph))
            except Exception:
                pass

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
        # Consolidation quality signal (Identity & Hygiene)
        try:
            from .edge_types import is_body_channel_node
            illegal_isa = 0
            for u, v, ed in self.archivist.graph.edges(data=True):
                if (ed or {}).get("relation_type") == "is-a":
                    if is_body_channel_node(u) or is_body_channel_node(v):
                        illegal_isa += 1
            if illegal_isa > 0 and hasattr(self.archivist, "repair_identity_edges"):
                self.archivist.repair_identity_edges()
            narr_n = sum(1 for n in self.archivist.graph.nodes if str(n).startswith("narr:"))
            named_ep = sum(
                1 for n, d in self.archivist.graph.nodes(data=True)
                if d.get("node_type") in ("epistemic_schema", "schema") and d.get("named")
            )
            self._last_consolidation_report = {
                "pulse": self.pulse_count,
                "illegal_body_is_a": illegal_isa,
                "narrative_nodes": narr_n,
                "named_epistemic": named_ep,
                "new_epistemic": len(new_epistemic_schemas or []),
                "repaired_body_isa": illegal_isa > 0,
            }
            print(f"Consolidation quality: {self._last_consolidation_report}")
        except Exception as e:
            logger.debug("consolidation quality failed: %s", e)

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
        # + focus-conditioned narrative retrieval (chained autobiographical nodes)
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

            focus_id = None
            try:
                if self.focus.thread is not None:
                    focus_id = self.focus.thread.target_id
                else:
                    focus_id = getattr(self.focus, "focus_id", None)
            except Exception:
                focus_id = None

            basin_key = self.synthesizer.get_current_basin_key()
            anchors = list(self.felt_state_anchors.get(basin_key, []) or [])
            retr_nodes = []
            try:
                if hasattr(self.self_narrative, "retrieval_node_ids"):
                    retr_nodes = self.self_narrative.retrieval_node_ids(
                        focus_id=focus_id, basin_anchors=anchors, top_k=8,
                    )
            except Exception as e:
                logger.warning("narrative retrieval_node_ids failed: %s", e)

            # Seed light focus residuals from autobiographical retrieval
            for nid in retr_nodes[:4]:
                try:
                    self.focus.boost_residual(nid, amount=0.4)
                except Exception:
                    pass

            lti_kwargs = dict(
                focus_id=focus_id,
                residual_totals=residual_totals,
                narrative_elements=getattr(self.self_narrative, "elements", {}),
                parental_nodes=parental_nodes[:20],
                felt_bound_schemas=felt_bound[:30],
            )
            # Only pass if the installed LTI supports the new arg
            try:
                import inspect
                if "narrative_retrieval_nodes" in inspect.signature(
                    self.long_term_interest.promote
                ).parameters:
                    lti_kwargs["narrative_retrieval_nodes"] = retr_nodes
            except Exception:
                pass

            lti_summary = self.long_term_interest.promote(**lti_kwargs)
            print(f"Consolidation: long-term interest {lti_summary} (narrative_retr={len(retr_nodes)})")
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

    def _others_path(self) -> str:
        import os
        return os.path.join(self._data_dir(), "others_state.json")

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
        if hasattr(self, "others") and self.others is not None:
            try:
                self.others.save_state(self._others_path())
            except Exception as e:
                print(f"others save failed: {e}")
        if getattr(self, "goals", None) is not None:
            try:
                self.goals.save()
            except Exception as e:
                print(f"goals save failed: {e}")
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
        if hasattr(self, "others") and self.others is not None:
            try:
                self.others.load_state(self._others_path())
            except Exception as e:
                print(f"others load failed: {e}")
        if getattr(self, "goals", None) is not None:
            try:
                self.goals.load()
            except Exception as e:
                print(f"goals load failed: {e}")
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
