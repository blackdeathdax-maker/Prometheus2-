import random

from .hormonal import BioSystem, Epoch
from .archivist import ArchivistModule, TIER_WORKING, TIER_TRUSTED
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

        print("Prometheus Core Initialized with Fatigue Cycling")

    # ------------------------------------------------------------------
    # External input entry point (used by app.py / tests)
    # ------------------------------------------------------------------
    def queue_input(self, text: str, source: str = "user"):
        self._input_queue.append((text, source))

    def _ingest(self, text: str, source: str):
        """Runs one piece of text through sensory + association + chronos
        linking. Shared by both externally-queued input and self-study."""
        self.sensory.ingest(text)
        basin_key = self.synthesizer.get_current_basin_key()
        felt_state = self.synthesizer.get_current_felt_state()
        anchor = None
        if felt_state != "Unformed":
            anchors = self.felt_state_anchors.get(basin_key, [])
            anchor = anchors[-1] if anchors else None

        result = self.association.place_node(text, definition="", source=source, context_node=anchor)
        node = result.get("term")

        # §2.1b item 4a: try to name any unnamed schemas tied to the felt
        # state active right now (schema naming trigger when user/dictionary
        # input provides a word while "in" that state).
        if node and source in ("user", "dictionary"):
            self.association.try_name_schemas(node, current_felt_state=felt_state)

        relations = self.sensory.detect_relational(text)
        if relations:
            self.association.link_relational(node, relations, source=source)

        if felt_state != "Unformed" and node:
            self.chronos.record_felt_state_link(basin_key, node)
            self.felt_state_anchors.setdefault(basin_key, []).append(node)

        # Explicit negation/correction (§3.4 mechanism 1): flag whatever
        # node was most recently active for gradual demotion at the next
        # Consolidation pass.
        text_lower = text.lower()
        if ("no, " in text_lower or "actually" in text_lower or "that's wrong" in text_lower) and anchor:
            self.archivist.flag_negation(anchor)

        return node

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

        self._update_fatigue(somatic)
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
        else (§5.1)."""
        target = self._select_self_study_target()
        if target is None:
            return

        expansions = self.sensory.lookup_expansion(target)
        if not expansions:
            return

        for child in expansions[:3]:
            definition = self.sensory.lookup_definition(child) or ""
            self.association.place_node(child, definition=definition, source="dictionary",
                                         context_node=target)
        self.archivist.store(target, source="dictionary")  # reinforce parent's last_reinforced

        # Small, scaled-down dopaminergic bump (§5.1, §9 risk 7) via the
        # same fast-layer pathway as any other event -- no bespoke
        # self-study fatigue tap.
        with self.bio.lock:
            self.bio._hormones["dopamine"] = min(
                1.0, self.bio._hormones["dopamine"] + self.SELF_STUDY_DOPAMINE_BUMP
            )

    def _select_self_study_target(self):
        """(a) active/trusted nodes with few children, or (b) emotionally
        salient nodes weighted by *current* felt state (§5.1) -- historical
        emotional weighting stays inside Consolidation, not here."""
        graph = self.archivist.graph
        if graph.number_of_nodes() == 0:
            return None

        candidates = [
            n for n, d in graph.nodes(data=True)
            if d.get("tier", 0) >= TIER_WORKING and graph.out_degree(n) < 3 and not d.get("is_schema")
        ]
        if candidates:
            return random.choice(candidates)

        # (b) fallback: whatever node is currently anchoring the felt
        # state, if any.
        key = self.synthesizer.get_current_basin_key()
        anchors = self.felt_state_anchors.get(key, [])
        if anchors:
            return anchors[-1]

        # Last resort: any node at all.
        return random.choice(list(graph.nodes)) if graph.number_of_nodes() else None

    # ------------------------------------------------------------------
    # Fatigue / state cycling
    # ------------------------------------------------------------------
    def _update_fatigue(self, somatic):
        self.fatigue = min(1.0, self.fatigue + somatic.urgency * self.FATIGUE_GROWTH_RATE)

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

        # §4C: the single checkpoint call for this pass -- everything
        # above mutates the graph and/or hormonal state without saving
        # individually (see the "No self.save() here" comments in
        # archivist.py, reflector.py, and hormonal.py's step()). This is
        # the one clock persistence is gated to.
        self.archivist.save()
        self.bio.save_state()

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
        regulating_nodes = self.archivist.eligible_regulation_nodes(anchors or None)

        if not regulating_nodes:
            # §4.2: legitimate state (nothing eligible yet), not an error.
            return

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
