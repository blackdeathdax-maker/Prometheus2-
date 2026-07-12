# Prometheus — Living Design Spec

**Purpose of this document:** single source of truth for the project's design decisions. Paste the relevant section into any new conversation (with Claude, Grok, Gemini, etc.) instead of re-explaining from memory. Update this doc whenever a decision changes — don't let decisions live only in chat history.

**Last updated:** 2026-07-09 (rev. 16 — final export for handoff)
**Status:** Design phase complete. No code has been written yet. Every structural/architectural question raised during design has been resolved; what remains is numeric tuning (~15 items) that can only be determined empirically, by running a vertical slice. See §10 for the full remaining list, and §9 for standing risks to watch during implementation.

**Contents:** §1 Project Concept · §2 Two Webs (2.1 Emotional Web, 2.1a Basin Formation, 2.1b Complex Schemas, 2.2 Knowledge Web, 2.3 Hierarchy Placement, 2.4 Trust vs. Depth) · §3 Trust Tier System · §4 Regulation Cross-Link · §4A reflector.py · §4B UI/Dashboard · §4C Persistence · §4D Tick Scheduling · §5 Fatigue-Driven State Cycling (5.1 Self-Study, 5.2 Pruning Trigger) · §6 Developmental Epochs · §6A Node/Edge Data Schema · §7 Module Responsibilities · §8 Immediate Tasks · §9 Known Risks · §10 Open Questions

---

## 1. Project Concept

Prometheus is a neuromorphic cognitive simulation: an agent that develops emotional and cognitive maturity through simulated biology rather than fixed rules. Hidden hormonal/somatic state drives a two-web knowledge system that grows, consolidates, and prunes itself over time, progressing through three developmental epochs gated by demonstrated competence — not timers or event counts.

**Hard boundary:** An LLM may eventually serve as a *speech/expression layer only* (translating internal state into language). It must never be the cognitive engine. All reasoning, growth, and regulation logic must remain deterministic, biologically-modeled logic — not delegated to a generative model. This boundary is easy to erode by accident once an LLM is in the pipeline for any purpose — treat it as a hard constraint in every future design/coding session.

**Core emergence principle — the hidden/visible boundary is about agency, not display.** The rule that "the executive layer cannot read core.py" is not a UI-privacy rule — it exists because the whole premise of emergence in this project is that felt states, graph structure, and self-regulation behavior must be *inferred and named by the agent through experience*, never read directly off a raw variable. An agent with programmatic access to its own raw hormonal values would be a fundamentally different, and strictly worse, system relative to this design's goals. Consequently:
- A **human operator** may view raw core.py values freely (§4B Debug tab) — this changes nothing about the agent's cognition and is purely external instrumentation.
- **No module that participates in the agent's own decision loop** — including `prometheus.py`, the orchestrator — may ever route core.py's raw values into anything the agent's own logic conditions on. `prometheus.py` already only reads visible-layer felt states for orchestration decisions (transitions §6, regulation triggers §4.1–4.2); this is consistent with what's designed but is now stated as a named, non-negotiable principle rather than an incidental property.
- The Debug tab (§4B) is therefore the **only** sanctioned path from core.py to anything outside hormonal.py/core.py, and it is a dead end — read-only, display-only, with zero code path back into agent logic.

---

## 2. System Architecture: Two Webs

### 2.1 Emotional Web (the engine)
- Small, dense, slow-changing.
- Nodes represent felt-state signatures (clusters of somatic state translated into named states).
- Drives construction and weighting of the knowledge web.
- Also receives feedback from the knowledge web during Adolescence (regulation — see §4).

### 2.1a Felt-state formation — topographical basins (resolved)

Rejects both extremes considered earlier: **not** fixed hand-defined emotion categories (would graduate the agent to Adolescence-equivalent competence with no actual growth — the agent would just be classifying against pre-written labels), and **not** generic statistical clustering (introduces an opaque, boundary-drifting process in tension with the "no black-box in the engine" principle, and conflicts with §6.1's stability requirement). Instead: felt states are **attractor basins in a topographical landscape**, formed entirely from the agent's own lived trajectory through state-space — grounded in dynamical-systems models of emotion (state as basins the system settles into and revisits), not classification.

**Mechanism:**
1. **Composite axes (hand-defined, but not raw core.py fields).** synthesizer.py projects core.py's raw variables onto a small number of deterministic composite dimensions — a PAD-style (Pleasure/Arousal/Dominance) three-axis model: **arousal** (heart_rate + respiration_rate), **valence** (dopaminergic_tone − cortisol_load), and **dominance/control** (vascular_constriction + muscle_tension, both correlating with threat/reduced-control physiology — low dominance reads as "this is happening to me," high as "I can act on this"). Three axes rather than two specifically to avoid emotions collapsing together that shouldn't (e.g., anger and fear sit at nearly identical arousal/valence coordinates but differ sharply in dominance). These axes are chosen in advance (you're defining *what dimensions exist*), but this is categorically different from defining regions on them — the axes are a coordinate system, not an emotional vocabulary.
2. **No pre-defined regions.** Nothing marks any point on the arousal/valence plane as "Distress" or "Calm" in advance. The landscape starts flat/unformed — a Childhood agent genuinely has no stable felt states yet, because it hasn't lived enough to have any.
3. **Basin formation via dwell-time density.** chronos.py's logged trajectory through arousal-valence-dominance space builds a grid-based dwell-time histogram (deliberately the simplest inspectable option — no ML clustering, no arbitrary distance metric or cluster count). Peaks in accumulated density — places the trajectory has repeatedly visited and returned to — are candidate basins.
4. **Basin stabilization = the Childhood naming milestone.** A candidate basin becomes a genuine named felt state (and earns a knowledge-web node link, per §6.1) only once it's been revisited enough to be a real recurring pattern, evaluated the same way as everything else needing hysteresis in this design — this **is** the concrete mechanism behind §6.1's "reliably names its own emotional states," not a separate classification exercise layered on top of it.
5. **Basin birth/death (decay).** A basin that stops being revisited should flatten back out over time, mirroring non-reinforcement decay used elsewhere (§3.4, §4.5) — an emotional pattern the agent has outgrown can genuinely fade rather than being permanent once formed, for consistency with the rest of the design.
6. **Cadence.** Basin detection/refinement (histogram update, stabilization check, decay) happens **during Consolidation only** — same clock as trust, efficacy, re-parenting, and reflector — so a single noisy tick can never reshape what a felt state means. This is what satisfies §6.1's stability requirement mechanically.

**What synthesizer.py actually outputs, tick to tick:** the current point in arousal-valence-dominance space is checked against the *existing, stabilized* basin map (built as of the last Consolidation pass) to find the nearest/matching basin — that lookup is what produces the "current Felt State" the rest of the system reads. The landscape itself only changes at Consolidation; the lookup against it is cheap and live every tick.

**Open item carried into §10:** grid resolution for the dwell-time histogram, and the revisit-count/duration threshold for basin stabilization, are not yet numeric — same tuning category as fatigue thresholds, only resolvable empirically once a vertical slice is running.

**Scope boundary — basic emotions from biomarkers alone; complex emotions require §2.1b.** A three-axis PAD landscape gives robust coverage of *basic/body-state emotions* (joy, fear, anger, sadness, disgust, surprise, calm, and gradations) — these are physiologically definable and should genuinely differentiate well, including previously-collapsed pairs like anger vs. fear (separated by dominance). It does **not**, and structurally cannot, produce *complex/social/abstract emotions* (guilt, nostalgia, jealousy, pride) from richer biomarkers alone — these are defined by relational or narrative content (a self-concept, a past/present comparison, another person as a referent), not body state. See §2.1b: this design reaches complex emotions not through biomarkers but by recognizing that the knowledge web already builds the relational structure they require.

### 2.1b Complex Emotional Schemas — resolved

Complex/social emotions become reachable not by adding more biomarkers, but by recognizing that the knowledge web already builds the relational/schematic structure they require. A complex emotion is modeled as a **recurring co-occurrence of a stabilized basic basin (§2.1a) with a specific relational graph pattern**, not a new point in PAD space.

**Same principle as §2.1a's axes-vs-regions distinction, applied one layer up:** the four relational edge types below are a hand-defined *vocabulary of relation kinds* (like arousal/valence/dominance being hand-defined axes) — but which *combinations* of those edges constitute a recognized complex emotion is never hand-specified, and is discovered entirely from recurrence in the agent's own experience (like basin regions on the PAD axes). No fixed mapping of "these edges together = guilt" exists anywhere in the design.

**1. The self-node — a deliberate, necessary exception to "everything is earned."** A single, permanent `SELF` node is seeded directly into the Trusted tier at initialization, not earned through corroboration like every other node in the system. This is necessary because no experience can precede having a self to relate things to — it's the one deliberate axiom in an otherwise fully emergent design, worth flagging explicitly as an intentional exception rather than an inconsistency.

**2. New relational edge types**, extending §2.3's typing scheme beyond categorical (`is-a`/`part-of`/`associated-with`) into narrative/relational:
- `responsible-for` — SELF linked as agent of an action/outcome node (guilt, pride).
- `violates` — a node conflicts with a standard/value node linked to SELF (guilt, shame).
- `temporal-contrast` — a node relates to a past state differing from the current one, using timestamps chronos.py already logs (nostalgia).
- `concerns-other` — a node involves a distinct entity node representing someone other than SELF (jealousy, embarrassment, social emotions generally).

sensory.py detects candidates for these edges at ingestion using the same deterministic, pattern/keyword-level approach already used for explicit negation (§3.4) — no new NLP machinery, e.g. "I shouldn't have done that" flags `responsible-for` + `violates` candidates near the current event node.

**3. Detection is reflector.py's job**, and gives real justification to the consistency-auditing capability previously deferred to v2 (§4A) — not speculative anymore, now solving an actual capability gap. Reflector scans chronos.py's logged history (Consolidation-gated, same clock as everything else) for **any** repeated co-occurrence of a stabilized basin with a *consistent* relational edge pattern — genuinely emergent, not matched against pre-written combinations. Reflector has no advance knowledge that a given combination "means" guilt or pride; it only recognizes that a specific combination has recurred enough to count as a stable pattern (same hysteresis-over-N-recurrences principle used throughout this design) and creates a Schema Node for it. Psychology terms like "guilt-shaped" or "pride-shaped" are illustrative labels for *this document*, to help reason about whether the mechanism is working — they are never written into the detection logic itself.

**4. Storage — a new node class: Schema Nodes.** Once a co-occurrence pattern stabilizes (same hysteresis-over-N-recurrences principle used throughout this design), it becomes its own node — linking back to the basic basin(s) it's composed from and the specific relational edges that define it. A Schema Node is created **unnamed** — its existence is just "this recurring pattern is real and stable," nothing more.

**4a. Naming a Schema Node works exactly like §6.1, not by pre-assignment.** No combination of relational edges is hand-labeled "guilt" or "pride" in advance — that would just be fixed categories smuggled in through the relational side door instead of the PAD side door already rejected in §2.1a. A Schema Node earns a name only if and when the agent's actual dictionary/user input happens to link a word to it, through the same felt-state-to-knowledge-node linkage mechanism used for basic basins. If no such linkage ever occurs, the schema remains real, stable, and recognized by the system, but unnamed — a pattern the agent has experienced without yet having a word for it, which is arguably a more honest state than forcing a label onto it.

**5. Epoch placement — the concrete mechanism behind §6.2.** Schema formation requires a stabilized basic basin *and* accumulated relational edges *and* enough recurrence of their co-occurrence — strictly later-developing than basic basin naming (§6.1). Adopted as the actual competence check behind Adolescence → Maturity: "structural resilience" becomes *the agent has formed some number of stable Schema Nodes*, replacing the looser variance-based placeholder previously in §6.2 with a real, evaluable milestone.

**Known compounding risk:** this deepens the cold-start concern already flagged for basic basins (§9 risk 6) by one layer — Schema Nodes need relational edges to exist *and* recur *and* co-occur with a stabilized basin, so Adolescence-to-Maturity progression could be considerably slower or more data-hungry than basic Childhood naming. This is honest emergence, not a flaw, but should be expected rather than discovered as a surprise.

### 2.2 Knowledge/Schema Web (the content)
- Large, hierarchical.
- Built from three input sources, handled differently:
  - **Dictionary-sourced input** — treated as more authoritative, higher starting trust weight.
  - **User input** — provisional by default, must earn trust over time.
  - **Self-generated (idle autonomous expansion)** — see §5.1. Dictionary-sourced by content, but tagged distinctly so it doesn't count toward the diversity signal the way externally-corroborated edges do (prevents the agent gaming its own trust metrics via self-expansion).
- Structured with trust tiers (see §3) layered onto a hierarchical/abstraction structure.

### 2.3 Hierarchy placement mechanics (resolved)
Two placement paths, both producing **typed edges** (`is-a`, `part-of`, `associated-with` — not generic edges; typing is what makes the graph actually hierarchical rather than just densely connected):

1. **Dictionary-pattern parsing (primary, for dictionary input).** Definitional phrasing often contains the hierarchy for free ("blue: a color resembling the sky" → `blue is-a color`). sensory.py pattern-matches definitional structures ("X is a Y," "X: a type of Y," etc.) to extract explicit parent-child edges at ingestion — no inference model needed.
2. **Co-occurrence context (fallback, primarily for user input).** When no explicit relationship is parseable, the new node attaches to whichever existing node was most active (highest recent corroboration, or dominant in current felt-state context) at time of ingestion. Produces an `associated-with` edge, not a false `is-a` claim — co-occurrence is weaker evidence than a stated relationship and must not be mislabeled as one.
3. **Re-parenting.** Nodes placed via co-occurrence (option 2) are not permanently fixed. If a node accumulates enough independent corroboration to justify a firmer/different parent, re-evaluation happens **during Consolidation** (§5) — same pass that already re-evaluates trust, rather than a separate mechanism.

### 2.4 Trust tier vs. hierarchy depth — resolved: orthogonal axes
Trust (tier: Provisional/Working/Trusted, §3) and structural position (edge type/depth, §2.3) are **separate properties on the same node, not the same dimension.** A node can be structurally deep and still Tier 0 (asserted once, unconfirmed). A node can be shallow and Trusted (well-corroborated, simple fact). Collapsing these into one axis would falsely force "abstract" and "confident" to move together. Both properties are tracked independently per node.

---

## 3. Trust Tier System

### 3.1 Tiers
- **Tier 0 — Provisional**: newly created, low/no corroboration.
- **Tier 1 — Working**: corroborated by a small number of *diverse* sources, or dictionary-original.
- **Tier 2 — Trusted/Schema**: heavily corroborated, diverse sources, survived consolidation passes without contradiction.

### 3.2 Trust score inputs (not raw edge count alone)
1. **Edge count** — corroboration signal.
2. **Edge diversity** — edges from different sources/sessions count more than repeated edges from the same context (prevents hub-node inflation and repetition-gaming).
3. **Source weight at creation** — dictionary origin starts higher than user-asserted.
4. **Emotional-state-at-encoding** *(optional/stretch)* — nodes formed under regulated/calm felt-states get a small trust bonus; nodes formed under high-arousal/spike states start lower.

### 3.3 Promotion & Demotion — timing
- **Consolidation-gated, not live.** Tier changes are only evaluated during the Consolidation state (see §5), not during Learning.
- Rationale: matches the biological metaphor (structural reorganization happens offline, not while awake/encoding); provides free hysteresis against single-session noise; reuses an already-designed mechanism instead of adding a new evaluation pathway.
- Promotion/demotion require the threshold condition to hold across **N consecutive consolidation passes** (hysteresis), not a single pass. Exact N not yet tuned.

### 3.4 Demotion mechanics — v1 scope
Two mechanisms only, for now:
1. **Explicit negation/correction** — detected when user input contains negation/correction language referencing a recently-active node (e.g., "no, that's wrong," "actually X is not Y"). Detection is keyword/pattern-level (regex-adjacent), not semantic. No NLP model required.
2. **Non-reinforcement decay** — a node that isn't corroborated across successive consolidation passes loses trust gradually, using the same decay-toward-baseline pattern already used in the hormonal layer.

**Explicitly deferred (not in v1):**
- Structural contradiction detection (pre-defined exclusivity categories, e.g. "hot" vs "cold" as siblings).
- Semantic/embedding-based contradiction detection. Reasoning: even free/local embedding models (a) still function as an opaque neural component inside what's meant to be a transparent, deterministic engine, conflicting with the LLM-boundary principle in spirit; (b) solve "these are topically related" much better than "these disagree" — similarity ≠ contradiction; (c) require a new inference pipeline and new threshold-tuning burden. Revisit only if v1 in practice shows a real gap.
- Demotion should be **gradual, one tier at a time**, not multi-tier drops from a single contradiction — consistent with hysteresis used elsewhere.

---

## 4. Emotional Engine ↔ Knowledge Web Cross-Link (Regulation) — resolved

Regulation is how the knowledge web reaches back into the emotional engine, fulfilling Adolescence's design goal ("use knowledge nodes to regulate somatic states"). Built entirely from existing mechanisms — no new subsystem class introduced.

**4.1 Detection.** prometheus.py monitors felt-state intensity each tick against a spike threshold, using the same hysteresis-band pattern as the fatigue T1/T2 checks (§5) rather than a bespoke trigger. This naturally concentrates regulation activity in Adolescence, since that's the epoch defined by high surge/turmoil.

**4.2 Node selection.** From the stable felt-state → node anchor established in Childhood (§6.1), prometheus.py looks at connected knowledge-web nodes, **restricted to Working or Trusted tier only** (§3.1). Provisional nodes are excluded — an unconfirmed node has no business being relied on to talk the system down from a spike. This restriction falls directly out of the existing tier system, no new logic required.

**4.3 Dampening mechanism.** Regulation **accelerates core.py's existing fast-layer decay** (`decay_fast`, called at a higher-than-normal rate) rather than applying an arbitrary counter-delta. This models regulation as *returning to baseline faster*, not *canceling the feeling* — a better fit for what emotional regulation actually is.

**4.4 Dampening is capped, not instant.** Per-tick dampening strength is capped as a fraction of the spike, scaled by the regulating node's efficacy (§4.5). A single high-efficacy node cannot fully flatten a spike in one tick — this preserves genuine turbulence in Adolescence (undermining this would defeat the epoch's purpose) while still giving real regulation capacity. The cap should scale with epoch: Adolescence gets meaningful but partial capacity; Maturity's stability comes from baseline hardening (§6.2), not from regulation strength increasing further.

**4.5 Regulatory efficacy — a new, separate score.** Not folded into epistemic trust (§3) — a node can be a well-trusted fact while being useless as a coping mechanism, and vice versa. Two independent properties per node (same pattern as the trust/hierarchy-depth orthogonality in §2.4):
- **Epistemic trust** (§3) — is this true/corroborated.
- **Regulatory efficacy** — has using this node to self-soothe historically worked.

Efficacy is evaluated **during Consolidation only**, not live — same evaluation point as trust and re-parenting, one clock. Check whether felt-state intensity dropped faster than baseline decay alone would predict, over the ticks following a regulation attempt; success nudges efficacy up, failure nudges it down.

**4.6 Regulation costs fatigue.** A regulation attempt draws on the same fatigue economy as self-study (§5.1) rather than being a free action — actively engaging a coping strategy is effortful, not free, in real cognition. This also naturally self-limits regulation from being invoked every tick without needing a separate cooldown mechanism.

---

## 4A. reflector.py — resolved scope

Reflector reads the finished state of the graph and produces insight *about* it — metacognition, not cognition. It does not grow, place, score-for-trust, or regulate; those belong to association.py, archivist.py, and prometheus.py respectively (§2.3, §3, §4). Three responsibilities:

1. **Structural self-report.** Periodic summary of graph shape: dense vs. sparse regions, hub nodes, hierarchy balance, proportion of Provisional vs. Working vs. Trusted nodes. Feeds the dashboard's high-level view and can supply self-study's (§5.1) "active/trusted nodes with few children" targeting, rather than association.py identifying gaps ad hoc.
2. **Regulatory self-awareness.** Aggregates regulatory efficacy scores (§4.5) across all regulation-capable nodes to surface which actually work — the agent's model of its own coping strategies, not just their use.
3. **Complex-schema detection (§2.1b).** Scans chronos.py's logged history for recurring co-occurrence of a stabilized basic basin with a specific relational edge pattern (SELF + `responsible-for`/`violates`/`temporal-contrast`/`concerns-other`, §2.1b), forming Schema Nodes once a pattern stabilizes. This is genuine structural pattern-matching over existing typed data, not black-box inference — it absorbs and gives concrete purpose to the consistency-auditing capability originally sketched here, now solving an actual capability gap (reaching complex emotions) rather than a speculative one.

**Still explicitly deferred (not in v1):** flagging structurally ambiguous nodes unrelated to emotional schemas (e.g., conflicting `is-a` parents with no self-referential content) or "known true but unhelpful" nodes where high epistemic trust and low regulatory efficacy diverge. This narrower consistency-auditing case still carries the scope-creep risk flagged for embeddings-based contradiction detection (§3.4) and remains a v2 candidate.

**Cadence:** Consolidation-gated, same clock as trust promotion/demotion, re-parenting, and regulatory efficacy scoring (§3.3, §2.3, §4.5) — one clock for all offline reprocessing, consistent with the rest of the design, rather than a separate reporting cadence.

---

## 4B. UI / Dashboard — resolved

Tabbed layout (chosen over a single dense view — Streamlit's rerun model means a single view re-renders everything on every interaction; tabs let the graph panel stay static while other panels update independently).

1. **Graph tab** — the knowledge/schema web (Pyvis). Visual encoding: trust tier (§3.1) → node color/opacity (Provisional faint → Trusted solid); edge type — both categorical (§2.3, `is-a`/`part-of`/`associated-with`) and relational (§2.1b, `responsible-for`/`violates`/`temporal-contrast`/`concerns-other`) — → distinct line style per type, since edge typing is what makes this a hierarchy-with-relationships rather than a blob. Regulatory efficacy (§4.5) is **not** overlaid here — kept in the Reflection tab, since it's a philosophically distinct property from epistemic trust and cramming both onto one node's visual encoding gets muddy. **Open gap:** `basin`/`schema`/`self` node types (§6A) don't carry a tier, so tier-based coloring doesn't apply to them — their visual treatment on this tab is not yet defined (§10 item 21).
2. **State tab** — felt state (named, current, always visible — the point of synthesizer.py's translation layer); epoch name (Childhood/Adolescence/Maturity) with **no visible progress meter toward the next transition** — showing one would turn an earned developmental milestone into a bar to min-max, undermining the "earned, not counted" philosophy (§6); current operating mode (Learning/Consolidation/Pruning, §5); fatigue shown as an **abstracted level indicator** (e.g., low/medium/high) rather than a raw number — consistent with felt states already being abstracted from core.py, not a direct numeric leak.
3. **Reflection tab** — reflector.py's structural self-report + regulatory self-awareness + detected Schema Nodes (§4A, §2.1b — surfacing recognized recurring emotional patterns, named if the agent has linked a word to them, otherwise shown as unnamed stable schemas), with a visible "last updated" timestamp, since this is Consolidation-gated and should read as periodic self-assessment, not a live feed.
4. **Debug tab** — the one sanctioned exception to the hidden/visible boundary (§ System Level Boundaries, original spec). Displays raw core.py values directly (heart_rate, respiration_rate, dopaminergic_tone, cortisol_load, vascular_constriction, muscle_tension). Must be rendered visually distinct from the other tabs (e.g., warning color, explicit "raw internal state — not part of the cognitive model" labeling) so it's unmistakably an instrument panel, not part of the felt-state system. Read-only in this direction — debug tab data must never feed back into any other UI panel's logic.

**Task 1 fix, now with real content to render:** build the Pyvis network in memory, call `.generate_html()`, pass the resulting string directly to `st.components.v1.html()` — no filesystem write, avoiding the silent-failure mode on read-only/sandboxed filesystems or races with Streamlit's rerun cycle. Node/edge styling pulls from archivist.py's per-node tier and edge-type data at render time.

---

## 4C. Persistence — resolved

**Scope: single agent instance for v1.** One save file, no multi-instance/named-agent support. Simpler, and nothing in the design implies multi-instance was ever a goal — revisit only if a real need appears later.

**What persists (durable learning only):**
- Both graphs — knowledge web (nodes, typed edges including relational types from §2.1b, tier, source tag) and emotional web (SELF node, Schema Nodes).
- The basin landscape (§2.1a) — stabilized basin boundaries/dwell-time histogram, not raw trajectory.
- Regulatory efficacy scores (§4.5).
- Current epoch (§6).
- core.py's **slow-layer baseline only** (the permanently-shifted cortisol/vascular_constriction constants from the epoch SHIFT MECHANISM) — this is temperament, and is durable by definition.

**What does not persist in full:**
- **Fast-layer values** (current heart_rate, current dopaminergic_tone, etc.) — reset to resting baseline on restart. These are moment-to-moment state, not learning; a restarted agent shouldn't wake up mid-spike from before it was last closed.
- **chronos.py's raw log** — kept only as a **bounded rolling window** (enough ticks to support the §6.1/§6.2 evaluation windows), not the agent's full lifetime history. This follows the same principle used everywhere else in the design: durable memory is the *consolidated summary* (graphs, tiers, basins, schemas), not the raw event feed. Avoids unbounded disk growth without needing a separate pruning mechanism for the log itself.

**When saves happen:** checkpointed **only at Consolidation**, the same clock as trust/efficacy/basin/schema evaluation (§3.3, §4.5, §2.1a, §2.1b) — one clock for all offline reprocessing, now including persistence. Consequence, stated honestly rather than hidden: a crash mid-Learning loses whatever accumulated since the last Consolidation (new Provisional nodes, recent chronos entries) — but this isn't really a gap, since nothing is *durable* until it survives Consolidation's evaluation anyway, same as everything else in this design.

**Format:** plain JSON, not a binary/pickle format — networkx graphs serialize cleanly to node-link JSON, and JSON keeps the saved state human-inspectable, consistent with the project's no-black-box transparency principle and useful for debugging across the multi-tool workflow.

---

## 4D. Tick Scheduling under Streamlit — resolved

**Chosen: catch-up simulation, interaction-driven only (no background process for v1).** No separate persistent process. On each Streamlit rerun (app opened, user interacts), prometheus.py checks wall-clock time elapsed since the last recorded tick (from the §4C checkpoint) and advances the simulation by computing however many Learning/self-study/Consolidation cycles that elapsed time implies, all at once, before rendering. "Idle time" is real but experienced retroactively in a batch at next open, not continuously in the background.

**Why, over a genuine background process:** fits Streamlit's actual request-response execution model instead of fighting it — no persistent process to keep alive independently of the dashboard, no threading/session-state complexity (a well-known Streamlit pain point). The catch-up computation is largely code that has to exist anyway, since it's just "replay Consolidation-driven ticks until wall-clock time is caught up," reusing the same checkpointing logic from §4C rather than adding new machinery.

**Known tradeoff, stated honestly:** an agent opened infrequently (e.g., once a week) will show a sudden batch of accumulated self-study/development at open-time rather than organically gradual change. Acceptable for v1; **a genuine background process (a persistent service independent of Streamlit) is a legitimate v2 upgrade** if continuous real-time development becomes a priority — deliberately deferred, not because it's a bad idea, but because it's a materially bigger infrastructure lift than anything else in this build, closer in weight to the Streamlit learning curve itself.

**New tuning item:** a real-time-to-simulated-tick mapping (how many simulated ticks one elapsed minute/hour represents) is required for the catch-up computation — not yet numeric, same tuning category as fatigue thresholds (folded into §10).

---

## 5. Fatigue-Driven State Cycling

The system cycles between three operating states, determined by a fatigue signal (derived from existing biomarkers — cortisol_load, dopaminergic depletion, time-since-last-consolidation, etc. — exact composite formula not yet finalized).

- **Learning** (`fatigue < T1`): active encoding of new input (dictionary + user) into the knowledge web, driven by current felt state.
- **Consolidation** (`T1 ≤ fatigue < T2`): offline reprocessing of chronos.py's felt-state history; reinforces high-arousal experiences, flattens low-arousal ones; **only point at which trust tiers are promoted/demoted**; slow-layer hormonal baseline shifts happen here (not instantly at epoch transition).
- **Pruning** (`fatigue ≥ T2`): removes low-salience, unconsolidated nodes. Should peak in Adolescence (matches real adolescent synaptic pruning), not run at constant rate across all epochs.

**Required for stability (flagged, not yet implemented):**
- Hysteresis margin around T1/T2 to prevent flicker at boundaries.
- Fatigue must have its own recovery curve (drops during Consolidation) so the system self-cycles rather than ratcheting into permanent Pruning.
- Minimum-occurrence floor before any reliability/promotion check is eligible to fire (prevents trivial early-tick milestone completion).

**Open question:** are T1/T2 fixed constants, or epoch-dependent (e.g., adolescent pruning threshold lower than adult)? Not yet decided — flagged as coupling two systems that are easier to reason about independently at first pass.

### 5.1 Autonomous idle expansion (self-study) — resolved
During Learning state, when no external input (user or dictionary-triggered) is queued, the agent self-initiates dictionary expansion of existing nodes rather than sitting idle (e.g., an existing "colors" node gets children like "blue," "white" pulled autonomously from the dictionary).

- **Node selection**: a mix of (a) active/trusted nodes with few children (targets obvious gaps in things it already half-knows) and (b) emotionally salient nodes, weighted by **current** felt state at the moment of self-study (not historical emotional weighting — that behavior stays inside Consolidation, which already owns history-based reprocessing).
- **Mechanism, not a new fatigue tap**: self-study does **not** directly drain a fatigue counter. Instead, successful self-expansion triggers a small hormonal reaction in core.py (a modest dopaminergic tone bump — mild curiosity/satisfaction signal), through the same fast-layer pathway as any other event. Fatigue rises as a *consequence* of that hormonal activity, same as everything else — no bespoke self-study fatigue mechanism needed.
- **Magnitude must be scaled down** relative to externally-triggered hormonal deltas, so self-study reads as gentle background texture rather than producing felt-state territory that's supposed to represent significant external events. This also keeps external experience the dominant driver of the emotional engine by construction, without needing a separate decay-rate rule.
- **Self-limiting loop**: low fatigue → idle self-study → small hormone bump → fatigue rises → eventually crosses T1 → forced into Consolidation, where self-generated nodes get evaluated for trust like anything else → fatigue drops during Consolidation → cycle repeats. No separate cap or graph-size gate needed — reuses the existing fatigue clock.
- **Known risk**: this creates a *new instance* of the general runaway-loop risk (§9 item 1) — emotional salience picks a node → self-study produces more emotional signal → shifts what's salient next → compounding. Scaled-down magnitude is the primary mitigation; decay rate between self-study cycles must be fast enough relative to delta size to settle rather than compound. Not yet numerically verified — flag as a tuning item once the vertical slice is running.

### 5.2 Pruning Trigger — resolved

Pruning is the terminal stage of the same non-reinforcement mechanism already used for demotion (§3.4), not a separate judgment call.

- **Rule:** a node becomes eligible for removal when it (a) is at Tier 0 (Provisional) — whether newly created and never promoted, or demoted down to the floor via §3.4's non-reinforcement decay — **and** (b) has received no new corroboration (no new edge) across N_prune consecutive Consolidation passes since reaching or remaining at that floor. Demotion and pruning form one continuum: Trusted → Working → Provisional → pruned, all driven by the same non-reinforcement signal.
- **Scope:** knowledge-web, tier-bearing nodes only. `basin`/`schema`/`self` node types (§6A) don't carry a tier and are exempt from this rule — basins already have their own decay mechanism (§2.1a point 5), and the same non-reinforcement pattern extends naturally to schema nodes rather than routing through archivist's tier-based sweep.
- **Cascade cleanup:** removing a node also removes any edge referencing it as `source_id` or `target_id` (§6A) — no orphaned edges left pointing at nonexistent nodes.
- **Epoch-scaling:** intentionally left as an open numeric question rather than resolved here — the original spec calls for pruning to peak in Adolescence (mirroring real synaptic pruning), which would mean N_prune is lower (more aggressive) in Adolescence than Childhood/Maturity. This is the same fixed-vs-epoch-dependent-threshold question already open for T1/T2 (§5, §10 item 6) — tracked as one shared open decision rather than duplicated.

---

## 6. Developmental Epochs

Transitions are **earned via competence checks**, evaluated by `prometheus.py` (the orchestrator — see §7), not by hormonal.py and not by simple event counters.

### 6.1 Childhood → Adolescence
Gate: the agent **reliably and consistently** links a given felt-state signature to the **same** knowledge-web node across repeated occurrences (i.e., convergence — "naming" — not just linkage). This is now mechanically concrete via §2.1a: a felt state *is* a stabilized attractor basin in arousal-valence-dominance space, and "naming" is the act of that stabilized basin acquiring a durable, reused link to a knowledge-web node. Requirements:
- Basin stabilization (§2.1a) is itself a precondition — there's no "naming" to evaluate until a basin exists at all.
- Linkage (association exists between a stabilized basin and a node) is a further precondition; naming (link is stable/reused across repeated visits to that basin) is the actual gate — three layered competencies, not one.
- "Reliably" = consistency rate above some threshold over a rolling window of the last N occurrences of a given basin being revisited, using chronos.py's logged history as the evaluation window.
- Must have a minimum-occurrence floor before eligibility (prevents trivial completion with a sparse early graph or landscape).
- Exact consistency threshold, window size (N), minimum floor, and basin stabilization threshold (§2.1a): **not yet tuned**.

### 6.2 Adolescence → Maturity
Gate: now mechanically concrete via §2.1b — "cognitive schemas achieve structural resilience" means **the agent has formed some number of stable Schema Nodes** (complex emotional schemas: recurring co-occurrence of a stabilized basic basin with a consistent, emergent relational edge pattern — no specific pattern hand-designated in advance as "this counts"), evaluated with the same hysteresis-over-N-recurrences principle used throughout this design, rather than a raw count of regulation events. Regulation-event outcomes (§4.5) may still feed into which basins/schemas are considered stable, but the gate itself is Schema Node formation. Exact count/threshold of Schema Nodes required, and their own stabilization window: not yet numeric — same tuning category as §6.1's thresholds.

---

## 6A. Node / Edge Data Schema — resolved

Canonical field list, consolidating every property that has accumulated across this design. This is the shared contract for any tool writing code against the graphs — the single highest-risk gap for cross-tool drift if left implicit.

**Node — common fields (every node):**
- `id` — unique identifier.
- `label` — display name; **may be null/unset** for unnamed Schema Nodes (§2.1b) or freshly-created co-occurrence-placed nodes awaiting naming.
- `web` — `"knowledge"` or `"emotional"` (§2.1/§2.2).
- `node_type` — `"standard"` | `"basin"` | `"schema"` | `"self"` (the SELF node, §2.1b).
- `source` — `"dictionary"` | `"user"` | `"self_generated"` (§2.2, §5.1) — governs starting trust weight and diversity-signal exclusion.
- `tier` — `Provisional` | `Working` | `Trusted` (§3.1). **Not applicable to `basin`/`schema`/`self` node types** — trust tiers represent epistemic corroboration of facts; basins/schemas represent recurrence, a different kind of evidence. Kept as a separate concept rather than folding these node types into the tier system with special-cased defaults.
- `created_at` / `last_reinforced_at` — timestamps, needed for non-reinforcement decay (§3.4) and `temporal-contrast` edges (§2.1b).
- `corroboration_edges` — structure supporting the diversity signal (§3.2): distinguishes distinct sources/sessions, not just a raw count.
- `regulatory_efficacy` — float; **null/unset if never used in regulation**, not zero — zero would falsely imply "tried and failed" rather than "never attempted" (§4.5).

**Node — type-specific fields:**
- `basin` nodes: `pad_coordinates` (arousal, valence, dominance centroid), `dwell_density`, `stabilized: bool` (§2.1a).
- `schema` nodes: `component_basins` (links to basin nodes), `component_edges` (the relational-edge pattern that defines it), `stabilized: bool` (§2.1b).

**Edge — common fields:**
- `id` — unique identifier (edges are individually addressable, not just keyed by source/target/type — needed to distinguish repeat corroboration events over time from a single persistent edge, per §3.2's corroboration tracking).
- `source_id`, `target_id`.
- `edge_type` — `is-a` | `part-of` | `associated-with` (§2.3) | `responsible-for` | `violates` | `temporal-contrast` | `concerns-other` (§2.1b).
- `created_at`, `source` — same provenance tracking as nodes (dictionary/user/self_generated); edges also feed the diversity signal.
- `placement_method` — `"parsed"` | `"co-occurrence"` (§2.3) — determines re-parenting eligibility at Consolidation.

---

## 7. Module Responsibilities (Updated)

| Module | Layer | Responsibility |
|---|---|---|
| `core.py` | Hidden | Raw somatic variables (heart_rate, respiration_rate, dopaminergic_tone — fast layer; cortisol_load, vascular_constriction — slow layer; muscle_tension — fast layer, feeds dominance axis, §2.1a). No semantic meaning. **Known overload**: dopaminergic_tone currently feeds valence, the fatigue composite, and the self-study reward bump (§5.1) simultaneously — flagged in §9 as a potential coupling artifact, not yet resolved. |
| `hormonal.py` | Hidden | Fast/slow chemical decay math, Epoch enum. **Stays pure** — does not read association.py or chronos.py directly. Receives epoch-shift instructions from prometheus.py. |
| `synthesizer.py` | Boundary | Projects core.py raw state onto composite arousal/valence/dominance (PAD) axes and looks up the current point against the stabilized basin map (§2.1a) to produce the current named Felt State. Only output the rest of the system may read. Does not modify the basin map itself — that's a Consolidation-time operation. |
| `association.py` | Visible | Grows the knowledge/schema web from felt states + dictionary/user/self-generated input. Handles hierarchy placement (dictionary-pattern parsing + co-occurrence fallback, §2.3) and typed edges, including relational edge types (`responsible-for`, `violates`, `temporal-contrast`, `concerns-other`, §2.1b). Detects explicit negation for demotion (v1). |
| `archivist.py` | Visible | Pruning: executes the concrete trigger rule (§5.2 — Provisional + N_prune unreinforced Consolidation passes → removal, with cascade edge cleanup), trust-tier bookkeeping (promotion/demotion execution during Consolidation), re-parenting evaluation for co-occurrence-placed nodes (§2.3), regulatory efficacy scoring (§4.5). |
| `chronos.py` | Visible | Time-series log of felt-state history, arousal-valence-dominance trajectory (feeds §2.1a's dwell-time histogram), and decay steps — the evaluation window for milestone/consolidation checks. Also the source of timestamps for `temporal-contrast` edges (§2.1b). |
| `reflector.py` | Visible | **Resolved (§4A, extended by §2.1b):** structural self-report (graph shape, hub nodes, tier distribution) + regulatory self-awareness (aggregates efficacy scores, §4.5) + complex-schema detection (scans for recurring basin + relational-edge co-occurrence patterns, forming Schema Nodes, §2.1b — this absorbs and gives concrete purpose to the consistency-auditing capability previously deferred). Consolidation-gated. |
| `sensory.py` | Input | Ingests dictionary tokens, user input, and self-generated idle-expansion lookups (§5.1); tags each by source for trust-weighting purposes (§3.2) and diversity-signal exclusion for self-generated content. Also detects candidate relational edges (`responsible-for`, `violates`, `temporal-contrast`, `concerns-other`, §2.1b) via the same deterministic pattern/keyword approach used for negation detection (§3.4). |
| `prometheus.py` | Orchestrator | Owns cross-layer epoch transition checks (reads both hidden and visible layers); coordinates tick sequencing, including the §4D catch-up computation on each Streamlit rerun; detects regulation triggers and selects the regulating node (§4.1–4.2); owns save/load of the persistence checkpoint at Consolidation (§4C); is the *only* module permitted to see across the full hidden/visible boundary. |
| `app.py` | Interface | Streamlit entry point. Hosts the tabbed layout: Graph / State / Reflection / Debug (§4B). |
| `prometheus_dashboard.py` | Interface | Pyvis graph rendering for the Graph tab. **Task 1 fix**: use in-memory `generate_html()` instead of `net.show()`, which can silently fail to render on read-only/sandboxed filesystems or race with Streamlit's rerun cycle. Node/edge styling pulls tier and edge-type data from archivist.py at render time. |

---

## 8. Immediate Development Tasks (from original spec)
- [ ] **Task 1**: Refactor `prometheus_dashboard.py` to render Pyvis via in-memory `generate_html()`, per the tabbed UI design in §4B.
- [ ] **Task 2**: Define the Epoch enumeration and transition checkers — now clarified to live primarily in `prometheus.py` (cross-layer check) with the enum itself in `hormonal.py`.
- [ ] **Task 3 (new)**: Build the tabbed Streamlit layout (§4B) — Graph / State / Reflection / Debug — as the container for Task 1's dashboard.

---

## 9. Known Risks (carried forward, not yet mitigated in code)
1. **Runaway feedback loops** — felt states drive graph growth; without an explicit dampening term in the cross-link (§4), Adolescence risks getting stuck in permanent turmoil rather than resolving toward Maturity. **Second instance**: the self-study loop (§5.1) — salience picks a node, self-study produces hormonal signal, signal shifts future salience. Mitigated in design by scaled-down self-study hormone magnitude, but decay-rate-vs-delta-size balance is unverified until the system actually runs.
2. **Unbounded slow-layer drift** — cortisol/baseline variables need floors/decay tuned so a bad stretch doesn't permanently bake in a dysregulated temperament before Maturity's stabilization logic can apply.
3. **Layer boundary is a convention, not enforced** — Python won't stop `association.py` (or, more critically, `prometheus.py`) from importing `core.py` directly. No lint rule or runtime guard yet. This is the **highest-stakes risk in the design**, not a minor code-hygiene issue: per the Core Emergence Principle (§ Project Concept), any accidental path from core.py's raw values into the agent's own decision logic — even just prometheus.py conditioning a choice on cortisol_load directly instead of a synthesized felt state — undermines the central premise that felt states and behavior are *inferred*, not read off a variable. Real risk given multiple LLM tools writing different modules across sessions, since none of them can be assumed to preserve an unenforced convention. Worth a runtime guard (e.g., core.py's SomaticCore refusing to be imported outside an allowlist of hidden-layer modules) once implementation starts, not just a docstring warning.
4. **Cross-tool drift** — the largest practical risk identified in this whole design process. Mitigate by: (a) keeping this doc updated as the single source of truth, (b) pasting relevant sections into new sessions rather than relying on model memory, (c) keeping design and implementation conversations separate — implementation sessions should receive a finalized spec, not co-develop one.
5. **Self-generated trust gaming** — without the distinct tagging described in §2.2, an agent could inflate its own node's trust/diversity signals by repeatedly self-expanding around it. Mitigated in design (self-generated edges excluded from diversity signal) but needs enforcement at the archivist.py trust-scoring step, not just at creation.
6. **Childhood cold-start / basin sparsity** — since felt states are now emergent basins (§2.1a) rather than fixed categories, a young agent may have a genuinely flat, unformed landscape for a long stretch, with no basin stable enough to name. This is intended (real growth, not instant classification) but means Childhood could last far longer or shorter than expected depending on how much the agent's trajectory naturally revisits similar regions — worth watching once the vertical slice is running, since this directly determines how long Childhood *feels* in practice, not just architecturally.
7. **dopaminergic_tone overload** — this single variable currently feeds the valence axis (§2.1a), the fatigue composite (§5), and the self-study reward bump (§5.1) simultaneously. This risks an artifact where every self-study action mechanically nudges valence positive regardless of whether the expansion was meaningful, entangling "feeling good" with "just self-studied" rather than these being independently earned. Not yet resolved — worth revisiting once the vertical slice makes the entanglement visible in practice, or deciding it's an acceptable, even thematically appropriate, coupling (self-study genuinely is mildly pleasant by definition).

---

## 10. Open Questions Requiring Decisions Before Further Coding
1. ~~Two-web cross-link mechanics~~ — **Resolved (§4): regulation via accelerated fast-layer decay, capped by tier-restricted efficacy score, fatigue-costed, evaluated at Consolidation.**
2. ~~`reflector.py`'s actual job~~ — **Resolved (§4A): structural self-report + regulatory self-awareness + complex-schema detection (§2.1b), Consolidation-gated. Narrow node-ambiguity auditing still deferred to v2.**
3. ~~Trust-tier axis vs. abstraction-depth axis~~ — **Resolved (§2.4): orthogonal, tracked independently.**
4. Fatigue composite formula and T1/T2 thresholds — need concrete values or a tuning plan. **Now the highest-priority remaining item — all major architecture is resolved; what remains is numeric tuning.**
5. ~~Adolescence → Maturity gate~~ — **Resolved (§2.1b, §6.2): Schema Node formation, not variance-based.** Remaining: exact count/threshold of Schema Nodes required (folded into item 14 below).
6. Hysteresis window sizes (N consolidation passes) for both state-cycling and trust promotion/demotion — not yet numeric.
7. Self-study hormone delta magnitude and decay rate (§5.1) — needs actual numbers, likely only tunable empirically once a vertical slice is running.
8. Regulation dampening cap and its epoch-scaling curve (§4.4), and initial/default regulatory efficacy value for a newly-eligible node (§4.5) — not yet numeric.
9. Fatigue level abstraction thresholds for the State tab's low/medium/high indicator (§4B) — needs to map to the same T1/T2 bands as §5, not a separate scale.
10. Composite axis formulas (arousal/valence, §2.1a) — the general shape (heart_rate+respiration_rate; dopaminergic_tone−cortisol_load) is set, but exact weighting/normalization is not.
11. Dwell-time histogram grid resolution and basin stabilization threshold (revisit count/duration, §2.1a) — not yet numeric, same tuning category as fatigue thresholds.
12. Basin decay rate (§2.1a point 5) for non-reinforced felt states — not yet numeric.
13. Schema Node stabilization threshold (§2.1b) — how many recurrences of a basin+relational-edge-pattern co-occurrence before it counts as a stable schema — not yet numeric, same tuning category as basin stabilization.
14. Number/threshold of Schema Nodes required for the Adolescence → Maturity gate (§6.2) — not yet numeric.
15. sensory.py's pattern/keyword rules for detecting `responsible-for`/`violates`/`temporal-contrast`/`concerns-other` candidates (§2.1b) — approach is set (deterministic, same style as negation detection) but the actual rule set isn't written yet.
16. ~~Persistence~~ — **Resolved (§4C): single-instance, checkpointed at Consolidation, durable state = both graphs + basin landscape + efficacy scores + epoch + slow-layer baseline only, bounded rolling chronos window, plain JSON format.**
17. ~~Tick/scheduling loop under Streamlit~~ — **Resolved (§4D): catch-up simulation on rerun, no background process for v1. Background process deferred as a deliberate v2 upgrade.**
18. ~~Node/edge data schema~~ — **Resolved (§6A): full canonical field list defined for nodes and edges, including type-specific fields for basin/schema nodes and edge IDs for corroboration-event tracking.**
19. ~~Pruning's concrete trigger~~ — **Resolved (§5.2): terminal stage of non-reinforcement decay — Provisional + N_prune unreinforced Consolidation passes → removed, with cascade edge cleanup. Epoch-scaling of N_prune remains an open numeric question, tracked jointly with T1/T2 (item 6).**
20. Real-time-to-simulated-tick mapping for the §4D catch-up computation — not yet numeric.
21. **Visual encoding for `basin`/`schema`/`self` node types on the Graph tab (§4B)** — tier-based coloring doesn't apply to these node types (§6A), and no alternative encoding has been defined yet. Cosmetic, not architectural — flagged in conversation, safe to leave for implementation time.
