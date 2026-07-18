"""
Streamlit entry point (§7, §4B). Hosts the tabbed layout: Graph / State / Reflection / Debug (Task 3).
"""
import streamlit as st
from Prometheus.Prometheus import Prometheus
from prometheus_dashboard import render_graph_html

st.set_page_config(page_title="Prometheus", layout="wide")
st.title("Prometheus – Living Brain")

if "prom" not in st.session_state:
    st.session_state.prom = None

st.sidebar.header("Controls")
if st.sidebar.button("Start System", disabled=st.session_state.prom is not None):
    st.session_state.prom = Prometheus()
    st.sidebar.success("System started")

with st.sidebar.expander("Reset Persistent Memory"):
    st.caption(
        "Deletes every on-disk checkpoint (§4C): the knowledge graph, "
        "chronos's rolling log, hormonal's slow-layer baseline + epoch, "
        "and the basin/schema landscape. Cannot be undone."
    )
    confirm_reset = st.checkbox("I understand this permanently erases all memory", key="confirm_reset")
    if st.button("Reset Memory", disabled=not confirm_reset):
        removed = Prometheus.reset_persistent_memory()
        st.session_state.prom = None  # discard the live instance -- __init__ only
                                       # loads from disk once, at creation, so the
                                       # old in-memory state would otherwise survive
                                       # even after the files on disk are gone.
        st.success(f"Memory reset. Removed {len(removed)} file(s). Click 'Start System' to begin fresh.")

if st.session_state.prom is not None:
    prom = st.session_state.prom
    st.sidebar.subheader("Input")
    user_text = st.sidebar.text_area(
        "Say something to Prometheus", key="user_text", height=80
    )
    if st.sidebar.button("Send") and user_text.strip():
        prom.queue_input(user_text.strip(), source="user")
        st.sidebar.success("Queued for next pulse")

    if st.sidebar.button("Pulse"):
        prom.pulse()

    st.sidebar.caption(
        "Not §4D's real-time catch-up (still unimplemented, §10.20) -- "
        "a simple stopgap for running many ticks at once, e.g. to watch "
        "fatigue cycling or a slider change play out without clicking "
        "Pulse by hand."
    )
    batch_n = st.sidebar.number_input("Run N pulses", min_value=1, max_value=2000, value=50, step=10)
    if st.sidebar.button("Run Batch"):
        progress = st.sidebar.progress(0.0)
        status = st.sidebar.empty()
        for i in range(int(batch_n)):
            prom.pulse()
            frac = (i + 1) / batch_n
            progress.progress(frac)
            status.caption(
                f"Pulse {prom.pulse_count} -- state: {prom.state}, "
                f"fatigue: {prom.fatigue:.3f}, felt: {prom.synthesizer.get_current_felt_state()}"
            )
        st.sidebar.success(f"Ran {int(batch_n)} pulses.")

    st.sidebar.subheader("Stimulus")
    focus = st.sidebar.text_input("Focus", "Knowledge")
    intensity = st.sidebar.slider("Intensity", 0.0, 1.0, 0.7)
    if st.sidebar.button("Trigger Event"):
        prom.stimulus.trigger_internal_event(intensity, focus)
        st.sidebar.success("Event triggered")

    # Tabbed layout per §4B: Graph / State / Reflection / Debug
    tab_graph, tab_state, tab_reflection, tab_debug = st.tabs(
        ["Graph", "State", "Reflection", "Debug"]
    )

    # ================================================================
    # TAB: GRAPH -- Knowledge/Schema Web (Pyvis)
    # ================================================================
    with tab_graph:
        st.subheader("Knowledge / Schema Web")
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            new_node = st.text_input("New Node Name", key="new_node")
            if st.button("Add Node", key="add_btn") and new_node:
                prom.archivist.store(new_node, source="user")
                st.success(f"Added {new_node}")

            # Focused rendering (§11 pull-forward, this revision): default
            # view shows only the top-activation working-memory
            # neighborhood, not the entire live graph -- the actual fix
            # for §11's rendering-cost/readability problem at scale
            # (rendering everything doesn't scale past a few hundred
            # nodes, and heavy multi-parenting makes it unreadable
            # regardless of tuning). Full graph remains available as an
            # explicit opt-in for anyone who wants the complete picture.
            show_full = st.checkbox(
                "Show full graph (may be slow/unreadable at scale, §11)",
                value=False, key="show_full_graph",
            )
            focus_size = st.slider(
                "Focus size (top-K active nodes)", 10, 200,
                value=prom.WORKING_MEMORY_DEFAULT_SIZE, step=5,
                key="graph_focus_size", disabled=show_full,
            )

            if show_full:
                node_subset = None
            else:
                # Always include the current felt-state's anchors too, so
                # the focused view stays coherent with what's actually
                # driving behavior right now, not just historically
                # high-activation nodes that may no longer be relevant.
                key = prom.synthesizer.get_current_basin_key()
                current_anchors = prom.felt_state_anchors.get(key, [])
                node_subset = prom.archivist.working_memory_nodes(
                    top_k=focus_size, always_include=current_anchors,
                )
                st.caption(f"Showing {len(node_subset)} of {prom.archivist.graph.number_of_nodes()} nodes.")

            # In-memory generate_html() per the Task 1 fix -- no filesystem
            # write, so this can't silently fail or race across sessions.
            html = render_graph_html(prom.archivist, node_subset=node_subset)
            st.components.v1.html(html, height=700)

    # ================================================================
    # TAB: STATE -- Current felt state and epoch (§4B)
    # ================================================================
    with tab_state:
        st.subheader("Current State")
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            felt_state = prom.synthesizer.get_current_felt_state()
            st.metric("Felt State", felt_state)
            st.metric("Epoch", prom.bio.epoch.value)
            st.metric("Operating Mode", prom.state)

            # Fatigue shown as an abstracted level, not a raw number (§4B).
            if prom.fatigue < Prometheus.T1:
                fatigue_level = "Low"
            elif prom.fatigue < Prometheus.T2:
                fatigue_level = "Medium"
            else:
                fatigue_level = "High"
            st.metric("Fatigue", fatigue_level)

            # No visible progress meter toward the next epoch transition, per
            # §4B: showing one would turn an earned milestone into a bar to
            # min-max. Intentionally omitted, not an oversight.

    # ================================================================
    # TAB: REFLECTION -- Structural self-report + regulatory awareness
    # ================================================================
    with tab_reflection:
        st.subheader("Self-Report")
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            metrics = prom.reflector.observe()
            st.write(f"Last updated: pulse {prom.reflector.pulse_count}")
            st.json(metrics)

            st.subheader("Regulatory Self-Awareness (§4.5 / §4A)")
            reg_report = prom.reflector.regulatory_self_report()
            st.write(
                f"Regulation-capable nodes: {reg_report['regulation_capable_count']}"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Most effective")
                st.table(reg_report["most_effective"])
            with col2:
                st.caption("Least effective")
                st.table(reg_report["least_effective"])

            st.subheader("Complex Emotional Schemas (§2.1b)")
            schema_nodes = [
                (n, d)
                for n, d in prom.archivist.graph.nodes(data=True)
                if d.get("is_schema")
            ]
            if not schema_nodes:
                st.caption("No stable Schema Nodes formed yet.")
            else:
                for n, d in schema_nodes:
                    label = d.get("name") or f"(unnamed: {n})"
                    st.write(
                        f"**{label}** – basin: `{d.get('basin')}`, "
                        f"relations: {', '.join(d.get('relation_types', []))}"
                    )
                    if not d.get("named"):
                        word = st.text_input(
                            f"Give this pattern a name", key=f"name_{n}"
                        )
                        if st.button("Name it", key=f"btn_{n}") and word.strip():
                            prom.reflector.name_schema(n, word.strip())
                            st.success(f"Named {n} -> {word.strip()}")

            with st.expander("Schema formation progress (diagnostic)"):
                st.caption(
                    "Read-only view into why schemas do/don't exist yet, since "
                    "'no stable Schema Nodes' alone doesn't say whether the "
                    "system is close or nowhere near. A schema requires "
                    "relational edges (responsible-for/violates/temporal-"
                    "contrast/concerns-other) from typed text matching specific "
                    "keyword patterns -- self-study alone can never produce "
                    "these, only Send does."
                )
                candidate_report = prom.reflector.schema_candidate_report()
                st.metric(
                    "Relational-edge-bearing event nodes",
                    candidate_report["total_relational_event_nodes"],
                )
                st.metric(
                    "Dropped (occurred before any felt state had stabilized)",
                    candidate_report["dropped_unformed_felt_state"],
                    help=(
                        "A relational edge created while the current felt "
                        "state was still 'Unformed' is permanently excluded "
                        "from schema candidacy -- not retried later."
                    ),
                )
                if candidate_report["candidate_pairs"]:
                    st.write("Closest candidate patterns to stabilizing:")
                    for c in candidate_report["candidate_pairs"]:
                        st.write(
                            f"- felt state `{c['felt_state']}` + "
                            f"{', '.join(c['relation_types'])} "
                            f"— {c['count']}/{c['threshold']} occurrences "
                            f"({c['remaining']} more needed)"
                        )
                else:
                    st.caption(
                        "No candidate (felt_state, relation-type) pairs yet -- "
                        "either no relational edges exist, or all of them "
                        "occurred before any felt state had stabilized. Try "
                        "sending a few messages like \"I shouldn't have done "
                        "that\" or \"that was my fault\" while the system is "
                        "in the same felt state (check the State tab), "
                        "repeated 3+ times."
                    )

    # ================================================================
    # TAB: DEBUG -- Raw internal state (§4B, one sanctioned exception)
    # ================================================================
    with tab_debug:
        st.markdown(
            "<div style='background-color:#402020;padding:8px;border-radius:4px;'>"
            "<b>RAW INTERNAL STATE – NOT PART OF THE COGNITIVE MODEL</b><br>"
            "This tab is a read-only instrumentation panel. Nothing shown here ever "
            "feeds back into agent logic (Core Emergence Principle)."
            "</div>",
            unsafe_allow_html=True,
        )
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            st.subheader("Raw Somatic Variables (§2.1a, §7)")
            st.json(prom.bio.get_raw_variables())

            st.subheader("Hormonal State")
            st.json({k: round(v, 4) for k, v in prom.bio._hormones.items()})

            st.caption(
                f"Current basin key (arousal, valence, dominance): "
                f"{prom.synthesizer.get_current_basin_key()}"
            )
            st.caption(
                f"Stabilized basins: {len(prom.synthesizer.stabilized_basins)}"
            )

            st.divider()
            st.markdown(
                "<div style='background-color:#402020;padding:8px;border-radius:4px;'>"
                "<b>LIVE TUNING</b><br>"
                "These sliders mutate the running instance's constants directly -- "
                "no restart needed. Every value here is still an undecided "
                "placeholder per the design spec (§10); this panel exists so "
                "they can be tuned empirically instead of guessed in code."
                "</div>",
                unsafe_allow_html=True,
            )

            with st.expander("Fatigue / State Cycling (§5)"):
                prom.T1 = st.slider("T1 (Learning \u2192 Consolidation)", 0.0, 1.0, value=prom.T1, step=0.01)
                prom.T2 = st.slider("T2 (Consolidation \u2192 Pruning)", 0.0, 1.0, value=prom.T2, step=0.01)
                prom.HYSTERESIS = st.slider("Hysteresis margin", 0.0, 0.3, value=prom.HYSTERESIS, step=0.01)
                prom.FATIGUE_GROWTH_RATE = st.slider(
                    "Fatigue growth rate (\u00d7 urgency, per tick)", 0.0, 1.0,
                    value=prom.FATIGUE_GROWTH_RATE, step=0.01,
                )
                prom.FATIGUE_RECOVERY_CONSOLIDATION = st.slider(
                    "Consolidation recovery (fraction retained)", 0.0, 1.0,
                    value=prom.FATIGUE_RECOVERY_CONSOLIDATION, step=0.05,
                )
                prom.FATIGUE_RECOVERY_PRUNING = st.slider(
                    "Pruning recovery (fraction retained)", 0.0, 1.0,
                    value=prom.FATIGUE_RECOVERY_PRUNING, step=0.05,
                )
                prom.bio.HORMONE_DECAY_RATE = st.slider(
                    "Hormone decay rate (toward 0.5 baseline, per tick)", 0.0, 1.0,
                    value=prom.bio.HORMONE_DECAY_RATE, step=0.01,
                )

            with st.expander("Regulation (§4)"):
                prom.REGULATION_SPIKE_THRESHOLD = st.slider(
                    "Spike threshold (intensity)", 0.0, 1.0,
                    value=prom.REGULATION_SPIKE_THRESHOLD, step=0.01,
                )
                prom.REGULATION_HYSTERESIS = st.slider(
                    "Regulation hysteresis margin", 0.0, 0.3,
                    value=prom.REGULATION_HYSTERESIS, step=0.01,
                )
                prom.REGULATION_DAMPENING_CAP = st.slider(
                    "Dampening cap", 0.0, 1.0, value=prom.REGULATION_DAMPENING_CAP, step=0.01,
                )
                prom.REGULATION_FATIGUE_COST = st.slider(
                    "Fatigue cost per regulation attempt", 0.0, 0.5,
                    value=prom.REGULATION_FATIGUE_COST, step=0.01,
                )

            with st.expander("Self-Study (§5.1)"):
                prom.SELF_STUDY_DOPAMINE_BUMP = st.slider(
                    "Dopamine bump per self-study expansion", 0.0, 0.3,
                    value=prom.SELF_STUDY_DOPAMINE_BUMP, step=0.01,
                )

            with st.expander("Hormonal Reaction to Input (new, this revision)"):
                st.caption(
                    "Previously, real conversational input produced ZERO "
                    "hormonal response -- only self-study's faint trickle "
                    "and manual Stimulus events moved anything. This is "
                    "the fix: deterministic, rule-based reaction to real "
                    "input, keyed off message length and detected "
                    "relational/negation signals (no NLP/sentiment model)."
                )
                prom.ENGAGEMENT_DOPAMINE_BUMP = st.slider(
                    "Base engagement dopamine bump (per message)", 0.0, 0.3,
                    value=prom.ENGAGEMENT_DOPAMINE_BUMP, step=0.01,
                )
                prom.ENGAGEMENT_AROUSAL_SCALE = st.slider(
                    "Arousal scale (by message length, capped)", 0.0, 0.3,
                    value=prom.ENGAGEMENT_AROUSAL_SCALE, step=0.01,
                )
                prom.RELATIONAL_CORTISOL_BUMP = st.slider(
                    "Cortisol bump: violates / responsible-for", 0.0, 0.3,
                    value=prom.RELATIONAL_CORTISOL_BUMP, step=0.01,
                )
                prom.RELATIONAL_AROUSAL_BUMP = st.slider(
                    "Arousal bump: concerns-other", 0.0, 0.3,
                    value=prom.RELATIONAL_AROUSAL_BUMP, step=0.01,
                )
                prom.TEMPORAL_CONTRAST_DOPAMINE_DELTA = st.slider(
                    "Dopamine shift: temporal-contrast", 0.0, 0.3,
                    value=prom.TEMPORAL_CONTRAST_DOPAMINE_DELTA, step=0.01,
                )
                prom.NEGATION_CORTISOL_BUMP = st.slider(
                    "Cortisol bump: explicit negation/correction", 0.0, 0.3,
                    value=prom.NEGATION_CORTISOL_BUMP, step=0.01,
                )

            with st.expander("Activation / Working Memory (§11 pull-forward, new)"):
                prom.archivist.ACTIVATION_BOOST = st.slider(
                    "Activation boost per real-input touch", 0.0, 5.0,
                    value=prom.archivist.ACTIVATION_BOOST, step=0.1,
                )
                prom.ACTIVATION_BOOST_SELF_STUDY = st.slider(
                    "Activation boost per self-study touch", 0.0, 5.0,
                    value=prom.ACTIVATION_BOOST_SELF_STUDY, step=0.1,
                )
                prom.archivist.ACTIVATION_DECAY_RATE = st.slider(
                    "Activation decay rate (retained per Consolidation)", 0.0, 1.0,
                    value=prom.archivist.ACTIVATION_DECAY_RATE, step=0.05,
                )
                prom.archivist.ACTIVATION_CAP = st.slider(
                    "Activation cap (per node)", 1.0, 30.0,
                    value=prom.archivist.ACTIVATION_CAP, step=0.5,
                )
                prom.WORKING_MEMORY_DEFAULT_SIZE = st.slider(
                    "Graph tab default focus size (top-K)", 10, 200,
                    value=prom.WORKING_MEMORY_DEFAULT_SIZE, step=5,
                )

            with st.expander("Trust Tiers (§3)"):
                prom.archivist.WORKING_THRESHOLD = st.slider(
                    "Working-tier score threshold", 0.0, 3.0,
                    value=prom.archivist.WORKING_THRESHOLD, step=0.05,
                )
                prom.archivist.TRUSTED_THRESHOLD = st.slider(
                    "Trusted-tier score threshold", 0.0, 3.0,
                    value=prom.archivist.TRUSTED_THRESHOLD, step=0.05,
                )
                prom.archivist.DIVERSITY_WEIGHT = st.slider(
                    "Diversity weight", 0.0, 1.0, value=prom.archivist.DIVERSITY_WEIGHT, step=0.01,
                )
                prom.archivist.EDGE_COUNT_WEIGHT = st.slider(
                    "Edge-count weight", 0.0, 0.5, value=prom.archivist.EDGE_COUNT_WEIGHT, step=0.01,
                )
                prom.archivist.EDGE_COUNT_CAP = st.slider(
                    "Edge-count cap", 1, 30, value=prom.archivist.EDGE_COUNT_CAP, step=1,
                )
                prom.archivist.PROMOTION_HYSTERESIS_N = st.slider(
                    "Promotion hysteresis (consecutive passes)", 1, 10,
                    value=prom.archivist.PROMOTION_HYSTERESIS_N, step=1,
                )
                prom.archivist.DEMOTION_HYSTERESIS_N = st.slider(
                    "Demotion hysteresis (consecutive passes)", 1, 10,
                    value=prom.archivist.DEMOTION_HYSTERESIS_N, step=1,
                )
                prom.archivist.PRUNE_TIER0_CYCLES = st.slider(
                    "Pruning: Tier-0 cycles before eligible (§5.2)", 1, 30,
                    value=prom.archivist.PRUNE_TIER0_CYCLES, step=1,
                )

            with st.expander("Basin Formation (§2.1a)"):
                prom.synthesizer.STABILIZATION_THRESHOLD = st.slider(
                    "Basin stabilization threshold (revisit count)", 1, 20,
                    value=prom.synthesizer.STABILIZATION_THRESHOLD, step=1,
                )
                prom.synthesizer.DECAY_RATE = st.slider(
                    "Basin decay rate (retained per Consolidation)", 0.0, 1.0,
                    value=prom.synthesizer.DECAY_RATE, step=0.01,
                )
                prom.synthesizer.DESTABILIZATION_FLOOR = st.slider(
                    "Destabilization floor (density)", 0.0, 5.0,
                    value=prom.synthesizer.DESTABILIZATION_FLOOR, step=0.1,
                )

            with st.expander("Schema Formation (§2.1b)"):
                prom.reflector.SCHEMA_STABILIZATION_THRESHOLD = st.slider(
                    "Schema stabilization threshold (co-occurrence count)", 1, 20,
                    value=prom.reflector.SCHEMA_STABILIZATION_THRESHOLD, step=1,
                )

            with st.expander("Epoch Gates (§6)"):
                prom.NAMING_WINDOW = st.slider(
                    "Childhood: naming reliability window (N occurrences)", 5, 100,
                    value=prom.NAMING_WINDOW, step=1,
                )
                prom.NAMING_MIN_OCCURRENCES = st.slider(
                    "Childhood: minimum-occurrence floor", 1, 30,
                    value=prom.NAMING_MIN_OCCURRENCES, step=1,
                )
                prom.NAMING_CONSISTENCY_THRESHOLD = st.slider(
                    "Childhood: consistency threshold", 0.0, 1.0,
                    value=prom.NAMING_CONSISTENCY_THRESHOLD, step=0.05,
                )
                prom.SCHEMA_NODES_REQUIRED_FOR_MATURITY = st.slider(
                    "Adolescence \u2192 Maturity: Schema Nodes required", 1, 20,
                    value=prom.SCHEMA_NODES_REQUIRED_FOR_MATURITY, step=1,
                )
else:
    st.info("Click 'Start System' in the sidebar to begin.")
