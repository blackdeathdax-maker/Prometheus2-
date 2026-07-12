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

if st.session_state.prom is not None:
    prom = st.session_state.prom
    st.sidebar.subheader("Input")
    user_text = st.sidebar.text_area(
        "Say something to Prometheus", key="user_text", height=80
    )
    if st.sidebar.button("Send") and user_text.strip():
        try:
        st.sidebar.success("Queued for next pulse")
        # Existing call
        prom_queue_input(user_text.strip(), source="user")
    except Exception as e:
        st.error(f"Send error: {e}")
        st.code(traceback.format_exc(), language="python")
    if st.sidebar.button("Pulse"):
    try:
        prom.pulse()
        st.sidebar.success("Pulse completed")
    except Exception as e:
        st.error(f"Pulse error: {e}")
        st.code(traceback.format_exc(), language="python")
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

            # In-memory generate_html() per the Task 1 fix -- no filesystem
            # write, so this can't silently fail or race across sessions.
            html = render_graph_html(prom.archivist)
            st.html(html)   # or st.iframe if it's an external src
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
else:
    st.info("Click 'Start System' in the sidebar to begin.")
