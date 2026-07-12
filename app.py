"""
Streamlit entry point (§7, §4B). Hosts the tabbed layout: Graph / State / Reflection / Debug.
"""
import streamlit as st
import traceback

# Import Prometheus
try:
    from Prometheus.Prometheus import Prometheus
except ImportError:
    from Prometheus import Prometheus

from prometheus_dashboard import render_graph_html

st.set_page_config(page_title="Prometheus", layout="wide")
st.title("Prometheus – Living Brain")

if "prom" not in st.session_state:
    st.session_state.prom = None

st.sidebar.header("Controls")
if st.sidebar.button("Start System", disabled=st.session_state.prom is not None):
    try:
        st.session_state.prom = Prometheus()
        st.sidebar.success("System started")
    except Exception as e:
        st.error(f"Failed to start: {e}")
        st.code(traceback.format_exc(), language="python")

if st.session_state.prom is not None:
    prom = st.session_state.prom

    st.sidebar.subheader("Input")
    user_text = st.sidebar.text_area("Say something to Prometheus", key="user_text", height=80)

    if st.sidebar.button("Send") and user_text.strip():
        try:
            prom.queue_input(user_text.strip(), source="user")
            st.sidebar.success("Queued for next pulse")
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
    focus = st.sidebar.text_input("Focus", "Knowledge", key="focus_input")
    intensity = st.sidebar.slider("Intensity", 0.0, 1.0, 0.7, key="intensity_slider")
    if st.sidebar.button("Trigger Event"):
        try:
            if hasattr(prom, 'stimulus'):
                prom.stimulus.trigger_internal_event(intensity, focus)
                st.sidebar.success("Event triggered")
            else:
                st.sidebar.warning("Stimulus not available")
        except Exception as e:
            st.error(f"Stimulus error: {e}")
            st.code(traceback.format_exc(), language="python")

    # Tabs
    tab_graph, tab_state, tab_reflection, tab_debug = st.tabs(["Graph", "State", "Reflection", "Debug"])

    # GRAPH TAB
    with tab_graph:
        st.subheader("Knowledge / Schema Web")
        new_node = st.text_input("New Node Name", key="new_node")
        if st.button("Add Node") and new_node:
            try:
                prom.archivist.store(new_node, source="user")
                st.success(f"Added {new_node}")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        try:
            html = render_graph_html(prom.archivist)
            st.components.v1.html(html, height=750, scrolling=True)
        except Exception as e:
            st.error(f"Graph render error: {e}")
            st.code(traceback.format_exc(), language="python")
            # Fallback
            st.subheader("Raw Graph Data")
            g = prom.archivist.graph
            st.json({
                "nodes": len(g.nodes()),
                "edges": len(g.edges()),
                "sample_nodes": list(g.nodes())[:20]
            })

    # STATE TAB
    with tab_state:
        st.subheader("Current State")
        try:
            felt = prom.synthesizer.get_current_felt_state()
            st.metric("Felt State", felt)
            st.metric("Epoch", getattr(getattr(prom, 'bio', None), 'epoch', 'N/A'))
            st.metric("Node Count", prom.archivist.graph.number_of_nodes())
        except Exception as e:
            st.error(f"State error: {e}")

    # REFLECTION TAB (simplified)
    with tab_reflection:
        st.subheader("Reflection & Schemas")
        try:
            st.write("Check console or add more pulses for schemas.")
            metrics = prom.reflector.observe()
            st.json(metrics)
        except Exception as e:
            st.error(f"Reflection error: {e}")

    # DEBUG TAB
    with tab_debug:
        st.markdown("<div style='background:#402020;padding:8px;border-radius:4px;'><b>RAW STATE (Read-only)</b></div>", unsafe_allow_html=True)
        try:
            st.json(prom.bio.get_raw_variables())
        except Exception as e:
            st.error(str(e))

else:
    st.info("Click 'Start System' to begin.")
    fatigue = getattr(prom, 'fatigue', 0)
            if fatigue < getattr(Prometheus, 'T1', 0.3):
                fatigue_level = "Low"
            elif fatigue < getattr(Prometheus, 'T2', 0.7):
                fatigue_level = "Medium"
            else:
                fatigue_level = "High"
            st.metric("Fatigue", fatigue_level)
        except Exception as e:
            st.error(f"State display error: {e}")
            st.code(traceback.format_exc())

    # REFLECTION TAB
    with tab_reflection:
        st.subheader("Self-Report & Schemas")
        try:
            metrics = prom.reflector.observe()
            st.write(f"Last updated: pulse {getattr(prom.reflector, 'pulse_count', 'N/A')}")
            st.json(metrics)

            st.subheader("Complex Emotional Schemas")
            schema_nodes = [(n, d) for n, d in prom.archivist.graph.nodes(data=True) if d.get("is_schema")]
            if not schema_nodes:
                st.caption("No stable Schema Nodes yet.")
            else:
                for n, d in schema_nodes:
                    label = d.get("name") or f"(unnamed: {n})"
                    st.write(f"**{label}** – basin: {d.get('basin')}")
        except Exception as e:
            st.error(f"Reflection error: {e}")
            st.code(traceback.format_exc())

    # DEBUG TAB
    with tab_debug:
        st.markdown(
            "<div style='background-color:#402020;padding:8px;border-radius:4px;'>"
            "<b>RAW INTERNAL STATE – NOT PART OF THE COGNITIVE MODEL</b><br>"
            "Read-only instrumentation only."
            "</div>",
            unsafe_allow_html=True,
        )
        try:
            st.subheader("Raw Somatic Variables")
            st.json(prom.bio.get_raw_variables())
            st.subheader("Hormonal State")
            st.json({k: round(v, 4) for k, v in getattr(prom.bio, '_hormones', {}).items()})
        except Exception as e:
            st.error(f"Debug data error: {e}")
            st.code(traceback.format_exc())

else:
    st.info("Click 'Start System' in the sidebar to begin.")
