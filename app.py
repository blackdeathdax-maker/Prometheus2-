"""
Streamlit entry point (§7, §4B). Hosts the tabbed layout: Graph / State / Reflection / Debug.
"""
import streamlit as st
import traceback

try:
    from Prometheus.Prometheus import Prometheus
except ImportError:
    from Prometheus import Prometheus  # fallback

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
    user_text = st.sidebar.text_area(
        "Say something to Prometheus", key="user_text", height=80
    )

    if st.sidebar.button("Send") and user_text.strip():
        try:
            prom.queue_input(user_text.strip(), source="user")  # assuming this method
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
    focus = st.sidebar.text_input("Focus", "Knowledge")
    intensity = st.sidebar.slider("Intensity", 0.0, 1.0, 0.7)
    if st.sidebar.button("Trigger Event"):
        try:
            if hasattr(prom, 'stimulus'):
                prom.stimulus.trigger_internal_event(intensity, focus)
                st.sidebar.success("Event triggered")
            else:
                st.sidebar.warning("Stimulus module not available")
        except Exception as e:
            st.error(f"Stimulus error: {e}")
            st.code(traceback.format_exc(), language="python")

    # Tabbed layout
    tab_graph, tab_state, tab_reflection, tab_debug = st.tabs(
        ["Graph", "State", "Reflection", "Debug"]
    )

    with tab_graph:
        st.subheader("Knowledge / Schema Web")
        if prom is None:
            st.info("Start the system first.")
        else:
            new_node = st.text_input("New Node Name", key="new_node")
            if st.button("Add Node") and new_node:
                try:
                    prom.archivist.store(new_node, source="user")
                    st.success(f"Added {new_node}")
                except Exception as e:
                    st.error(f"Add node error: {e}")

            try:
                html = render_graph_html(prom.archivist)
                st.html(html)
            except Exception as e:
                st.error(f"Graph render error: {e}")
                st.code(traceback.format_exc())

    with tab_state:
        st.subheader("Current State")
        try:
            felt_state = prom.synthesizer.get_current_felt_state()
            st.metric("Felt State", felt_state)
            st.metric("Epoch", getattr(prom, 'bio', type('obj', (), {'epoch': type('e', (), {'value': 'N/A'})})()).epoch.value)
            # ... add other metrics with try/except as needed
        except Exception as e:
            st.error(f"State error: {e}")

    # (Reflection and Debug tabs can stay mostly as-is, but wrap heavy calls similarly)

    with tab_reflection:
        st.subheader("Self-Report")
        try:
            metrics = prom.reflector.observe()
            st.write(f"Last updated: pulse {getattr(prom.reflector, 'pulse_count', 'N/A')}")
            st.json(metrics)
            # ... rest of reflection
        except Exception as e:
            st.error(f"Reflection error: {e}")
            st.code(traceback.format_exc())

    with tab_debug:
        st.markdown("""<div style='background-color:#402020;padding:8px;border-radius:4px;'>
        <b>RAW INTERNAL STATE – NOT PART OF THE COGNITIVE MODEL</b>
        </div>""", unsafe_allow_html=True)
        try:
            st.json(prom.bio.get_raw_variables())
            st.json({k: round(v, 4) for k, v in getattr(prom.bio, '_hormones', {}).items()})
        except Exception as e:
            st.error(f"Debug error: {e}")

else:
    st.info("Click 'Start System' in the sidebar to begin.")
