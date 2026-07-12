"""
Streamlit entry point for Prometheus.
"""
import streamlit as st
import traceback

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

    tab_graph, tab_state, tab_reflection, tab_debug = st.tabs(["Graph", "State", "Reflection", "Debug"])

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
            st.error(f"Graph error: {e}")
            st.code(traceback.format_exc(), language="python")
            g = prom.archivist.graph
            st.json({"nodes": len(g.nodes()), "edges": len(g.edges())})

    with tab_state:
        st.subheader("Current State")
        try:
            st.metric("Felt State", prom.synthesizer.get_current_felt_state())
            st.metric("Nodes", prom.archivist.graph.number_of_nodes())
        except Exception as e:
            st.error(str(e))

    with tab_reflection:
        st.subheader("Reflection")
        try:
            st.json(prom.reflector.observe())
        except Exception as e:
            st.error(str(e))

    with tab_debug:
        st.subheader("Raw Debug")
        try:
            st.json(prom.bio.get_raw_variables())
        except Exception as e:
            st.error(str(e))

else:
    st.info("Click 'Start System' in the sidebar.")
