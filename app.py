"""
Streamlit entry point (§7, §4B). Hosts the tabbed layout: Graph / State / Reflection / Debug (Task 3).
"""
import time
import streamlit as st
from Prometheus.Prometheus import Prometheus
from Prometheus.archivist import SELF_NODE
from prometheus_dashboard import render_graph_html

st.set_page_config(page_title="Prometheus", layout="wide")
st.title("Prometheus – Living Brain")

if "prom" not in st.session_state:
    st.session_state.prom = None
if "semi_live" not in st.session_state:
    st.session_state.semi_live = False
if "last_auto_pulse_t" not in st.session_state:
    st.session_state.last_auto_pulse_t = 0.0
if "semi_live_pause_until" not in st.session_state:
    st.session_state.semi_live_pause_until = 0.0

st.sidebar.header("Controls")
if st.sidebar.button("Start System", disabled=st.session_state.prom is not None):
    st.session_state.prom = Prometheus()
    st.sidebar.success("System started")

with st.sidebar.expander("Reset Persistent Memory (intentional wipe only)"):
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
        # Give the next Learning pulse a chance to ingest before auto continues
        if st.session_state.get("semi_live_pause_on_send", True):
            st.session_state.semi_live_pause_until = time.time() + max(
                2.0, float(st.session_state.get("semi_live_interval", 3.0))
            )

    if st.sidebar.button("Pulse"):
        prom.pulse()

    # ------------------------------------------------------------------
    # Semi-live: auto-pulse while the browser session is open (§4D lite).
    # Not a headless daemon — closing the tab or toggling off stops the clock.
    # ------------------------------------------------------------------
    st.sidebar.subheader("Semi-live")
    st.sidebar.caption(
        "Auto-advances pulse() on a timer while this page is open. "
        "Pause/resume anytime. Not unsupervised 24/7 hosting."
    )
    semi = st.sidebar.toggle(
        "Auto-pulse",
        value=st.session_state.semi_live,
        key="semi_live_toggle",
    )
    st.session_state.semi_live = bool(semi)
    interval_s = st.sidebar.slider(
        "Interval (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
        key="semi_live_interval",
    )
    pulses_per_tick = st.sidebar.number_input(
        "Pulses per tick",
        min_value=1,
        max_value=25,
        value=1,
        step=1,
        key="semi_live_ppt",
    )
    pause_on_send = st.sidebar.checkbox(
        "Pause briefly after Send",
        value=True,
        key="semi_live_pause_on_send",
    )

    # Drive the clock: prefer streamlit-autorefresh; else timed st.rerun fallback.
    if st.session_state.semi_live:
        has_autorefresh = False
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(
                interval=int(float(interval_s) * 1000),
                key="semi_live_autorefresh",
            )
            has_autorefresh = True
        except ImportError:
            has_autorefresh = False

        now = time.time()
        paused = now < float(st.session_state.semi_live_pause_until)
        if not paused and (now - float(st.session_state.last_auto_pulse_t)) >= float(interval_s):
            n = int(pulses_per_tick)
            for _ in range(n):
                prom.pulse()
            st.session_state.last_auto_pulse_t = now
        st.sidebar.caption(
            f"{'⏸ Paused' if paused else '▶ Running'} · "
            f"pulse {prom.pulse_count} · {prom.state} · "
            f"pressure {prom.fatigue:.3f} {prom.state}"
        )
        if not has_autorefresh:
            # Heavier on the server; install streamlit-autorefresh when possible.
            time.sleep(min(0.5, float(interval_s)))
            st.rerun()

    st.sidebar.caption(
        "Batch run remains available for fast lab soaks. "
        "Prefer semi-live for continuous-time behavior."
    )
    st.sidebar.caption(
        "Not §4D's full real-time catch-up -- semi-live is the intentional "
        "middle step (timer while the session is open)."
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

    st.sidebar.subheader("Parental Feedback (§13.2)")
    st.sidebar.caption(
        "Implicit guidance, 'mirror neuron' style -- colors whatever the "
        "system is currently anchored to (check State tab's Felt State) "
        "rather than naming or asserting anything directly."
    )
    pf_col1, pf_col2 = st.sidebar.columns(2)
    with pf_col1:
        if st.button("Approval", key="pf_approval"):
            result = prom.give_parental_reaction("approval")
            names = ", ".join(result.get("anchors_colored") or []) or "(none)"
            st.sidebar.success(f"Approval → {names}")
        if st.button("Warmth", key="pf_warmth"):
            result = prom.give_parental_reaction("warmth")
            names = ", ".join(result.get("anchors_colored") or []) or "(none)"
            st.sidebar.success(f"Warmth → {names}")
    with pf_col2:
        if st.button("Disapproval", key="pf_disapproval"):
            result = prom.give_parental_reaction("disapproval")
            names = ", ".join(result.get("anchors_colored") or []) or "(none)"
            st.sidebar.success(f"Disapproval → {names}")
        if st.button("Concern", key="pf_concern"):
            result = prom.give_parental_reaction("concern")
            names = ", ".join(result.get("anchors_colored") or []) or "(none)"
            st.sidebar.success(f"Concern → {names}")

    # Tabbed layout per §4B: Graph / State / Reflection / Working Memory / Debug
    tab_graph, tab_state, tab_reflection, tab_working_memory, tab_debug = st.tabs(
        ["Graph", "State", "Reflection", "Working Memory", "Debug"]
    )

    # ================================================================
    # TAB: GRAPH -- Knowledge/Schema Web (Pyvis)
    # ================================================================
    with tab_graph:
        st.subheader("Knowledge / Schema Web")
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            st.markdown("##### Search nodes & schemas")
            sq_col, lim_col = st.columns([3, 1])
            with sq_col:
                search_q = st.text_input(
                    "Query",
                    value="",
                    key="graph_search_q",
                    placeholder="substring match on id, name, definition",
                    label_visibility="collapsed",
                )
            with lim_col:
                search_lim = st.number_input("Max hits", 5, 100, 25, 5, key="graph_search_lim")

            if search_q.strip() and hasattr(prom, "search_graph"):
                hits = prom.search_graph(search_q.strip(), limit=int(search_lim))
                st.caption(f"{len(hits)} hit(s) — expand lists neighbors without drawing the full graph")
                for i, h in enumerate(hits):
                    label = h.get("name") or h["id"]
                    meta = f"{h['kind']} · tier {h['tier']} · act {h['activation']}"
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.write(f"**{label}**")
                        st.caption(f"`{h['id']}` — {meta}")
                    with c2:
                        if st.button("Pin", key=f"pin_hit_{i}"):
                            if prom.pin_search_hit(h["id"]):
                                st.session_state["graph_pin_id"] = h["id"]
                                st.success("Pinned")
                    with c3:
                        if st.button("Expand", key=f"exp_hit_{i}"):
                            st.session_state["search_expand_id"] = h["id"]
                    if st.session_state.get("search_expand_id") == h["id"] and hasattr(prom, "node_neighborhood"):
                        nb = prom.node_neighborhood(h["id"], max_each=15)
                        with st.expander(f"Neighbors of `{h['id']}` (lists only)", expanded=True):
                            st.caption(
                                f"parents {nb.get('parent_count', 0)} · "
                                f"children {nb.get('child_count', 0)} · "
                                f"related {nb.get('related_count', 0)} (showing up to 15 each)"
                            )
                            # True taxonomic is-a only
                            if nb.get("parents"):
                                st.markdown("**Parents (this is-a …)**")
                                for row in nb["parents"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm}) · {row.get('relation')}")
                            if nb.get("children"):
                                st.markdown("**Children (… is-a this)**")
                                for row in nb["children"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm}) · {row.get('relation')}")
                            # Membership / anatomy — never labeled is-a
                            if nb.get("part_of"):
                                st.markdown("**Part of (part-of / composed-of whole)**")
                                for row in nb["part_of"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm}) · {row.get('relation')}")
                            if nb.get("has_parts"):
                                st.markdown("**Has parts (composed-of / reverse part-of)**")
                                for row in nb["has_parts"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm}) · {row.get('relation')}")
                            if nb.get("member_of_schemas"):
                                st.markdown("**Member of schemas**")
                                for row in nb["member_of_schemas"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm})")
                            if nb.get("schema_members"):
                                st.markdown("**Schema members (if this is a schema)**")
                                for row in nb["schema_members"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm})")
                            if nb.get("related"):
                                st.markdown("**Other edges**")
                                for row in nb["related"]:
                                    nm = row.get("name") or row["id"]
                                    st.write(f"- `{row['id']}` ({nm}) · {row.get('relation')}")
                            if not any([
                                nb.get("parents"), nb.get("children"),
                                nb.get("part_of"), nb.get("has_parts"),
                                nb.get("member_of_schemas"), nb.get("schema_members"),
                                nb.get("related"),
                            ]):
                                st.caption("No edges yet.")
                if st.session_state.get("graph_pin_id"):
                    st.info(f"Pinned for attention: `{st.session_state['graph_pin_id']}`")
            elif search_q.strip():
                st.caption("search_graph not available on this instance — update Prometheus.py")

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
                current_anchors = list(prom.felt_state_anchors.get(key, []))
                pin = st.session_state.get("graph_pin_id")
                if pin:
                    current_anchors.append(pin)
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
            st.metric(
                "Bias", prom.executive.current_bias,
                help=(
                    "§13.1: now drives self-study targeting, not just logged. "
                    "EXPLORE prefers fresher, low-activation content; "
                    "STABILIZE prefers established, high-activation content; "
                    "NEUTRAL matches the original default weighting."
                ),
            )

            # Fatigue shown as an abstracted level, not a raw number (§4B).
            soft = getattr(prom, "SLEEP_SOFT_MIN", Prometheus.T1)
            hard = getattr(prom, "SLEEP_HARD_MAX", 0.92)
            if prom.state == "Sleep":
                fatigue_level = f"Sleep ({getattr(prom, 'sleep_stage', '?')})"
            elif prom.fatigue < soft:
                fatigue_level = "Low"
            elif prom.fatigue < getattr(prom, "T2", 0.75):
                fatigue_level = "Medium"
            else:
                fatigue_level = "High / push"
            st.metric("Sleep pressure", f"{prom.fatigue:.2f} ({fatigue_level})")
            if hasattr(prom, "_last_urgency"):
                st.caption(f"Urgency {getattr(prom, '_last_urgency', 0):.2f} · debt {getattr(prom, 'sleep_debt', 0):.2f} · micro-day {getattr(prom, 'micro_day_pulse', 0)}/{getattr(prom, 'MICRO_DAY_PULSES', 60)}")

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

            st.subheader("Epistemic Schemas (§13.3, new -- Tier 1)")
            st.caption(
                "Knowledge-cluster schemas, distinct from the emotional "
                "schemas above -- formed from nodes that get touched "
                "together repeatedly (self-study cycles, real input), not "
                "from relational edges or felt state. Unnamed until a real "
                "dictionary assertion ties back to enough members (§13.3.1) "
                "-- never autonomously generated, and no manual naming "
                "control here by design (unlike somatic schemas above, "
                "naming an epistemic cluster is about hierarchy, not "
                "felt-state co-occurrence, so a free-text override wouldn't "
                "fit the same earned-naming semantics)."
            )
            epistemic_nodes = [
                (n, d) for n, d in prom.archivist.graph.nodes(data=True)
                if d.get("node_type") == "epistemic_schema"
            ]
            if not epistemic_nodes:
                st.caption("No stable Epistemic Schema Nodes formed yet.")
            else:
                for n, d in epistemic_nodes:
                    label = d.get("name") or f"(unnamed: {n})"
                    st.write(
                        f"**{label}** – tier {d.get('abstraction_level', 1)}, "
                        f"{d.get('member_count', '?')} member(s)"
                    )

            with st.expander("Epistemic cluster progress (diagnostic)"):
                st.caption(
                    "Read-only view into co-activation tracking and cluster "
                    "candidates, same 'make it checkable' pattern as the "
                    "somatic schema diagnostic above."
                )
                epi_report = prom.reflector.epistemic_schema_report()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tracked co-activation pairs", epi_report["total_co_activation_pairs"])
                with col2:
                    st.metric("Stabilized pairs", epi_report["stabilized_pairs"])
                st.metric("Schemas formed / named", f"{epi_report['schemas_formed']} / {epi_report['schemas_named']}")
                if epi_report["candidate_clusters"]:
                    st.write("Closest candidate clusters to stabilizing:")
                    for c in epi_report["candidate_clusters"]:
                        members_preview = ", ".join(c["members"])
                        st.write(
                            f"- {c['size']}/{c['threshold']} members "
                            f"({c['remaining']} more needed): {members_preview}"
                        )
                else:
                    st.caption(
                        "No candidate clusters yet -- co-activation builds "
                        "up from nodes being touched together repeatedly, "
                        "mostly through self-study cycles. Give it more "
                        "pulses, or check whether candidates already have "
                        "a dominant shared dictionary parent (in which case "
                        "clustering is correctly skipped as redundant)."
                    )

            st.subheader("Identity / SELF")
            st.caption("Body parts, felt, narrative parts, process tags — unified hub surface.")
            try:
                idr = prom.get_identity_hub_report() if hasattr(prom, "get_identity_hub_report") else {}
            except Exception as e:
                idr = {"error": str(e)}
            if idr and not idr.get("error"):
                felt = idr.get("felt") or {}
                st.caption(
                    f"felt={felt.get('label') or '—'} · "
                    f"A={felt.get('arousal')} V={felt.get('valence')} D={felt.get('dominance')}"
                )
                body = idr.get("body") or []
                if body:
                    st.write("**Body**")
                    for b in body[:8]:
                        st.write(f"- `{b.get('node')}` val={b.get('value')} · {b.get('relation')}")
                narr = idr.get("narrative_parts") or []
                if narr:
                    st.write("**Narrative parts**")
                    for n in narr[:6]:
                        st.write(f"- `{n.get('node')}` w={n.get('weight')}")
                tags = idr.get("process_tags") or []
                if tags:
                    st.write("**Process**")
                    for t in tags[:6]:
                        st.write(f"- `{t.get('node')}` ({t.get('op')})")
            else:
                st.caption("Identity hub report unavailable.")

            st.subheader("Active Thread")
            st.caption(
                "Short-horizon now: focus, goals, derived intent, body prediction error. "
                "In-memory only (persistence later with live head)."
            )
            try:
                atr = prom.get_active_thread_report() if hasattr(prom, "get_active_thread_report") else {}
            except Exception as e:
                atr = {"error": str(e)}
            if atr and not atr.get("error"):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    st.metric("Intent", atr.get("intent") or "—")
                with ac2:
                    st.metric("|body_error|", atr.get("max_abs_body_error") or 0)
                with ac3:
                    st.metric("Age (pulses)", atr.get("age") or 0)
                st.caption(
                    f"focus=`{atr.get('focus_id')}` · goals={atr.get('goal_ids')} · "
                    f"barren={atr.get('barren_focus')} · {atr.get('note')}"
                )
                if atr.get("last_ops"):
                    st.caption("last ops: " + " → ".join(str(x) for x in atr.get("last_ops")[-8:]))
            else:
                st.caption("Active Thread not available on this build.")

            st.subheader("Allostasis / Affect")
            st.caption(
                "Adaptive set-points + pain/pleasure surface (caps & escapes). "
                "Not emotion taxonomy — body channels only."
            )
            try:
                alr = prom.get_allostasis_report() if hasattr(prom, "get_allostasis_report") else {}
            except Exception as e:
                alr = {"error": str(e)}
            if alr and not alr.get("error"):
                a1, a2, a3 = st.columns(3)
                with a1:
                    st.metric("Pain", alr.get("pain") or 0)
                with a2:
                    st.metric("Pleasure", alr.get("pleasure") or 0)
                with a3:
                    st.metric("Escape", alr.get("last_escape") or "—")
                st.caption(
                    f"note={alr.get('note')} · drivers={alr.get('setpoint_drivers')} · "
                    f"pain_streak={alr.get('high_pain_streak')}"
                )
            else:
                st.caption("Allostasis not available on this build.")

            st.subheader("World stub / Act outcomes")
            st.caption(
                "Package A: tiny internal world slots + operator→body/world deltas. "
                "Thin C: outcome confidence (rises when effect matches plan)."
            )
            try:
                wr = prom.get_world_report() if hasattr(prom, "get_world_report") else {}
            except Exception as e:
                wr = {"error": str(e)}
            try:
                ar = prom.get_act_report() if hasattr(prom, "get_act_report") else {}
            except Exception as e:
                ar = {"error": str(e)}
            if wr and not wr.get("error") and wr.get("slots"):
                slots = wr.get("slots") or {}
                w1, w2, w3 = st.columns(3)
                with w1:
                    st.metric("object_near", round(float(slots.get("object_near") or 0), 3))
                with w2:
                    st.metric("obstacle", round(float(slots.get("obstacle") or 0), 3))
                with w3:
                    st.metric("goal_cue", round(float(slots.get("goal_cue") or 0), 3))
                st.caption(
                    f"last_op={wr.get('last_op') or ar.get('last_op')} · "
                    f"world_deltas={wr.get('last_deltas') or ar.get('world_deltas')}"
                )
            else:
                st.caption("World stub not available on this build (deploy world_stub.py).")
            if ar and not ar.get("error"):
                st.caption(
                    f"act last_op=`{ar.get('last_op')}` · "
                    f"body_deltas={ar.get('body_deltas')} · "
                    f"pending={ar.get('pending_body')}"
                )
                conf = ar.get("confidence_top") or []
                if conf:
                    st.write("**Outcome confidence (top)**")
                    for c in conf[:8]:
                        st.write(f"- `{c.get('key')}` → {c.get('confidence')}")
                causal = ar.get("causal") or {}
                if causal:
                    st.caption(
                        f"causal B: focus=`{causal.get('focus_id')}` → "
                        f"anchor=`{causal.get('anchor')}` · made={causal.get('made')} · "
                        f"{causal.get('note')}"
                    )
                    for t in (causal.get("targets") or [])[:6]:
                        st.write(f"- {t.get('rel')} → `{t.get('to')}` (Δ={t.get('dv')})")

            st.subheader("Plan (compositional)")
            st.caption(
                "Package D: means from high-confidence / scored causes only. "
                "Legacy low-conf edges ignored."
            )
            try:
                pr = prom.get_plan_report() if hasattr(prom, "get_plan_report") else {}
            except Exception as e:
                pr = {"error": str(e)}
            if pr and pr.get("active"):
                st.caption(
                    f"goal=`{pr.get('goal_id')}` · next=`{pr.get('next_means_id')}` · "
                    f"op={pr.get('suggested_op')} · {pr.get('note')}"
                )
                for i, s in enumerate(pr.get("steps") or []):
                    mark = "→" if i == pr.get("next_index") else "·"
                    st.write(
                        f"{mark} `{s.get('means_id')}` -{s.get('relation')}→ "
                        f"`{s.get('target_id')}` [{s.get('suggested_op')}] "
                        f"w={s.get('weight')} c={s.get('confidence')}"
                    )
            else:
                st.caption((pr or {}).get("note") or "No active plan (need goal + high-conf causes).")

            st.subheader("Schema expectations")
            st.caption("Map retrieve: members + high-conf causes under focus kind.")
            try:
                se = prom.get_schema_expectations() if hasattr(prom, "get_schema_expectations") else {}
            except Exception as e:
                se = {"error": str(e)}
            if se and not se.get("error"):
                st.caption(
                    f"root=`{se.get('root')}` · anchor=`{se.get('anchor')}` · {se.get('note')}"
                )
                for m in (se.get("members") or [])[:6]:
                    st.write(f"· member `{m.get('id')}` ({m.get('rel')})")
                for c in (se.get("causes") or [])[:6]:
                    tag = "scored" if c.get("scored") else "conf"
                    st.write(
                        f"· {c.get('rel')} `{c.get('id')}` "
                        f"c={c.get('confidence')} L={c.get('link_L')} [{tag}]"
                    )
            else:
                st.caption("No expectations for current focus.")
            # Expectation error write-back
            try:
                eer = getattr(prom, "_last_expectation_error", None) or {}
            except Exception:
                eer = {}
            if eer and eer.get("focus"):
                st.caption(
                    f"Info credit: hits={eer.get('hits')} misses={len(eer.get('misses') or [])} "
                    f"ratio={eer.get('miss_ratio')} info={eer.get('info_score')} "
                    f"reliability={eer.get('reliability')} explore={eer.get('explore_nudge')}"
                )
                if eer.get("info_credit_op"):
                    st.caption(
                        f"info→L: op={eer.get('info_credit_op')} improved={eer.get('info_improved')}"
                    )
                if eer.get("boosted"):
                    st.write("boosted: " + ", ".join(f"`{x}`" for x in (eer.get("boosted") or [])[:8]))
                if eer.get("promoted"):
                    st.write("promoted: " + ", ".join(f"`{x}`" for x in (eer.get("promoted") or [])[:8]))
                if eer.get("demoted"):
                    st.write("demoted: " + ", ".join(f"`{x}`" for x in (eer.get("demoted") or [])[:8]))

            st.subheader("Episodes (F)")
            st.caption(
                "Minimal act traces (lemma|op). Consolidation replays consistent successes → mild L re-credit."
            )
            try:
                er = prom.get_episode_report() if hasattr(prom, "get_episode_report") else {}
            except Exception as e:
                er = {"error": str(e)}
            if er and not er.get("error"):
                st.caption(f"buffer={er.get('buffer', 0)}")
                for p in (er.get("top_patterns") or [])[:6]:
                    st.write(
                        f"· `{p.get('key')}` ×{p.get('count')} net={p.get('net_improve')}"
                    )
                lr = er.get("last_replay") or {}
                if lr.get("replayed"):
                    st.caption(f"last replay: {lr.get('replayed')} patterns")
            else:
                st.caption("No episode log yet.")

            st.subheader("Goals / Commitments")
            st.caption(
                "Explicit commitments opened by focus dwell. Schema goals track "
                "member/nested growth; node goals track residual cool-down. "
                "REGULATE + high body_error soft-delays satisfy."
            )
            try:
                gr = prom.get_goals_report() if hasattr(prom, "get_goals_report") else {"active": [], "recent_history": []}
            except Exception as e:
                gr = {"active": [], "recent_history": [], "error": str(e)}
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.metric("Active goals", len(gr.get("active") or []))
            with gcol2:
                st.metric("Recent closed", len(gr.get("recent_history") or []))
            if gr.get("active"):
                for g in gr["active"]:
                    tag = "schema" if g.get("is_schema_goal") else "node"
                    st.write(
                        f"- **`{g.get('target_id')}`** ({tag}) "
                        f"str={g.get('strength')} dwell={g.get('dwell_pulses')} "
                        f"members={g.get('members')} nested={g.get('nested')} "
                        f"growth={g.get('growth_events')}"
                    )
            else:
                st.caption("No active goals — hold focus on a node/schema for ~8+ pulses.")
            if gr.get("recent_history"):
                with st.expander("Recent goal history"):
                    for h in gr["recent_history"]:
                        st.write(
                            f"- `{h.get('target_id')}` → **{h.get('status')}** "
                            f"({h.get('reason')}) @ pulse {h.get('pulse')}"
                        )

            with st.expander("Soak checks / consolidation quality"):
                if st.button("Run soak checks", key="soak_btn"):
                    try:
                        sc = prom.run_soak_checks() if hasattr(prom, "run_soak_checks") else {}
                        st.json(sc)
                    except Exception as e:
                        st.error(str(e))
                try:
                    cr = prom.get_consolidation_report() if hasattr(prom, "get_consolidation_report") else {}
                    if cr:
                        st.caption("Last consolidation")
                        st.json(cr)
                except Exception:
                    pass
                try:
                    mh = prom.get_module_health_report() if hasattr(prom, "get_module_health_report") else {}
                    if mh:
                        missing = [k for k, v in mh.items() if not v]
                        st.caption("Modules OK" if not missing else f"Missing: {missing}")
                except Exception:
                    pass

            st.subheader("Directed Working Memory (§14, new)")
            st.caption(
                "What's actually 'in mind' right now -- SELF + current basin "
                "+ up to 7 schema-cluster slots, narrowed by emotional "
                "intensity and populated by epoch-weighted admission (real "
                "input vs. self-study content). Distinct from the simpler "
                "top-K activation panel below (§11) -- this is the richer "
                "model that also gates self-study targeting in Childhood."
            )
            wm = prom.get_current_working_memory()
            wm_col1, wm_col2, wm_col3 = st.columns(3)
            with wm_col1:
                st.metric("Epoch", prom.bio.epoch.value)
            with wm_col2:
                st.metric("Schema slot capacity", f"{wm['capacity']} / {prom.working_memory.MAX_SCHEMA_SLOTS}")
            with wm_col3:
                st.metric("Slots filled", len(wm["slots"]))
            st.write(f"**SELF** (permanent) + **basin** (privileged) + current slots:")
            if wm["slots"]:
                for slot in wm["slots"]:
                    data = prom.archivist.graph.nodes.get(slot, {})
                    node_type = data.get("node_type", "standard")
                    label = data.get("name") or slot
                    user_linked = " (user-linked)" if prom.working_memory.is_user_linked(slot) else ""
                    st.write(f"- `{label}` ({node_type}){user_linked}")
            else:
                st.caption("No schema slots filled yet.")
            with st.expander("Why this capacity right now?"):
                arousal, valence, dominance = prom.synthesizer.get_current_basin_key()
                st.caption(
                    f"Current basin: arousal={arousal:.2f}, valence={valence:.2f}, "
                    f"dominance={dominance:.2f}. Below "
                    f"{prom.working_memory.LOW_AROUSAL_THRESHOLD:.2f} arousal, full "
                    f"baseline capacity (7) applies regardless of valence. Above "
                    f"that, negative valence narrows hard toward a floor of "
                    f"{prom.working_memory.HIGH_NEGATIVE_SLOT_FLOOR}; positive "
                    f"valence narrows only mildly."
                )

            with st.expander("Activation / Working Memory (§11 pull-forward, diagnostic)"):
                st.caption(
                    "Real activation numbers, so 'is focus actually working' is "
                    "checkable instead of just eyeballed on the Graph tab. Also "
                    "shows felt_state_anchors' bounded window size -- this was "
                    "previously an unbounded list that could silently grow into "
                    "the hundreds over long runs and swamp the top-K focus "
                    "filter; it's now capped per basin (fixed this revision)."
                )
                activation_report = prom.reflector.activation_report()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total nodes", activation_report["total_nodes"])
                with col2:
                    st.metric(
                        "Nodes with nonzero activation",
                        activation_report["nodes_with_nonzero_activation"],
                    )
                st.caption(f"Anchor window size (per felt state): {prom.ANCHOR_WINDOW_SIZE}")
                key = prom.synthesizer.get_current_basin_key()
                current_anchor_count = len(prom.felt_state_anchors.get(key, []))
                st.caption(
                    f"Current felt state's anchor count: {current_anchor_count} "
                    f"(capped at {prom.ANCHOR_WINDOW_SIZE})"
                )
                if activation_report["top_active"]:
                    st.write("Top active nodes:")
                    for name, act, node_type in activation_report["top_active"]:
                        st.write(f"- `{name}` ({node_type}): {act:.2f}")
                else:
                    st.caption("No nodes have any activation yet.")

            with st.expander("Valence Coloring / Parental Feedback (§13.2, diagnostic)"):
                st.caption(
                    "Parental buttons tint **at most 3** targets: current focus, "
                    "then recent user topics — not the whole basin history. "
                    "Sidebar shows exact node ids after each press. "
                    "Real accumulated coloring; a node's coloring only "
                    "ever moves when it was the current felt-state anchor at "
                    "the moment a Parental Feedback button was clicked -- "
                    "nothing here is a hand-assigned valence lookup, it's "
                    "purely a record of repeated co-occurrence."
                )
                if getattr(prom, "last_parental_feedback", None):
                    lpf = prom.last_parental_feedback
                    st.info(
                        f"Last parental: **{lpf.get('reaction')}** @ pulse {lpf.get('pulse')} → "
                        + (", ".join(lpf.get("anchors_colored") or []) or "(none)")
                    )
                st.subheader("Parental marks (what was actually colored)")
                if hasattr(prom, "parental_coloring_report"):
                    pcr = prom.parental_coloring_report()
                    if pcr.get("last_feedback"):
                        st.caption(
                            f"Last: **{pcr['last_feedback'].get('reaction')}** on "
                            f"`{pcr['last_feedback'].get('anchors_colored')}` @ pulse "
                            f"{pcr['last_feedback'].get('pulse')}"
                        )
                    st.caption(f"Nodes with parental/valence mark: {pcr.get('marked_nodes', 0)}")
                    if pcr.get("top"):
                        st.json(pcr["top"][:12])
                    else:
                        st.caption("None yet — use sidebar parental buttons while focus is on a topic.")
                coloring_report = prom.reflector.valence_coloring_report()
                st.metric("Total colored nodes", coloring_report["total_colored_nodes"])
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Most positive")
                    if coloring_report["most_positive"]:
                        for name, val in coloring_report["most_positive"]:
                            st.write(f"- `{name}`: {val:+.2f}")
                    else:
                        st.caption("None yet.")
                with col2:
                    st.caption("Most negative")
                    if coloring_report["most_negative"]:
                        for name, val in coloring_report["most_negative"]:
                            st.write(f"- `{name}`: {val:+.2f}")
                    else:
                        st.caption("None yet.")

            with st.expander("SELF / OTHER relational activity (diagnostic)"):
                st.caption(
                    "SELF and OTHER only ever gain edges through relational "
                    "detection (§2.1b) -- typed input matching specific "
                    "keyword patterns. Self-study and regulation both exclude "
                    "SELF/OTHER by design (axioms, not growable dictionary "
                    "concepts), so if you're testing mostly via Pulse/Run "
                    "Batch, both will look permanently frozen -- that's "
                    "expected, not a bug. Three of the four relation types "
                    "(responsible-for/violates/temporal-contrast) route "
                    "through SELF; only concerns-other routes through OTHER "
                    "-- so third-person-heavy messages (\"he/she/they...\") "
                    "grow OTHER, while self-referential ones (\"I did...\", "
                    "\"my fault...\", \"I shouldn't have...\") grow SELF."
                )
                self_other = prom.reflector.self_other_report()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SELF outgoing relational edges", self_other["self"]["total"])
                    if self_other["self"]["by_type"]:
                        st.write(self_other["self"]["by_type"])
                    if self_other["self"]["most_recent"]:
                        st.caption("Most recent:")
                        for ts, rel, target in self_other["self"]["most_recent"]:
                            st.caption(f"  {rel} -> `{target}`")
                with col2:
                    st.metric("OTHER outgoing relational edges", self_other["other"]["total"])
                    if self_other["other"]["by_type"]:
                        st.write(self_other["other"]["by_type"])
                    if self_other["other"]["most_recent"]:
                        st.caption("Most recent:")
                        for ts, rel, target in self_other["other"]["most_recent"]:
                            st.caption(f"  {rel} -> `{target}`")
                if self_other["self"]["total"] == 0 and self_other["other"]["total"] == 0:
                    st.caption(
                        "Zero relational edges anywhere yet. Try sending "
                        "(via Send, not Pulse/Run Batch) something like "
                        "\"I shouldn't have done that\" or \"that was my "
                        "fault\" to grow SELF, or \"my friend did that\" to "
                        "grow OTHER."
                    )

    # ================================================================
    # TAB: WORKING MEMORY -- Graph view of the actual 7-slot working
    # memory (§18 pull-forward, new), distinct from the Graph tab's
    # top-K-by-activation focus view and from the Reflection tab's
    # text-only slot summary (§14) -- this renders what's actually "in
    # mind" right now as a graph, and surfaces the Self-Narrative (§16,
    # new) alongside it, since narrative-linked nodes now feed the same
    # anchor pool that determines what lands in these slots.
    # ================================================================
    with tab_working_memory:
        st.subheader("Working Memory")
        if prom is None:
            st.info("Start the system from the sidebar first.")
        else:
            st.caption(
                "Graph rendering of the actual 7-slot working memory "
                "(§14): SELF + current basin + up to 7 schema-cluster "
                "slots. Distinct from the Graph tab's 'Focus size' view, "
                "which shows the top-K most-active nodes generally -- "
                "this shows specifically what get_working_memory() "
                "currently holds, the same content already listed as "
                "text on the Reflection tab, rendered visually here. "
                "Each slot's immediate graph neighbors are also shown "
                "(bounded, most-recent-first) so real edges are visible "
                "instead of the view looking artificially disconnected -- "
                "slots are chosen by independent activation scoring, not "
                "by being graph-adjacent to each other, so without this "
                "a node can have confirmed edges that simply have nowhere "
                "to render. The **Slot contents** list below is the exact "
                "slot set, unaffected by this -- the graph shows more "
                "nodes than that list for context."
            )

            wm = prom.get_current_working_memory()
            key = prom.synthesizer.get_current_basin_key()
            basin_id = prom.synthesizer.stabilized_basins.get(key)

            wm_core = {SELF_NODE}
            if basin_id and basin_id in prom.archivist.graph:
                wm_core.add(basin_id)
            wm_core.update(s for s in wm["slots"] if s in prom.archivist.graph)

            # Bug fix (this session): rendering *only* the slot set makes
            # the graph look artificially empty/disconnected even when the
            # underlying data is healthy -- working-memory slots are
            # chosen by independent activation-based scoring, not by being
            # graph-adjacent to each other, so a node like "Emotions" can
            # have real, confirmed edges (to its co-occurrence anchor, a
            # self-study-derived child, etc.) that simply don't render
            # because the OTHER endpoint didn't also score into the top-7
            # and so isn't in node_subset -- render_graph_html() only
            # draws an edge when BOTH endpoints are present. This is the
            # exact same rendering blind spot archivist.working_memory_
            # nodes() already fixed for SELF/OTHER specifically (see that
            # method's own docstring) -- generalized here to every node in
            # THIS subset, not just SELF/OTHER, since the slot members
            # deserve the same treatment. Bounded per-node (not
            # unconditionally all neighbors), same "bounded, not
            # unbounded" principle as everything else in this design.
            MAX_NEIGHBORS_PER_NODE = 5
            graph = prom.archivist.graph
            node_subset = set(wm_core)
            for n in wm_core:
                if n not in graph:
                    continue
                neighbor_edges = list(graph.out_edges(n, data=True)) + list(graph.in_edges(n, data=True))
                neighbor_edges.sort(key=lambda e: e[2].get("created_at", ""), reverse=True)
                for u, v, _d in neighbor_edges[:MAX_NEIGHBORS_PER_NODE]:
                    node_subset.add(v if u == n else u)

            wm_col1, wm_col2, wm_col3 = st.columns(3)
            with wm_col1:
                st.metric("Epoch", prom.bio.epoch.value)
            with wm_col2:
                st.metric("Schema slot capacity", f"{wm['capacity']} / {prom.working_memory.MAX_SCHEMA_SLOTS}")
            with wm_col3:
                st.metric("Nodes shown", f"{len(node_subset)} ({len(wm_core)} slot members + neighbors)")

            html = render_graph_html(prom.archivist, node_subset=node_subset)
            st.components.v1.html(html, height=500)

            st.write("**Slot contents:**")
            if wm["slots"]:
                for slot in wm["slots"]:
                    data = prom.archivist.graph.nodes.get(slot, {})
                    node_type = data.get("node_type", "standard")
                    label = data.get("name") or slot
                    user_linked = " (user-linked)" if prom.working_memory.is_user_linked(slot) else ""
                    st.write(f"- `{label}` ({node_type}){user_linked}")
            else:
                st.caption("No schema slots filled yet.")

            st.divider()
            st.subheader("Operators / prediction")
            st.caption("Internal moves: HOLD · RETURN · EXPAND · RELEASE · SETTLE")
            try:
                opr = prom.get_operators_report() if hasattr(prom, "get_operators_report") else {}
            except Exception as e:
                opr = {"error": str(e)}
            if opr.get("last_operator"):
                st.write(
                    f"**{opr['last_operator']}** — {opr.get('last_note','')} "
                    f"(predict: {opr.get('last_predict',{})})"
                )
            # Evidence spine: L (log-odds) vs bias-led choice
            ev = opr.get("evidence") or {}
            Lsc = opr.get("L_scores") or (ev.get("L") if ev else {}) or {}
            if Lsc:
                led = "evidence_led" if opr.get("evidence_led") else "bias_or_weak_L"
                st.caption(f"Evidence L (log-odds) · decision={led} · θ={ev.get('theta', 0.85)}")
                st.write(" · ".join(f"{k}={v}" for k, v in Lsc.items()))
            eps = opr.get("episodes") or []
            if eps:
                with st.expander(f"Recent episodes ({len(eps)})"):
                    for e in reversed(eps[-12:]):
                        st.write(
                            f"p{e.get('pulse')} **{e.get('operator')}** "
                            f"{e.get('predict')} focus=`{e.get('focus')}` "
                            f"{e.get('detail') or ''}"
                        )

            st.subheader("Stream of Consciousness")
            st.caption(
                "Moment-to-moment trace from focus, felt state, working memory, "
                "and goals — template-only, not an LLM monologue. Updates every pulse."
            )
            try:
                soc = prom.get_stream_report(last_n=15) if hasattr(prom, "get_stream_report") else {}
            except Exception as e:
                soc = {"beats": [], "latest": "", "error": str(e)}
            if soc.get("monologue"):
                st.markdown("**Inner monologue**")
                st.write(soc["monologue"])
            if soc.get("latest"):
                st.caption("Latest beat")
                st.info(soc["latest"])
            beats = soc.get("beats") or []
            if beats:
                with st.expander(f"Recent stream ({len(beats)} beats)", expanded=False):
                    for b in reversed(beats):
                        kind = b.get("kind") or "pulse"
                        tag = "⚡" if kind == "event" else "·"
                        st.write(f"{tag} **p{b.get('pulse')}** — {b.get('line')}")
            else:
                st.caption("No stream beats yet — run a few pulses.")

            st.subheader("Self-Narrative (§16, new)")
            st.caption(
                "Compressed, decaying record of what has turned out to "
                "matter -- distinct from the slot-based working memory "
                "above (short-horizon, 'what's in mind right now'). "
                "Elements at or above the salience floor feed into the "
                "same anchor pool that determines working-memory/self-"
                "study/regulation candidacy, so narratively significant "
                "content gets a better shot at showing up above even "
                "when it isn't the current felt state's own anchor."
            )
            narrative_report = prom.get_narrative_report()
            st.metric("Total narrative elements", narrative_report["total_elements"])
            if narrative_report["top_elements"]:
                for el in narrative_report["top_elements"]:
                    st.write(f"**{el['type_label']}** (weight={el['weight']:.2f}): {el['description']}")
                    with st.expander("Raw data", expanded=False):
                        st.write(f"`{el['element_id']}`")
                        st.write(f"Linked nodes: {', '.join(f'`{n}`' for n in el['linked_nodes'])}")
                        st.write(f"First formed: {el['formed_at']}")
                        st.write(f"Last reinforced: {el['last_reinforced_at']}")
            else:
                st.caption("No narrative elements formed yet.")

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
            st.subheader("Body surface (what cognition may sense)")
            st.caption("heart / breath / tension / sweat / gut / energy / warmth — not hormones.")
            st.json(prom.bio.get_raw_variables())

            st.subheader("Fast modulators (hidden — Debug only)")
            if hasattr(prom, "get_modulator_report"):
                st.caption("Salience / encode / alert / settle → gates + body gusts; mind does not see these names.")
                st.json(prom.get_modulator_report())
            st.subheader("Hormonal State (hidden motor — Debug only, not cognition)")
            st.json({k: round(v, 4) for k, v in prom.bio._hormones.items()})

            st.caption(
                f"Current basin key (arousal, valence, dominance): "
                f"{prom.synthesizer.get_current_basin_key()}"
            )
            st.caption(
                f"Stabilized basins: {len(prom.synthesizer.stabilized_basins)}"
            )
            st.subheader("Somatic topography (basin map)")
            st.caption("Basins and transitions — not raw hormone gauges.")
            st.json(prom.get_somatic_topo_report())

            st.subheader("Felt anchors (linkable felt identities)")
            st.caption("Stable ids over PAD; body_mean is heart/breath/tension/sweat/gut/energy/warmth only — never hormones.")
            if hasattr(prom, "get_felt_anchor_report"):
                st.json(prom.get_felt_anchor_report())
            else:
                st.caption("Felt anchors not wired on this instance.")

            st.subheader("Schema ↔ felt binds")
            st.caption("Phase B: schema ↔ felt place by recurrence; focus rehydrates absorbed children.")
            if hasattr(prom, "get_schema_felt_report"):
                st.json(prom.get_schema_felt_report())
            else:
                st.caption("Schema-felt binder not wired.")
            st.subheader("Focus / Residuals (§13.y)")
            if hasattr(prom, "get_focus_report"):
                st.json(prom.get_focus_report())

                st.subheader("Long-term interest themes")
                st.caption("Orientation across sleep — biases curiosity; earned from focus/narrative/parental recurrence.")
                if hasattr(prom, "get_long_term_interest_report"):
                    st.json(prom.get_long_term_interest_report())

            else:
                st.caption("Focus module not wired on this instance.")

                st.subheader("Last collapse summary (§13.4)")
                st.json(getattr(prom, "last_collapse_summary", {}))

            absorbed_parents = [
                {"parent": n, "absorbed_count": len(d.get("absorbed") or [])}
                for n, d in prom.archivist.graph.nodes(data=True)
                if d.get("absorbed")
            ]
            if absorbed_parents:
                st.subheader("Parents with absorbed children")
                st.json(absorbed_parents[:30])

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

            with st.expander("Sleep pressure / cycle"):
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
                prom.SELF_STUDY_AROUSAL_BUMP = st.slider(
                    "Adrenaline bump per self-study expansion (new -- "
                    "previously arousal/dominance never moved from "
                    "autonomous activity at all)", 0.0, 0.1,
                    value=prom.SELF_STUDY_AROUSAL_BUMP, step=0.001,
                )

            with st.expander("Bias-Modulated Self-Study Targeting (§13.1, new)"):
                st.caption(
                    "Previously, executive.py's EXPLORE/STABILIZE/NEUTRAL "
                    "bias signal was computed every tick and only ever "
                    "logged -- self-study picked targets the same way "
                    "regardless of bias. Now EXPLORE favors fresher, "
                    "low-activation content; STABILIZE favors established, "
                    "high-activation content; NEUTRAL reproduces the "
                    "original default exactly."
                )
                prom.SELF_STUDY_PROVISIONAL_PROB_EXPLORE = st.slider(
                    "P(provisional pool) under EXPLORE", 0.0, 1.0,
                    value=prom.SELF_STUDY_PROVISIONAL_PROB_EXPLORE, step=0.05,
                )
                prom.SELF_STUDY_PROVISIONAL_PROB_STABILIZE = st.slider(
                    "P(provisional pool) under STABILIZE", 0.0, 1.0,
                    value=prom.SELF_STUDY_PROVISIONAL_PROB_STABILIZE, step=0.05,
                )
                prom.SELF_STUDY_PROVISIONAL_PROB_NEUTRAL = st.slider(
                    "P(provisional pool) under NEUTRAL (original default: 0.6)", 0.0, 1.0,
                    value=prom.SELF_STUDY_PROVISIONAL_PROB_NEUTRAL, step=0.05,
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

            with st.expander("Parental Feedback / Valence Coloring (§13.2, new)"):
                prom.PARENTAL_APPROVAL_DOPAMINE = st.slider(
                    "Dopamine bump: Approval", 0.0, 0.3,
                    value=prom.PARENTAL_APPROVAL_DOPAMINE, step=0.01,
                )
                prom.PARENTAL_DISAPPROVAL_CORTISOL = st.slider(
                    "Cortisol bump: Disapproval", 0.0, 0.3,
                    value=prom.PARENTAL_DISAPPROVAL_CORTISOL, step=0.01,
                )
                prom.PARENTAL_WARMTH_DOPAMINE = st.slider(
                    "Dopamine bump: Warmth", 0.0, 0.3,
                    value=prom.PARENTAL_WARMTH_DOPAMINE, step=0.01,
                )
                prom.PARENTAL_WARMTH_CORTISOL_RELIEF = st.slider(
                    "Cortisol relief: Warmth", 0.0, 0.3,
                    value=prom.PARENTAL_WARMTH_CORTISOL_RELIEF, step=0.01,
                )
                prom.PARENTAL_CONCERN_AROUSAL = st.slider(
                    "Arousal bump: Concern", 0.0, 0.3,
                    value=prom.PARENTAL_CONCERN_AROUSAL, step=0.01,
                )
                prom.PARENTAL_CONCERN_CORTISOL = st.slider(
                    "Cortisol bump: Concern", 0.0, 0.3,
                    value=prom.PARENTAL_CONCERN_CORTISOL, step=0.01,
                )
                prom.VALENCE_COLORING_STEP = st.slider(
                    "Valence coloring step (per reaction, per anchor)", 0.0, 0.5,
                    value=prom.VALENCE_COLORING_STEP, step=0.01,
                )
                prom.VALENCE_COLORING_CAP = st.slider(
                    "Valence coloring cap (per node)", 0.5, 3.0,
                    value=prom.VALENCE_COLORING_CAP, step=0.1,
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

            with st.expander("Epistemic Schema Formation (§13.3, new)"):
                prom.CO_ACTIVATION_RECENCY_WINDOW = st.slider(
                    "Co-activation recency window (recent distinct anchors "
                    "paired per touch -- new, fixes pairs only ever getting "
                    "one chance to co-occur)", 0, 10,
                    value=prom.CO_ACTIVATION_RECENCY_WINDOW, step=1,
                )
                prom.archivist.CO_ACTIVATION_STABILIZATION_THRESHOLD = st.slider(
                    "Co-activation stabilization threshold (pair touch count)", 1, 20,
                    value=prom.archivist.CO_ACTIVATION_STABILIZATION_THRESHOLD, step=1,
                )
                prom.archivist.CO_ACTIVATION_DECAY_RATE = st.slider(
                    "Co-activation decay rate (retained per Consolidation)", 0.0, 1.0,
                    value=prom.archivist.CO_ACTIVATION_DECAY_RATE, step=0.05,
                )
                prom.archivist.CO_ACTIVATION_PRUNE_FLOOR = st.slider(
                    "Co-activation prune floor (below this, pair is dropped)", 0.0, 5.0,
                    value=prom.archivist.CO_ACTIVATION_PRUNE_FLOOR, step=0.1,
                )
                prom.reflector.EPISTEMIC_MIN_CLUSTER_SIZE = st.slider(
                    "Minimum cluster size", 2, 15,
                    value=prom.reflector.EPISTEMIC_MIN_CLUSTER_SIZE, step=1,
                )
                prom.reflector.EPISTEMIC_NAME_MIN_COVERAGE = st.slider(
                    "Minimum member coverage for an earned name", 1, 10,
                    value=prom.reflector.EPISTEMIC_NAME_MIN_COVERAGE, step=1,
                )

            with st.expander("Directed Working Memory (§14, new)"):
                st.caption(
                    "SELF + basin + up to 7 schema-cluster slots, narrowed "
                    "by emotional intensity, gated by developmental epoch. "
                    "Now drives self-study targeting directly (hard-gated "
                    "in Childhood, boosted elsewhere) -- not just logged."
                )
                wm_mod = prom.working_memory
                wm_mod.MAX_SCHEMA_SLOTS = st.slider(
                    "Hard ceiling on schema slots", 3, 12,
                    value=wm_mod.MAX_SCHEMA_SLOTS, step=1,
                )
                wm_mod.MIN_SCHEMA_SLOTS = st.slider(
                    "Soft lower bound (never fewer than this)", 0, 3,
                    value=wm_mod.MIN_SCHEMA_SLOTS, step=1,
                )
                wm_mod.HIGH_NEGATIVE_SLOT_FLOOR = st.slider(
                    "High-negative/high-arousal narrowing floor", 1, 5,
                    value=wm_mod.HIGH_NEGATIVE_SLOT_FLOOR, step=1,
                )
                wm_mod.LOW_AROUSAL_THRESHOLD = st.slider(
                    "Low-arousal threshold (below this: full baseline capacity)", 0.0, 1.0,
                    value=wm_mod.LOW_AROUSAL_THRESHOLD, step=0.05,
                )
                wm_mod.POSITIVE_NARROWING_MAX = st.slider(
                    "Positive-valence narrowing (max slots removed)", 0, 5,
                    value=wm_mod.POSITIVE_NARROWING_MAX, step=1,
                )
                st.caption("Epoch-weighted admission (§14.2) -- probability a slot favors user-linked content:")
                wm_mod.CHILDHOOD_USER_PRIORITY = st.slider(
                    "Childhood user priority", 0.0, 1.0,
                    value=wm_mod.CHILDHOOD_USER_PRIORITY, step=0.05,
                )
                wm_mod.ADOLESCENCE_USER_PRIORITY = st.slider(
                    "Adolescence user priority", 0.0, 1.0,
                    value=wm_mod.ADOLESCENCE_USER_PRIORITY, step=0.05,
                )
                wm_mod.MATURITY_USER_PRIORITY = st.slider(
                    "Maturity user priority", 0.0, 1.0,
                    value=wm_mod.MATURITY_USER_PRIORITY, step=0.05,
                )
                wm_mod.MATURITY_RESERVED_SLOTS = st.slider(
                    "Maturity reserved user-linked slots (never fully displaced)", 0, 3,
                    value=wm_mod.MATURITY_RESERVED_SLOTS, step=1,
                )
                wm_mod.CALM_PRIORITY_SOFTENING_MAX = st.slider(
                    "Calm-state admission softening (§14.3 'exploratory expansion' "
                    "-- loosens restriction, does NOT raise the slot ceiling)", 0.0, 1.0,
                    value=wm_mod.CALM_PRIORITY_SOFTENING_MAX, step=0.05,
                )
                wm_mod.BASIN_COOCCURRENCE_BONUS = st.slider(
                    "Basin co-occurrence ranking bonus (§14.4)", 0.0, 20.0,
                    value=wm_mod.BASIN_COOCCURRENCE_BONUS, step=0.5,
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
