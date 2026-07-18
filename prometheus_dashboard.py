"""
prometheus_dashboard.py -- Interface layer (§7).

Pyvis graph rendering for app.py's Graph tab (§4B). This file was
referenced by app.py (`from prometheus_dashboard import render_graph_html`)
and described in §7's module table across every spec revision, but was
never actually present in the project -- reconstructed here.

Two things implemented per the spec:
  - Task 1 fix (§4B, §7): build the network in memory and call
    .generate_html() directly rather than net.show(), which can silently
    fail to render on read-only/sandboxed filesystems or race with
    Streamlit's rerun cycle.
  - Visual encoding (§4B, §10 item 21): edge color/style by type (grouped
    by category), node shape by node_type, opacity by tier for standard
    nodes, basin nodes colored by their own valence, schema nodes gray
    until named. All pulled from edge_types.py's shared style map rather
    than re-declared here, so the vocabulary and its rendering can't drift
    apart the way stimulus.py's edge type once did from §6A's list.
"""
import logging
from typing import Optional

from pyvis.network import Network

from Prometheus.edge_types import (
    EDGE_STYLE, DEFAULT_EDGE_STYLE, NODE_SHAPE, TIER_OPACITY,
    SCHEMA_UNNAMED_COLOR, SCHEMA_NAMED_COLOR, SELF_COLOR,
    NODE_STANDARD, NODE_BASIN, NODE_SCHEMA, NODE_SELF,
    basin_color,
)

logger = logging.getLogger(__name__)

# Base hue for standard (non-schema, non-basin, non-self) nodes; tier
# controls opacity on top of this, not a different hue per tier, so
# "trusted" reads as "solid version of the same thing" not "different
# thing" (§4B: "node color/opacity, Provisional faint -> Trusted solid").
_STANDARD_BASE_COLOR = (90, 140, 200)  # r, g, b


def _hex_with_opacity(rgb, opacity: float) -> str:
    r, g, b = rgb
    return f"rgba({r},{g},{b},{max(0.15, min(1.0, opacity))})"


def _node_visual(node: str, data: dict) -> dict:
    """Returns {shape, color} for one node, per node_type (§6A/§10 item 21).
    node_type may be absent on graphs saved before this field existed --
    falls back to inferring it from is_schema/known SELF-node id rather
    than crashing, so older checkpoints still render sensibly."""
    node_type = data.get("node_type")
    if node_type is None:
        if data.get("is_schema"):
            node_type = NODE_SCHEMA
        elif node == "SELF":
            node_type = NODE_SELF
        else:
            node_type = NODE_STANDARD

    shape = NODE_SHAPE.get(node_type, NODE_SHAPE[NODE_STANDARD])

    if node_type == NODE_SELF:
        return {"shape": shape, "color": SELF_COLOR}

    if node_type == NODE_SCHEMA:
        color = SCHEMA_NAMED_COLOR if data.get("named") else SCHEMA_UNNAMED_COLOR
        return {"shape": shape, "color": color}

    if node_type == NODE_BASIN:
        pad = data.get("pad_coordinates") or (0.0, 0.0, 0.0)
        valence = pad[1] if len(pad) > 1 else 0.0
        return {"shape": shape, "color": basin_color(valence)}

    # Standard: tier -> opacity, per §4B ("Provisional faint -> Trusted
    # solid"). Regulatory efficacy is deliberately NOT encoded here (§4B:
    # kept in the Reflection tab -- a philosophically distinct property
    # from epistemic trust, and cramming both onto one node's visual
    # encoding gets muddy).
    tier = data.get("tier", 0)
    opacity = TIER_OPACITY.get(tier, TIER_OPACITY[0])
    return {"shape": shape, "color": _hex_with_opacity(_STANDARD_BASE_COLOR, opacity)}


def _edge_visual(relation_type: str) -> dict:
    return EDGE_STYLE.get(relation_type, DEFAULT_EDGE_STYLE)


def render_graph_html(archivist, node_subset: Optional[set] = None,
                       height: str = "700px", width: str = "100%") -> str:
    """
    Builds the Pyvis network from archivist.graph and returns the rendered
    HTML as a string via generate_html() -- no filesystem write (Task 1
    fix), so this can't silently fail on a read-only/sandboxed filesystem
    or race with Streamlit's rerun cycle. app.py passes the result
    straight to st.components.v1.html().

    `node_subset` (new, this revision -- §11 pull-forward): when supplied,
    only nodes in the subset are rendered (typically
    archivist.working_memory_nodes()'s output), and only edges where both
    endpoints are in the subset. This is the actual fix for §11's
    rendering-cost/readability problem at scale -- rendering the full live
    graph every time doesn't scale past a few hundred nodes and multi-
    parenting makes it unreadable regardless of tuning; a bounded
    neighborhood sidesteps both. None (default) renders the full graph,
    preserving old behavior for callers that want it (e.g. an explicit
    "show full graph" opt-in).
    """
    net = Network(height=height, width=width, directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut()

    graph = archivist.graph
    nodes_to_render = graph.nodes(data=True) if node_subset is None else (
        (n, d) for n, d in graph.nodes(data=True) if n in node_subset
    )

    for node, data in nodes_to_render:
        visual = _node_visual(node, data)
        label = data.get("name") or node
        title_bits = [f"tier: {data.get('tier', 0)}", f"source: {data.get('source', 'unknown')}",
                      f"activation: {data.get('activation', 0.0):.2f}"]
        if data.get("is_schema"):
            title_bits.append(f"named: {data.get('named', False)}")
        net.add_node(
            node,
            label=str(label),
            shape=visual["shape"],
            color=visual["color"],
            title=" | ".join(title_bits),
        )

    # MultiDiGraph: iterate with keys so parallel edges of different
    # relation_type between the same two nodes (§2.1b's own requirement --
    # an event node can carry both responsible-for and violates at once)
    # are each added individually, not collapsed.
    for u, v, key, data in graph.edges(keys=True, data=True):
        if node_subset is not None and (u not in node_subset or v not in node_subset):
            continue
        relation_type = data.get("relation_type", "associated-with")
        style = _edge_visual(relation_type)
        net.add_edge(
            u, v,
            color=style["color"],
            width=float(style["width"]),
            dashes=(style["dashes"] != "false"),
            title=relation_type,
        )

    try:
        return net.generate_html()
    except Exception as e:  # pragma: no cover -- defensive, per Task 1's
        # own rationale: this must never silently fail to render.
        logger.warning("render_graph_html failed: %s", e)
        return f"<p>Graph rendering failed: {e}</p>"
