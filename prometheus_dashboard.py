"""
Graph tab rendering (§4B / §7). Exposes render_graph_html(archivist) which
app.py's tabbed layout calls. Kept as a separate module per the module
responsibility table so app.py stays focused on tab layout/orchestration.
"""
from pyvis.network import Network

# Trust tier -> visual encoding (§4B: "trust tier -> node color/opacity,
# Provisional faint -> Trusted solid")
TIER_COLORS = {
    0: "#4a4a6a",  # Provisional -- faint
    1: "#6a6ac8",  # Working
    2: "#8a8aff",  # Trusted -- solid
}
TIER_LABELS = {0: "Provisional", 1: "Working", 2: "Trusted"}

# Edge type -> line style (§4B: "distinct line style" per edge type)
EDGE_DASHES = {
    "is-a": False,
    "part-of": False,
    "associated-with": True,
    "associated_with": True,
    "related_to": True,
    "responsible-for": True,
    "violates": True,
    "temporal-contrast": True,
    "concerns-other": True,
}

SELF_COLOR = "#ffcc66"
SCHEMA_COLOR_NAMED = "#66ffcc"
SCHEMA_COLOR_UNNAMED = "#3a6a5a"


def render_graph_html(archivist) -> str:
    """
    Builds the Pyvis network fully in memory and returns the HTML string
    directly via generate_html() -- no filesystem write, no re-open from
    disk (Task 1 fix): the previous implementation could silently fail to
    render on a read-only/sandboxed filesystem, or race against
    Streamlit's rerun cycle if two sessions wrote the same path
    concurrently.

    Regulatory efficacy (§4.5) is deliberately NOT encoded on this graph
    (kept in the Reflection tab per §4B) -- it's a philosophically
    distinct property from epistemic trust and cramming both onto one
    node's visual encoding gets muddy.
    """
    net = Network(height="700px", width="100%", bgcolor="#1e1e1e", font_color="#ffffff", directed=True)
    net.toggle_physics(True)

    graph = archivist.graph
    if graph is None or len(graph.nodes) == 0:
        net.add_node("empty", label="No nodes yet", color="#555555")
        return net.generate_html()

    for node, data in graph.nodes(data=True):
        tier = data.get("tier", 0)
        is_schema = data.get("is_schema", False)

        if node == "SELF":
            color = SELF_COLOR
            label = "SELF"
        elif is_schema:
            named = data.get("named", False)
            color = SCHEMA_COLOR_NAMED if named else SCHEMA_COLOR_UNNAMED
            label = data.get("name") or f"(unnamed schema: {node})"
        else:
            color = TIER_COLORS.get(tier, TIER_COLORS[0])
            label = str(node)

        title_lines = [str(node), f"Tier: {TIER_LABELS.get(tier, 'Provisional')}",
                       f"Source: {data.get('source', 'unknown')}"]
        if is_schema:
            title_lines.append(f"Basin: {data.get('basin')}")
            title_lines.append(f"Relations: {', '.join(data.get('relation_types', []))}")
            title_lines.append(f"Named: {data.get('named', False)}")
        title = "\n".join(title_lines)

        net.add_node(str(node), label=label, color=color, title=title)

    for u, v, data in graph.edges(data=True):
        relation_type = data.get("relation_type", "associated-with")
        dashes = EDGE_DASHES.get(relation_type, True)
        net.add_edge(str(u), str(v), label=relation_type, dashes=dashes)

    return net.generate_html()
