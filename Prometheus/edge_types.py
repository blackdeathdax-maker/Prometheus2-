"""
edge_types.py -- shared vocabulary (spec §6A / §10 item 18).

Every module that creates or reads a typed edge or a node_type must import
from here rather than typing a literal string. This is the fix for the
cross-tool-drift instance found in review: stimulus.py had invented its own
"related_to" edge type that never appeared in §6A's canonical list, because
nothing existed to import instead.

Also owns the Graph-tab color/style map (§4B, §10 item 21) so visual
encoding lives next to the vocabulary it encodes, not duplicated into
prometheus_dashboard.py by hand.
"""
from typing import Dict

# ---------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------
# Categorical / hierarchical (§2.3). associated-with is deliberately the
# weakest of the three -- co-occurrence, not a stated relationship.
EDGE_IS_A = "is-a"
EDGE_PART_OF = "part-of"
EDGE_ASSOCIATED_WITH = "associated-with"

# Relational / narrative (§2.1b). These fire off SELF/OTHER onto an event
# node and are what reflector.py's schema detection scans for.
EDGE_RESPONSIBLE_FOR = "responsible-for"
EDGE_VIOLATES = "violates"
EDGE_TEMPORAL_CONTRAST = "temporal-contrast"
EDGE_CONCERNS_OTHER = "concerns-other"

# Structural (new, this revision). Schema Nodes link back to their
# component basin/event nodes (reflector.detect_schemas) -- this is a
# permanent compositional fact about what the schema *is made of*, not a
# tentative co-occurrence placement, and was previously mislabeled
# associated-with, which made it indistinguishable from a re-parenting-
# eligible placement it structurally isn't (it never carries
# placement="cooccurrence", so it was only accidentally safe from being
# treated as one).
EDGE_COMPOSED_OF = "composed-of"

CATEGORICAL_EDGE_TYPES = frozenset({EDGE_IS_A, EDGE_PART_OF, EDGE_ASSOCIATED_WITH})
RELATIONAL_EDGE_TYPES = frozenset({
    EDGE_RESPONSIBLE_FOR, EDGE_VIOLATES, EDGE_TEMPORAL_CONTRAST, EDGE_CONCERNS_OTHER,
})
STRUCTURAL_EDGE_TYPES = frozenset({EDGE_COMPOSED_OF})
ALL_EDGE_TYPES = CATEGORICAL_EDGE_TYPES | RELATIONAL_EDGE_TYPES | STRUCTURAL_EDGE_TYPES

# Edge types that count toward epistemic trust corroboration (§3.2).
# Relational and structural edges represent recurrence/composition, not
# independent confirmation of a fact -- counting them inflated a node's
# trust score for reasons unrelated to whether it's true (e.g. a node
# frequently on the receiving end of `violates` edges was drifting toward
# Trusted purely for showing up in guilt-shaped patterns).
TRUST_BEARING_EDGE_TYPES = CATEGORICAL_EDGE_TYPES

# ---------------------------------------------------------------------
# Node types (§6A: node_type field -- standard | basin | schema | self)
# ---------------------------------------------------------------------
NODE_STANDARD = "standard"
NODE_BASIN = "basin"
NODE_SCHEMA = "schema"
NODE_SELF = "self"

# ---------------------------------------------------------------------
# Graph tab visual encoding (§4B, §10 item 21)
# ---------------------------------------------------------------------
# Edge styling: color grouped by category (so the grouping itself reads
# visually), line style/dash pattern distinguishes individual types within
# a category. Consumed by prometheus_dashboard.py at render time.
EDGE_STYLE: Dict[str, Dict[str, str]] = {
    # Categorical -- blue family, solid. associated-with is deliberately
    # thin/faint: it's the weakest claim (§2.3), and should visually read
    # as tentative, not equal in weight to a parsed is-a/part-of edge.
    EDGE_IS_A:             {"color": "#2c5da8", "width": "2.5", "dashes": "false"},
    EDGE_PART_OF:          {"color": "#5c8fd6", "width": "2.0", "dashes": "false"},
    EDGE_ASSOCIATED_WITH:  {"color": "#a9c2e8", "width": "1.0", "dashes": "false"},

    # Relational -- orange/red family, dashed (narrative, not structural).
    # Distinct dash pattern per type since combinations on one event node
    # are what §2.1b's schema detection actually looks for.
    EDGE_RESPONSIBLE_FOR:   {"color": "#c1440e", "width": "1.5", "dashes": "[2,2]"},
    EDGE_VIOLATES:          {"color": "#b3121b", "width": "1.5", "dashes": "[6,3]"},
    EDGE_TEMPORAL_CONTRAST: {"color": "#d68a1c", "width": "1.5", "dashes": "[1,3]"},
    EDGE_CONCERNS_OTHER:    {"color": "#e0a11a", "width": "1.5", "dashes": "[4,2,1,2]"},

    # Structural -- purple, dotted. Visually signals "membership fact,"
    # not a corroboration edge.
    EDGE_COMPOSED_OF:      {"color": "#7a4bb0", "width": "1.5", "dashes": "[1,1]"},
}
DEFAULT_EDGE_STYLE = {"color": "#999999", "width": "1.0", "dashes": "false"}

# Node styling: node_type picks shape + base color; tier (standard nodes
# only) picks opacity. basin nodes use their own PAD valence to color
# (informative, not arbitrary); schema nodes are gray until named.
NODE_SHAPE: Dict[str, str] = {
    NODE_STANDARD: "dot",
    NODE_BASIN: "diamond",
    NODE_SCHEMA: "hexagon",
    NODE_SELF: "star",
}
TIER_OPACITY = {0: 0.35, 1: 0.65, 2: 1.0}  # Provisional / Working / Trusted
SCHEMA_UNNAMED_COLOR = "#888888"
SCHEMA_NAMED_COLOR = "#2e8b57"
SELF_COLOR = "#d4af37"


def basin_color(valence: float) -> str:
    """Warm-gradient color keyed to a basin's valence centroid (-1..1) --
    reuses data the node already carries (pad_coordinates, §6A) rather than
    assigning an arbitrary fixed color to every basin alike."""
    v = max(-1.0, min(1.0, valence))
    if v >= 0:
        # positive valence -> warm yellow/green
        r = int(255 - 120 * v)
        g = int(200 + 55 * v)
        b = 90
    else:
        # negative valence -> cool blue/violet
        r = int(150 + 60 * v)
        g = 90
        b = int(200 - 40 * v)
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
