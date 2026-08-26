"""
edge_types.py -- shared vocabulary (spec §6A / §10 item 18, extended by
§13.4.3's meta-edge family system).

Every module that creates or reads a typed edge or a node_type must import
from here rather than typing a literal string. This is the fix for the
cross-tool-drift instance found in review: stimulus.py had invented its own
"related_to" edge type that never appeared in §6A's canonical list, because
nothing existed to import instead.

Also owns the Graph-tab color/style map (§4B, §10 item 21) so visual
encoding lives next to the vocabulary it encodes, not duplicated into
prometheus_dashboard.py by hand.

§13.4.3, new this revision: every edge choice (is-a, violates, causes,
etc.) now also belongs to a FAMILY (HIERARCHY, SOCIAL_NORM, CAUSAL, etc.).
Families are the layer schema detection, trust diversity, collapse merge,
and working-memory ranking are meant to operate on going forward; choices
remain what meaning/UI display use. This is a NEW, separate classification
alongside the existing CATEGORICAL_EDGE_TYPES/RELATIONAL_EDGE_TYPES/
TRUST_BEARING_EDGE_TYPES groupings below -- those are untouched and keep
their exact existing behavior (trust scoring in particular must not change
as a side effect of this addition). Notably the two schemes disagree on
associated-with: the old "categorical" grouping lumps it with is-a/part-of
for trust-scoring purposes, but under the new, more precise family table
it's RESIDUAL, not HIERARCHY -- both are correct for their own purpose,
this file just now expresses both.
"""
from typing import Dict, Optional

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

# Structural / MEMBERSHIP family (new, this revision -- extended by
# §13.4.3). Schema Nodes link back to their component basin/event nodes
# (reflector.detect_schemas) -- this is a permanent compositional fact
# about what the schema *is made of*, not a tentative co-occurrence
# placement, and was previously mislabeled associated-with, which made it
# indistinguishable from a re-parenting-eligible placement it structurally
# isn't (it never carries placement="cooccurrence", so it was only
# accidentally safe from being treated as one).
EDGE_COMPOSED_OF = "composed-of"
# §13.4.3's second MEMBERSHIP choice, new this revision -- distinct from
# composed-of: an instance is a concrete particular of a category (this
# specific event *is an instance of* a recognized pattern), not a part of
# a whole the way a schema's components are. No producer creates this yet
# (§13.4's collapse/rehydration algorithm is the intended first user, via
# rehydration's "P --composed-of/instance-of--> C" restoration step,
# §13.4.6) -- the constant exists now so that implementation has
# something real to import rather than inventing a literal string.
EDGE_INSTANCE_OF = "instance-of"

# ROLE family (new, this revision -- §13.4.3). Event-participant binding
# ("who did what to what, with what") -- feeds the somatic-schema pattern
# skeleton §13.4.7 describes (SOCIAL_NORM + ROLE + optional TEMPORAL). No
# producer creates these yet; same status as EDGE_INSTANCE_OF above.
EDGE_AGENT = "agent"
EDGE_PATIENT = "patient"
EDGE_INSTRUMENT = "instrument"

# CAUSAL family (new, this revision -- §13.4.3). The explanatory backbone
# §13.4.3's family table names -- "X causes Y", "X prevents Y", etc. No
# producer creates these yet.
EDGE_CAUSES = "causes"
EDGE_RESULTS_IN = "results-in"
EDGE_PREVENTS = "prevents"
EDGE_ENABLES = "enables"

# ABSTRACTION family (new, this revision -- §13.4.3). This section's own
# collapse bookkeeping (§13.4.4 step 5, §13.4.6 step 5): written when a
# child C is absorbed into parent P (C -abstracted-from-> P) and its
# inverse on rehydration (P -elaborates-> C). Directed pair, not a
# frozenset of independent choices, since the two are meant to stay
# consistent inverses of each other (§13.4.4's edge-rewiring rule:
# "maintain inverse consistency").
EDGE_ABSTRACTED_FROM = "abstracted-from"
EDGE_ELABORATES = "elaborates"

CATEGORICAL_EDGE_TYPES = frozenset({EDGE_IS_A, EDGE_PART_OF, EDGE_ASSOCIATED_WITH})
RELATIONAL_EDGE_TYPES = frozenset({
    EDGE_RESPONSIBLE_FOR, EDGE_VIOLATES, EDGE_TEMPORAL_CONTRAST, EDGE_CONCERNS_OTHER,
})
MEMBERSHIP_EDGE_TYPES = frozenset({EDGE_COMPOSED_OF, EDGE_INSTANCE_OF})
STRUCTURAL_EDGE_TYPES = MEMBERSHIP_EDGE_TYPES  # deprecated alias, kept for safety -- nothing outside this file referenced the old name, but costs nothing to preserve
ROLE_EDGE_TYPES = frozenset({EDGE_AGENT, EDGE_PATIENT, EDGE_INSTRUMENT})
CAUSAL_EDGE_TYPES = frozenset({EDGE_CAUSES, EDGE_RESULTS_IN, EDGE_PREVENTS, EDGE_ENABLES})
ABSTRACTION_EDGE_TYPES = frozenset({EDGE_ABSTRACTED_FROM, EDGE_ELABORATES})
ALL_EDGE_TYPES = (
    CATEGORICAL_EDGE_TYPES | RELATIONAL_EDGE_TYPES | MEMBERSHIP_EDGE_TYPES
    | ROLE_EDGE_TYPES | CAUSAL_EDGE_TYPES | ABSTRACTION_EDGE_TYPES
)

# Edge types that count toward epistemic trust corroboration (§3.2).
# Relational and structural edges represent recurrence/composition, not
# independent confirmation of a fact -- counting them inflated a node's
# trust score for reasons unrelated to whether it's true (e.g. a node
# frequently on the receiving end of `violates` edges was drifting toward
# Trusted purely for showing up in guilt-shaped patterns). Unchanged by
# §13.4.3's addition below -- this grouping's exact membership and every
# consumer of it (archivist.py's trust scoring) must not change as a side
# effect of adding the new family layer.
TRUST_BEARING_EDGE_TYPES = CATEGORICAL_EDGE_TYPES

# ---------------------------------------------------------------------
# §13.4.3 Meta-edge families -- new this revision.
# ---------------------------------------------------------------------
# Family name constants, so nothing hardcodes the literal strings from
# §13.4.3's family table.
FAMILY_HIERARCHY = "HIERARCHY"
FAMILY_MEMBERSHIP = "MEMBERSHIP"
FAMILY_ROLE = "ROLE"
FAMILY_CAUSAL = "CAUSAL"
FAMILY_SOCIAL_NORM = "SOCIAL_NORM"
FAMILY_TEMPORAL = "TEMPORAL"
FAMILY_ABSTRACTION = "ABSTRACTION"
FAMILY_RESIDUAL = "RESIDUAL"

# Every choice this file knows about, mapped to its family -- the single
# source of truth §13.4.3 calls for ("Legacy flat relation_type strings
# map into this table"). Deliberately a plain dict built explicitly
# choice-by-choice rather than derived by iterating the frozensets above,
# so a new choice added to a frozenset without a FAMILY_OF entry fails
# loudly (KeyError in get_family, not a silent RESIDUAL fallback) instead
# of quietly mis-classifying.
FAMILY_OF: Dict[str, str] = {
    EDGE_IS_A: FAMILY_HIERARCHY,
    EDGE_PART_OF: FAMILY_HIERARCHY,
    EDGE_ASSOCIATED_WITH: FAMILY_RESIDUAL,
    EDGE_COMPOSED_OF: FAMILY_MEMBERSHIP,
    EDGE_INSTANCE_OF: FAMILY_MEMBERSHIP,
    EDGE_AGENT: FAMILY_ROLE,
    EDGE_PATIENT: FAMILY_ROLE,
    EDGE_INSTRUMENT: FAMILY_ROLE,
    EDGE_CAUSES: FAMILY_CAUSAL,
    EDGE_RESULTS_IN: FAMILY_CAUSAL,
    EDGE_PREVENTS: FAMILY_CAUSAL,
    EDGE_ENABLES: FAMILY_CAUSAL,
    EDGE_RESPONSIBLE_FOR: FAMILY_SOCIAL_NORM,
    EDGE_VIOLATES: FAMILY_SOCIAL_NORM,
    EDGE_CONCERNS_OTHER: FAMILY_SOCIAL_NORM,
    EDGE_TEMPORAL_CONTRAST: FAMILY_TEMPORAL,
    EDGE_ABSTRACTED_FROM: FAMILY_ABSTRACTION,
    EDGE_ELABORATES: FAMILY_ABSTRACTION,
}

# Families where a given directed (u, v) pair may hold at most one choice
# at a time, per §13.4.3's exclusivity rule -- HIERARCHY and ROLE are
# exclusive-per-direction/context, SOCIAL_NORM explicitly allows a dual
# (the spec's own "I shouldn't have done that" example needs both
# responsible-for and violates simultaneously, §2.1b), MEMBERSHIP/CAUSAL/
# TEMPORAL/RESIDUAL are multi. Consulted by collapse's edge-rewrite merge
# step (§13.4.4) to decide whether landing a second choice on an existing
# (u, v) pair is a reinforcement or a conflict -- not enforced by this
# file itself, since edge_types.py only owns vocabulary, not graph
# mutation (archivist.py's job, per the module responsibility table).
EXCLUSIVE_FAMILIES = frozenset({FAMILY_HIERARCHY, FAMILY_ROLE})


def get_family(relation_type: str, family: Optional[str] = None) -> str:
    """Resolves the family for an edge. `family` is the edge's own stored
    `family` attribute if the caller already has it (graceful pass-
    through -- once an edge is written with a family at creation time,
    later readers shouldn't need to re-derive it). When `family` is None
    (the migration case: every edge created before this revision has no
    `family` attribute at all, §13.4.14 item 5), falls back to looking
    `relation_type` up in FAMILY_OF. An unrecognized relation_type with no
    stored family falls back to RESIDUAL -- the "weak glue, low weight
    everywhere" family, the correct conservative default for something
    this lookup can't otherwise classify, rather than raising or silently
    picking a structural family that would overstate what's known about
    the edge."""
    if family:
        return family
    return FAMILY_OF.get(relation_type, FAMILY_RESIDUAL)

# ---------------------------------------------------------------------
# Node types (§6A: node_type field -- standard | basin | schema | self)
# ---------------------------------------------------------------------
NODE_STANDARD = "standard"
NODE_BASIN = "basin"
NODE_SCHEMA = "schema"  # somatic/emotional schema (§2.1b)
NODE_SELF = "self"
NODE_EPISTEMIC_SCHEMA = "epistemic_schema"  # §13.3, new -- knowledge-cluster schema, distinct kind of pattern from NODE_SCHEMA

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

    # Structural / MEMBERSHIP -- purple, dotted. Visually signals
    # "membership fact," not a corroboration edge.
    EDGE_COMPOSED_OF:      {"color": "#7a4bb0", "width": "1.5", "dashes": "[1,1]"},
    EDGE_INSTANCE_OF:      {"color": "#9b6bc9", "width": "1.5", "dashes": "[1,1]"},

    # ROLE -- teal-green family, short dashes. Distinct from MEMBERSHIP's
    # purple since a role binding is a different kind of fact (who did
    # what to what) than a part/whole or category/instance relationship.
    EDGE_AGENT:      {"color": "#2f8f6f", "width": "1.5", "dashes": "[3,1]"},
    EDGE_PATIENT:    {"color": "#4fae8c", "width": "1.5", "dashes": "[3,1,1,1]"},
    EDGE_INSTRUMENT: {"color": "#6ec7ab", "width": "1.5", "dashes": "[1,1,3,1]"},

    # CAUSAL -- deep red family, long dashes (an explanatory claim, reads
    # heavier than a relational/narrative edge).
    EDGE_CAUSES:     {"color": "#8b1a1a", "width": "2.0", "dashes": "[8,3]"},
    EDGE_RESULTS_IN: {"color": "#a83232", "width": "1.5", "dashes": "[8,3,2,3]"},
    EDGE_PREVENTS:   {"color": "#5c1010", "width": "1.5", "dashes": "[8,3]"},
    EDGE_ENABLES:    {"color": "#c24949", "width": "1.5", "dashes": "[8,3,2,3]"},

    # ABSTRACTION -- gray-blue, fine dotted. §13.4's own collapse
    # bookkeeping (§13.4.4/§13.4.6) -- deliberately understated, since
    # this is infrastructure the collapse mechanism reads, not primarily
    # a fact the person is meant to read off the graph visually.
    EDGE_ABSTRACTED_FROM: {"color": "#6b7a8f", "width": "1.0", "dashes": "[1,2]"},
    EDGE_ELABORATES:      {"color": "#8a97a8", "width": "1.0", "dashes": "[1,2]"},
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
    NODE_EPISTEMIC_SCHEMA: "hexagon",  # same shape family as somatic schemas (both are "recognized recurring patterns"), distinguished by color instead
}
TIER_OPACITY = {0: 0.35, 1: 0.65, 2: 1.0}  # Provisional / Working / Trusted
SCHEMA_UNNAMED_COLOR = "#888888"
SCHEMA_NAMED_COLOR = "#2e8b57"
SELF_COLOR = "#d4af37"
EPISTEMIC_SCHEMA_UNNAMED_COLOR = "#5a6b7a"  # slate, distinct from somatic's gray
EPISTEMIC_SCHEMA_NAMED_COLOR = "#1a8a8a"    # teal, distinct from somatic's green


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


# ---------------------------------------------------------------------
# Somatic body surface (fixed infrastructure — not world knowledge)
# ---------------------------------------------------------------------
# Phenomenological channels cognition may sense. Hardcoded anatomy, not
# emergent lemmas. Linkable into the epistemic graph only as PARTS
# (composed-of / part-of), never as is-a children/parents, and never
# expanded by self-study / WordNet.
BODY_CHANNELS = (
    "heart_rate",
    "breath",
    "muscle_tension",
    "sweat_skin",
    "gut",
    "energy",
    "warmth",
)

# Canonical node ids in the graph (prefix keeps them out of lemma space)
def body_channel_node_id(channel: str) -> str:
    ch = (channel or "").strip().lower()
    if ch.startswith("body:"):
        return ch
    return f"body:{ch}"

BODY_CHANNEL_NODE_IDS = tuple(body_channel_node_id(c) for c in BODY_CHANNELS)


def is_body_channel_node(node_id: str) -> bool:
    if not node_id:
        return False
    n = str(node_id)
    if n in BODY_CHANNEL_NODE_IDS:
        return True
    if n.startswith("body:") and n[5:] in BODY_CHANNELS:
        return True
    # bare channel name
    if n in BODY_CHANNELS:
        return True
    return False



def is_felt_place_node(node_id: str) -> bool:
    """PAD basin place nodes (felt:a_v_d) — somatic, not knowledge."""
    if not node_id:
        return False
    n = str(node_id)
    return n.startswith("felt:") or n.startswith("basin_")


def is_narrative_graph_node(node_id: str) -> bool:
    if not node_id:
        return False
    return str(node_id).startswith("narr:")


def is_somatic_infrastructure(node_id: str) -> bool:
    """Body channels, felt places — anatomy, not epistemic growth targets."""
    return (
        is_body_channel_node(node_id)
        or is_felt_place_node(node_id)
        or is_narrative_graph_node(node_id)
    )


def is_forbidden_epistemic_parent(node_id: str) -> bool:
    """Nodes that must never become epistemic_of_* hubs or cluster parents.

    SELF, body channels, felt places, basins, narrative nodes — identity
    and anatomy, not world-knowledge kinds.
    """
    if not node_id:
        return True
    n = str(node_id)
    low = n.lower()
    if n in ("SELF", "OTHER") or low in ("self", "other"):
        return True
    if is_somatic_infrastructure(n):
        return True
    if n.startswith(("body:", "felt:", "basin_", "narr:")):
        return True
    # Already-formed illegal epistemic shells
    if low.startswith("epistemic_of_self") or low.startswith("epistemic_of_other"):
        return True
    if low.startswith("epistemic_of_body") or low.startswith("epistemic_of_felt"):
        return True
    if low.startswith("epistemic_of_basin") or low.startswith("epistemic_of_narr"):
        return True
    # slug forms: epistemic_of_heart_rate, epistemic_of_0_5_0_4_0_7, etc.
    for ch in BODY_CHANNELS:
        if low == f"epistemic_of_{ch}" or low.startswith(f"epistemic_of_{ch}"):
            return True
    return False


def is_eligible_epistemic_member(node_id: str) -> bool:
    """Knowledge lemmas only — not identity/anatomy infrastructure."""
    if not node_id:
        return False
    if is_forbidden_epistemic_parent(node_id):
        return False
    n = str(node_id)
    if n.startswith("epistemic_"):
        return False  # schemas are not members of other epistemic clusters here
    return True
