"""
self_narrative.py -- Visible layer (§7, §16 of the design addendum).

Owns the Self-Narrative: a single, slowly-evolving, bounded collection of
Narrative Elements -- NOT prose, NOT a chat log. Each element is a
structured pointer at real graph entities (a Schema Node, a relational
event node, a basin pair) plus a decaying salience weight. This is the
"longer consolidated current state" the person asked for, as distinct
from chronos.py's bounded rolling log (raw ticks) and the Active Thread
(not yet built -- short-horizon, single, volatile "what am I doing right
now"). The Self-Narrative is the opposite timescale: what has turned out
to matter, across many Consolidation passes, not what's dominant this
moment.

Consolidation-gated only, same "one clock, not several" principle as
every other offline-reprocessing mechanism in this design. Nothing here
is a black box: every trigger (§16.3) reads data this codebase already
computes elsewhere (schema formation, valence coloring, co-activation,
relational edges, basin dwell-density) -- no new inference, no embedding/
similarity machinery, consistent with §3.4's standing rejection of that
category of mechanism for this engine.

Implemented against archivist.py/reflector.py/synthesizer.py/edge_types.py
as they actually exist in this codebase, not the addendum's abstract
sketch -- concretely: "high-stickiness recurrence" (§16.3 trigger 3) uses
co-activation-pair degree as its proxy (a real, already-computed signal),
not a new reinforcement counter; the working-memory and regulation
influence channels (§16.5.1/§16.5.2) are both implemented as one merge
into Prometheus.py's existing anchor pool (the same mechanism this
session's felt_state_anchors/_global_protected_anchors fix already
established and validated), since the real get_working_memory()/
eligible_regulation_nodes() don't have a separate always_include-style
hook the way archivist.working_memory_nodes() does. The schema-formation
discount (§16.5.3) is NOT wired in this revision -- flagged honestly as
deferred rather than rushed against reflector.py's already carefully-
tuned detection thresholds without dedicated review.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from .archivist import SELF_NODE, OTHER_NODE, TIER_PROVISIONAL
from .edge_types import (
    EDGE_IS_A, EDGE_PART_OF, EDGE_COMPOSED_OF, RELATIONAL_EDGE_TYPES,
)

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
NARRATIVE_STATE_PATH = os.path.join(_DATA_DIR, "self_narrative.json")

_STRUCTURAL_ANCESTOR_EDGE_TYPES = frozenset({EDGE_IS_A, EDGE_PART_OF, EDGE_COMPOSED_OF})

ELEMENT_SOMATIC_SCHEMA = "somatic_schema"
ELEMENT_EPISTEMIC_SCHEMA = "epistemic_schema"
ELEMENT_RELATIONAL_PATTERN = "relational_pattern"
ELEMENT_PARENTAL_COLORING = "parental_coloring"
ELEMENT_BASIN_SHIFT = "basin_shift"


class NarrativeModule:
    """
    §16. Reads archivist.graph, synthesizer's basin state, and reflector's
    per-pass schema-formation results; owns nothing outside its own
    bounded element set and a small amount of basin-shift/reinforcement
    bookkeeping. Never grows the graph, never scores trust, never
    regulates -- same "reads the finished state, produces insight about
    it" posture reflector.py already established for itself (§4A).
    """

    # §21-category tuning placeholders throughout -- none of these are
    # claimed-final values, same status as every other constant in this
    # design. Exposed as instance attributes (not just class constants)
    # so a Debug-tab slider could tune them live, matching the existing
    # pattern used everywhere else in this codebase.
    NARRATIVE_ELEMENT_CAP = 200
    NARRATIVE_STICKINESS_THRESHOLD = 3          # co-activation-pair degree
    NARRATIVE_COLORING_THRESHOLD = 0.5          # fraction of valence_coloring's 1.0 cap
    NARRATIVE_RELATIONAL_THRESHOLD = 2          # recurrence count, deliberately lower than schema stabilization (§16.3 trigger 5)
    NARRATIVE_BASIN_SHIFT_WINDOW = 20           # Consolidation passes
    NARRATIVE_DECAY_RATE = 0.9                  # multiplicative, per Consolidation pass
    NARRATIVE_WEIGHT_CAP = 10.0
    NARRATIVE_ABSORPTION_FLOOR = 1.0            # below this, try absorption into a structural ancestor
    NARRATIVE_PRUNE_FLOOR = 0.1                 # below this (and no absorption target), drop entirely
    NARRATIVE_WM_SALIENCE_FLOOR = 2.0           # threshold for linked_nodes_above_floor()
    NARRATIVE_NEGATION_PENALTY = 2.0            # flat weight subtraction, larger than normal decay (§16.6)
    NARRATIVE_AFFECT_AMPLIFICATION = 2.0        # multiplier on trigger-5 increment under high intensity (§16.6)
    REINFORCE_STEP = 1.0                        # base weight increment per trigger firing

    def __init__(self, archivist, synthesizer):
        self.archivist = archivist
        self.synthesizer = synthesizer
        self.elements: Dict[str, dict] = {}
        self._next_id = 0

        # §16.3 trigger 3 bookkeeping: nodes already credited as "sticky"
        # this lifetime. Bug fix (found via testing, this session): this
        # was originally a `set` -- credit once, then permanently excluded
        # from ever reinforcing again, even as real co-activation degree
        # kept genuinely growing. That directly undermined §16's own
        # "Robust: survives many Consolidation cycles" requirement --
        # every trigger-3/5 element got exactly one reinforcement ever,
        # then decayed to nothing within ~22 passes regardless of whether
        # the underlying pattern was still actively recurring. Now a dict
        # of node -> degree-at-last-credit, so a node whose degree keeps
        # climbing (genuinely new co-activation partners, not a static
        # unchanged condition) keeps earning fresh reinforcement.
        self._stickiness_credited: Dict[str, int] = {}

        # §16.3 trigger 5 bookkeeping: per-target relational-edge-pattern
        # recurrence counts, keyed by (target_node, frozenset(relation_types)).
        # Separate from reflector's own schema-detection counting --
        # deliberately a lower, faster bar (§16.3), not a duplicate of it.
        self._relational_recurrence: Dict[tuple, int] = {}
        # Same fix as _stickiness_credited above: dict of key -> count-at-
        # last-credit, not a permanent one-shot set.
        self._relational_credited: Dict[tuple, int] = {}

        # §16.3 trigger 6 bookkeeping: recent history of "the dominant
        # stabilized basin as of this pass", for comparing against
        # NARRATIVE_BASIN_SHIFT_WINDOW passes ago.
        self._dominant_basin_history: List[Optional[str]] = []

        # Pulse-time stream of consciousness (not consolidation-only)
        self.stream: List[dict] = []
        self.STREAM_CAP = 80
        self._stream_last_line = ""
        self.load()

    # ------------------------------------------------------------------
    # Main Consolidation-pass entry point
    # ------------------------------------------------------------------
    def evaluate(self, new_somatic_schema_ids: List[str], new_epistemic_schema_ids: List[str],
                 current_intensity: float = 0.0) -> Dict[str, int]:
        """
        Called once per Consolidation pass, from prometheus.py's
        _run_consolidation(), after reflector's own schema-detection
        passes have run (so this pass's newly-formed schemas are
        available as trigger 1/2 material the same cycle, not a pass
        late -- same ordering rationale already used elsewhere in that
        method). `current_intensity` is synthesizer.get_current_intensity()
        at the moment of evaluation, used only for §16.6's affect
        amplification -- the already-synthesized composite signal, never
        a raw hidden-layer value (Core Emergence Principle).

        Returns a small summary dict for logging, matching the existing
        convention (archivist.run_consolidation_pass() etc).
        """
        created = 0
        reinforced = 0

        for schema_id in new_somatic_schema_ids:
            if self._reinforce_or_create(ELEMENT_SOMATIC_SCHEMA, [schema_id]):
                created += 1
            else:
                reinforced += 1

        for schema_id in new_epistemic_schema_ids:
            if self._reinforce_or_create(ELEMENT_EPISTEMIC_SCHEMA, [schema_id]):
                created += 1
            else:
                reinforced += 1

        c, r = self._trigger_stickiness()
        created += c
        reinforced += r

        c, r = self._trigger_parental_coloring()
        created += c
        reinforced += r

        c, r = self._trigger_relational_recurrence(current_intensity)
        created += c
        reinforced += r

        c, r = self._trigger_basin_shift()
        created += c
        reinforced += r

        absorbed, pruned = self._decay_and_absorb()

        return {
            "created": created, "reinforced": reinforced,
            "absorbed": absorbed, "pruned": pruned,
            "total_elements": len(self.elements),
        }

    # ------------------------------------------------------------------
    # §16.3 triggers
    # ------------------------------------------------------------------

    def record_goal_event(self, event: str, target_id: str, detail: str = "", pulse: int = 0) -> bool:
        """Narrative beat for goal OPEN / satisfied / failed.

        Links SELF + target so continuity tracks commitments.
        """
        try:
            nodes = ["SELF", target_id] if target_id else ["SELF"]
            nodes = [n for n in nodes if n]
            # Prefer epistemic element if target is a schema
            el_type = ELEMENT_EPISTEMIC_SCHEMA
            try:
                nt = self.archivist.graph.nodes.get(target_id, {}).get("node_type")
                if nt not in ("epistemic_schema", "schema"):
                    el_type = ELEMENT_SOMATIC_SCHEMA
            except Exception:
                pass
            # Use reinforce_or_create if available
            if hasattr(self, "_reinforce_or_create"):
                created = self._reinforce_or_create(el_type, nodes)
            else:
                created = False
            # Stamp a lightweight annotation on the freshest element if possible
            try:
                for eid, el in list(getattr(self, "elements", {}).items())[-3:]:
                    if target_id in (el.get("nodes") or []):
                        el["goal_event"] = event
                        el["goal_detail"] = detail
                        el["goal_pulse"] = pulse
                        break
            except Exception:
                pass
            return bool(created)
        except Exception as e:
            logger.warning("record_goal_event failed: %s", e)
            return False

    def _trigger_stickiness(self) -> tuple:
        """Trigger 3: co-activation-pair degree crossing threshold, using
        archivist.stabilized_co_activation_pairs() -- a real, already-
        computed signal, not a new counter. A node's "stickiness" here is
        how many OTHER stabilized-pair partners it has, i.e. how central
        it is in the current co-activation structure."""
        degree: Dict[str, int] = {}
        for a, b in self.archivist.stabilized_co_activation_pairs():
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

        created = reinforced = 0
        for node, d in degree.items():
            if d < self.NARRATIVE_STICKINESS_THRESHOLD:
                continue
            if d <= self._stickiness_credited.get(node, 0):
                continue  # no NEW recurrence since last credit -- don't re-fire on a static condition
            self._stickiness_credited[node] = d
            node_type = self.archivist.graph.nodes.get(node, {}).get("node_type")
            element_type = ELEMENT_EPISTEMIC_SCHEMA if node_type == "epistemic_schema" else ELEMENT_SOMATIC_SCHEMA
            if self._reinforce_or_create(element_type, [node]):
                created += 1
            else:
                reinforced += 1
        return created, reinforced

    def _trigger_parental_coloring(self) -> tuple:
        """Trigger 4: |valence_coloring| crossing threshold. Re-fires
        (reinforces) on every pass a node stays above threshold -- unlike
        stickiness, a strongly-colored node staying strongly colored is
        itself ongoing evidence, not a one-time fact, so continuous
        reinforcement here (rather than a credited-once set) is
        intentional."""
        created = reinforced = 0
        for node, data in self.archivist.graph.nodes(data=True):
            coloring = data.get("valence_coloring", 0.0)
            if abs(coloring) < self.NARRATIVE_COLORING_THRESHOLD:
                continue
            if self._reinforce_or_create(ELEMENT_PARENTAL_COLORING, [node], sign=coloring):
                created += 1
            else:
                reinforced += 1
        return created, reinforced

    def _trigger_relational_recurrence(self, current_intensity: float) -> tuple:
        """Trigger 5: repeated SELF/OTHER-anchored relational edges,
        deliberately below reflector's own schema-stabilization threshold
        (§16.3) -- this is meant to pick up "this keeps happening to me"
        earlier than full schema formation. Groups by (target, relation
        type set) same as reflector.detect_schemas()'s own grouping, so
        the two mechanisms are counting the same underlying pattern, just
        at different bars."""
        graph = self.archivist.graph
        event_relations: Dict[str, set] = {}
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation_type")
            if rel in RELATIONAL_EDGE_TYPES and u in (SELF_NODE, OTHER_NODE):
                event_relations.setdefault(v, set()).add(rel)

        created = reinforced = 0
        for target, relation_set in event_relations.items():
            key = (target, frozenset(relation_set))
            self._relational_recurrence[key] = self._relational_recurrence.get(key, 0) + 1
            count = self._relational_recurrence[key]
            if count < self.NARRATIVE_RELATIONAL_THRESHOLD:
                continue
            if count <= self._relational_credited.get(key, 0):
                continue  # no NEW recurrence since last credit -- don't re-fire on a static condition
            self._relational_credited[key] = count
            # §16.6 affect amplification: emotionally charged experiences
            # leave a stronger mark. Only affects the INITIAL creation
            # increment for this trigger, not the base REINFORCE_STEP used
            # everywhere else -- deliberately narrow, matching the spec's
            # own scoping of this as a trigger-5-specific amplification.
            step = self.REINFORCE_STEP
            if current_intensity > 0.6:
                step *= self.NARRATIVE_AFFECT_AMPLIFICATION
            if self._reinforce_or_create(ELEMENT_RELATIONAL_PATTERN, [target], step=step):
                created += 1
            else:
                reinforced += 1
        return created, reinforced

    def _trigger_basin_shift(self) -> tuple:
        """Trigger 6: comparing the highest-dwell-density stabilized basin
        now versus NARRATIVE_BASIN_SHIFT_WINDOW passes ago. Uses
        synthesizer.basin_grid (dwell density) restricted to keys already
        in synthesizer.stabilized_basins (only named felt states count as
        "dominant," not raw unstabilized density)."""
        stabilized = self.synthesizer.stabilized_basins
        grid = self.synthesizer.basin_grid
        dominant = None
        if stabilized:
            dominant = max(
                (k for k in stabilized if k in grid),
                key=lambda k: grid.get(k, 0.0),
                default=None,
            )
        dominant_id = stabilized.get(dominant) if dominant is not None else None

        self._dominant_basin_history.append(dominant_id)
        if len(self._dominant_basin_history) > self.NARRATIVE_BASIN_SHIFT_WINDOW + 1:
            self._dominant_basin_history.pop(0)

        created = reinforced = 0
        if len(self._dominant_basin_history) > self.NARRATIVE_BASIN_SHIFT_WINDOW:
            past = self._dominant_basin_history[0]
            current = self._dominant_basin_history[-1]
            if past and current and past != current:
                if self._reinforce_or_create(ELEMENT_BASIN_SHIFT, [past, current]):
                    created += 1
                else:
                    reinforced += 1
        return created, reinforced

    # ------------------------------------------------------------------
    # Element creation / reinforcement
    # ------------------------------------------------------------------
    def _element_id_for(self, element_type: str, linked_nodes: List[str]) -> str:
        return f"narr_{element_type}_{'_'.join(sorted(linked_nodes))}"

    def _reinforce_or_create(self, element_type: str, linked_nodes: List[str],
                              sign: Optional[float] = None, step: Optional[float] = None) -> bool:
        """Returns True if a new element was created, False if an
        existing one was reinforced instead. `step` overrides
        REINFORCE_STEP for this call only (used by trigger 5's affect
        amplification)."""
        step = self.REINFORCE_STEP if step is None else step
        eid = self._element_id_for(element_type, linked_nodes)
        now = datetime.now().isoformat()
        if eid in self.elements:
            el = self.elements[eid]
            el["weight"] = min(self.NARRATIVE_WEIGHT_CAP, el["weight"] + step)
            el["last_reinforced_at"] = now
            if sign is not None:
                el["sign"] = sign
            return False

        if len(self.elements) >= self.NARRATIVE_ELEMENT_CAP:
            self._evict_lowest_weight()

        self.elements[eid] = {
            "element_id": eid,
            "element_type": element_type,
            "linked_nodes": list(linked_nodes),
            "weight": step,
            "sign": sign,
            "formed_at": now,
            "last_reinforced_at": now,
            "absorbed_from": [],
            "predecessors": [],   # narrative chain: earlier related elements
            "successors": [],     # narrative chain: later related elements
        }
        # Attempt light chaining: link to recent high-weight elements that
        # share a node or are the same type (deterministic co-occurrence).
        self._try_chain(eid)
        return True

    def _evict_lowest_weight(self):
        """Hard cap enforcement (§16.2), distinct from decay-driven
        absorption/pruning (§16.4) -- only triggers if the bounded set is
        genuinely full at creation time, not part of the normal per-pass
        lifecycle."""
        if not self.elements:
            return
        lowest_id = min(self.elements, key=lambda k: self.elements[k]["weight"])
        del self.elements[lowest_id]

    # ------------------------------------------------------------------
    # §16.4 Decay & Absorption
    # ------------------------------------------------------------------
    def _structural_parent(self, node: str) -> Optional[str]:
        """First is-a/part-of/composed-of predecessor of `node` -- i.e.
        its structural parent. Edge direction convention in this codebase
        (archivist.link(parent, term, edge_type)) creates parent->term,
        so a node's structural ancestor is found via its IN-edges of
        these types, not its out-edges."""
        graph = self.archivist.graph
        if node not in graph:
            return None
        for u, _v, data in graph.in_edges(node, data=True):
            if data.get("relation_type") in _STRUCTURAL_ANCESTOR_EDGE_TYPES:
                return u
        return None

    def _element_covering(self, node: str) -> Optional[str]:
        """Finds an existing element whose linked_nodes already includes
        `node`, if any."""
        for eid, el in self.elements.items():
            if node in el["linked_nodes"]:
                return eid
        return None

    def _decay_and_absorb(self) -> tuple:
        """Consolidation-gated multiplicative decay (same shape as basin/
        co-activation/regulatory-efficacy decay throughout this design),
        with absorption into a structural ancestor's element when one
        already exists and is itself tracked (§16.4's "or is eligible to
        become one" clause is deliberately NOT implemented here --
        speculatively re-running all six triggers against an arbitrary
        ancestor on every absorption check would be expensive and hard to
        verify; absorbing only into an ALREADY-tracked ancestor element is
        a safe, honest simplification of the addendum's fuller spec, not
        a silent shortfall)."""
        absorbed = 0
        pruned = 0
        for eid in list(self.elements.keys()):
            el = self.elements[eid]
            el["weight"] *= self.NARRATIVE_DECAY_RATE
            if el["weight"] >= self.NARRATIVE_ABSORPTION_FLOOR:
                continue

            # Try absorption into a structural ancestor's element first.
            absorbed_here = False
            for node in el["linked_nodes"]:
                parent = self._structural_parent(node)
                if parent is None:
                    continue
                parent_eid = self._element_covering(parent)
                if parent_eid is None or parent_eid == eid:
                    continue
                target = self.elements[parent_eid]
                target["weight"] = min(self.NARRATIVE_WEIGHT_CAP, target["weight"] + el["weight"])
                target["absorbed_from"].append(eid)
                target["last_reinforced_at"] = datetime.now().isoformat()
                del self.elements[eid]
                absorbed += 1
                absorbed_here = True
                break
            if absorbed_here:
                continue

            if el["weight"] < self.NARRATIVE_PRUNE_FLOOR:
                del self.elements[eid]
                pruned += 1
        return absorbed, pruned

    # ------------------------------------------------------------------
    # §16.6 Revision -- negation & (partial) affect amplification
    # ------------------------------------------------------------------
    def apply_negation_penalty(self, node: str):
        """§16.6: called from prometheus.py wherever archivist.
        flag_negation() already fires (§3.4 mechanism 1) -- if the negated
        node is covered by an existing narrative element, that element
        takes an immediate, larger-than-normal-decay cut, rather than
        just waiting for the next ordinary decay pass. A correction to
        something narratively significant should land harder than an
        ordinary fact getting demoted."""
        eid = self._element_covering(node)
        if eid is None:
            return
        self.elements[eid]["weight"] = max(0.0, self.elements[eid]["weight"] - self.NARRATIVE_NEGATION_PENALTY)

    # ------------------------------------------------------------------
    # §16.5 Influence channels -- working memory (16.5.1) + regulation
    # (16.5.2), unified into one merge point. See module docstring for
    # why this differs from the addendum's original three-separate-
    # channels sketch: the real get_working_memory()/
    # eligible_regulation_nodes() don't have an always_include-style hook
    # the way archivist.working_memory_nodes() does, so both are served
    # by feeding into the same expanded anchor pool this session's
    # felt_state_anchors fix already established. §16.5.3 (schema-
    # formation discount) is not wired -- see module docstring.
    # ------------------------------------------------------------------
    def linked_nodes_above_floor(self) -> List[str]:
        """Real graph nodes referenced by elements at or above
        NARRATIVE_WM_SALIENCE_FLOOR -- meant to be merged into
        prometheus.py's _get_unique_anchors() output, the same way
        _global_protected_anchors already is."""
        result = []
        for el in self.elements.values():
            if el["weight"] >= self.NARRATIVE_WM_SALIENCE_FLOOR:
                result.extend(el["linked_nodes"])
        return list(dict.fromkeys(n for n in result if n in self.archivist.graph))

    # ------------------------------------------------------------------
    # Diagnostics (matches the existing *_report() naming convention
    # already used by reflector.py -- activation_report,
    # valence_coloring_report, etc.)
    # ------------------------------------------------------------------
    _ELEMENT_TYPE_LABELS = {
        ELEMENT_SOMATIC_SCHEMA: "Recurring emotional pattern",
        ELEMENT_EPISTEMIC_SCHEMA: "Knowledge cluster",
        ELEMENT_RELATIONAL_PATTERN: "Repeated self-relevant experience",
        ELEMENT_PARENTAL_COLORING: "Strongly felt association",
        ELEMENT_BASIN_SHIFT: "Emotional shift",
    }

    def _pad_description(self, node: str) -> str:
        """Plain, deterministic PAD-coordinate-to-adjective mapping for
        basin nodes -- basins are never given human names by this design
        (§2.1a: felt states are earned/named only via §6.1's knowledge-
        node linkage, which most basins never receive), so a raw
        `basin_0.5_0.2_0.5`-style id is the only identifier most basins
        ever have. This turns that id's own coordinates into a short,
        readable phrase instead -- no new data, just decoding what the id
        already encodes.

        Bug fix (found via testing): an earlier 2-bucket-per-axis version
        of this let two genuinely different basins collapse into an
        identical-sounding description (e.g. a basin-shift element
        reading "from a high-arousal, positive, in-control state to a
        high-arousal, positive, in-control state" -- technically
        accurate per-axis, but reads as no shift at all when the actual
        coordinates clearly differed). Finer 3-level bucketing per axis
        plus the raw numbers alongside fixes this -- still deliberately
        plain/mechanical, not evocative prose, consistent with this
        project's preference for boring, checkable descriptions over
        flavorful ones that could misrepresent a bare coordinate."""
        data = self.archivist.graph.nodes.get(node, {})
        if data.get("name"):
            return data["name"]
        pad = data.get("pad_coordinates")
        if pad is None and node.startswith("basin_"):
            try:
                pad = tuple(float(x) for x in node[len("basin_"):].split("_"))
            except ValueError:
                pad = None
        if pad is None or len(pad) < 3:
            return node
        arousal, valence, dominance = pad[0], pad[1], pad[2]

        def bucket(v: float, low: str, mid: str, high: str, low_hi: float, high_lo: float) -> str:
            if v < low_hi:
                return low
            if v >= high_lo:
                return high
            return mid

        arousal_word = bucket(arousal, "low-arousal", "moderate-arousal", "high-arousal", 0.35, 0.65)
        valence_word = bucket(valence, "negative", "mixed/neutral", "positive", -0.2, 0.2)
        dominance_word = bucket(dominance, "overwhelmed", "somewhat in-control", "in-control", 0.35, 0.65)
        return (f"{arousal_word} ({arousal:.2f}), {valence_word} ({valence:.2f}), "
                f"{dominance_word} ({dominance:.2f})")

    def _describe_element(self, el: dict) -> str:
        """Resolves an element's linked_nodes into a human-readable
        description using data that already exists on the graph (name
        fields, member_count, relation_types, basin coordinates) --
        display-time only, never stored, never influences any decision
        logic. The raw element_id/linked_nodes are still returned
        alongside this in report()'s output for anyone who wants the
        exact underlying data."""
        etype = el["element_type"]
        nodes = el["linked_nodes"]
        graph = self.archivist.graph

        def label(n: str) -> str:
            data = graph.nodes.get(n, {})
            return data.get("name") or n

        if etype in (ELEMENT_SOMATIC_SCHEMA, ELEMENT_EPISTEMIC_SCHEMA):
            node = nodes[0]
            data = graph.nodes.get(node, {})
            if data.get("name"):
                return f'named "{data["name"]}"'
            if etype == ELEMENT_EPISTEMIC_SCHEMA:
                members = [
                    v for _u, v, d in graph.out_edges(node, data=True)
                    if d.get("relation_type") == EDGE_COMPOSED_OF
                ]
                preview = ", ".join(label(m) for m in members[:4])
                more = f" (+{len(members) - 4} more)" if len(members) > 4 else ""
                return f"not yet named -- {data.get('member_count', len(members))} member(s): {preview}{more}"
            relation_types = data.get("relation_types", [])
            basin_label = self._pad_description(data.get("basin", ""))
            return f"not yet named -- felt during {basin_label}, involving {', '.join(relation_types)}"

        if etype == ELEMENT_RELATIONAL_PATTERN:
            node = nodes[0]
            rel_types = sorted({
                d.get("relation_type") for u, v, d in graph.in_edges(node, data=True)
                if u in (SELF_NODE, OTHER_NODE) and d.get("relation_type") in RELATIONAL_EDGE_TYPES
            })
            return f'"{label(node)}" ({", ".join(rel_types) if rel_types else "relational"})'

        if etype == ELEMENT_PARENTAL_COLORING:
            node = nodes[0]
            sign = el.get("sign") or 0.0
            feeling = "warmly regarded" if sign >= 0 else "uneasily regarded"
            return f'"{label(node)}" -- {feeling} ({sign:+.2f})'

        if etype == ELEMENT_BASIN_SHIFT and len(nodes) >= 2:
            return f"from {self._pad_description(nodes[0])} to {self._pad_description(nodes[1])}"

        return ", ".join(label(n) for n in nodes)


    # ------------------------------------------------------------------
    # Stream of consciousness (§16 companion) -- pulse-time, not prose LLM
    # ------------------------------------------------------------------
    def record_stream_beat(
        self,
        pulse: int,
        focus_id: Optional[str] = None,
        felt_state: str = "",
        basin_key: str = "",
        wm_slots: Optional[List[str]] = None,
        goal_targets: Optional[List[str]] = None,
        bias: str = "",
        state: str = "",
        residual_top: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Append one short deterministic SoC line from live cognitive state.

        Not a chat log and not the consolidated Self-Narrative. This is the
        moment-to-moment trace: what is online, felt, aimed-at, in mind.
        Template-only (no LLM). Skips near-duplicate consecutive lines.
        """
        wm_slots = list(wm_slots or [])
        goal_targets = list(goal_targets or [])
        residual_top = list(residual_top or [])

        def label(n: str) -> str:
            if not n:
                return ""
            d = self.archivist.graph.nodes.get(n, {})
            name = d.get("name")
            if name and str(name).strip():
                return str(name).strip()
            if str(n).startswith("epistemic_of_"):
                return str(n)[len("epistemic_of_"):].replace("_", " ")
            if str(n).startswith("basin_"):
                return self._pad_description(n)
            return str(n)

        felt = felt_state if felt_state and felt_state != "Unformed" else "something unformed"
        focus = label(focus_id) if focus_id else ""
        goals = [label(g) for g in goal_targets[:3] if g]
        # WM: skip SELF-like, prefer short labels
        mind = []
        for s in wm_slots:
            if s in (SELF_NODE, OTHER_NODE, "SELF", "OTHER"):
                continue
            lab = label(s)
            if lab and lab not in mind:
                mind.append(lab)
            if len(mind) >= 4:
                break

        parts = []
        # Felt climate
        if felt_state == "Unformed" or not felt_state:
            parts.append("A haze; nothing has settled yet")
        else:
            parts.append(f"This {felt} state holds")

        # Focus
        if focus:
            parts.append(f"attention on {focus}")
        else:
            parts.append("attention drifts")

        # Goals
        if goals:
            parts.append("reaching for " + ", ".join(goals))

        # Working memory contents
        if mind:
            parts.append("in mind: " + ", ".join(mind[:4]))

        # Residual tension tips
        tips = [label(x) for x in residual_top[:2] if x and label(x) not in (focus,)]
        tips = [t for t in tips if t]
        if tips:
            parts.append("pull from " + " and ".join(tips))

        # Operating mode color
        if state == "Sleep" or (state and "Sleep" in str(state)):
            parts.append("the body is offline, sorting")
        elif bias in ("FORCE_EXPLORE", "BIAS_EXPLORE"):
            parts.append("a restlessness to look further")
        elif bias == "BIAS_STABILIZE":
            parts.append("a need to settle what is known")

        line = "; ".join(parts) + "."
        # De-dupe consecutive identical traces
        if line == getattr(self, "_stream_last_line", ""):
            return None
        self._stream_last_line = line

        beat = {
            "pulse": pulse,
            "line": line,
            "focus": focus_id,
            "felt": felt_state,
            "basin": basin_key,
            "wm": mind[:4],
            "goals": goal_targets[:3],
            "ts": datetime.now().isoformat(),
        }
        self.stream.append(beat)
        if len(self.stream) > self.STREAM_CAP:
            self.stream = self.stream[-self.STREAM_CAP:]
        return line

    def stream_report(self, last_n: int = 12) -> Dict:
        """Rolling stream-of-consciousness buffer for UI / diagnostics."""
        beats = list(self.stream[-last_n:]) if self.stream else []
        return {
            "count": len(self.stream),
            "cap": self.STREAM_CAP,
            "latest": beats[-1]["line"] if beats else "",
            "beats": beats,
        }

    def report(self, top_n: int = 10) -> Dict:
        ranked = sorted(self.elements.values(), key=lambda el: el["weight"], reverse=True)
        return {
            "total_elements": len(self.elements),
            "top_elements": [
                {
                    "element_id": el["element_id"],
                    "element_type": el["element_type"],
                    "type_label": self._ELEMENT_TYPE_LABELS.get(el["element_type"], el["element_type"]),
                    "description": self._describe_element(el),
                    "linked_nodes": el["linked_nodes"],
                    "weight": round(el["weight"], 3),
                    "sign": el["sign"],
                    "formed_at": el["formed_at"],
                    "last_reinforced_at": el["last_reinforced_at"],
                }
                for el in ranked[:top_n]
            ],
        }


    # ------------------------------------------------------------------
    # Narrative chaining + retrieval (new)
    # ------------------------------------------------------------------
    CHAIN_MAX_LINKS = 4          # max predecessors/successors per element
    CHAIN_MIN_WEIGHT = 1.5       # only chain to elements above this weight
    CHAIN_RECENCY_WINDOW = 12    # only consider elements reinforced recently (pass count proxy via weight)

    def _try_chain(self, new_eid: str) -> None:
        """Link a newly created/reinforced element to a few related prior
        elements that share linked_nodes or element_type. Deterministic,
        no embeddings. Builds short autobiographical chains.
        """
        if new_eid not in self.elements:
            return
        new_el = self.elements[new_eid]
        new_nodes = set(new_el["linked_nodes"])
        candidates = []
        for eid, el in self.elements.items():
            if eid == new_eid:
                continue
            if el.get("weight", 0) < self.CHAIN_MIN_WEIGHT:
                continue
            shared = new_nodes.intersection(el.get("linked_nodes") or [])
            type_match = 1.0 if el.get("element_type") == new_el.get("element_type") else 0.0
            score = len(shared) * 2.0 + type_match + min(el["weight"], 5.0) * 0.15
            if score >= 1.0:
                candidates.append((score, eid))
        candidates.sort(key=lambda t: -t[0])
        for _score, pred_id in candidates[: self.CHAIN_MAX_LINKS]:
            pred = self.elements.get(pred_id)
            if not pred:
                continue
            # bidirectional soft links (lists, capped)
            if new_eid not in pred.get("successors", []):
                pred.setdefault("successors", []).append(new_eid)
                pred["successors"] = pred["successors"][-self.CHAIN_MAX_LINKS:]
            if pred_id not in new_el.get("predecessors", []):
                new_el.setdefault("predecessors", []).append(pred_id)
                new_el["predecessors"] = new_el["predecessors"][-self.CHAIN_MAX_LINKS:]

    def retrieve_for_focus(
        self,
        focus_id: Optional[str] = None,
        basin_anchors: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[dict]:
        """Return the most relevant narrative elements for current focus /
        basin anchors. Used by focus residual seeding and LTI promotion.
        Ranking = weight * (1 + shared_nodes + chain_bonus).
        """
        seed_nodes = set()
        if focus_id:
            seed_nodes.add(focus_id)
        for n in basin_anchors or []:
            seed_nodes.add(n)
        if not seed_nodes and not self.elements:
            return []

        scored = []
        for eid, el in self.elements.items():
            w = float(el.get("weight") or 0)
            if w < 0.5:
                continue
            linked = set(el.get("linked_nodes") or [])
            shared = len(linked & seed_nodes)
            # chain bonus: if any predecessor/successor also touches seed
            chain_hit = 0
            for other_id in (el.get("predecessors") or []) + (el.get("successors") or []):
                other = self.elements.get(other_id)
                if other and set(other.get("linked_nodes") or []) & seed_nodes:
                    chain_hit += 1
            score = w * (1.0 + shared * 1.5 + min(chain_hit, 3) * 0.4)
            if score <= 0:
                continue
            scored.append((score, {
                "element_id": eid,
                "element_type": el.get("element_type"),
                "linked_nodes": list(linked),
                "weight": round(w, 3),
                "score": round(score, 3),
                "predecessors": list(el.get("predecessors") or []),
                "successors": list(el.get("successors") or []),
            }))
        scored.sort(key=lambda t: -t[0])
        return [row for _s, row in scored[:top_k]]

    def retrieval_node_ids(
        self,
        focus_id: Optional[str] = None,
        basin_anchors: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> List[str]:
        """Flattened unique graph node ids from retrieve_for_focus, for
        easy merge into focus residual boosts or LTI promotion."""
        rows = self.retrieve_for_focus(focus_id, basin_anchors, top_k=top_k)
        out = []
        seen = set()
        for row in rows:
            for n in row.get("linked_nodes") or []:
                if n not in seen and n in self.archivist.graph:
                    seen.add(n)
                    out.append(n)
        return out

    def chain_report(self, top_n: int = 10) -> dict:
        """Diagnostic: elements that participate in chains."""
        chained = [
            el for el in self.elements.values()
            if el.get("predecessors") or el.get("successors")
        ]
        chained.sort(key=lambda e: -float(e.get("weight") or 0))
        return {
            "chained_count": len(chained),
            "total_elements": len(self.elements),
            "top": [
                {
                    "element_id": el["element_id"],
                    "type": el.get("element_type"),
                    "weight": round(float(el.get("weight") or 0), 3),
                    "predecessors": el.get("predecessors") or [],
                    "successors": el.get("successors") or [],
                    "linked_nodes": el.get("linked_nodes") or [],
                }
                for el in chained[:top_n]
            ],
        }


    # ------------------------------------------------------------------
    # Persistence (§16.8) -- same JSON-per-module pattern as archivist.py/
    # synthesizer.py/chronos.py, checkpointed once at end-of-Consolidation
    # by prometheus.py, not on every element mutation.
    # ------------------------------------------------------------------
    def save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            data = {
                "elements": self.elements,
                "relational_recurrence": {
                    f"{k[0]}||{'|'.join(sorted(k[1]))}": v
                    for k, v in self._relational_recurrence.items()
                },
                "relational_credited": {
                    f"{k[0]}||{'|'.join(sorted(k[1]))}": v
                    for k, v in self._relational_credited.items()
                },
                "stickiness_credited": self._stickiness_credited,
                "dominant_basin_history": self._dominant_basin_history,
            }
            with open(NARRATIVE_STATE_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            logger.warning("NarrativeModule.save failed: %s", e)

    def load(self):
        if not os.path.exists(NARRATIVE_STATE_PATH):
            return
        try:
            with open(NARRATIVE_STATE_PATH, "r") as f:
                data = json.load(f)
            self.elements = data.get("elements", {})
            # Migration: older saves lack chain fields
            for el in self.elements.values():
                el.setdefault("predecessors", [])
                el.setdefault("successors", [])
            for k_str, v in data.get("relational_recurrence", {}).items():
                target, rels = k_str.split("||")
                rel_set = frozenset(rels.split("|")) if rels else frozenset()
                self._relational_recurrence[(target, rel_set)] = v
            credited_raw = data.get("relational_credited", {})
            if isinstance(credited_raw, list):
                # Backward compatibility with the old one-shot-set format
                # (pre-fix saves) -- treat any previously-credited key as
                # credited at its current recurrence count, so it doesn't
                # immediately re-fire on load just because the format
                # changed; genuinely new recurrence beyond that still
                # re-triggers normally going forward.
                for k_str in credited_raw:
                    target, rels = k_str.split("||")
                    rel_set = frozenset(rels.split("|")) if rels else frozenset()
                    self._relational_credited[(target, rel_set)] = self._relational_recurrence.get(
                        (target, rel_set), self.NARRATIVE_RELATIONAL_THRESHOLD)
            else:
                for k_str, v in credited_raw.items():
                    target, rels = k_str.split("||")
                    rel_set = frozenset(rels.split("|")) if rels else frozenset()
                    self._relational_credited[(target, rel_set)] = v
            stickiness_raw = data.get("stickiness_credited", {})
            if isinstance(stickiness_raw, list):
                # Same backward-compatibility handling as above.
                self._stickiness_credited = {n: self.NARRATIVE_STICKINESS_THRESHOLD for n in stickiness_raw}
            else:
                self._stickiness_credited = stickiness_raw
            self._dominant_basin_history = data.get("dominant_basin_history", [])
        except (json.JSONDecodeError, OSError, TypeError, ValueError, KeyError) as e:
            logger.warning("NarrativeModule.load failed, starting fresh: %s", e)
