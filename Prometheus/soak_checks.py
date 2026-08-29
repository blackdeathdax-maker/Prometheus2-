"""
soak_checks.py — Minimal invariant harness (Identity & Hygiene).

Call `run_soak_checks(prom)` after a soak or from Debug.
Returns a dict of pass/fail booleans + detail strings.
No network, no long graph walks beyond what's needed.
"""
from __future__ import annotations

from typing import Any, Dict, List


def run_soak_checks(prom) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "checks": []}

    def add(name: str, passed: bool, detail: str = ""):
        out["checks"].append({"name": name, "pass": bool(passed), "detail": detail})
        if not passed:
            out["ok"] = False

    try:
        from .archivist import SELF_NODE
        from .edge_types import is_body_channel_node
    except Exception:
        SELF_NODE = "SELF"
        is_body_channel_node = lambda n: str(n).startswith("body:")

    g = getattr(getattr(prom, "archivist", None), "graph", None)
    if g is None:
        add("graph_present", False, "no archivist.graph")
        return out
    add("graph_present", True, f"nodes={g.number_of_nodes()}")

    # SELF present
    add("self_present", SELF_NODE in g, "")

    # SELF → some body:*
    body_links = 0
    if SELF_NODE in g:
        for _u, v, ed in g.out_edges(SELF_NODE, data=True):
            if is_body_channel_node(v) or str(v).startswith("body:"):
                body_links += 1
    add("self_body_edges", body_links > 0, f"count={body_links}")

    # No is-a involving body:*
    illegal_isa = 0
    for u, v, ed in g.edges(data=True):
        if (ed or {}).get("relation_type") != "is-a":
            continue
        if is_body_channel_node(u) or is_body_channel_node(v):
            illegal_isa += 1
    add("no_body_is_a", illegal_isa == 0, f"illegal={illegal_isa}")

    # Narrative elements exist (optional soft)
    narr = sum(1 for n in g.nodes if str(n).startswith("narr:"))
    add("narrative_nonneg", narr >= 0, f"narr={narr}")

    # Active Thread API
    has_at = hasattr(prom, "get_active_thread_report")
    add("active_thread_api", has_at, "")
    if has_at:
        try:
            r = prom.get_active_thread_report() or {}
            add("active_thread_keys", "intent" in r, str(list(r.keys())[:8]))
        except Exception as e:
            add("active_thread_keys", False, str(e))

    # Identity hub API
    has_id = hasattr(prom, "get_identity_hub_report")
    add("identity_hub_api", has_id, "")

    return out
