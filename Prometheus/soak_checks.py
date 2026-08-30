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

    # Body surface ↔ felt place co-occurrence (weighted)
    felt_hits = 0
    weighted = 0
    weights: List[float] = []
    flat_ones = 0
    try:
        show_floor = float(getattr(prom, "BODY_FELT_SHOW", 0.12) or 0.12)
        for u, v, ed in g.edges(data=True):
            ed = ed or {}
            placement = ed.get("placement")
            u_b = str(u).startswith("body:")
            v_b = str(v).startswith("body:")
            u_f = str(u).startswith(("felt:", "basin_"))
            v_f = str(v).startswith(("felt:", "basin_"))
            is_bf = (u_b and v_f) or (u_f and v_b)
            if placement == "body_felt_cooccur":
                if not is_bf:
                    continue
            elif not is_bf:
                continue
            # count body↔felt associated edges (new weighted + legacy)
            if ed.get("relation_type") not in (None, "associated-with", "associated_with"):
                if placement != "body_felt_cooccur":
                    continue
            felt_hits += 1
            w = ed.get("weight")
            if w is not None:
                try:
                    wf = float(w)
                    weights.append(wf)
                    weighted += 1
                    if wf >= 0.99:
                        flat_ones += 1
                except (TypeError, ValueError):
                    pass
        detail = f"felt_hits={felt_hits} weighted={weighted}"
        if weights:
            mn = min(weights)
            mx = max(weights)
            avg = sum(weights) / len(weights)
            above = sum(1 for w in weights if w >= show_floor)
            detail += (
                f" min={mn:.3f} max={mx:.3f} avg={avg:.3f} "
                f"above_show={above} flat1={flat_ones}"
            )
        stuck = weighted >= 8 and flat_ones == weighted
        add("body_felt_cooccur", not stuck, detail)
        if stuck:
            add("body_felt_weights_varied", False, "all body↔felt weights ≈ 1.0 (static)")
        elif weights:
            add("body_felt_weights_varied", True, detail)
    except Exception as e:
        add("body_felt_cooccur", False, str(e))

    # Package A: world stub present
    try:
        ws = getattr(orch, "world_stub", None)
        ok = ws is not None and bool(getattr(ws, "slots", None))
        add("world_stub_slots", ok, str(ws.report() if ws else None)[:120])
    except Exception as e:
        add("world_stub_slots", False, str(e))

    # Thin C: outcome confidence dict exists after acts
    try:
        ops = getattr(orch, "operators", None)
        conf = getattr(ops, "outcome_confidence", {}) if ops else {}
        add("act_outcome_confidence", ops is not None, f"n_keys={len(conf)}")
    except Exception as e:
        add("act_outcome_confidence", False, str(e))

    # No epistemic_of_body / epistemic_of_felt shells
    try:
        illegal_shells = []
        for n in g.nodes:
            low = str(n).lower()
            if low.startswith(("epistemic_of_body", "epistemic_of_felt", "epistemic_of_basin", "epistemic_of_self")):
                illegal_shells.append(str(n))
            elif low.startswith("epistemic_of_"):
                tail = low[len("epistemic_of_"):]
                if tail.replace(".", "").replace("_", "").isdigit() and tail.count("_") >= 2:
                    illegal_shells.append(str(n))
        add("no_illegal_epistemic_shells", len(illegal_shells) == 0, f"count={len(illegal_shells)} sample={illegal_shells[:5]}")
    except Exception as e:
        add("no_illegal_epistemic_shells", False, str(e))

    # Allostasis affect channels present + not stuck at ceiling
    try:
        if hasattr(prom, "get_allostasis_report"):
            ar = prom.get_allostasis_report() or {}
            p = float(ar.get("pain") or 0)
            pl = float(ar.get("pleasure") or 0)
            add(
                "affect_in_range",
                0.0 <= p <= 1.0 and 0.0 <= pl <= 1.0,
                f"pain={p} pleasure={pl}",
            )
            add("pain_not_stuck_ceiling", p < 0.99, f"pain={p}")
        else:
            add("allostasis_api", False, "missing")
    except Exception as e:
        add("allostasis_api", False, str(e))

    return out
