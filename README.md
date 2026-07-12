# Prometheus

Implementation of `prometheus_design_spec.md` (rev. 11) on top of the
existing partial codebase. Run with:

```
pip install -r requirements.txt
streamlit run app.py
```

(Set `PROMETHEUS_DATA_DIR` to control where the JSON state files land;
defaults to `prometheus/data/`.)

## What changed from the codebase you had

- **`prometheus/chronos.py` — new file.** `prometheus.py` and
  `reflector.py` both imported `ChronosModule` from a module that didn't
  exist anywhere in the project, so nothing could actually run. It now
  implements the §7 responsibility: rolling pulse history, the
  felt-state→node link log that §6.1's naming-reliability gate reads from,
  structural-report summaries for `reflector.py`, and timestamp lookups
  for `temporal-contrast` edges (§2.1b).

- **`archivist.py` — real trust-tier bookkeeping (§3).** Promotion/
  demotion now runs from an actual score (source weight + edge diversity +
  edge count, §3.2) with N-consecutive-pass hysteresis (§3.3), gradual
  one-tier demotion (§3.4), explicit-negation flagging, regulatory
  efficacy scoring (§4.5), re-parenting candidate detection + execution
  (§2.3 mechanism 3), and the concrete pruning trigger (§10 item 19: still
  Tier 0 after N consolidation cycles). The `SELF` node is seeded at
  Trusted tier on init — the one deliberate axiom in the design (§2.1b
  item 1). The graph is now a `MultiDiGraph`, not `DiGraph` — a plain
  DiGraph silently collapses multiple relation types between the same two
  nodes into one overwritten edge, which breaks §2.1b's requirement that
  an event node can carry more than one relational edge at once (its own
  "I shouldn't have done that" example needs both `responsible-for` and
  `violates` simultaneously).

- **`sensory.py` — dictionary-pattern hierarchy parsing (§2.3 mechanism
  1)** added (`parse_hierarchy`), and relational-edge detection extended
  from one match to all four types in §2.1b (`detect_relational` now
  returns a list).

- **`association.py` — hierarchy placement actually implemented.**
  `place_node()` does path 1 (dictionary-pattern parsing) then falls back
  to path 2 (co-occurrence, tagged so it's never confused with an is-a
  claim), plus `run_reparenting_pass()` and `link_relational()` for the
  SELF/OTHER-anchored edges schema detection depends on.

- **`synthesizer.py` — basin decay (§2.1a point 5)** added to
  `consolidate_basins()`, so an unrevisited basin can flatten back out and
  de-stabilize, not just grow. Also exposes the raw basin key so
  `prometheus.py` can log felt-state→node links.

- **`reflector.py` — the two pieces §4A described but didn't implement:**
  `regulatory_self_report()` (§4.5 aggregation) and `detect_schemas()`
  (§2.1b Schema Node formation from recurring basin + relational-edge
  co-occurrence), plus `name_schema()` for the "only named if the agent's
  own input links a word to it" rule (§2.1b item 4a).

- **`prometheus.py` — rewired as the actual orchestrator:** self-study
  (§5.1), tier+anchor-restricted regulation with efficacy tracking (§4),
  a real Consolidation pass that calls all of the above in one place, the
  Pruning state now does something, and `maybe_advance_epoch()` evaluates
  the real §6.1/§6.2 gates instead of being a documented no-op.

- **`app.py`** gained a text-input path that actually feeds the
  sensory→association→chronos pipeline, plus a Reflection tab showing
  regulatory self-awareness and Schema Nodes (with a way to name them).

## What's still open (honestly, per the spec's own §9/§10)

Nothing here invents numbers the spec explicitly left undecided —
fatigue thresholds, hysteresis window sizes, basin/schema stabilization
counts, regulation dampening curves, etc. are all still the placeholders
the spec calls "not yet numeric," now wired to real code paths that will
make them tunable once you're watching the system run, per §10.

Two structural gaps called out in §10 are intentionally *not* solved
here, since the spec itself treats them as open design questions rather
than implementation debt: the single- vs. multi-instance persistence
question (item 16 — the JSON-per-module pattern used here is single-
instance only), and a background/scheduling loop independent of
Streamlit's request-response cycle (item 17 — self-study and fatigue
decay here only advance on `pulse()`, i.e. on user interaction, which is
the "idle just means no new input this tick" reading, not a true
background process).

A canonical node/edge schema (§10 item 18) is still implicit in the code
rather than written down as its own document — worth doing before handing
this to another tool/session, per the cross-tool-drift risk (§9 item 4).
