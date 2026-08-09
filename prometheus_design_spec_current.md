# Prometheus Design Specification — Current Build

**Document status:** Working spec reflecting implemented architecture as of 2026-08-09  
**Project:** Prometheus2 (`Prometheus – Living Brain`)  
**Codename intent:** A pulse-driven cognitive architecture aimed at schema formation, collapse/abstraction, somatic grounding, and directed attention — not an LLM wrapper.

---

## 0. One-line mission

Build a **bounded, compressible, goal-directed** agent whose knowledge densifies into nested schemas, whose felt state is a real landscape (not labels alone), and whose attention is residual-driven — with language and speech as **skins**, not the mind.

---

## 1. Core emergence principle

- Agent logic may condition on **synthesized** outputs (felt state, intensity, basin key, trust tiers, focus report).
- Agent logic must **not** treat Debug panels, raw hormone dumps, or UI instrumentation as cognitive inputs.
- No pre-assigned emotion names as causal drivers; basins and schemas are **earned** by recurrence.
- Dictionary / WordNet is a **source of structure**, not a substitute for experience.
- LLM (if any) is **I/O skin only** — never the reasoning core.

---

## 2. Runtime modes

| Mode | Owner of `pulse()` | When to use |
|------|----------------------|-------------|
| **Manual / batch** | User clicks or “Run N pulses” | Lab soaks, regression |
| **Semi-live** | Streamlit timer while page open | Continuous time in browser lab |
| **Headless live** | `python -m Prometheus.live` / `run_live.py` | Full live clock; UI optional |
| **Mobile** | Phone as **remote control** only | Status + input; core stays on host |

**Non-goals for Streamlit:** unsupervised 24/7 process, smooth 60fps 3D viz, speech daemon.

**Architecture target:**

```text
Headless core (process)
    ├── data/ checkpoints + live_status.json
    ├── optional inbox file / API for input
    ├── Streamlit lab UI (optional)
    └── Future: thin mobile page + viz client
```

---

## 3. Module map

| Module | Responsibility |
|--------|----------------|
| `hormonal.py` / BioSystem | Hormones, epoch, somatic raw variables, homeostasis |
| `synthesizer.py` | PAD projection, basin grid, stabilized basins, intensity |
| `somatic_topo.py` | Basin dwell + **transition graph** (somatic topography data) |
| `archivist.py` | Epistemic MultiDiGraph, trust, activation, collapse, co-activation |
| `association.py` | Place nodes, hierarchy parse, relational link, reparenting |
| `sensory.py` | Ingest, relational detect, WordNet expand, hierarchy parse |
| `chronos.py` | Pulse / felt timeline |
| `reflector.py` | Somatic complex schemas, epistemic clusters, schema ids/names |
| `executive.py` | Bias from intensity |
| `working_memory.py` | Limited slots, emotional gating, epoch user priority |
| `focus.py` | Residuals, sticky focus, prediction error, max-age, cooldown |
| `self_narrative.py` | Narrative elements over graph |
| `Prometheus.py` | Orchestrator: queue, pulse, fatigue states, consolidation |
| `live.py` | Headless pulse loop |
| `app.py` | Streamlit lab + semi-live |
| `prometheus_dashboard.py` | Pyvis HTML for epistemic graph |

---

## 4. Pulse pipeline (Learning tick)

1. `bio.step()` — hormone flux + homeostasis  
2. `synthesizer.update_from_core(raw)` — basin key + felt state  
3. `somatic_topo.record(basin_key)` — dwell + transition  
4. Intensity → executive bias; regulation hysteresis if spiked  
5. Reflector directive may override bias  
6. **Learning only:** dequeue user input → `_ingest`, else `_self_study`  
7. Chronos record  
8. Fatigue update + state machine (`Learning` / `Consolidation` / `Pruning`)  
9. Epoch check  
10. `focus.tick(...)` — residual decay, prediction, sticky focus, stagnation/max-age  

---

## 5. Fatigue three-state machine

```text
Learning  --fatigue ≥ T1-->  Consolidation
Consolidation --fatigue ≥ T2-->  Pruning
Consolidation --fatigue < T1−H-->  Learning
Pruning --after prune, fatigue < T2−H-->  Consolidation
```

| Symbol | Role |
|--------|------|
| T1 | Enter consolidation |
| T2 | Enter pruning (must be reachable under consolidation recovery) |
| H | Hysteresis band |
| Growth | Fatigue rise scaled by intensity (and related factors) |
| Recovery | Multiplicative retain on consolidation/pruning (higher = less drop) |

**Tuning note:** If T2 is high and consolidation recovery is strong (large drop), Pruning never runs. Starting lab values that often work: T1≈0.35, T2≈0.50, H≈0.05, growth≈0.28, consol recovery≈0.72.

**Pruning** = trust/leaf maintenance pass on archivist (distinct from **schema collapse**).

---

## 6. Epistemic graph

### 6.1 Nodes

- Ordinary terms / events (user sentences, dictionary lemmas)
- `SELF` (axiom, Trusted seed)
- `OTHER` (social placeholder)
- Schema nodes (somatic complex + epistemic clusters)
- Optional basin-linked anchors

### 6.2 Trust tiers

| Tier | Name | Role |
|------|------|------|
| 0 | Provisional | New / weakly corroborated |
| 1 | Working | Corroborated enough to use |
| 2 | Trusted | Stable |

Promotion/demotion uses source weights, diversity, edges, **hysteresis** (not single-pass flips).

### 6.3 Edge families (meta)

| Family | Examples | Notes |
|--------|----------|------|
| HIERARCHY | is-a, part-of | Self-study / parse heavy |
| MEMBERSHIP | composed-of, instance-ish | Schema membership |
| ROLE | responsible-for | Relational intake |
| SOCIAL_NORM | violates | Relational intake |
| CAUSAL | causes, results-in | **Write path still thin** |
| RESIDUAL | associated-with | Glue / co-occurrence |

### 6.4 Relational intake (§2.1b)

`sensory.detect_relational` → `association.link_relational`:

- `violates`, `responsible-for`, `temporal-contrast`, `concerns-other`
- Keyword-expanded lists (no LLM)
- SELF or OTHER as endpoints; felt_state stamped when known

### 6.5 Collapse (§13.4)

- Eligible neglected children under hierarchy/schema parents
- **Remove children; preserve / rewire relational edges to parent**
- `absorbed` list on parent for rehydration later
- Protected: SELF/OTHER, WM slots, narrative floor nodes, **current focus**
- Always log `last_collapse_summary` (including zeros)

### 6.6 Naming hygiene

- Epistemic schema **ids**: short `epistemic_of_{slug}_{hash}` or `epistemic_{hash}`
- Human gloss in `name` / `definition` attributes — **not** full WordNet gloss as id
- Legacy long ids may remain until merge/reset

---

## 7. Schemas

### 7.1 Epistemic schemas

- Co-activation → connected components → cluster (Working+ members only)
- **Admission gates (quality repair):**
  - mean pairwise coherence ≥ `EPISTEMIC_MIN_COHERENCE` (token Jaccard + optional WordNet hypernym bonus)
  - lemma-like member ratio ≥ `EPISTEMIC_MIN_LEMMA_RATIO` (rejects sentence-soup clusters)
- **Ids:** short stable `epistemic_of_{slug}` or `epistemic_{hash}` — never full gloss as id
- **Naming (delayed):** schemas are created **unnamed**; `name` is set only when:
  - coherence still holds
  - context diversity ≥ `EPISTEMIC_NAME_MIN_CONTEXTS` (distinct sources / felt stamps)
  - a lemma-like member meets frequency ≥ `EPISTEMIC_NAME_MIN_FREQ`
- **Unnamed expiry:** after `EPISTEMIC_UNNAMED_MAX_CYCLES` consolidations stagnant and not improving coherence → **dissolve schema wrapper only**; members and their edges remain
- Membership via composed-of edges
- Formation / naming / expiry on **Consolidation**

### 7.2 Complex emotional (somatic) schemas

- Pattern: **stabilized basin + relational edge set**
- Requires recurrence (e.g. 3 matching events) — not one script pass
- Candidates shown in Reflection when below threshold

### 7.3 Nested abstraction (design intent)

- Depth in nest matters more than raw use-count alone
- Deep leaves become disposable; edges/roles promote to parent
- Rehydration on demand when focus/query needs children back

---

## 8. Working memory (§14)

- Hard cap on schema slots (emotionally gated capacity)
- Epoch-weighted user priority (childhood high → maturity lower)
- Basin co-occurrence bonus
- Focus bonus from `focus.py`
- SELF + current basin context privileged

---

## 9. Focus / residuals / prediction (§13.y)

### 9.1 Residuals

| Channel | Source |
|---------|--------|
| act | Activation / boost on touch |
| unc | Uncertainty (tier + structural sparsity) |
| pred | Schema expectation gap |
| par | Parental reaction |
| basin / schema | Soft bonuses in composite score |

### 9.2 Sticky focus

- One primary focus thread
- Min residency before normal switch
- Challenger needs margin
- **Hard max age** (e.g. 100): force leave even if act is hot (breaks self-study feedback locks)
- **Soft stagnation**: old + cold act → slash pred
- **Cooldown** after leave so the same node cannot immediately re-win

### 9.3 Prediction quality

- `expected_families` EMA on schemas at Consolidation
- Error weighted: hierarchy/membership high; role/causal/social lower until write paths mature
- Soften expectations that stay missing under long focus
- Inject pred residual sparsely (not every pulse) to avoid fixed points

Goals are **not** strings: a goal is whatever keeps winning sticky focus until tension falls.

---

## 10. Somatic system

### 10.1 Axes

```text
arousal    ← f(adrenaline, cortisol proxies)
valence    ← f(dopamine / related)
dominance  ← f(adrenaline, testosterone proxies)
```

Homeostasis pulls hormones toward baseline each pulse. Autonomous self-study must apply **non-trivial** arousal bumps or A/D stay flat.

### 10.2 Basins

- Grid over PAD; stabilization threshold; decay of neglected basins
- Felt state name = stabilized basin id or `Unformed`

### 10.3 Somatic topography (`somatic_topo.py`)

**This is the map; raw hormone JSON is not.**

- Nodes: basin keys  
- Edges: observed transitions  
- Consolidation: decay weak transitions  
- Report: top dwell basins, top transitions, current basin  

Future: 3D viz client; optional bias into focus/schema from neighborhood.

---

## 11. Self-narrative (§16)

- Elements formed/absorbed/pruned on Consolidation
- Links into epistemic nodes; protected from collapse when above floor
- Report for Reflection / Working Memory tabs

---

## 12. Streamlit lab UI

Tabs: **Graph | State | Reflection | Working Memory | Debug**

| Area | Content |
|------|---------|
| Graph | Pyvis; prefer top-K / WM neighborhood over full graph |
| State | Felt, bias, fatigue, epoch |
| Reflection | Self-report, regulatory efficacy, complex schema candidates, epistemic schemas |
| WM | Slots + narrative |
| Debug | Raw somatic, hormones, **somatic topo report**, focus/residuals, collapse summary, absorbed parents, live sliders |

**Semi-live sidebar:** Auto-pulse, interval, pulses-per-tick, pause-after-send.

---

## 13. Headless live (`live.py`)

```bash
python run_live.py --interval 2.0 --pulses-per-tick 1
python -m Prometheus.live --input-file data/inbox.txt --max-pulses 5000
```

- SIGINT/SIGTERM → best-effort save + status `running: false`
- Status: `data/live_status.json`
- Inbox: append lines → `queue_input`

---

## 14. Mobile

- Phone = **remote control**, not host  
- Core on VPS/RPi/always-on box  
- Mobile UI: large status + text send + parental (thin page or simplified Streamlit)  
- Smooth WM graph + 3D basin map → **dedicated viz frontend**, not Streamlit

---

## 15. Speech center (deferred)

**Integrate only after** live core is stable and a text I/O path already exists.

| Layer | Role |
|-------|------|
| STT | → same as `queue_input` |
| TTS | ← short reports from focus/narrative/state |
| Not | Second reasoning engine |

Optional later: prosody biased by arousal; silence while prediction residual high.

---

## 16. Safety / product posture

- No tools until behavior is predictable and gated  
- No claim of consciousness as fact; “type of mind-model / pseudo-conscious architecture” is the honest frame  
- Emergence does not remove operator responsibility  

---

## 17. Implementation status (2026-08-08)

| Item | Status |
|------|--------|
| Epistemic graph + trust + activation | Implemented |
| Self-study / WordNet expand | Implemented |
| Relational detect + link | Implemented (expanded keywords) |
| Collapse + absorbed | Implemented |
| Focus + residuals + pred + max-age/cooldown | Implemented |
| Working memory | Implemented |
| Complex schema candidates | Implemented (needs recurrence) |
| Epistemic schema clusters + naming hygiene | Implemented (short ids) |
| Schema coherence gate + delayed naming + unnamed expiry | Implemented |
| Felt anchors (linkable basins) | Not done — next |
| Differentiated hormone drive | Partial (arousal bump raised; full role map not done) |
| Somatic topo data | Implemented |
| Semi-live Streamlit | Implemented |
| Headless live loop | Implemented (`live.py`) |
| Graph iteration safety (reparent list) | Fixed |
| Causal write path | Thin / incomplete |
| Focus-triggered rehydrate | Not done |
| Headless ↔ Streamlit API client | Not done |
| Mobile thin UI | Not done |
| 3D topo / smooth graph viz | Not done |
| Speech | Deferred |
| LLM skin | Deferred |

---

## 18. Recommended roadmap

1. **Validate** headless live for multi-hour runs + inbox input  
2. **Clean graph** restart under current code for keeper baseline  
3. **Thin status/input API or file protocol** for phone  
4. **Causal edges** when language supports them; raise pred weights after  
5. **Rehydrate** on focus/query  
6. **Viz client** (WM + 3D basins) talking to live core  
7. **Speech skin** last among I/O features  

Avoid broad fatigue tuning until live dynamics are the normal case.

---

## 19. Design invariants (do not violate casually)

1. One primary clock for consolidation-class work  
2. Collapse preserves relational meaning on parents  
3. Focus cannot self-lock forever (max-age + cooldown)  
4. Prediction must not demand structure the graph cannot grow  
5. Somatic topo ≠ hormone gauge list  
6. UI is not cognition  
7. Live core ≠ browser session  

---

## 20. Document history

| Rev | Notes |
|-----|--------|
| rev18 family | Prior basin/schema/WM/narrative design |
| **Current (this file)** | + schema coherence, delayed naming, unnamed expiry; arousal bump; roadmap for felt anchors |

---

*End of current build specification.*
