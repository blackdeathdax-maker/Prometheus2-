"""
chronos.py -- Visible layer (§7).

"Time-series log of felt-state history, arousal-valence-dominance
trajectory (feeds §2.1a's dwell-time histogram), and decay steps -- the
evaluation window for milestone/consolidation checks. Also the source of
timestamps for `temporal-contrast` edges (§2.1b)."

This module previously did not exist at all in the codebase even though
both prometheus.py and reflector.py imported ChronosModule from it -- the
whole system could not run without it. That is the bug this file fixes.
"""
import json
import logging
import os
from collections import Counter, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
CHRONOS_LOG_PATH = os.path.join(_DATA_DIR, "chronos_log.json")

# How much pulse history to retain in memory. Unbounded history was
# explicitly rejected by the persistence discussion (§10 item 16: "not
# fast-layer or full raw history") -- a rolling window is the correct
# shape for this, not a growing-forever list.
DEFAULT_MAXLEN = 5000


class ChronosModule:
    """
    Owns the rolling log prometheus.py writes to every pulse, and the
    evaluation windows other modules read from:
      - reflector.py's structural self-report (§4A) uses get_state_summary()
      - synthesizer.py's basin stabilization (§2.1a) is fed the raw AVD
        trajectory via record_pulse()
      - prometheus.py's §6.1 Childhood->Adolescence gate uses
        naming_reliability()
      - sensory.py/association.py's `temporal-contrast` edges (§2.1b) use
        get_timestamp_for() / the timestamps already stored on entries
    """

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        self.history: deque = deque(maxlen=maxlen)
        # (timestamp, basin_key, node) -- the raw material for §6.1's
        # "reliably links a given felt-state signature to the same
        # knowledge-web node across repeated occurrences" check.
        self.felt_state_links: deque = deque(maxlen=maxlen)
        self.load()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_pulse(self, somatic, bias: str, felt_state: Optional[str] = None,
                      avd: Optional[Tuple[float, float, float]] = None):
        """Called once per pulse by prometheus.py. `somatic` may be a
        SomaticReadout (has .to_dict()) or a plain dict."""
        if hasattr(somatic, "to_dict"):
            snapshot = somatic.to_dict()
        elif isinstance(somatic, dict):
            snapshot = somatic
        else:
            snapshot = {"urgency": getattr(somatic, "urgency", 0.0),
                        "tension": getattr(somatic, "tension", 0.0)}

        entry = {
            "timestamp": datetime.now().isoformat(),
            "somatic": snapshot,
            "bias": bias,
            "felt_state": felt_state,
            "avd": list(avd) if avd else None,
        }
        self.history.append(entry)

    def record_felt_state_link(self, basin_key: Tuple[float, float, float], node: str):
        """Records that, at this moment, a stabilized basin's live lookup
        (synthesizer.py) resolved to `node` being the active/associated
        knowledge node. This is the log §6.1's naming-reliability check
        reads from -- naming is "the link is stable/reused across repeated
        visits to that basin", which can only be evaluated from a history
        of such links, not a single tick."""
        self.felt_state_links.append((datetime.now().isoformat(), basin_key, node))

    # ------------------------------------------------------------------
    # Reflector-facing summary (§4A structural self-report)
    # ------------------------------------------------------------------
    def get_state_summary(self, window: int = 10) -> Dict[str, float]:
        """Trend/acceleration signals over the last `window` pulses.
        Used by reflector.py's spinning/stagnant detection."""
        recent = list(self.history)[-window:]
        if len(recent) < 3:
            return {"urgency_trend": 0.0, "tension_acceleration": 0.0}

        urgencies = [e["somatic"].get("urgency", 0.0) for e in recent]
        tensions = [e["somatic"].get("tension", 0.0) for e in recent]

        urgency_trend = urgencies[-1] - urgencies[0]
        # Discrete second derivative as a simple acceleration proxy.
        tension_acceleration = (tensions[-1] - tensions[-2]) - (tensions[-2] - tensions[-3])

        return {
            "urgency_trend": urgency_trend,
            "tension_acceleration": tension_acceleration,
        }

    # ------------------------------------------------------------------
    # §6.1 Childhood -> Adolescence gate support
    # ------------------------------------------------------------------
    def naming_reliability(self, basin_key: Tuple[float, float, float],
                            window: int = 20, min_occurrences: int = 5
                            ) -> Tuple[Optional[str], float, int]:
        """
        Consistency rate of a basin resolving to the *same* node across its
        last `window` occurrences (§6.1: "reliably and consistently...
        across repeated occurrences"). Returns (dominant_node, consistency,
        occurrence_count). dominant_node is None if the minimum-occurrence
        floor (§6.1, §5 stability requirement) hasn't been met yet --
        callers must treat that as "not eligible", not "failed".
        """
        occurrences = [n for (_, k, n) in self.felt_state_links if k == basin_key][-window:]
        if len(occurrences) < min_occurrences:
            return None, 0.0, len(occurrences)
        counts = Counter(occurrences)
        node, count = counts.most_common(1)[0]
        return node, count / len(occurrences), len(occurrences)

    def all_linked_basins(self) -> List[Tuple[float, float, float]]:
        """Distinct basin keys that have at least one recorded link, for
        callers that need to sweep every candidate rather than check one
        basin at a time."""
        return list({k for (_, k, _n) in self.felt_state_links})

    # ------------------------------------------------------------------
    # §2.1b temporal-contrast support
    # ------------------------------------------------------------------
    def find_past_state(self, lookback: int = 50) -> Optional[Dict]:
        """Returns a past pulse entry for `temporal-contrast` edges
        (nostalgia: "a node relates to a past state differing from the
        current one, using timestamps chronos.py already logs"). Simple
        recency-window lookup -- no semantic matching, consistent with the
        deterministic/no-black-box principle used everywhere else."""
        if len(self.history) <= lookback:
            return None
        return self.history[-lookback]

    # ------------------------------------------------------------------
    # Persistence -- same pattern as archivist.py/hormonal.py: each module
    # owns its own JSON file under PROMETHEUS_DATA_DIR (§10 item 16 is
    # still open at the *architecture* level -- single vs multi-instance,
    # exactly what's included in a checkpoint -- but a module resetting to
    # nothing on every restart is strictly worse than this, so basic
    # continuity is provided here rather than left totally undesigned.)
    # ------------------------------------------------------------------
    def save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(CHRONOS_LOG_PATH, "w") as f:
                json.dump({
                    "history": list(self.history)[-500:],  # bounded checkpoint
                    "felt_state_links": list(self.felt_state_links)[-500:],
                }, f, default=str)
        except OSError as e:
            logger.warning("ChronosModule.save failed: %s", e)

    def load(self):
        if os.path.exists(CHRONOS_LOG_PATH):
            try:
                with open(CHRONOS_LOG_PATH, "r") as f:
                    data = json.load(f)
                for e in data.get("history", []):
                    self.history.append(e)
                for link in data.get("felt_state_links", []):
                    ts, key, node = link
                    self.felt_state_links.append((ts, tuple(key), node))
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                logger.warning("ChronosModule.load failed, starting fresh: %s", e)
