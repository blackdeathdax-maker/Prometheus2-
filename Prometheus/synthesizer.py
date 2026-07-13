from collections import defaultdict
from typing import Dict, Tuple
import json
import logging
import os

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
BASIN_STATE_PATH = os.path.join(_DATA_DIR, "basin_state.json")


class SynthesizerModule:
    """
    Boundary module (§7). Projects hidden-layer raw variables onto the
    composite arousal/valence/dominance axes (§2.1a) and looks the current
    point up against the *stabilized* basin map to produce the current
    named Felt State. The basin map itself only changes during
    Consolidation (consolidate_basins); the per-tick lookup against it is
    cheap and live every tick, per spec.
    """

    # Grid resolution for the dwell-time histogram. Not yet tuned per spec
    # §10 item 11 -- 1 decimal place is a placeholder, revisit empirically.
    GRID_RESOLUTION = 1
    # Minimum revisits before a candidate basin counts as stabilized.
    # Placeholder per §10 item 11 (not yet numeric in the spec either).
    STABILIZATION_THRESHOLD = 3
    # §2.1a point 5: a basin that stops being revisited should flatten
    # back out, mirroring non-reinforcement decay used elsewhere (§3.4,
    # §4.5). Applied once per Consolidation pass, same cadence as the rest
    # of basin formation (§2.1a point 6).
    DECAY_RATE = 0.85
    # Bug fix: this was 1.0, which is higher than a single fresh visit
    # (basin_grid starts a new key at exactly 1.0). Since decay applies
    # every Consolidation pass and 1.0*0.85=0.85 < 1.0, ANY bin that
    # hadn't already reached ~4 hits got deleted on its very first decay
    # pass -- before it could ever accumulate toward STABILIZATION_THRESHOLD
    # (3). Combined with Consolidation firing every few ticks (post-fatigue
    # -fix), this meant basins essentially never survived long enough to
    # stabilize, regardless of how long the system ran -- reproducing
    # exactly the reported "consistently Unformed" symptom. The floor
    # should catch basins that WERE stabilized and have since gone
    # unused for a long time, not erase fresh candidates on their first
    # non-reinforced pass. Still a §10 tuning placeholder, but 1.0 was
    # not just "untuned," it was structurally wrong.
    DESTABILIZATION_FLOOR = 0.2

    def __init__(self):
        self.basin_grid = defaultdict(float)
        self.stabilized_basins: Dict[Tuple[float, float, float], str] = {}
        self.current_felt_state = "Unformed"
        self._current_key: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.load_state()

    def _project_axes(self, raw: Dict[str, float]) -> Tuple[float, float, float]:
        """Composite axis formulas per §2.1a. Exact weighting/normalization
        not yet finalized (§10 item 10) -- this is a first-pass average."""
        arousal = max(0.0, min(1.0, (raw.get("heart_rate", 0.5) + raw.get("respiration_rate", 0.5)) / 2))
        valence = max(-1.0, min(1.0, raw.get("dopaminergic_tone", 0.5) - raw.get("cortisol_load", 0.5)))
        dominance = max(0.0, min(1.0, (raw.get("vascular_constriction", 0.5) + raw.get("muscle_tension", 0.5)) / 2))
        return arousal, valence, dominance

    def _bin_key(self, arousal: float, valence: float, dominance: float) -> Tuple[float, float, float]:
        return (round(arousal, 1), round(valence, 1), round(dominance, 1))

    def update_from_core(self, raw_variables: Dict[str, float]):
        """
        Call with BioSystem.get_raw_variables(), NOT the raw hormone dict.
        """
        arousal, valence, dominance = self._project_axes(raw_variables)
        key = self._bin_key(arousal, valence, dominance)
        self.basin_grid[key] += 1
        self._current_key = key

        # Live, cheap lookup against the already-stabilized map (§2.1a).
        self.current_felt_state = self.stabilized_basins.get(key, "Unformed")

    def get_current_felt_state(self) -> str:
        return self.current_felt_state

    def get_current_basin_key(self) -> Tuple[float, float, float]:
        """Exposes the raw (arousal, valence, dominance) bin key for the
        *current* tick -- prometheus.py uses this to log felt-state ->
        knowledge-node links into chronos.py, which is the evaluation
        window §6.1's naming-reliability gate reads from. This is not a
        core.py raw-variable leak (§ Core Emergence Principle): it's the
        already-synthesized composite key, not a hidden-layer value."""
        return self._current_key

    def get_current_intensity(self) -> float:
        """Legitimate, boundary-crossing continuous signal for anything
        that needs a spike/threshold check (§4.1 regulation, executive
        bias) rather than a raw hidden-layer value. Uses the arousal
        component of the current basin key -- arousal is the "how
        activated" axis in the PAD model (§2.1a), and it's already the
        product of synthesizer.py's projection, same legitimacy argument
        as get_current_basin_key() above. This exists specifically so
        prometheus.py and executive.py never need to read
        BioSystem.get_somatic_readout()/somatic.urgency directly, which
        hormonal.py's own docstring prohibits ("nothing in here should be
        read directly by any module that participates in the agent's
        decision loop -- only synthesizer.py's composite projection may
        leave this layer")."""
        return self._current_key[0]

    def consolidate_basins(self):
        """
        Consolidation-only basin stabilization (§2.1a point 6). Peaks in
        dwell-time density that have been revisited enough become named
        felt states. Basins that stop being revisited decay (§2.1a point
        5) and can de-stabilize back into "Unformed" if density falls far
        enough -- an emotional pattern the agent has outgrown can
        genuinely fade rather than being permanent once formed.
        """
        newly_stabilized = 0
        for key, count in self.basin_grid.items():
            if count >= self.STABILIZATION_THRESHOLD and key not in self.stabilized_basins:
                basin_id = f"basin_{key[0]}_{key[1]}_{key[2]}"
                self.stabilized_basins[key] = basin_id
                newly_stabilized += 1

        # Decay every key's density toward zero; a key that hasn't been
        # revisited this pass simply doesn't get reinforced, so repeated
        # non-revisits shrink it out.
        destabilized = 0
        for key in list(self.basin_grid.keys()):
            self.basin_grid[key] *= self.DECAY_RATE
            if self.basin_grid[key] < self.DESTABILIZATION_FLOOR:
                del self.basin_grid[key]
                if key in self.stabilized_basins:
                    del self.stabilized_basins[key]
                    destabilized += 1

        print(
            f"Consolidation: {newly_stabilized} new basin(s), "
            f"{destabilized} destabilized, {len(self.stabilized_basins)} total stabilized."
        )

    # ------------------------------------------------------------------
    # Persistence (§4C). Previously entirely missing -- the whole
    # emotional web (basin_grid, stabilized_basins) lived purely in
    # memory and reset to empty on every fresh SynthesizerModule()
    # instantiation, meaning basin-formation progress never survived an
    # app restart, and a genuine "reset memory" action had no dedicated
    # target for this module (it just happened by accident on restart,
    # while archivist.py/hormonal.py's data would silently persist and
    # get out of sync with a "reset" emotional web). Same JSON-file
    # pattern as archivist.py/hormonal.py/chronos.py. Checkpointed once,
    # at the end of Consolidation, by prometheus.py -- not on every tick.
    # ------------------------------------------------------------------
    @staticmethod
    def _key_to_str(key: Tuple[float, float, float]) -> str:
        """Tuple keys aren't valid JSON object keys -- encode losslessly
        as a delimited string, decoded back to a tuple on load."""
        return "|".join(str(v) for v in key)

    @staticmethod
    def _str_to_key(s: str) -> Tuple[float, float, float]:
        parts = s.split("|")
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    def save_state(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            data = {
                "basin_grid": {self._key_to_str(k): v for k, v in self.basin_grid.items()},
                "stabilized_basins": {self._key_to_str(k): v for k, v in self.stabilized_basins.items()},
            }
            with open(BASIN_STATE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("SynthesizerModule.save_state failed: %s", e)

    def load_state(self):
        if os.path.exists(BASIN_STATE_PATH):
            try:
                with open(BASIN_STATE_PATH, "r") as f:
                    data = json.load(f)
                for k_str, v in data.get("basin_grid", {}).items():
                    self.basin_grid[self._str_to_key(k_str)] = v
                for k_str, basin_id in data.get("stabilized_basins", {}).items():
                    self.stabilized_basins[self._str_to_key(k_str)] = basin_id
            except (json.JSONDecodeError, OSError, TypeError, ValueError, IndexError) as e:
                logger.warning("SynthesizerModule.load_state failed, starting fresh: %s", e)
