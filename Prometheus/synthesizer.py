from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import math

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

    Extended for mixed / competing affect:
      - primary basin = nearest stabilized (or current key if unformed)
      - secondary basin = next-nearest stabilized within a distance window
      - conflict_score in [0, 1] when primary and secondary pull in opposing
        valence or dominance directions. This is a legitimate boundary
        output (already-synthesized), safe for modulators / focus / WM.
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

    # Mixed-affect / competing-basin parameters (new).
    # Max PAD L2 distance for a second basin to count as "active competitor".
    SECONDARY_MAX_DIST = 0.55
    # Minimum dwell density ratio (secondary / primary) to be considered real.
    SECONDARY_MIN_RATIO = 0.25
    # How strongly opposing valence vs dominance contribute to conflict.
    CONFLICT_VALENCE_WEIGHT = 0.65
    CONFLICT_DOMINANCE_WEIGHT = 0.35

    def __init__(self):
        self.GRID_RESOLUTION = SynthesizerModule.GRID_RESOLUTION
        self.STABILIZATION_THRESHOLD = SynthesizerModule.STABILIZATION_THRESHOLD
        self.DECAY_RATE = SynthesizerModule.DECAY_RATE
        self.DESTABILIZATION_FLOOR = SynthesizerModule.DESTABILIZATION_FLOOR
        self.SECONDARY_MAX_DIST = SynthesizerModule.SECONDARY_MAX_DIST
        self.SECONDARY_MIN_RATIO = SynthesizerModule.SECONDARY_MIN_RATIO
        self.CONFLICT_VALENCE_WEIGHT = SynthesizerModule.CONFLICT_VALENCE_WEIGHT
        self.CONFLICT_DOMINANCE_WEIGHT = SynthesizerModule.CONFLICT_DOMINANCE_WEIGHT

        self.basin_grid = defaultdict(float)
        self.stabilized_basins: Dict[Tuple[float, float, float], str] = {}
        self.current_felt_state = "Unformed"
        self._current_key: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Mixed-affect live state (recomputed every update_from_core)
        self._primary_key: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._secondary_key: Optional[Tuple[float, float, float]] = None
        self._secondary_felt_state: str = "None"
        self._conflict_score: float = 0.0

        self.load_state()

    def _project_axes(self, raw: Dict[str, float]) -> Tuple[float, float, float]:
        """PAD from phenomenological body channels only (§Phase A).
        Never reads hormone names — only heart/breath/tension/sweat/gut/energy/warmth.
        """
        heart = raw.get("heart_rate", 0.5)
        breath = raw.get("breath", raw.get("respiration_rate", 0.5))
        sweat = raw.get("sweat_skin", 0.5)
        tension = raw.get("muscle_tension", 0.5)
        gut = raw.get("gut", 0.5)
        energy = raw.get("energy", 0.5)
        warmth = raw.get("warmth", 0.5)

        # Activated body → arousal
        arousal = max(0.0, min(1.0, 0.35 * heart + 0.30 * breath + 0.20 * sweat + 0.15 * energy))
        # Pleasant settled vs distressed gut/cold — valence in [-1, 1]
        valence = max(-1.0, min(1.0, (0.45 * warmth + 0.35 * energy) - (0.40 * gut + 0.25 * sweat)))
        # Stance: energy + inverse overwhelm (tension+gut high → low dominance)
        dominance = max(0.0, min(1.0, 0.40 * energy + 0.30 * (1.0 - tension) + 0.30 * (1.0 - gut)))
        return arousal, valence, dominance

    def _bin_key(self, arousal: float, valence: float, dominance: float) -> Tuple[float, float, float]:
        return (
            round(arousal, self.GRID_RESOLUTION),
            round(valence, self.GRID_RESOLUTION),
            round(dominance, self.GRID_RESOLUTION),
        )

    def _pad_dist(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )

    def _find_competitors(self, key: Tuple[float, float, float]):
        """Return (primary_key, secondary_key_or_None, conflict_score).

        Primary = nearest stabilized basin (or the live key itself if none).
        Secondary = next-nearest stabilized basin within SECONDARY_MAX_DIST
        that has meaningful dwell relative to primary.
        Conflict rises when primary and secondary disagree on valence sign
        or dominance polarity.
        """
        if not self.stabilized_basins:
            return key, None, 0.0

        ranked: List[Tuple[float, Tuple[float, float, float]]] = []
        for sk in self.stabilized_basins:
            ranked.append((self._pad_dist(key, sk), sk))
        ranked.sort(key=lambda t: t[0])

        primary = ranked[0][1]
        secondary = None
        conflict = 0.0

        if len(ranked) >= 2:
            d2, cand = ranked[1]
            if d2 <= self.SECONDARY_MAX_DIST:
                prim_dwell = self.basin_grid.get(primary, 1.0)
                sec_dwell = self.basin_grid.get(cand, 0.0)
                if prim_dwell > 0 and (sec_dwell / prim_dwell) >= self.SECONDARY_MIN_RATIO:
                    secondary = cand
                    # Opposing valence (sign disagreement) + dominance spread
                    v_diff = abs(primary[1] - secondary[1])  # valence in [-1,1]
                    d_diff = abs(primary[2] - secondary[2])  # dominance in [0,1]
                    # Extra boost when signs of valence actually oppose
                    sign_oppose = 1.0 if (primary[1] * secondary[1] < 0) else 0.4
                    conflict = min(1.0, (
                        self.CONFLICT_VALENCE_WEIGHT * v_diff * sign_oppose
                        + self.CONFLICT_DOMINANCE_WEIGHT * d_diff
                    ))
                    # Scale by how close the competitor is (nearer = more conflict)
                    proximity = 1.0 - (d2 / max(1e-6, self.SECONDARY_MAX_DIST))
                    conflict = min(1.0, conflict * (0.5 + 0.5 * proximity))

        return primary, secondary, conflict

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

        # Mixed-affect: primary / secondary / conflict
        primary, secondary, conflict = self._find_competitors(key)
        self._primary_key = primary
        self._secondary_key = secondary
        self._secondary_felt_state = (
            self.stabilized_basins.get(secondary, "None") if secondary else "None"
        )
        self._conflict_score = conflict

        # Prefer the named primary basin when the exact key is still Unformed
        # but a nearby stabilized basin exists (smooths early Childhood).
        if self.current_felt_state == "Unformed" and primary in self.stabilized_basins:
            self.current_felt_state = self.stabilized_basins[primary]

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

    def get_primary_basin_key(self) -> Tuple[float, float, float]:
        return self._primary_key

    def get_secondary_basin_key(self) -> Optional[Tuple[float, float, float]]:
        return self._secondary_key

    def get_secondary_felt_state(self) -> str:
        return self._secondary_felt_state

    def get_conflict_score(self) -> float:
        """0..1 ambivalence / competing-affect signal. Safe boundary output."""
        return self._conflict_score

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
    # Persistence (§4C).
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
