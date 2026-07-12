import json
import threading
import math
import os
import logging
from enum import Enum
from typing import Dict
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Persistence lives under the package's own data/ folder by default so this
# runs anywhere (previous hardcoded '/content/drive/MyDrive/Prometheus/...'
# only worked inside a specific Colab session with Drive mounted). Override
# with the PROMETHEUS_DATA_DIR env var for a different deployment.
_DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
BIOSYSTEM_STATE_PATH = os.path.join(_DATA_DIR, "biosystem_state.json")


class Epoch(str, Enum):
    """
    Developmental epoch enum (spec §6 / Task 2). Lives here per the module
    responsibility table (§7): "the enum itself in hormonal.py". hormonal.py
    only stores/reports the current epoch -- it never decides transitions.
    Epoch-shift instructions are issued by prometheus.py, which is the only
    module allowed to evaluate the cross-layer competence gates in §6.
    """
    CHILDHOOD = "Childhood"
    ADOLESCENCE = "Adolescence"
    MATURITY = "Maturity"


class SomaticReadout(BaseModel):
    vitality: float = Field(default=0.5, ge=0.0, le=1.0)
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)
    tension: float = Field(default=0.4, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, float]:
        # pydantic v1's .dict() / v2's .model_dump() compatibility shim
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class BioSystem:
    """
    Hidden layer. Owns raw hormonal state + fast/slow decay math. Per the
    Core Emergence Principle, nothing in here should be read directly by
    any module that participates in the agent's decision loop -- only
    synthesizer.py's composite projection may leave this layer.

    Epoch is stored here (instructed by prometheus.py) but this class does
    NOT evaluate transitions itself -- see prometheus.py.
    """

    # §4C persistence split: fast-layer hormones reset to their __init__
    # resting-baseline values on restart (a restarted agent shouldn't wake
    # up mid-spike from before it was last closed); only slow-layer
    # hormones (temperament) persist. Matches decay_fast()'s own
    # fast-layer set below, plus dopamine (dopaminergic_tone is fast-layer
    # per the original core.py module table) which decay_fast doesn't
    # currently touch but which self-study's reward bump modifies directly.
    FAST_LAYER = ("adrenaline", "cortisol", "dopamine")
    SLOW_LAYER = ("oxytocin", "testosterone", "estrogen", "thyroxine", "serotonin")

    def __init__(self):
        # Tunable decay rate, exposed as an instance attribute (not a bare
        # literal inside _compute_hormonal_flux) so the Debug tab's
        # sliders can adjust it live. This is the rate every hormone
        # decays toward its 0.5 baseline each tick -- the single biggest
        # lever on how fast urgency (and therefore fatigue, §5) climbs.
        # Still an undecided placeholder (§10), same as everywhere else.
        self.HORMONE_DECAY_RATE = 0.05

        self._hormones = {
            "adrenaline": 0.3,
            "cortisol": 0.4,
            "dopamine": 0.6,
            "serotonin": 0.65,
            "oxytocin": 0.5,
            "testosterone": 0.55,
            "estrogen": 0.5,
            "thyroxine": 0.6,
        }
        self.somatic = SomaticReadout()
        self.epoch = Epoch.CHILDHOOD
        self.lock = threading.Lock()
        self.logger = []
        self.load_state()

    def step(self, external_input: Dict = None) -> SomaticReadout:
        """Recursive step: Hormones <-> Body."""
        with self.lock:
            flux = self._compute_hormonal_flux()
            self.somatic = self._compute_somatic_readout(flux)
            self._apply_homeostasis()

            self.logger.append({
                "timestamp": datetime.now().isoformat(),
                "hormones": {k: round(v, 3) for k, v in self._hormones.items()},
                "somatic": self.somatic.to_dict(),
            })
            # No self.save_state() here (§4C) -- step() runs every single
            # pulse. Persistence is Consolidation-gated only; prometheus.py
            # calls save_state() once at the end of a Consolidation pass,
            # same checkpoint call as archivist.save().
        return self.somatic

    def _compute_hormonal_flux(self) -> Dict:
        """Metabolic decay + external influence."""
        for h in self._hormones:
            baseline = 0.5
            rate = self.HORMONE_DECAY_RATE
            self._hormones[h] = self._hormones[h] * (1 - rate) + baseline * rate
        return self._hormones

    def _compute_somatic_readout(self, hormones: Dict) -> SomaticReadout:
        """Non-linear mapping."""
        urgency = math.tanh(hormones["adrenaline"] + hormones["cortisol"] * 1.2)
        vitality = math.tanh(hormones["dopamine"] + hormones["serotonin"] + hormones["thyroxine"] * 0.5)
        stability = math.tanh(hormones["serotonin"] + hormones["oxytocin"] * 0.8)
        tension = math.tanh(hormones["cortisol"] + hormones["testosterone"] * 0.6)
        return SomaticReadout(
            vitality=max(0.1, min(1.0, vitality)),
            stability=max(0.1, min(1.0, stability)),
            urgency=max(0.0, min(1.0, urgency)),
            tension=max(0.0, min(1.0, tension)),
        )

    def _apply_homeostasis(self):
        """Compensatory adjustment."""
        if self.somatic.urgency > 0.85:
            self._hormones["serotonin"] = min(1.0, self._hormones["serotonin"] + 0.15)
            self._hormones["cortisol"] *= 0.7
        if self.somatic.vitality < 0.2:
            self._hormones["dopamine"] = min(1.0, self._hormones["dopamine"] + 0.25)

    def get_somatic_readout(self) -> SomaticReadout:
        """Blind external access -- already-composited somatic values."""
        return self.somatic

    def get_raw_variables(self) -> Dict[str, float]:
        """
        Hidden-layer only. Exposes the spec-named raw somatic variables
        (§7 core.py responsibility row) that synthesizer.py projects onto
        the arousal/valence/dominance axes (§2.1a). This is the ONE
        legitimate consumer of this method -- prometheus.py and every
        other module must go through synthesizer.py's output instead.

        These are derived from the underlying endocrine hormones rather
        than tracked as separate raw fields, since BioSystem models an
        endocrine layer rather than direct vitals. The mapping is a
        deliberate, documented approximation:
          - heart_rate        ~ adrenaline (sympathetic drive)
          - respiration_rate  ~ cortisol   (stress-linked respiration)
          - dopaminergic_tone = dopamine
          - cortisol_load     = cortisol
          - vascular_constriction ~ adrenaline (vasoconstrictive)
          - muscle_tension     ~ testosterone (correlates with tension/dominance)
        """
        h = self._hormones
        return {
            "heart_rate": h["adrenaline"],
            "respiration_rate": h["cortisol"],
            "dopaminergic_tone": h["dopamine"],
            "cortisol_load": h["cortisol"],
            "vascular_constriction": h["adrenaline"],
            "muscle_tension": h["testosterone"],
        }

    def decay_fast(self, rate: float = 0.5):
        """
        Accelerated fast-layer decay used by regulation (§4.3). Regulation
        should call this rather than mutating hormones directly (unlike the
        old _apply_regulation flat multiply in prometheus.py) so the
        "returning to baseline faster" model in the spec is literal.
        `rate` in (0, 1]; higher = faster return toward baseline.
        """
        with self.lock:
            rate = max(0.0, min(1.0, rate))
            for h in ("adrenaline", "cortisol"):  # fast-layer hormones
                baseline = 0.5
                self._hormones[h] = self._hormones[h] * (1 - rate) + baseline * rate

    def shift_slow_baseline(self, deltas: Dict[str, float]):
        """
        Slow-layer baseline shifts (§5: "slow-layer hormonal baseline
        shifts happen here [Consolidation] not instantly at epoch
        transition"). Only touches the slow-layer hormones so fast-layer
        state (which regulation/decay_fast owns) is never perturbed by a
        Consolidation-time baseline nudge.
        """
        with self.lock:
            for h in ("oxytocin", "testosterone", "estrogen", "thyroxine", "serotonin"):
                if h in deltas:
                    self._hormones[h] = max(0.0, min(1.0, self._hormones[h] + deltas[h]))

    def save_state(self):
        """§4C: only slow-layer hormones (temperament) + epoch are
        durable. Fast-layer hormones and the last somatic reading are
        deliberately NOT written -- a restarted agent should wake up at
        resting baseline, not mid-spike from whenever it was last closed."""
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            slow_layer_snapshot = {h: self._hormones[h] for h in self.SLOW_LAYER}
            with open(BIOSYSTEM_STATE_PATH, "w") as f:
                json.dump({
                    "slow_layer_hormones": slow_layer_snapshot,
                    "epoch": self.epoch.value,
                    "log_count": len(self.logger),
                }, f, indent=2)
        except OSError as e:
            logger.warning("BioSystem.save_state failed: %s", e)

    def load_state(self):
        """§4C: restores slow-layer baseline + epoch only. Fast-layer
        hormones stay at their __init__ resting-baseline defaults --
        loading never touches them. Also reads the old pre-§4C save
        format's "hormones" key as a slow-layer-only fallback for
        backward compatibility with checkpoints written before this fix,
        rather than silently discarding them."""
        if os.path.exists(BIOSYSTEM_STATE_PATH):
            try:
                with open(BIOSYSTEM_STATE_PATH, "r") as f:
                    data = json.load(f)
                slow_layer_snapshot = data.get("slow_layer_hormones") or {
                    k: v for k, v in data.get("hormones", {}).items() if k in self.SLOW_LAYER
                }
                for h in self.SLOW_LAYER:
                    if h in slow_layer_snapshot:
                        self._hormones[h] = slow_layer_snapshot[h]
                epoch_value = data.get("epoch")
                if epoch_value:
                    try:
                        self.epoch = Epoch(epoch_value)
                    except ValueError:
                        logger.warning("Unknown epoch '%s' in saved state; keeping default.", epoch_value)
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.warning("BioSystem.load_state failed, starting fresh: %s", e)
