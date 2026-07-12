from collections import deque


class ExecutiveModule:
    """
    Visible layer. Tracks drift in the synthesized intensity signal
    (synthesizer.get_current_intensity(), §4.1) to bias Learning toward
    exploration or stabilization. Per the Core Emergence Principle, this
    must never read BioSystem.get_somatic_readout() or any other hidden-
    layer value directly -- prometheus.py passes the already-synthesized
    intensity float in, same as it does for regulation.
    """

    def __init__(self, bio_system=None, archivist=None, config=None):
        # `bio_system`/`archivist` kept for signature compatibility with
        # existing callers; bio_system is intentionally unused for any
        # decision logic (see class docstring) -- bias_processing takes a
        # plain intensity float, not a bio reference to read from.
        self.bio = bio_system
        self.archivist = archivist
        self.config = config or {"deadband": 0.015}
        self.intensity_history = deque(maxlen=8)
        self.current_bias = "BIAS_NEUTRAL"

    def bias_processing(self, intensity: float) -> str:
        """`intensity` must be synthesizer.get_current_intensity() --
        never a raw somatic/hormonal value."""
        self.intensity_history.append(intensity)
        if len(self.intensity_history) < 2:
            return self.current_bias

        drift = self.intensity_history[-1] - self.intensity_history[-2]
        if drift > self.config["deadband"]:
            self.current_bias = "BIAS_EXPLORE"
        elif drift < -self.config["deadband"]:
            self.current_bias = "BIAS_STABILIZE"
        return self.current_bias
