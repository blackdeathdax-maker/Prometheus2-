from collections import deque


class ExecutiveModule:
    def __init__(self, bio_system=None, archivist=None, config=None):
        self.bio = bio_system
        self.archivist = archivist
        self.config = config or {"deadband": 0.015}
        self.somatic_history = deque(maxlen=8)
        self.current_bias = "BIAS_NEUTRAL"

    def bias_processing(self, somatic=None):
        if somatic is None and self.bio:
            somatic = self.bio.get_somatic_readout()

        if hasattr(somatic, "to_dict"):
            snapshot = somatic.to_dict()
        elif hasattr(somatic, "dict"):
            snapshot = somatic.dict()
        else:
            snapshot = {"urgency": getattr(somatic, "urgency", 0.0)}

        self.somatic_history.append(snapshot)
        if len(self.somatic_history) < 2:
            return self.current_bias

        drift = self.somatic_history[-1]["urgency"] - self.somatic_history[-2]["urgency"]
        if drift > self.config["deadband"]:
            self.current_bias = "BIAS_EXPLORE"
        elif drift < -self.config["deadband"]:
            self.current_bias = "BIAS_STABILIZE"
        return self.current_bias
