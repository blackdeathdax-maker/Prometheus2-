import random


class SyntheticStimulusEngine:
    """
    Test/demo harness for injecting synthetic hormonal events and chaotic
    graph data without going through the normal sensory/association
    ingestion path. Not part of the agent's own cognition -- useful for
    exercising the fatigue cycle and regulation logic in isolation.
    """

    def __init__(self, bio, archivist, reflector):
        self.bio = bio
        self.archivist = archivist
        self.reflector = reflector
        self.event_log = []

    def trigger_internal_event(self, intensity=0.6, focus="General"):
        with self.bio.lock:
            self.bio._hormones["adrenaline"] = min(
                1.0, self.bio._hormones.get("adrenaline", 0.3) + intensity * 0.4
            )
            self.bio._hormones["cortisol"] = min(
                1.0, self.bio._hormones.get("cortisol", 0.4) + intensity * 0.3
            )
        # Tagged self_generated (§2.2) so it's excluded from the diversity
        # signal at trust-scoring time (§9 risk 5) -- a synthetic stimulus
        # must not be able to inflate a node's own trust.
        self.archivist.store(f"Internal_{focus}", {"relations": {"related_to": focus}}, source="self_generated")
        self.event_log.append({"intensity": intensity, "focus": focus})
        print(f"Internal Event: {focus} (intensity={intensity})")

    def inject_chaotic_data(self, count=50):
        print(f"Injecting {count} chaotic nodes...")
        for _ in range(count):
            noise = f"ChaosNode_{random.randint(1000, 9999)}"
            self.archivist.store(noise, source="self_generated")
        print("Chaotic injection complete.")

    def log_state(self, pulse):
        print(f"\n=== Pulse {pulse} State ===")
        print(f"Bio Urgency: {self.bio.get_somatic_readout().urgency:.3f}")
        print(f"Archivist Nodes: {len(self.archivist.graph.nodes())}")
        print("=" * 40)
