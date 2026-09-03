"""
felt_templates.py — Felt-state / emotion binding (FELT_STATE_BINDING_SPEC).

Layers (all simultaneous):
  1. PAD position — always live (synthesizer)
  2. 7-channel signed deviation from allostatic setpoint — always live
  3. Template match — optional naming when taught template clears L threshold

Feeling does not depend on naming. Naming writes edges to SELF + body channels.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Seven grounding channels (pain/pleasure are separate affect surface)
FELT_CHANNELS = (
    "heart_rate",
    "breath",
    "muscle_tension",
    "sweat_skin",
    "gut",
    "energy",
    "warmth",
)

# Qualitative → signed deviation from setpoint (teaching)
HIGH_WORDS = (
    "high", "elevated", "racing", "pounding", "quick", "sharp", "forceful",
    "rigid", "clenched", "explosive", "expansive", "widespread", "hot",
    "strong", "full", "vital", "fluttering", "spiking", "up", "mobilized",
    "wet", "clammy", "tight", "churning", "fast", "rapid",
)
LOW_WORDS = (
    "slow", "slowed", "low", "depleted", "minimal", "absent", "cool", "cold",
    "limp", "loose", "hollow", "heavy", "restrained", "shallow", "draining",
    "down", "less", "weak", "calm", "settled", "soft", "dry", "held",
    "dropping", "sighing",
)

CHANNEL_ALIASES = {
    "heart_rate": ["heart", "heartbeat", "heart rate", "pulse"],
    "breath": ["breath", "breathing", "breathe", "respiration", "sighing"],
    "muscle_tension": [
        "tension", "tense", "muscle", "clench", "tight", "limp", "loose",
        "rigid", "bracing", "heavy",
    ],
    "sweat_skin": ["sweat", "sweating", "sweaty", "perspir", "clammy", "damp", "dry"],
    "gut": ["gut", "stomach", "nausea", "butterflies", "belly", "hollow", "churning"],
    "energy": [
        "energy", "energetic", "tired", "fatigue", "exhausted", "wired",
        "depleted", "expansive", "mobilized", "jittery", "sluggish",
    ],
    "warmth": [
        "warm", "warmth", "hot", "cold", "chill", "flush", "cooling", "glowing",
    ],
}

EMOTION_HEADS = (
    "joy", "sadness", "anger", "fear", "disgust", "surprise",
    "guilt", "pride", "jealousy", "compassion", "anxiety",
    "elation", "shame", "grief", "calm",
)

# Evidence spine for template match (same materials as op L)
L_HIT = 0.28
L_MISS = 0.22
L_DECAY = 0.997
L_MIN, L_MAX = -3.0, 3.0
THETA_MATCH = 0.85
SMOOTH_ALPHA = 0.45  # 2–3 pulse-ish EMA on live snapshot
PAD_PREFILTER = 0.55  # max PAD distance to shortlist (normalized)
CHANNEL_MATCH_MAX = 2.2  # sum abs signed-dev distance; below → near-match


@dataclass
class FeltTemplate:
    name: str
    pad: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # A, V, D reference
    # channel -> signed deviation from setpoint at teach time (or explicit)
    channel_dev: Dict[str, float] = field(default_factory=dict)
    ref_magnitude: float = 1.0
    source: str = "user"
    log_odds: float = 0.0
    match_count: int = 0


class FeltTemplateStore:
    """Teach, smooth, match, commit felt-state templates."""

    def __init__(self):
        self.templates: Dict[str, FeltTemplate] = {}
        self._smooth_dev: Dict[str, float] = {}
        self._smooth_pad: Optional[Tuple[float, float, float]] = None
        self.last_match: Dict[str, Any] = {}
        self.active_name: Optional[str] = None
        self.active_intensity: float = 0.0

    # ----- teaching -----
    def parse_and_teach(
        self,
        text: str,
        *,
        pad: Optional[Tuple[float, float, float]] = None,
        setpoints: Optional[Dict[str, float]] = None,
        body: Optional[Dict[str, float]] = None,
    ) -> Optional[FeltTemplate]:
        """Parse 'Sadness: heart rate slowed, energy depleted…' into a template."""
        if not text or not str(text).strip():
            return None
        t = str(text).strip().lower()
        name = None
        head = t.split(":")[0].strip() if ":" in t else (t.split()[0] if t.split() else "")
        head = re.sub(r"[^a-z_]", "", head.replace("-", "_"))
        for em in EMOTION_HEADS:
            if head == em or t.startswith(em + ":") or t.startswith(em + " "):
                name = em
                break
        if not name:
            # "joy is an emotion" alone is not a somatic template
            return None

        # Need at least one channel cue
        channel_dev: Dict[str, float] = {}
        for ch, aliases in CHANNEL_ALIASES.items():
            if not any(a in t for a in aliases):
                continue
            pos = min((t.find(a) for a in aliases if a in t), default=-1)
            window = t[max(0, pos - 32): pos + 48] if pos >= 0 else t
            if any(w in window for w in LOW_WORDS):
                channel_dev[ch] = -0.55
            elif any(w in window for w in HIGH_WORDS):
                channel_dev[ch] = 0.55
            else:
                # mentioned without clear pole — mild high as "salient"
                channel_dev[ch] = 0.25

        if not channel_dev:
            return None

        # Optional: refine magnitude from live body vs setpoint at teach time
        sp = setpoints or {}
        bd = body or {}
        mags = []
        for ch, dev in list(channel_dev.items()):
            if ch in bd and ch in sp:
                live_dev = float(bd[ch]) - float(sp[ch])
                # If live agrees in sign, use magnitude
                if live_dev * dev > 0:
                    channel_dev[ch] = max(-0.9, min(0.9, live_dev))
                    mags.append(abs(live_dev))
        ref_mag = max(0.35, sum(mags) / len(mags)) if mags else 0.55

        pad_ref = pad or (0.5, 0.5, 0.5)
        # Soft PAD prior from channel face (very light — not ontology)
        a = 0.5 + 0.15 * (channel_dev.get("heart_rate", 0) + channel_dev.get("energy", 0)) / 2
        v = 0.5 + 0.2 * (
            channel_dev.get("warmth", 0) * 0.5
            + channel_dev.get("energy", 0) * 0.3
            - channel_dev.get("muscle_tension", 0) * 0.2
        )
        d = 0.5 + 0.15 * (
            channel_dev.get("energy", 0) - channel_dev.get("gut", 0) * 0.5
        )
        if pad is None:
            pad_ref = (
                max(0.05, min(0.95, a)),
                max(0.05, min(0.95, v)),
                max(0.05, min(0.95, d)),
            )

        tmpl = FeltTemplate(
            name=name,
            pad=pad_ref,
            channel_dev=dict(channel_dev),
            ref_magnitude=float(ref_mag),
            source="user",
            log_odds=0.0,
        )
        self.templates[name] = tmpl
        return tmpl

    # ----- live smoothing + match -----
    def observe(
        self,
        body: Dict[str, float],
        setpoints: Dict[str, float],
        pad: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """Smooth snapshot, update L per template, commit if θ crossed."""
        # Signed deviations
        raw_dev = {}
        for ch in FELT_CHANNELS:
            obs = float(body.get(ch, setpoints.get(ch, 0.5)))
            sp = float(setpoints.get(ch, 0.5))
            raw_dev[ch] = obs - sp

        # EMA smooth
        for ch, d in raw_dev.items():
            prev = self._smooth_dev.get(ch, d)
            self._smooth_dev[ch] = (1.0 - SMOOTH_ALPHA) * prev + SMOOTH_ALPHA * d
        if self._smooth_pad is None:
            self._smooth_pad = pad
        else:
            a0, v0, d0 = self._smooth_pad
            a, v, d = pad
            self._smooth_pad = (
                (1.0 - SMOOTH_ALPHA) * a0 + SMOOTH_ALPHA * a,
                (1.0 - SMOOTH_ALPHA) * v0 + SMOOTH_ALPHA * v,
                (1.0 - SMOOTH_ALPHA) * d0 + SMOOTH_ALPHA * d,
            )

        # Decay all L
        for tmpl in self.templates.values():
            tmpl.log_odds = max(L_MIN, min(L_MAX, tmpl.log_odds * L_DECAY))

        if not self.templates:
            self.active_name = None
            self.active_intensity = 0.0
            self.last_match = {"matched": None, "reason": "no_templates"}
            return self.last_match

        # PAD shortlist
        pa, pv, pd = self._smooth_pad
        shortlist = []
        for name, tmpl in self.templates.items():
            ta, tv, td = tmpl.pad
            dist = math.sqrt((pa - ta) ** 2 + (pv - tv) ** 2 + (pd - td) ** 2)
            if dist <= PAD_PREFILTER or len(self.templates) <= 4:
                shortlist.append((dist, name, tmpl))
        if not shortlist:
            shortlist = [
                (
                    math.sqrt(
                        (pa - t.pad[0]) ** 2
                        + (pv - t.pad[1]) ** 2
                        + (pd - t.pad[2]) ** 2
                    ),
                    n,
                    t,
                )
                for n, t in self.templates.items()
            ]
        shortlist.sort(key=lambda x: x[0])

        best_name = None
        best_chan_dist = 1e9
        best_intensity = 0.0
        near = []
        for dist_pad, name, tmpl in shortlist[:8]:
            chan_dist = 0.0
            n = 0
            dir_agree = 0
            live_mag = 0.0
            for ch, ref_dev in tmpl.channel_dev.items():
                live = float(self._smooth_dev.get(ch, 0.0))
                chan_dist += abs(live - ref_dev)
                n += 1
                if live * ref_dev > 0 and abs(live) > 0.05:
                    dir_agree += 1
                    live_mag += abs(live)
            if n == 0:
                continue
            chan_dist /= n
            # Near-match if channel distance small and some direction agreement
            is_near = chan_dist < (CHANNEL_MATCH_MAX / max(3, n)) and dir_agree >= max(
                1, n // 2
            )
            if is_near:
                tmpl.log_odds = max(
                    L_MIN, min(L_MAX, tmpl.log_odds + L_HIT)
                )
                near.append(name)
                intensity = 0.0
                if tmpl.ref_magnitude > 1e-6:
                    intensity = min(2.0, (live_mag / max(1, dir_agree)) / tmpl.ref_magnitude)
                if chan_dist < best_chan_dist:
                    best_chan_dist = chan_dist
                    best_name = name
                    best_intensity = intensity
            else:
                tmpl.log_odds = max(L_MIN, min(L_MAX, tmpl.log_odds - L_MISS * 0.5))

        matched = None
        intensity = 0.0
        if best_name:
            tmpl = self.templates[best_name]
            if tmpl.log_odds >= THETA_MATCH:
                matched = best_name
                intensity = best_intensity
                tmpl.match_count += 1
                self.active_name = matched
                self.active_intensity = intensity
            else:
                # accumulating evidence — not yet named
                if self.active_name == best_name and tmpl.log_odds < THETA_MATCH * 0.5:
                    self.active_name = None
                    self.active_intensity = 0.0
        else:
            self.active_name = None
            self.active_intensity = 0.0

        self.last_match = {
            "matched": matched,
            "candidate": best_name,
            "L": {
                n: round(t.log_odds, 3) for n, t in self.templates.items()
            },
            "intensity": round(intensity, 3) if matched else 0.0,
            "near": near,
            "smooth_dev": {k: round(v, 3) for k, v in self._smooth_dev.items()},
            "pad": tuple(round(x, 3) for x in (self._smooth_pad or (0.5, 0.5, 0.5))),
            "theta": THETA_MATCH,
        }
        return self.last_match

    def report(self) -> dict:
        return {
            "templates": {
                n: {
                    "pad": t.pad,
                    "channel_dev": t.channel_dev,
                    "ref_magnitude": t.ref_magnitude,
                    "L": round(t.log_odds, 3),
                    "match_count": t.match_count,
                }
                for n, t in self.templates.items()
            },
            "active": self.active_name,
            "intensity": self.active_intensity,
            "last_match": dict(self.last_match or {}),
        }
