"""
others.py -- Multiple named social entities (lightweight, deterministic).

Beyond the single axiomatic OTHER node, this module tracks named people
extracted from text (my friend Sam, my boss, she/Alex, etc.). Each named
other is a normal graph node (tier Working when first mentioned) with:
  - node_type hint / is_other flag
  - per-entity valence_coloring (same accumulator as parental feedback)
  - mention count + last_seen

No theory-of-mind model. Just durable identity + relational history so
concerns-other / social-norm edges and narrative can attach to specific
people instead of only the generic OTHER placeholder.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Generic OTHER stays as the fallback when no specific name is found.
GENERIC_OTHER = "OTHER"

# Deterministic extraction patterns — conservative, keyword-level only.
_NAMED_PATTERNS = [
    # my friend Sam / my sister Jane / my boss Alex
    re.compile(
        r"\b(?:my|our)\s+(?:friend|sister|brother|mother|father|mom|dad|"
        r"partner|boss|colleague|coworker|neighbor|teacher|doctor|"
        r"husband|wife|son|daughter|cousin|uncle|aunt)\s+"
        r"([A-Z][a-z]{1,20})\b"
    ),
    # Sam said / Alex told me / Jordan asked
    re.compile(
        r"\b([A-Z][a-z]{1,20})\s+(?:said|told|asked|called|texted|wrote|"
        r"wanted|needed|helped|hurt|left|came)\b"
    ),
    # with Sam / to Alex / from Jordan
    re.compile(
        r"\b(?:with|to|from|about|for)\s+([A-Z][a-z]{1,20})\b"
    ),
]

# Lowercase kinship / role words that should become other nodes even
# without a proper name (still more specific than generic OTHER).
_ROLE_OTHERS = frozenset({
    "friend", "sister", "brother", "mother", "father", "mom", "dad",
    "partner", "boss", "colleague", "coworker", "neighbor", "teacher",
    "husband", "wife", "son", "daughter", "cousin",
})

# Pronouns map to the *most recently mentioned* named other when possible;
# otherwise fall back to generic OTHER.
_PRONOUNS = frozenset({"he", "she", "him", "her", "they", "them"})


def _normalize_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    # Keep display form Title Case for the graph id stability
    return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()


class OthersRegistry:
    """
    Owns extraction + registration of named other entities.
    Graph mutations go through archivist; this only decides *which*
    other-id to use and tracks lightweight per-other stats.
    """

    MAX_NAMED_OTHERS = 40
    MENTION_BOOST_COLORING = 0.02  # tiny positive nudge on re-mention

    def __init__(self, archivist):
        self.archivist = archivist
        # name -> {mentions, last_pulse, role_hint}
        self.named: Dict[str, dict] = {}
        self._last_specific: Optional[str] = None  # for pronoun resolution

    def extract_names(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (other_id, how) from text.
        how: 'proper' | 'role' | 'pronoun'
        """
        if not text or not isinstance(text, str):
            return []
        found: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        # Proper names via patterns (case-sensitive on the name capture)
        for pat in _NAMED_PATTERNS:
            for m in pat.finditer(text):
                name = _normalize_name(m.group(1))
                if not name or name.lower() in _PRONOUNS:
                    continue
                # Skip common false positives
                if name.lower() in {
                    "i", "me", "my", "the", "a", "an", "and", "or", "but",
                    "when", "then", "that", "this", "what", "who", "how",
                    "yes", "no", "ok", "okay", "hi", "hey",
                }:
                    continue
                if name not in seen:
                    seen.add(name)
                    found.append((name, "proper"))

        # Role words without proper name (lowercase scan)
        low = text.lower()
        for role in _ROLE_OTHERS:
            # "my boss" / "the boss" / "boss said"
            if re.search(rf"\b(?:my|the|our)\s+{role}\b", low) or re.search(
                rf"\b{role}\s+(?:said|told|asked|called)\b", low
            ):
                # Role-only id: other_boss, other_friend, ...
                rid = f"other_{role}"
                if rid not in seen:
                    seen.add(rid)
                    found.append((rid, "role"))

        # Pronoun → most recent specific other
        if any(re.search(rf"\b{p}\b", low) for p in _PRONOUNS):
            if self._last_specific and self._last_specific not in seen:
                found.append((self._last_specific, "pronoun"))
            elif GENERIC_OTHER not in seen and not found:
                found.append((GENERIC_OTHER, "pronoun"))

        return found

    def ensure_other(self, other_id: str, role_hint: str = "", pulse: int = 0) -> str:
        """Ensure a named other exists in the graph; return the node id used."""
        if not other_id or other_id == GENERIC_OTHER:
            # generic OTHER is seeded by archivist
            return GENERIC_OTHER

        graph = self.archivist.graph
        if other_id not in graph:
            # Cap total named others
            if len(self.named) >= self.MAX_NAMED_OTHERS:
                # evict least-mentioned
                if self.named:
                    weakest = min(self.named.items(), key=lambda t: t[1].get("mentions", 0))[0]
                    # do not delete graph node (may have edges); just stop tracking new growth
                    pass
            from .archivist import TIER_WORKING
            self.archivist.store(other_id, source="social", tier=TIER_WORKING)
            if other_id in graph:
                graph.nodes[other_id]["is_other"] = True
                graph.nodes[other_id]["other_role"] = role_hint or ""
                graph.nodes[other_id]["valence_coloring"] = graph.nodes[other_id].get(
                    "valence_coloring", 0.0
                )

        stats = self.named.setdefault(
            other_id, {"mentions": 0, "last_pulse": 0, "role_hint": role_hint or ""}
        )
        stats["mentions"] = stats.get("mentions", 0) + 1
        stats["last_pulse"] = pulse
        if role_hint and not stats.get("role_hint"):
            stats["role_hint"] = role_hint

        self._last_specific = other_id
        return other_id

    def process_text(self, text: str, pulse: int = 0) -> List[str]:
        """Extract + ensure all named others in text. Returns list of other ids."""
        extracted = self.extract_names(text)
        ids = []
        for name, how in extracted:
            role = ""
            if how == "role" and name.startswith("other_"):
                role = name[len("other_"):]
            oid = self.ensure_other(name, role_hint=role, pulse=pulse)
            ids.append(oid)
        return list(dict.fromkeys(ids))  # unique, order preserved

    def color_other(self, other_id: str, delta: float, cap: float = 1.0) -> None:
        """Per-other valence coloring (same shape as parental feedback)."""
        if not other_id or other_id not in self.archivist.graph:
            return
        self.archivist.nudge_valence_coloring(other_id, delta, cap=cap)

    def top_others(self, n: int = 12) -> List[dict]:
        rows = sorted(
            self.named.items(),
            key=lambda t: (-t[1].get("mentions", 0), -t[1].get("last_pulse", 0)),
        )[:n]
        out = []
        for oid, st in rows:
            coloring = 0.0
            if oid in self.archivist.graph:
                coloring = float(self.archivist.graph.nodes[oid].get("valence_coloring", 0.0))
            out.append({
                "id": oid,
                "mentions": st.get("mentions", 0),
                "role_hint": st.get("role_hint", ""),
                "last_pulse": st.get("last_pulse", 0),
                "valence_coloring": round(coloring, 3),
            })
        return out

    def report(self) -> dict:
        return {
            "named_count": len(self.named),
            "last_specific": self._last_specific,
            "top": self.top_others(15),
        }

    def save_state(self, path: str) -> None:
        import json, os
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "named": self.named,
                    "last_specific": self._last_specific,
                }, f, indent=2)
        except OSError as e:
            logger.warning("OthersRegistry.save_state failed: %s", e)

    def load_state(self, path: str) -> None:
        import json, os
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.named = dict(data.get("named") or {})
            self._last_specific = data.get("last_specific")
        except Exception as e:
            logger.warning("OthersRegistry.load_state failed: %s", e)
