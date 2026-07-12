import logging
import re
from typing import List, Optional, Tuple

from .core import Message

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.corpus import wordnet
    try:
        nltk.download("wordnet", quiet=True)
        _WORDNET_AVAILABLE = True
    except Exception as e:  # network unavailable, offline sandbox, etc.
        logger.warning("WordNet download failed, self-study lookups disabled: %s", e)
        _WORDNET_AVAILABLE = False
except ImportError:
    logger.warning("nltk not installed, self-study lookups disabled.")
    _WORDNET_AVAILABLE = False


# §2.3 mechanism 1: dictionary-pattern parsing. Definitional phrasing often
# contains the hierarchy for free ("blue: a color resembling the sky" ->
# `blue is-a color`). These are plain regexes, not an inference model --
# consistent with "no inference model needed" in the spec.
_HIERARCHY_PATTERNS = [
    re.compile(r"^\s*a[n]?\s+type\s+of\s+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*a[n]?\s+kind\s+of\s+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*is\s+a[n]?\s+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*a[n]?\s+(.+)$", re.IGNORECASE),  # "blue: a color resembling the sky"
]

# §2.3 mechanism 1 fallback for "part of" phrasing -> part-of edge instead
# of is-a.
_PART_OF_PATTERNS = [
    re.compile(r"^\s*(?:a\s+|an\s+)?part\s+of\s+(.+)$", re.IGNORECASE),
]


def _first_noun_phrase(remainder: str) -> str:
    """Definitional remainders are often a longer clause ("a color
    resembling the sky") -- the hierarchy parent is just the head noun, so
    take the text up to the first clause boundary (comma, "resembling",
    "that", "which", or the first stop word cluster) as a cheap, keyword
    -level approximation. No semantic parsing."""
    remainder = remainder.strip().rstrip(".")
    for stop in (",", " that ", " which ", " resembling ", " used ", " with "):
        idx = remainder.find(stop)
        if idx > 0:
            remainder = remainder[:idx]
            break
    return remainder.strip()


class SensoryModule:
    """
    Input layer (§7). Ingests dictionary/user/self-generated input,
    extracts hierarchy edges from definitional phrasing (§2.3 mechanism 1),
    and detects candidate relational edges (§2.1b, §3.4) -- all
    deterministic pattern/keyword matching, no NLP/embedding model in the
    detection path itself, consistent with the no-black-box-in-the-engine
    principle.
    """

    def __init__(self):
        self.relations = []

    def ingest(self, text: str):
        if not isinstance(text, str):
            raise TypeError(f"SensoryModule.ingest expects str, got {type(text)}")
        msg = Message(source="sensory", content=text)
        for rel in self.detect_relational(text):
            self.relations.append(rel)
        return msg

    # ------------------------------------------------------------------
    # §2.3 mechanism 1: hierarchy extraction from dictionary definitions.
    # ------------------------------------------------------------------
    def parse_hierarchy(self, definition: str) -> Optional[Tuple[str, str]]:
        """
        Given a definition string (e.g. "a color resembling the sky", or
        "part of a larger group"), returns (parent_node, edge_type) if a
        hierarchy relationship is parseable, else None. edge_type is
        'is-a' or 'part-of' per §2.3's typed-edge scheme -- never a
        generic edge.
        """
        if not definition:
            return None
        text = definition.strip()

        for pat in _PART_OF_PATTERNS:
            m = pat.match(text)
            if m:
                parent = _first_noun_phrase(m.group(1))
                if parent:
                    return parent, "part-of"

        for pat in _HIERARCHY_PATTERNS:
            m = pat.match(text)
            if m:
                parent = _first_noun_phrase(m.group(1))
                if parent:
                    return parent, "is-a"

        return None

    # ------------------------------------------------------------------
    # §2.1b relational edge candidates (extended beyond the original
    # single-relation version to cover all four edge types the spec
    # defines). Returns a list because a single sentence can trigger more
    # than one candidate (spec's own example: "I shouldn't have done that"
    # flags both `responsible-for` and `violates`).
    # ------------------------------------------------------------------
    def detect_relational(self, text: str) -> List[str]:
        text = text.lower()
        found = []

        # violates -- conflicts with a standard/value linked to SELF
        if "should not" in text or "shouldn't" in text or "wrong" in text or "not supposed to" in text:
            found.append("violates")

        # responsible-for -- SELF as agent of an action/outcome
        if "i did" in text or "my fault" in text or " i caused" in f" {text}" or "i made" in text:
            found.append("responsible-for")

        # temporal-contrast -- relates a node to a differing past state
        # (nostalgia). Keyword-level only, per spec: "using timestamps
        # chronos.py already logs" rather than any semantic comparison.
        if "used to" in text or "back then" in text or "remember when" in text or "before, " in text:
            found.append("temporal-contrast")

        # concerns-other -- involves a distinct entity other than SELF
        # (jealousy, embarrassment, social emotions generally). Cheap
        # heuristic: third-person pronoun or "they/he/she/someone" present.
        if re.search(r"\b(he|she|they|someone|another person|my friend|my sister|my brother)\b", text):
            found.append("concerns-other")

        return found

    def lookup_expansion(self, node: str):
        """Real dictionary lookup using WordNet for self-study expansion.
        Returns an empty list rather than raising if WordNet isn't
        available, so self-study degrades gracefully offline instead of
        crashing the pulse loop."""
        if not _WORDNET_AVAILABLE:
            return []
        synonyms = []
        for syn in wordnet.synsets(node):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return list(set(synonyms))[:5]

    def lookup_definition(self, node: str) -> Optional[str]:
        """WordNet gloss for `node`, used as the definitional text
        parse_hierarchy() runs against during self-study expansion (§5.1)
        so autonomously-added nodes get hierarchy placement too, not just
        user/dictionary-triggered ones."""
        if not _WORDNET_AVAILABLE:
            return None
        synsets = wordnet.synsets(node)
        if not synsets:
            return None
        return synsets[0].definition()
