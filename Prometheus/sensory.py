import logging
import re
from typing import List, Optional, Tuple

from .core import Message
from .edge_types import (
    EDGE_IS_A, EDGE_PART_OF, EDGE_VIOLATES, EDGE_RESPONSIBLE_FOR,
    EDGE_TEMPORAL_CONTRAST, EDGE_CONCERNS_OTHER,
)

logger = logging.getLogger(__name__)

# Dictionary source (§2.2, §5.1) is WordNet via nltk.corpus.wordnet --
# glosses (lookup_definition), hyponyms (lookup_expansion, self-study
# children), hypernyms (lookup_hypernym, re-parenting), and synonyms
# (lookup_synonyms, canonicalization) all come from the same corpus. This
# is why nltk is a requirement even though nothing else in the design uses
# NLP -- previously undocumented anywhere in the spec.

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

# Self-study word-extraction fallback (§5.1, new -- see lookup_expansion's
# docstring for the bug this fixes). Common function words plus the
# relational-detection trigger words themselves (§2.1b: "shouldn't",
# "wrong", "fault", "did", "made", "caused", "used", etc.) are excluded
# from candidacy, since those are grammatical scaffolding or the SIGNAL
# that flagged the sentence as relational, not its substantive topic. Not
# an exhaustive list -- a small, deterministic, hand-maintained set in the
# same spirit as the equally small, fixed relational-keyword lists already
# used elsewhere in sensory.py (§3.4, §2.1b), not a claim of linguistic
# completeness.
_EXPANSION_STOPWORDS = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "the", "a", "an", "is", "was", "were", "be", "been", "being", "am", "are",
    "this", "that", "these", "those", "and", "or", "but", "not", "no",
    "to", "of", "in", "on", "at", "for", "with", "as", "so", "too",
    "my", "your", "his", "its", "our", "their",
    "do", "did", "does", "done", "have", "has", "had",
    "should", "shouldn't", "wrong", "fault", "caused", "made",
    "used", "back", "then", "before", "remember", "when",
    "someone", "another", "person", "friend", "sister", "brother",
})


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
                    return parent, EDGE_PART_OF

        for pat in _HIERARCHY_PATTERNS:
            m = pat.match(text)
            if m:
                parent = _first_noun_phrase(m.group(1))
                if parent:
                    return parent, EDGE_IS_A

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
            found.append(EDGE_VIOLATES)

        # responsible-for -- SELF as agent of an action/outcome
        if "i did" in text or "my fault" in text or " i caused" in f" {text}" or "i made" in text:
            found.append(EDGE_RESPONSIBLE_FOR)

        # temporal-contrast -- relates a node to a differing past state
        # (nostalgia). Keyword-level only, per spec: "using timestamps
        # chronos.py already logs" rather than any semantic comparison.
        if "used to" in text or "back then" in text or "remember when" in text or "before, " in text:
            found.append(EDGE_TEMPORAL_CONTRAST)

        # concerns-other -- involves a distinct entity other than SELF
        # (jealousy, embarrassment, social emotions generally). Cheap
        # heuristic: third-person pronoun or "they/he/she/someone" present.
        if re.search(r"\b(he|she|they|someone|another person|my friend|my sister|my brother)\b", text):
            found.append(EDGE_CONCERNS_OTHER)

        return found

    def lookup_expansion(self, node: str):
        """Self-study expansion (§5.1): "an existing 'colors' node gets
        children like 'blue,' 'white' pulled autonomously from the
        dictionary." That's a hyponym relationship (subtype), NOT a
        synonym one -- a prior version of this method returned
        syn.lemmas() (same-synset synonyms, e.g. "colour"/"coloring" for
        "colors"), which can never actually produce the parent->child
        taxonomy §5.1 describes, since synonyms aren't children. Fixed to
        use syn.hyponyms(), with the old synonym behavior preserved
        separately in lookup_synonyms() for its own, different, use
        (canonicalization/dedup -- see that method's docstring).

        Multi-word fallback (new, this revision -- bug found while
        investigating "self-study never expands nodes connected to
        SELF"). Any node created from real typed input (association.
        place_node() uses the whole message as the node name, §2.2) is a
        full sentence, and wordnet.synsets() has no entry for a full
        sentence -- it always returned [] for these, silently marking
        every such node barren after one self-study attempt and
        permanently excluding it. This wasn't actually SELF-specific: it
        affected every real user message equally, self-referential or
        not, since the root cause is "sentence, not a WordNet lemma," not
        anything about which relational edges the node happens to carry.
        Now tries the whole node name first (covers genuine multi-word
        WordNet entries like "New York"), then falls through to trying
        each individual significant word (skipping _EXPANSION_STOPWORDS --
        common function words and the relational-detection trigger words
        themselves, neither of which is the sentence's actual topic),
        stopping at the first word that yields real hyponyms.

        Returns an empty list rather than raising if WordNet isn't
        available, so self-study degrades gracefully offline instead of
        crashing the pulse loop."""
        if not _WORDNET_AVAILABLE:
            return []

        def hyponyms_for(candidate: str) -> List[str]:
            found = []
            for syn in wordnet.synsets(candidate):
                for hyponym in syn.hyponyms():
                    for lemma in hyponym.lemmas():
                        found.append(lemma.name().replace("_", " "))
            return found

        children = hyponyms_for(node)
        if not children and " " in node:
            for word in node.split():
                word = word.strip(".,!?;:'\"").lower()
                if not word or word in _EXPANSION_STOPWORDS:
                    continue
                children = hyponyms_for(word)
                if children:
                    break

        return list(set(children))[:5]

    def lookup_synonyms(self, node: str):
        """Same-synset synonyms (e.g. "color" -> "colour," "coloring").
        NOT used for self-study expansion (a synonym is the same concept
        wearing a different word, not a child of it) -- intended use is
        canonicalization/dedup: association.py can check this before
        creating a new node, so "color" and "colour" reinforce one shared
        node instead of splitting corroboration across two. Not yet wired
        into place_node()'s create path -- flagged as an open item, same
        shape as §11's archive rehydration problem, just showing up
        earlier in the live graph rather than after archiving exists."""
        if not _WORDNET_AVAILABLE:
            return []
        synonyms = []
        for syn in wordnet.synsets(node):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().replace("_", " "))
        return list(set(synonyms))[:5]

    def lookup_hypernym(self, node: str) -> Optional[str]:
        """First WordNet hypernym (the broader category `node` belongs
        to), e.g. "blue" -> "color". Used by association.py's re-parenting
        pass (§2.3 mechanism 3) as the authoritative firmer parent for a
        co-occurrence-placed node, instead of trying to regex-parse a
        gloss with parse_hierarchy() -- WordNet's own taxonomy already
        gives the answer directly, no pattern-matching needed. Returns
        None if there's no hypernym (already a root concept) or WordNet
        is unavailable."""
        if not _WORDNET_AVAILABLE:
            return None
        synsets = wordnet.synsets(node)
        if not synsets:
            return None
        hypernyms = synsets[0].hypernyms()
        if not hypernyms:
            return None
        return hypernyms[0].lemmas()[0].name().replace("_", " ")

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
