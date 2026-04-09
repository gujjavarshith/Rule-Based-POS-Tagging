"""
Innovation: compound context rules.

Detects multi-word expressions and adjusts tagging by looking at sequences
of tokens.

Notebook-03 error analysis findings used to improve this module:
  - PROPN→NOUN: 470 errors (biggest error type) — many capitalized words at
    sentence-initial position (pos=0) fall through to default NOUN because
    capitalization_rule skips pos=0.  Added sentence-initial PROPN detection.
  - Added EWT-specific proper noun bigrams observed in error examples:
    "Wall Street", "Prime Minister", "al-Qaeda" variants, news agency names.
  - Compound adposition set extended with patterns from SCONJ→ADP errors.
"""

from typing import Dict, List, Optional, Set, Tuple


# ── Common sentence-starting words that are NOT proper nouns ─────────────
# Capitalized words at position 0 that reach this rule should be PROPN
# unless they are in this list.
_SENTENCE_STARTERS: Set[str] = {
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each", "all", "both",
    "few", "many", "much", "another", "other",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "who", "what", "which", "there",
    # Conjunctions / adverbs
    "but", "and", "or", "nor", "so", "yet", "for",
    "however", "therefore", "although", "though",
    "while", "when", "if", "because", "since", "unless",
    "after", "before", "once", "as",
    # Common sentence openers that are not names
    "in", "on", "at", "to", "from", "with", "by", "of",
    "about", "for", "not", "just", "also", "even", "still",
    # Interrogatives
    "how", "why", "where", "what", "when", "which", "who",
    # Numbers spelled out
    "one", "two", "three", "four", "five",
    # Other frequent non-PROPN sentence starters
    "well", "ok", "yes", "no", "please", "thanks",
    "here", "now", "then", "thus",
}

# ── Known multi-word proper nouns (lowercased) ───────────────────────────
# Extended with EWT-specific bigrams from notebook-03 error analysis
_PROPER_BIGRAMS: Set[Tuple[str, str]] = {
    # US cities / places
    ("new", "york"), ("los", "angeles"), ("san", "francisco"),
    ("las", "vegas"), ("san", "diego"), ("san", "jose"),
    ("new", "orleans"), ("new", "jersey"), ("new", "mexico"),
    ("west", "virginia"), ("north", "carolina"), ("south", "carolina"),
    ("rhode", "island"), ("new", "hampshire"),
    # World cities / places
    ("hong", "kong"), ("buenos", "aires"), ("rio", "de"),
    ("cape", "town"), ("sri", "lanka"), ("el", "salvador"),
    ("costa", "rica"),
    # Countries / regions
    ("united", "states"), ("united", "kingdom"), ("united", "nations"),
    ("saudi", "arabia"), ("new", "zealand"), ("north", "korea"),
    ("south", "korea"), ("south", "africa"), ("middle", "east"),
    # Organizations / institutions
    ("wall", "street"), ("white", "house"), ("supreme", "court"),
    ("al", "qaeda"), ("al", "jazeera"), ("new", "york"),
    ("associated", "press"), ("fox", "news"), ("cnn", "news"),
    # Political titles (from error analysis: "Prime Minister" → PROPN PROPN)
    ("prime", "minister"), ("vice", "president"), ("attorney", "general"),
    ("secretary", "general"), ("chief", "justice"),
}

# ── Compound adpositions (extended from notebook-03 SCONJ→ADP errors) ────
_COMPOUND_ADP: Set[Tuple[str, str]] = {
    ("because", "of"), ("instead", "of"), ("according", "to"),
    ("due", "to"), ("prior", "to"), ("next", "to"),
    ("close", "to"), ("thanks", "to"), ("regardless", "of"),
    ("in", "front"), ("on", "top"), ("out", "of"),
    ("as", "well"), ("such", "as"), ("apart", "from"),
    ("ahead", "of"), ("as", "of"), ("as", "per"),
    ("in", "lieu"), ("in", "spite"), ("in", "terms"),
    ("on", "behalf"), ("on", "account"), ("with", "respect"),
}

# ── Phrasal verb particles ──────────────────────────────────────────────
_VERB_PARTICLES = frozenset({
    "up", "out", "off", "on", "in", "down", "away",
    "back", "over", "about", "around", "through",
})


def apply(word: str, context: Optional[List[str]] = None,
          words: Optional[List[str]] = None,
          position: int = -1) -> Optional[str]:
    """
    Use compound / multi-word context to refine tagging.

    Modes
    -----
    Simple (words=None): returns None — no context available.
    Full   (words+position given): checks bigrams and sentence-initial PROPN.

    Parameters
    ----------
    word : str
        Current token.
    context : list of str, optional
        Tag context.
    words : list of str, optional
        Full sentence word list.
    position : int
        Current word's index in the sentence.

    Returns
    -------
    str or None
    """
    # Simple mode — no sentence context available
    if words is None or position < 0:
        return None

    n = len(words)
    w_lower = word.lower()

    # ── Sentence-initial PROPN detection ───────────────────────────────
    # Notebook-03: PROPN→NOUN is the #1 error (470 cases).  Many happen at
    # sentence position 0 where capitalization_rule returns None.
    # If the word is capitalized, at position 0, and not a common sentence
    # starter → tag it PROPN.
    if position == 0 and word[0].isupper() and w_lower not in _SENTENCE_STARTERS:
        return "PROPN"

    # ── Check current + next form a known proper bigram ────────────────
    if position + 1 < n:
        bigram = (w_lower, words[position + 1].lower())
        if bigram in _PROPER_BIGRAMS:
            return "PROPN"

    # ── Check prev + current form a known proper bigram ────────────────
    if position > 0:
        bigram = (words[position - 1].lower(), w_lower)
        if bigram in _PROPER_BIGRAMS:
            return "PROPN"

    # ── Compound adpositions → ADP ──────────────────────────────────────
    if position + 1 < n:
        bigram = (w_lower, words[position + 1].lower())
        if bigram in _COMPOUND_ADP:
            return "ADP"

    if position > 0:
        bigram = (words[position - 1].lower(), w_lower)
        if bigram in _COMPOUND_ADP:
            return "ADP"

    # ── Phrasal verb particles ──────────────────────────────────────────
    if context and position > 0:
        prev_tag = context[position - 1] if position - 1 < len(context) else None
        if prev_tag == "VERB" and w_lower in _VERB_PARTICLES:
            return "ADP"

    return None
