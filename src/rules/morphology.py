"""
Morphology-based POS rules.

Uses English suffix and prefix patterns to guess the part of speech for
unknown words (those not in the lexicon).
"""

import re
from typing import List, Optional


#  Suffix → Tag mappings (longest-match first) 

_SUFFIX_RULES = [
    # Adverbs
    ("ly",    "ADV"),

    # Nouns – derivational suffixes
    ("tion",  "NOUN"),
    ("sion",  "NOUN"),
    ("ment",  "NOUN"),
    ("ness",  "NOUN"),
    ("ity",   "NOUN"),
    ("ence",  "NOUN"),
    ("ance",  "NOUN"),
    ("ship",  "NOUN"),
    ("dom",   "NOUN"),
    ("ist",   "NOUN"),
    ("ism",   "NOUN"),
    ("ology", "NOUN"),
    ("ure",   "NOUN"),
    ("age",   "NOUN"),
    ("ery",   "NOUN"),
    ("ling",  "NOUN"),
    ("ette",  "NOUN"),
    ("eer",   "NOUN"),

    # Adjectives
    ("ous",   "ADJ"),
    ("ious",  "ADJ"),
    ("eous",  "ADJ"),
    ("ive",   "ADJ"),
    ("able",  "ADJ"),
    ("ible",  "ADJ"),
    ("ful",   "ADJ"),
    ("less",  "ADJ"),
    ("al",    "ADJ"),
    ("ical",  "ADJ"),
    ("ish",   "ADJ"),
    ("like",  "ADJ"),
    ("esque", "ADJ"),
    ("ent",   "ADJ"),
    ("ant",   "ADJ"),
    ("ary",   "ADJ"),
    ("ory",   "ADJ"),

    # Verbs
    ("ize",   "VERB"),
    ("ise",   "VERB"),
    ("ify",   "VERB"),
    ("ate",   "VERB"),
    ("en",    "VERB"),
    ("ing",   "VERB"),
    ("ed",    "VERB"),
]

# Sort by suffix length descending so longer suffixes match first
_SUFFIX_RULES.sort(key=lambda x: len(x[0]), reverse=True)

# Minimum word length to apply suffix rule (avoids tagging short words like "in")
_MIN_WORD_LEN = 5


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Guess POS using English morphological suffixes.

    Only fires on words ≥ 5 characters long to avoid false positives.

    Parameters
    ----------
    word : str
        Surface form.
    context : list of str, optional
        Unused.

    Returns
    -------
    str or None
    """
    if len(word) < _MIN_WORD_LEN:
        return None

    w = word.lower()

    for suffix, tag in _SUFFIX_RULES:
        if w.endswith(suffix):
            # Extra guard: the stem before the suffix should be ≥ 2 chars
            stem = w[: -len(suffix)]
            if len(stem) >= 2:
                return tag

    return None
