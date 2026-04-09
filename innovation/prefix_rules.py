"""
Innovation: prefix-based POS rules.

Uses common English prefixes to help guess the part of speech for OOV words
that have already passed lexicon, capitalization, and morphology rules.

Notebook-05 OOV analysis findings:
  - ~40 % of OOV words are PROPN → caught by capitalization rule (fires first)
  - ~30 % are NOUN with clear suffixes → caught by morphology rule (fires first)
  - This module targets the remaining lowercase OOV words with no suffix signal.

Prefixes removed vs original (high false-positive rate on NOUNs):
  "pre"  → president/pressure/present/prevent → mostly NOUNs/VERBs
  "de"   → defense/debate/describe/deputy/deal → mostly NOUNs
  "sub"  → subject/substance/subsequent → mostly NOUNs/ADVs
  "en"   → energy/enter/enough/entire → mostly NOUNs/ADJs
  "em"   → empty/embassy → mostly ADJs/NOUNs
  "dis"  → discount/display/distance/district → mostly NOUNs
"""

from typing import List, Optional


# Reliable prefix → tag mappings only
_PREFIX_RULES = {
    # Negative / reversative → ADJ
    "un":      "ADJ",    # unhappy, unclear, unusual, unresolved, unprecedented
    "non":     "ADJ",    # nonprofit, nonexistent, nonfiction, nonviolent

    # Repetition → VERB
    "mis":     "VERB",   # misunderstand, mislead, mispronounce, miscalculate
    "re":      "VERB",   # rebuild, reconsider, rewrite, renegotiate

    # Degree / direction → VERB
    "over":    "VERB",   # overcome, overlook, overestimate, overrule, overturn
    "under":   "VERB",   # underestimate, underpay, undermine, underperform
    "out":     "VERB",   # outperform, outlast, outweigh, outmaneuver

    # Scope / degree → ADJ
    "super":   "ADJ",    # supernatural, superhuman, supercharged
    "ultra":   "ADJ",    # ultramodern, ultraviolet, ultrasonic
    "semi":    "ADJ",    # semifinal, semicircular, semiautomatic
    "multi":   "ADJ",    # multilingual, multimedia, multifaceted
    "inter":   "ADJ",    # international, interactive, interdisciplinary
    "trans":   "ADJ",    # transatlantic, transgender, transcontinental
    "anti":    "ADJ",    # antisocial, antibiotic, antivirus, antiterrorism
    "post":    "ADJ",    # postmodern, postwar, postoperative
    "counter": "NOUN",   # counterargument, counterpart, counterattack
}

# Minimum stem length (characters after stripping the prefix).
# Tighter than the original to reduce residual false-positives.
_MIN_STEM: dict = {
    "re":      4,   # total ≥ 6  — avoids "real", "rely", "read"
    "out":     4,   # total ≥ 7  — avoids "outer", "outfit"
    "mis":     4,   # total ≥ 7  — avoids "mist"
    "un":      3,   # total ≥ 5
    "non":     3,   # total ≥ 6
    "over":    3,   # total ≥ 7
    "under":   3,   # total ≥ 8
    "super":   3,
    "ultra":   3,
    "semi":    3,
    "multi":   3,
    "inter":   3,
    "trans":   3,
    "anti":    3,
    "post":    3,
    "counter": 3,
}


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Guess POS based on common English prefixes.

    Only fires on lowercase OOV words that have already passed the
    lexicon, capitalization, and morphology rules.

    Parameters
    ----------
    word : str
        Surface form.
    context : list, optional
        Unused.

    Returns
    -------
    str or None
    """
    # Capitalized words reach here only when at sentence position 0
    # (capitalization rule skips pos-0). Do not tag them with prefix
    # rules — compound_context handles sentence-initial proper nouns.
    if word[0].isupper():
        return None

    w = word.lower()

    # Check longest prefix first for most specific match
    for prefix in sorted(_PREFIX_RULES, key=len, reverse=True):
        if not w.startswith(prefix):
            continue
        stem = w[len(prefix):]
        min_stem = _MIN_STEM.get(prefix, 3)
        if len(stem) >= min_stem:
            return _PREFIX_RULES[prefix]

    return None
