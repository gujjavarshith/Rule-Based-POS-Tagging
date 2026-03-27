"""
Punctuation and numeral detection rules.

Handles tokens that are clearly PUNCT, SYM, or NUM based on their surface form.
"""

import re
from typing import List, Optional


# ── Patterns ─────────────────────────────────────────────────────────────
# Standard punctuation characters
_PUNCT_RE = re.compile(
    r'^[\.\,\;\:\!\?\"\'\`\-\–\—\(\)\[\]\{\}\/\\…\u2018\u2019\u201C\u201D]+$'
)

# Symbols (currency, math, misc)
_SYM_RE = re.compile(
    r'^[\$\€\£\¥\%\+\=\<\>\|\^~\&\*\@\#\°\±\×\÷§©®™]+$'
)

# Numbers: integers, decimals, negative, percentages, ordinals, fractions
_NUM_RE = re.compile(
    r'^-?[\d][\d,]*\.?[\d]*%?$'          # 123  1,000  3.14  -5  50%
    r'|^[\d]+/[\d]+$'                     # 1/2  3/4
    r'|^[\d]+(st|nd|rd|th)$'              # 1st  2nd  3rd  4th
    r'|^(one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
    r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
    r'eighty|ninety|hundred|thousand|million|billion|trillion|'
    r'zero|dozen|half)$',                 # spelled-out numbers
    re.IGNORECASE
)

# Roman numerals
_ROMAN_RE = re.compile(
    r'^[IVXLCDM]+$'
)


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Return PUNCT, SYM, or NUM if the word matches; otherwise None.

    Parameters
    ----------
    word : str
        The surface form of the token.
    context : list of str, optional
        Surrounding words (unused by this rule).

    Returns
    -------
    str or None
    """
    # Punctuation
    if _PUNCT_RE.match(word):
        return "PUNCT"

    # Symbols
    if _SYM_RE.match(word):
        return "SYM"

    # Numerals
    if _NUM_RE.match(word):
        return "NUM"

    # Short Roman numerals (II, III, IV, etc.) but not single letters like I
    if len(word) >= 2 and _ROMAN_RE.match(word):
        return "NUM"

    return None
