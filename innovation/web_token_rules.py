"""
Innovation: web-token POS rules.

Handles URLs, email addresses, hashtags, @mentions, timestamps, dates,
and repeated punctuation — token types common in web text (EWT) that
are often missed by the baseline rules.

Notebook-03 error analysis findings:
  - PUNCT→SYM: 40 errors — "***", ">>", "<<", "<", ">" tagged SYM by
    baseline punct_num, but UD EWT labels them PUNCT.
  - NUM→NOUN: 54 errors — timestamps (07:17, 10:13) and dates
    (03/23/2001) not caught by baseline NUM regex.
"""

import re
from typing import List, Optional


# ── Web / social patterns ─────────────────────────────────────────────────

_URL_RE = re.compile(
    r'^(https?://|www\.)[^\s]+$', re.IGNORECASE
)

_EMAIL_RE = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

_HASHTAG_RE = re.compile(
    r'^#[a-zA-Z_]\w+$'
)

_MENTION_RE = re.compile(
    r'^@[a-zA-Z_]\w+$'
)

_FILE_EXT_RE = re.compile(
    r'^\S+\.(html?|php|asp|jsp|xml|json|csv|txt|pdf|docx?|xlsx?|pptx?|'
    r'jpe?g|png|gif|svg|mp[34]|avi|mov|zip|tar|gz|py|js|css|rb|java|cpp)$',
    re.IGNORECASE,
)

_EMOJI_RE = re.compile(
    r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F900-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F]+$'
)

# ── Timestamp / date patterns (notebook-03: 54 NUM→NOUN errors) ──────────

# HH:MM or HH:MM:SS
_TIME_RE = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')

# MM/DD/YYYY, MM/DD/YY, YYYY-MM-DD, MM-DD-YYYY
_DATE_RE = re.compile(
    r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$'
    r'|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}$'
)

# ── Repeated / decorative punctuation (notebook-03: 40 PUNCT→SYM errors)
# Strings of *, -, =, #, ~ used as section separators in EWT emails/web
# Gold tag in UD EWT is PUNCT for these, but baseline punct_num emits SYM.
_REPEATED_PUNCT_RE = re.compile(
    r'^[*\-=~#_+]{2,}$'   # "***", "---", "===", "~~~" etc.
    r'|^[<>]{1,2}$'        # "<", ">", "<<", ">>"
    r'|^//$'               # "//"
    r'|^\\\\$'             # "\\"
)


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Tag web-specific tokens.

    Priority (first match wins):
      PUNCT — repeated/decorative punctuation that baseline tags as SYM
      NUM   — timestamps and dates
      X     — URLs, emails, file extensions, hashtags
      PROPN — @mentions
      SYM   — emoji

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
    # Repeated punctuation / angle-brackets → PUNCT (fixes PUNCT→SYM errors)
    if _REPEATED_PUNCT_RE.match(word):
        return "PUNCT"

    # Timestamps and dates → NUM (fixes NUM→NOUN errors)
    if _TIME_RE.match(word):
        return "NUM"
    if _DATE_RE.match(word):
        return "NUM"

    # URLs
    if _URL_RE.match(word):
        return "X"

    # Emails
    if _EMAIL_RE.match(word):
        return "X"

    # File extensions / paths
    if _FILE_EXT_RE.match(word):
        return "X"

    # Hashtags
    if _HASHTAG_RE.match(word):
        return "X"

    # @mentions
    if _MENTION_RE.match(word):
        return "PROPN"

    # Emoji
    if _EMOJI_RE.match(word):
        return "SYM"

    return None
