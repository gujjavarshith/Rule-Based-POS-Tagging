"""
Capitalization-based POS rules.

Uses casing patterns to identify proper nouns (PROPN) and other tags.
"""

from typing import List, Optional


def apply(word: str, context: Optional[List[str]] = None,
          position: int = 0) -> Optional[str]:
    """
    Guess POS based on capitalization.

    Parameters
    ----------
    word : str
        Surface form.
    context : list, optional
        Unused.
    position : int
        0-based index of the token in the sentence.

    Returns
    -------
    str or None
    """
    # Skip very short tokens (e.g., "I")
    if len(word) <= 1:
        return None

    # All uppercase → PROPN (e.g., "NASA", "FBI") unless very short
    if word.isupper() and len(word) >= 2:
        return "PROPN"

    # Titlecase (first letter upper, rest lower) and NOT sentence-initial
    if word[0].isupper() and not word.isupper():
        if position > 0:
            # Non-sentence-initial capitalized word → PROPN
            return "PROPN"
        # Sentence-initial: can't tell from capitalization alone
        return None

    return None
