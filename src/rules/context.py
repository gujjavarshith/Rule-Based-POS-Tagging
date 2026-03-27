"""
Context-based POS rules.

Uses the tags of surrounding words (already assigned by higher-priority rules
or the lexicon) to disambiguate the current token.
"""

from typing import List, Optional


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Infer POS from the tags of neighbouring tokens.

    ``context`` is a list of already-assigned tags for the **full sentence**.
    The current word's position is indicated by a ``None`` entry in the list.

    Convention:  context = [tag_0, tag_1, ..., None, ..., tag_n]
                 The ``None`` marks the position of the word being tagged.

    Parameters
    ----------
    word : str
        Surface form.
    context : list of str or None
        Tag sequence for the sentence, with None at the current position.

    Returns
    -------
    str or None
    """
    if context is None:
        return None

    # Find the current position (marked as None)
    try:
        pos = context.index(None)
    except ValueError:
        return None

    prev_tag = context[pos - 1] if pos > 0 else None
    prev2_tag = context[pos - 2] if pos > 1 else None
    next_tag = context[pos + 1] if pos + 1 < len(context) else None

    w = word.lower()

    #  Rule 1: After DET → expect NOUN or ADJ 
    if prev_tag == "DET":
        # "the/a + <word>" – likely NOUN if no adjective suffix
        if _looks_adjective(w):
            return "ADJ"
        return "NOUN"

    #  Rule 2: After ADP → expect NOUN / DET / PROPN 
    if prev_tag == "ADP":
        if w[0].isupper() and pos > 0:
            return "PROPN"
        return "NOUN"

    #  Rule 3: After AUX → expect VERB or ADV 
    if prev_tag == "AUX":
        if w.endswith("ly"):
            return "ADV"
        return "VERB"

    # Rule 4: After "to" (PART) → expect VERB 
    if prev_tag == "PART" and pos > 0:
        # Check if the previous word was actually "to"
        return "VERB"

    #  Rule 5: ADJ ADJ → second one might be NOUN 
    if prev_tag == "ADJ" and prev2_tag == "ADJ":
        return "NOUN"

    #  Rule 6: DET ADJ → next is NOUN 
    if prev_tag == "ADJ" and prev2_tag == "DET":
        return "NOUN"

    #  Rule 7: VERB + ADV/ADP pattern 
    if prev_tag == "VERB":
        if w.endswith("ly"):
            return "ADV"

    #  Rule 8: NUM + <word> → NOUN (e.g., "five dogs") 
    if prev_tag == "NUM":
        return "NOUN"

    return None


def _looks_adjective(word: str) -> bool:
    """Quick check for adjective-like suffixes."""
    adj_suffixes = (
        "ous", "ive", "able", "ible", "ful", "less",
        "al", "ical", "ish", "ant", "ent", "ary", "ory",
    )
    return any(word.endswith(s) for s in adj_suffixes)
