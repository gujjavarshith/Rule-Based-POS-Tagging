"""
Closed-class word rules.

Determiners, pronouns, conjunctions, auxiliaries, adpositions, particles,
and interjections are (mostly) closed classes that can be enumerated.
"""

from typing import List, Optional


# Closed-class word lists 

DETERMINERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each", "all", "both",
    "few", "many", "much", "several", "enough",
    "another", "other", "such", "what", "which", "whatever",
    "whichever", "either", "neither",
})

PRONOUNS = frozenset({
    # Personal
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # Demonstrative / Relative / Interrogative
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    "whoever", "whomever", "whatever", "whichever",
    # Indefinite
    "someone", "somebody", "something",
    "anyone", "anybody", "anything",
    "everyone", "everybody", "everything",
    "no one", "nobody", "nothing",
    "one", "ones",
    # Reflexive / Other
    "each", "other",
})

COORD_CONJ = frozenset({
    "and", "but", "or", "nor", "for", "yet", "so",
    "both", "either", "neither",
    "&",
})

SUBORD_CONJ = frozenset({
    "if", "because", "since", "while", "although", "though",
    "unless", "until", "before", "after", "when", "whenever",
    "where", "wherever", "whether", "whereas", "as",
    "once", "than", "that", "till", "so",
})

AUXILIARIES = frozenset({
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "'s", "'re", "'ve", "'ll", "'d", "'m",
    "s", "re", "ve", "ll", "d", "m",  # without curly apostrophe
    "'s", "'re", "'ve", "'ll", "'d", "'m",
})

ADPOSITIONS = frozenset({
    "in", "on", "at", "by", "for", "with", "from", "to", "of",
    "about", "above", "across", "after", "against", "along",
    "among", "around", "before", "behind", "below", "beneath",
    "beside", "besides", "between", "beyond", "but", "concerning",
    "despite", "down", "during", "except", "following",
    "inside", "into", "like", "near", "off", "onto",
    "opposite", "out", "outside", "over", "past", "per",
    "plus", "regarding", "round", "since", "through",
    "throughout", "till", "toward", "towards", "under",
    "underneath", "unlike", "until", "up", "upon", "via",
    "within", "without",
})

PARTICLES = frozenset({
    "not", "n't", "n't", "'t",
    "to",  # infinitive marker (handled contextually below)
    "'s",  # possessive
    "up", "out", "off", "on", "in", "down", "away", "back",
    "over", "about", "around",
})

INTERJECTIONS = frozenset({
    "oh", "ah", "wow", "hey", "hi", "hello", "bye", "goodbye",
    "yes", "no", "yeah", "yep", "nope", "please", "thanks",
    "sorry", "oops", "ouch", "hmm", "huh", "uh", "um", "well",
    "ok", "okay", "bravo", "cheers", "alas", "hooray",
})


# Priority-ordered checks 

def _is_negation(word: str) -> bool:
    """Check if word is a negation particle."""
    return word.lower() in {"not", "n't", "n't", "'t"}


def apply(word: str, context: Optional[List[str]] = None) -> Optional[str]:
    """
    Return a closed-class POS tag if the word belongs to a known set.

    Priority: INTJ → PART(negation) → DET → PRON → AUX → CCONJ → SCONJ → ADP

    Some words appear in multiple lists (e.g., "that" can be DET, PRON, SCONJ).
    The priority order above resolves most common cases; the lexicon or context
    rules handle the rest.

    Parameters
    ----------
    word : str
        Surface form of the token.
    context : list of str, optional
        Surrounding words (unused here; context rules refine later).

    Returns
    -------
    str or None
    """
    w = word.lower()

    # Negation particles are unambiguously PART
    if _is_negation(w):
        return "PART"

    # Interjections (rare but unambiguous)
    if w in INTERJECTIONS:
        return "INTJ"

    # Auxiliaries — check before ADP because "to" can be both
    if w in AUXILIARIES and w not in {"to"}:
        return "AUX"

    # Determiners (checked before PRON because overlap on this/that/these/those)
    if w in DETERMINERS and w not in PRONOUNS:
        return "DET"

    # Coordinating conjunctions
    if w in COORD_CONJ and w not in {"for", "so", "both", "either", "neither"}:
        return "CCONJ"

    return None
