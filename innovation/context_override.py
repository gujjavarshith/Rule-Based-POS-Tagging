"""
Innovation: context-based tag correction (post-processing pass).

Applied AFTER all other rules have assigned a first-pass tag to every token.
Implements Brill-style transformation rules derived from error analysis
in notebooks 03 and 05.

Transformation rules (ordered by estimated impact from error analysis):

1.  "'s" AUX → PART   (83 errors): possessive — prev tag NOUN/PROPN/PRON
2.  "to" PART → ADP  (198 errors): prepositional — next tag DET/NOUN/PROPN/PRON/ADJ/NUM
3.  "have/has/had/do/does/did/are/is/am/were/was/be/been"
     AUX → VERB       (~258 errors): main verb use — next tag DET/NOUN/ADJ/PRON/NUM/INTJ
4.  "that" SCONJ → PRON/DET  (fixes over-generalisation from compound_context
                              or lexicon: if "that" next is NOUN/ADJ → DET;
                              if next is DET/PRON/NUM after NOUN → PRON)
     "that" PRON → SCONJ     (93 errors): complementiser — prev NOUN/VERB/ADV
                              and next is VERB/AUX/PRON/ADV/DET
5.  "no"/"No" INTJ → DET     (39 errors): when next tag is NOUN/ADJ/NUM
6.  PROPN chain               remaining PROPN→NOUN: if tag[i]==PROPN and
                              tag[i±1]==NOUN and word[i±1] is capitalised → PROPN
7.  "'s"/"s" possessive after name — already PART from rule 1
8.  Emails/URLs tagged X when gold is PROPN — web_token already handles this;
    do not apply here.
"""

from typing import List


# ── AUX-or-VERB disambiguation words ─────────────────────────────────────
# These words are in the closed_class AUXILIARIES set so they get tagged AUX
# in Pass 1.  Context can tell us when they are main verbs (VERB).
_AUX_OR_VERB = frozenset({
    "have", "has", "had", "having",
    "do", "does", "did",
    "is", "are", "am", "was", "were", "be", "been", "being",
})

# Tags that follow an AUX-or-VERB word when it's acting as a main VERB
# (not as an auxiliary).  e.g. "I have a book" → have=VERB next=DET
_VERB_NEXT_TAGS = frozenset({
    "DET", "NOUN", "PROPN", "PRON", "ADJ", "NUM", "INTJ",
})

# Tags that follow an AUX-or-VERB word when it's acting as AUX
# (followed by a verbal complement).
_AUX_NEXT_TAGS = frozenset({
    "VERB", "AUX", "ADV", "PART",
})


def apply(words: List[str], tags: List[str]) -> List[str]:
    """
    Apply context-based correction rules to a fully-tagged sentence.

    Parameters
    ----------
    words : list of str
        Surface forms.
    tags : list of str
        Tags produced by all previous rules.

    Returns
    -------
    list of str
        Corrected tag sequence (same length).
    """
    tags = list(tags)   # work on a copy
    n = len(tags)

    for i, (w, t) in enumerate(zip(words, tags)):
        w_lower = w.lower()
        prev_tag = tags[i - 1] if i > 0 else None
        next_tag = tags[i + 1] if i + 1 < n else None
        next2_tag = tags[i + 2] if i + 2 < n else None

        # ── Rule 1: possessive "'s" / "s"  AUX → PART ────────────────────
        # Pattern: NOUN|PROPN + 's → possessive PART
        # "John's book", "company's plan", "Today's news"
        # Do NOT fire after PRON: "it's", "he's", "she's" → 's = AUX
        # Evidence: 83 PART→AUX errors, prev=NOUN/PROPN (not PRON)
        if t == "AUX" and w_lower in ("'s", "\u2019s", "s") and \
                prev_tag in ("NOUN", "PROPN", "ADJ", "X"):
            tags[i] = "PART"
            continue

        # ── Rule 2: "to" PART → ADP ───────────────────────────────────────
        # Pattern: to + DET|NOUN|PROPN|PRON|ADJ|NUM|X  → prepositional ADP
        # Evidence: 198 ADP→PART errors; all next tags = DET/NOUN/PROPN/PRON
        # Keep PART when next = VERB (infinitive: "want to go")
        if t == "PART" and w_lower in ("to", "ta", "TO") and \
                next_tag in ("DET", "NOUN", "PROPN", "PRON", "ADJ", "NUM",
                             "X", "INTJ", "SCONJ"):
            tags[i] = "ADP"
            continue

        # ── Rule 3: have/do  AUX → VERB ──────────────────────────────────
        # "I have a car" (have=VERB), "Do you have money" (have=VERB)
        # Keep AUX for "is/are/am/was/were/be/been" — these are almost always
        # AUX in UD even before DET ("it is the plan" → is=AUX).
        # Evidence: 258 VERB→AUX errors, mostly have/do/did/had
        if t == "AUX" and w_lower in ("have", "has", "had", "having",
                                       "do", "does", "did"):
            if next_tag in ("DET", "NOUN", "PROPN", "ADJ", "NUM", "INTJ") and \
                    next_tag not in _AUX_NEXT_TAGS:
                tags[i] = "VERB"
                continue

        # ── Rule 4a: "that" PRON → SCONJ (complementiser) ────────────────
        # "the evidence that the group ..."  / "I think that he ..."
        # prev = NOUN|VERB|ADV|PUNCT and next = VERB|AUX|PRON|DET|ADV|NOUN
        # Evidence: 93 PRON→SCONJ errors
        if t == "PRON" and w_lower == "that" and \
                prev_tag in ("NOUN", "VERB", "ADV", "PUNCT", "ADP") and \
                next_tag in ("VERB", "AUX", "PRON", "DET", "ADV", "NOUN", "ADJ",
                             "PROPN", "PART"):
            tags[i] = "SCONJ"
            continue

        # ── Rule 4b: "that" SCONJ → DET (demonstrative before noun) ──────
        # "that book", "that idea" — next tag is NOUN/ADJ/PROPN and prev is
        # not NOUN (otherwise it's a relative clause head)
        if t == "SCONJ" and w_lower == "that" and \
                next_tag in ("NOUN", "ADJ", "PROPN", "NUM") and \
                prev_tag not in ("NOUN", "VERB", "PROPN"):
            tags[i] = "DET"
            continue

        # ── Rule 5: "no"/"No" INTJ → DET ─────────────────────────────────
        # "no problem", "no reason", "no other" — next is NOUN/ADJ/DET/NUM
        # Evidence: 39 DET→INTJ errors (closed_class tags "no" as INTJ)
        if t == "INTJ" and w_lower == "no" and \
                next_tag in ("NOUN", "PROPN", "ADJ", "DET", "NUM", "ADV", "PRON"):
            tags[i] = "DET"
            continue

        # ── Rule 6: "thanks"/"Thanks"/"Cheers" INTJ when NOUN ────────────
        # "Thanks for the tip", "Cheers mate" - standalone → INTJ (correct)
        # Only override to NOUN when followed by ADP (thanks for / cheers to)
        if t == "INTJ" and w_lower in ("thanks", "cheers") and \
                next_tag == "ADP":
            tags[i] = "NOUN"
            continue

        # ── Rule 7: PROPN chain ────────────────────────────────────────────
        # If tags[i]==PROPN and adjacent word is capitalised and tagged NOUN
        # → likely part of the same proper noun phrase.
        # Evidence: PROPN→NOUN still the biggest error (324 remaining).
        if t == "PROPN":
            # Forward: next word capitalised and tagged NOUN
            if i + 1 < n and tags[i + 1] == "NOUN" and \
                    words[i + 1][0].isupper() and len(words[i + 1]) > 1:
                tags[i + 1] = "PROPN"
            # Backward: prev word capitalised and tagged NOUN
            if i > 0 and tags[i - 1] == "NOUN" and \
                    words[i - 1][0].isupper() and len(words[i - 1]) > 1:
                tags[i - 1] = "PROPN"

        # ── Rule 8: "well" INTJ when prev=AUX/VERB → ADV ─────────────────
        # "works well", "went well" → ADV not INTJ
        if t == "INTJ" and w_lower == "well" and \
                prev_tag in ("VERB", "AUX", "ADV"):
            tags[i] = "ADV"
            continue

    return tags
