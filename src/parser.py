"""
CoNLL-U format parser for Universal Dependencies data.

Reads .conllu files and produces structured sentence/token objects.
"""

from collections import namedtuple
from typing import List


# Each token in a CoNLL-U sentence
Token = namedtuple("Token", [
    "id",       # int   – word index (1-based)
    "form",     # str   – surface form
    "lemma",    # str   – lemma
    "upos",     # str   – Universal POS tag
    "xpos",     # str   – language-specific tag
    "feats",    # str   – morphological features
    "head",     # str   – head index
    "deprel",   # str   – dependency relation
    "deps",     # str   – enhanced deps
    "misc",     # str   – miscellaneous
])


def parse_conllu(filepath: str) -> List[List[Token]]:
    """
    Parse a CoNLL-U file and return a list of sentences.

    Each sentence is a list of Token namedtuples.
    Skips comment lines (starting with #) and multiword tokens (IDs like 1-2).

    Parameters
    ----------
    filepath : str
        Path to the .conllu file.

    Returns
    -------
    list of list of Token
        Parsed sentences.
    """
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Blank line → end of sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Skip comment lines
            if line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                continue

            # Skip multiword tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue

            token = Token(
                id=int(token_id),
                form=fields[1],
                lemma=fields[2],
                upos=fields[3],
                xpos=fields[4],
                feats=fields[5],
                head=fields[6],
                deprel=fields[7],
                deps=fields[8],
                misc=fields[9],
            )
            current_sentence.append(token)

    # Handle file that doesn't end with a blank line
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def get_forms_and_tags(sentences: List[List[Token]]):
    """
    Extract parallel lists of (forms, gold_tags) per sentence.

    Returns
    -------
    list of (list of str, list of str)
        Each element is (word_forms, gold_upos_tags) for one sentence.
    """
    result = []
    for sent in sentences:
        forms = [tok.form for tok in sent]
        tags = [tok.upos for tok in sent]
        result.append((forms, tags))
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <conllu_file>")
        sys.exit(1)

    sents = parse_conllu(sys.argv[1])
    print(f"Parsed {len(sents)} sentences")
    print(f"Total tokens: {sum(len(s) for s in sents)}")
    # Show first sentence
    if sents:
        print("\nFirst sentence:")
        for tok in sents[0]:
            print(f"  {tok.id:>3}  {tok.form:<20} {tok.upos}")
