"""
Lexicon module — builds a word-to-tag frequency dictionary from training data.

The lexicon assigns each word its most-frequently-observed UPOS tag. It also
supports case-insensitive fallback and pickle-based serialization.
"""

import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .parser import Token


class Lexicon:
    """Word-level POS lexicon built from annotated training data."""

    def __init__(self):
        # {lowercase_word: {tag: count}}
        self.word_tag_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # {lowercase_word: most_frequent_tag}
        self.word_tag: Dict[str, str] = {}
        # Overall tag frequencies (used as prior)
        self.tag_counts: Dict[str, int] = defaultdict(int)
        self.total_tokens: int = 0

    # Building
    def build(self, sentences: List[List[Token]]) -> "Lexicon":
        """
        Populate the lexicon from parsed training sentences.

        Parameters
        ----------
        sentences : list of list of Token
            Output of ``parse_conllu``.

        Returns
        -------
        Lexicon
            ``self``, for chaining.
        """
        for sent in sentences:
            for tok in sent:
                word = tok.form.lower()
                tag = tok.upos
                self.word_tag_counts[word][tag] += 1
                self.tag_counts[tag] += 1
                self.total_tokens += 1

        # Resolve most-frequent tag per word
        for word, tag_counts in self.word_tag_counts.items():
            self.word_tag[word] = max(tag_counts, key=tag_counts.get)

        return self

    # Lookup
    def lookup(self, word: str) -> Optional[str]:
        """
        Look up the most-frequent tag for *word*.

        Tries exact lowercase match first. Returns ``None`` if unknown.
        """
        return self.word_tag.get(word.lower())

    def get_tag_distribution(self, word: str) -> Optional[Dict[str, int]]:
        """Return the full tag-count distribution for *word*, or None."""
        key = word.lower()
        if key in self.word_tag_counts:
            return dict(self.word_tag_counts[key])
        return None

    def is_ambiguous(self, word: str, threshold: float = 0.85) -> bool:
        """
        Return True if *word* has no single dominant tag.

        A word is considered ambiguous when its most-frequent tag accounts for
        less than *threshold* of all observations.
        """
        dist = self.get_tag_distribution(word)
        if dist is None:
            return True  # unknown words are maximally ambiguous
        total = sum(dist.values())
        dominant = max(dist.values())
        return (dominant / total) < threshold

    @property
    def vocabulary_size(self) -> int:
        return len(self.word_tag)

    # Persistence
    def save(self, filepath: str) -> None:
        """Serialize the lexicon to a pickle file."""
        data = {
            "word_tag_counts": dict(self.word_tag_counts),
            "word_tag": self.word_tag,
            "tag_counts": dict(self.tag_counts),
            "total_tokens": self.total_tokens,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "Lexicon":
        """Load a previously-saved lexicon."""
        lex = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        lex.word_tag_counts = defaultdict(
            lambda: defaultdict(int), data["word_tag_counts"]
        )
        lex.word_tag = data["word_tag"]
        lex.tag_counts = defaultdict(int, data["tag_counts"])
        lex.total_tokens = data["total_tokens"]
        return lex

    # Utilities
    def coverage(self, sentences: List[List[Token]]) -> Tuple[int, int, float]:
        """
        Compute lexicon coverage on a set of sentences.

        Returns (known_count, total_count, coverage_pct).
        """
        known = 0
        total = 0
        for sent in sentences:
            for tok in sent:
                total += 1
                if tok.form.lower() in self.word_tag:
                    known += 1
        pct = (known / total * 100) if total else 0.0
        return known, total, pct

    def __repr__(self):
        return (
            f"Lexicon(vocab={self.vocabulary_size:,}, "
            f"tokens={self.total_tokens:,}, "
            f"tags={len(self.tag_counts)})"
        )
