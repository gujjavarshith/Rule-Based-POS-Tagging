"""
Main tagger pipeline — composes lexicon lookup with cascading rules.

Priority order:
  1. Punctuation / Numeral rules
  2. Closed-class word rules
  3. Lexicon lookup (most-frequent tag)
  4. Capitalization rules
  5. Morphology (suffix) rules
  6. Context rules
  7. Default: NOUN
"""

from typing import Dict, List, Optional, Tuple

from .lexicon import Lexicon
from .rules import (
    capitalization_rule,
    closed_class_rule,
    context_rule,
    morphology_rule,
    punct_num_rule,
)


# Default tag when nothing matches
_DEFAULT_TAG = "NOUN"


class RuleBasedTagger:
    """
    A cascading rule-based POS tagger.

    Applies rules in priority order; the first rule that returns a non-None
    tag wins.
    """

    def __init__(self, lexicon: Lexicon, use_innovation: bool = False):
        self.lexicon = lexicon
        self.use_innovation = use_innovation

        # Innovation modules (loaded lazily)
        self._innovation_rules = []
        if use_innovation:
            self._load_innovation_rules()

    # Innovation extension
    def _load_innovation_rules(self):
        """Import innovation rule modules if available."""
        try:
            from innovation.prefix_rules import apply as prefix_apply
            self._innovation_rules.append(("prefix", prefix_apply))
        except ImportError:
            pass
        try:
            from innovation.web_token_rules import apply as web_apply
            self._innovation_rules.append(("web_token", web_apply))
        except ImportError:
            pass
        try:
            from innovation.compound_context import apply as compound_apply
            self._innovation_rules.append(("compound_ctx", compound_apply))
        except ImportError:
            pass

    # Tagging
    def tag_sentence(self, words: List[str]) -> List[str]:
        """
        Assign UPOS tags to a list of surface forms.

        Parameters
        ----------
        words : list of str
            Token surface forms for one sentence.

        Returns
        -------
        list of str
            Predicted UPOS tags (same length as *words*).
        """
        n = len(words)
        tags: List[Optional[str]] = [None] * n

        # Pass 1: Deterministic rules (punct, closed-class) 
        for i, w in enumerate(words):
            # 1. Punctuation / Numeral
            tag = punct_num_rule(w)
            if tag:
                tags[i] = tag
                continue

            # 2. Closed-class
            tag = closed_class_rule(w)
            if tag:
                tags[i] = tag
                continue

        # Pass 2: Lexicon + form-based rules 
        for i, w in enumerate(words):
            if tags[i] is not None:
                continue

            # 3. Lexicon lookup
            tag = self.lexicon.lookup(w)
            if tag:
                tags[i] = tag
                continue

            # 4. Innovation rules (if enabled, before capitalization)
            if self.use_innovation:
                for _name, rule_fn in self._innovation_rules:
                    tag = rule_fn(w)
                    if tag:
                        tags[i] = tag
                        break
                if tags[i] is not None:
                    continue

            # 5. Capitalization
            tag = capitalization_rule(w, position=i)
            if tag:
                tags[i] = tag
                continue

            # 6. Morphology (suffix)
            tag = morphology_rule(w)
            if tag:
                tags[i] = tag
                continue

        #  Pass 3: Context rules for remaining unknowns 
        for i, w in enumerate(words):
            if tags[i] is not None:
                continue

            # Build context window (tags assigned so far)
            ctx = list(tags)
            ctx[i] = None  # mark current position
            tag = context_rule(w, ctx)
            if tag:
                tags[i] = tag
                continue

            # 7. Default
            tags[i] = _DEFAULT_TAG

        return tags

    def tag_corpus(
        self, sentences: List[Tuple[List[str], List[str]]]
    ) -> List[Tuple[List[str], List[str], List[str]]]:
        """
        Tag an entire corpus.

        Parameters
        ----------
        sentences : list of (words, gold_tags)

        Returns
        -------
        list of (words, gold_tags, pred_tags)
        """
        results = []
        for words, gold in sentences:
            pred = self.tag_sentence(words)
            results.append((words, gold, pred))
        return results

    # Ablation support
    def tag_sentence_ablation(
        self, words: List[str], disable: Optional[str] = None
    ) -> List[str]:
        """
        Tag with one rule module disabled (for ablation study).

        Parameters
        ----------
        disable : str or None
            One of: 'punct_num', 'closed_class', 'lexicon',
            'capitalization', 'morphology', 'context', 'innovation'.
        """
        n = len(words)
        tags: List[Optional[str]] = [None] * n

        for i, w in enumerate(words):
            if disable != "punct_num":
                tag = punct_num_rule(w)
                if tag:
                    tags[i] = tag
                    continue

            if disable != "closed_class":
                tag = closed_class_rule(w)
                if tag:
                    tags[i] = tag
                    continue

        for i, w in enumerate(words):
            if tags[i] is not None:
                continue

            if disable != "lexicon":
                tag = self.lexicon.lookup(w)
                if tag:
                    tags[i] = tag
                    continue

            if disable != "capitalization":
                tag = capitalization_rule(w, position=i)
                if tag:
                    tags[i] = tag
                    continue

            if disable != "morphology":
                tag = morphology_rule(w)
                if tag:
                    tags[i] = tag
                    continue

        for i, w in enumerate(words):
            if tags[i] is not None:
                continue

            if disable != "context":
                ctx = list(tags)
                ctx[i] = None
                tag = context_rule(w, ctx)
                if tag:
                    tags[i] = tag
                    continue

            tags[i] = _DEFAULT_TAG

        return tags
