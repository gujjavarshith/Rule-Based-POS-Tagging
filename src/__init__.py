"""
Rule-Based POS Tagger for UD English Web Treebank.
"""

from .parser import parse_conllu, Token
from .lexicon import Lexicon
from .tagger import RuleBasedTagger
from .evaluate import Evaluator
