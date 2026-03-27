"""Rule modules for the rule-based POS tagger."""

from .punct_num import apply as punct_num_rule
from .closed_class import apply as closed_class_rule
from .morphology import apply as morphology_rule
from .context import apply as context_rule
from .capitalization import apply as capitalization_rule

__all__ = [
    "punct_num_rule",
    "closed_class_rule",
    "morphology_rule",
    "context_rule",
    "capitalization_rule",
]
