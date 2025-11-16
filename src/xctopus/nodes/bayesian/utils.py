"""
DEPRECATED: This file is kept for backward compatibility only.

TextPreprocessor has been moved to bayesian.core.text_preprocessor
as it is a core component, not a utility.

For new code, import from:
    from xctopus.nodes.bayesian.core import TextPreprocessor

Or use the backward-compatible import:
    from xctopus.nodes.bayesian.utils import TextPreprocessor
"""

import warnings
warnings.warn(
    "Importing TextPreprocessor from bayesian.utils is deprecated. "
    "Use 'from xctopus.nodes.bayesian.core import TextPreprocessor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from core for backward compatibility
from .core import TextPreprocessor

__all__ = ['TextPreprocessor']
