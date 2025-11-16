"""
Xctopus Pipeline Module

This module provides the XctopusPipeline manager and related components
for orchestrating the complete Xctopus workflow.
"""

from .pipeline import XctopusPipeline
from .config import PipelineConfig

__all__ = [
    'XctopusPipeline',
    'PipelineConfig',
]

