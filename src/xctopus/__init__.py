"""
Xctopus - Hierarchical Continuous Learning Framework

This package provides a modular architecture for continuous learning
using Transformers and Bayesian Nodes.

Quick Start:
    from xctopus import XctopusPipeline
    
    pipeline = XctopusPipeline('data.csv')
    pipeline.run()

Or use the CLI:
    xctopus-run data.csv
"""

# Pipeline API (new, recommended)
try:
    from .pipeline import XctopusPipeline, PipelineConfig
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

# Backward compatibility: existing exports
try:
    from .nodes.bayesian.bayesian_node import BayesianNode
    from .nodes.bayesian.bayesian_filter import BayesianFilter
    from .nodes.transformer.transformer import TransformerNode
    _HAS_NODES = True
except ImportError:
    _HAS_NODES = False

# Build __all__ based on what's available
__all__ = []

if _HAS_PIPELINE:
    __all__.extend(['XctopusPipeline', 'PipelineConfig'])

if _HAS_NODES:
    __all__.extend(['BayesianNode', 'BayesianFilter', 'TransformerNode'])