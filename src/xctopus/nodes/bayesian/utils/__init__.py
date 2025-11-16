"""
Utilities for Bayesian nodes including cluster analysis.

Note: TextPreprocessor has been moved to bayesian.core.text_preprocessor
as it is a core component. It is still available here for backward compatibility.
"""

# Import cluster analyzer
from .cluster_analyzer import ClusterAnalyzer, ClusterAnalysisConfig

# Import clustering configuration
from .clustering_config import DynamicClusteringConfig

# Import cluster utilities
from .cluster_utils import (
    identify_orphan_clusters,
    evaluate_cluster_quality,
    adaptive_merge_clusters,
    calculate_cluster_centroids,
    renumber_clusters,
    calculate_cluster_coherence,
    assign_outliers_to_clusters,
    merge_clusters_by_ratio,
    calculate_dynamic_threshold,
    calculate_dynamic_node_threshold,
    detect_domain_from_path,
    load_domain_mapping,
    calculate_domain_purity,
    determine_node_configuration,
    get_current_layers,
    get_current_lora_rank,
    update_node_configuration,
    batch_update_node_configurations,
    analyze_cluster_distribution,
    find_similar_cluster_pairs,
    calculate_cluster_internal_similarity,
    analyze_cluster_quality_per_cluster,
    extract_embeddings_from_nodes,
    identify_large_clusters,
    fine_tune_cluster_with_lora
)

# Import TextPreprocessor from core (core component, not a utility)
from ..core import TextPreprocessor

# Import LearningAuditor
from .learning_auditor import LearningAuditor

# Import LoRAAuditor
from .lora_auditor import LoRAAuditor

# Import LearningEvaluator
from .learning_evaluator import LearningEvaluator

__all__ = [
    'ClusterAnalyzer', 
    'ClusterAnalysisConfig', 
    'TextPreprocessor', 
    'DynamicClusteringConfig',
    'identify_orphan_clusters',
    'evaluate_cluster_quality',
    'adaptive_merge_clusters',
    'calculate_cluster_centroids',
    'renumber_clusters',
    'calculate_cluster_coherence',
    'assign_outliers_to_clusters',
    'merge_clusters_by_ratio',
    'calculate_dynamic_threshold',
    'calculate_dynamic_node_threshold',
    'detect_domain_from_path',
    'load_domain_mapping',
    'calculate_domain_purity',
    'determine_node_configuration',
    'get_current_layers',
    'get_current_lora_rank',
    'update_node_configuration',
    'batch_update_node_configurations',
    'analyze_cluster_distribution',
    'find_similar_cluster_pairs',
    'calculate_cluster_internal_similarity',
    'analyze_cluster_quality_per_cluster',
    'extract_embeddings_from_nodes',
    'identify_large_clusters',
    'fine_tune_cluster_with_lora',
    'LearningAuditor',
    'LoRAAuditor',
    'LearningEvaluator'
]

