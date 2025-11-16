"""
Analysis Step - Cluster Fragmentation Analysis

Encapsulates the cluster fragmentation analysis functionality.
This step can run independently (no dependencies on other steps).
"""

from typing import Dict, Any, Optional
from .base import PipelineStep
from xctopus.nodes.bayesian.utils.cluster_analyzer import (
    ClusterAnalyzer,
    ClusterAnalysisConfig
)


class AnalysisStep(PipelineStep):
    """
    Step 0: Cluster fragmentation analysis.
    
    Encapsulates: scripts/deprecated/00_cluster_analyze_fragmentation.py (legacy)
    
    This step analyzes cluster distribution and fragmentation without requiring
    any previous steps to be executed. It can be used for initial dataset analysis.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        results = pipeline.run(step='analysis')
        print(f"Clusters detected: {results['statistics']['num_clusters']}")
    """
    
    def get_required_steps(self):
        """
        Analysis step doesn't require any previous steps.
        
        Returns:
            list: Empty list (no dependencies)
        """
        return []
    
    def validate_inputs(self, pipeline, **kwargs):
        """
        Validate that required inputs are available.
        
        Args:
            pipeline: XctopusPipeline instance
            **kwargs: Step-specific parameters
        
        Raises:
            ValueError: If dataset_path is missing or invalid
        """
        dataset_path = kwargs.get('dataset_path') or pipeline.dataset_path
        
        if not dataset_path:
            raise ValueError(
                "dataset_path is required for analysis. "
                "Provide it as argument or set pipeline.dataset_path"
            )
        
        import os
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset file not found: {dataset_path}")
    
    def execute(self, dataset_path, pipeline, **kwargs):
        """
        Execute cluster fragmentation analysis.
        
        Args:
            dataset_path: Path to dataset CSV file
            pipeline: XctopusPipeline instance (for accessing preprocessor if available)
            **kwargs: Additional options:
                - save_plots: Whether to save visualization plots (default: True)
                - compute_advanced_metrics: Whether to compute advanced metrics (default: True)
                - enable_adaptive_merge: Whether to enable adaptive cluster merging (default: False)
                - export_format: Format for export ('json', 'csv', None) (default: None)
        
        Returns:
            dict: Analysis results with keys:
                - 'statistics': Cluster statistics
                - 'orphans': Orphan clusters identified
                - 'problems': Detected problems
                - 'recommendations': Generated recommendations
                - 'advanced_metrics': Advanced clustering metrics (if computed)
        """
        self.validate_inputs(pipeline, dataset_path=dataset_path)
        
        # Get or create TextPreprocessor from pipeline
        text_preprocessor = None
        if hasattr(pipeline, '_get_preprocessor'):
            try:
                text_preprocessor = pipeline._get_preprocessor()
            except Exception:
                # If preprocessor can't be created, ClusterAnalyzer will create its own
                pass
        
        # Configure analysis
        analysis_config = ClusterAnalysisConfig(
            save_plots=kwargs.get('save_plots', True),
            compute_advanced_metrics=kwargs.get('compute_advanced_metrics', True),
            enable_adaptive_merge=kwargs.get('enable_adaptive_merge', False),
            text_columns=pipeline.text_columns if hasattr(pipeline, 'text_columns') else None,
            join_with=pipeline.join_with if hasattr(pipeline, 'join_with') else '\n'
        )
        
        # Create analyzer
        analyzer = ClusterAnalyzer(
            config=analysis_config,
            text_preprocessor=text_preprocessor
        )
        
        # Run full analysis
        export_format = kwargs.get('export_format', None)
        summary = analyzer.run_full_analysis(
            dataset_path=dataset_path,
            text_preprocessor=text_preprocessor,
            export_format=export_format
        )
        
        # Store results in pipeline
        pipeline.results['analysis'] = summary
        
        return summary

