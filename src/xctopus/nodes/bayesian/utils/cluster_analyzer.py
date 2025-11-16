"""
Cluster Analysis Module

Provides ClusterAnalyzer class for analyzing cluster distribution and fragmentation
in Xctopus knowledge nodes.
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import TextPreprocessor from core (core component, not a utility)
from ..core import TextPreprocessor
from ..bayesian_filter import FilterBayesianNode

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ClusterAnalysisConfig:
    """
    Configuration for cluster analysis.
    
    Attributes:
        small_cluster_threshold: Minimum size for small clusters (default: 5)
        medium_cluster_threshold: Minimum size for medium clusters (default: 15)
        orphan_min_size: Minimum size to not be considered orphan (default: 3)
        over_fragmentation_ratio: Ratio threshold for over-fragmentation warning (default: 0.3)
        excess_clusters_ratio: Ratio threshold for excess clusters warning (default: 0.3)
        low_avg_size: Average size threshold for low average warning (default: 5.0)
        save_plots: Whether to save visualization plots (default: True)
        plot_format: Format for saved plots: 'png', 'pdf', 'svg' (default: 'png')
        show_plots: Whether to display plots interactively (default: False)
        text_columns: Column names to use for text extraction (None = auto-detect 'text')
        join_with: Separator for joining multiple text columns (default: '\n')
        compute_advanced_metrics: Whether to compute advanced clustering metrics (default: False)
        # Adaptive merge configuration
        enable_adaptive_merge: bool = False
        max_merge_iterations: int = 10
        fusion_percentile: float = 25
        semantic_fusion_threshold: float = 0.85
        min_clusters_target: int = 5
        min_texts_per_cluster: int = 5
    """
    small_cluster_threshold: int = 5
    medium_cluster_threshold: int = 15
    orphan_min_size: int = 3
    over_fragmentation_ratio: float = 0.3
    excess_clusters_ratio: float = 0.3
    low_avg_size: float = 5.0
    save_plots: bool = True
    plot_format: str = 'png'
    show_plots: bool = False
    text_columns: Optional[List[str]] = None
    join_with: str = '\n'
    compute_advanced_metrics: bool = False
    # Adaptive merge configuration
    enable_adaptive_merge: bool = False
    max_merge_iterations: int = 10
    fusion_percentile: float = 25
    semantic_fusion_threshold: float = 0.85
    min_clusters_target: int = 5
    min_texts_per_cluster: int = 5


class ClusterAnalyzer:
    """
    Class for analyzing cluster distribution and fragmentation.
    
    This class provides a unified interface for:
    - Loading datasets (CSV files or DataFrames)
    - Encoding texts using TextPreprocessor
    - Assigning clusters using FilterBayesianNode
    - Computing statistics and metrics
    - Identifying orphan clusters
    - Detecting fragmentation problems
    - Generating recommendations
    - Visualizing distributions
    - Exporting results
    
    Example:
        >>> from xctopus.nodes.bayesian.utils.cluster_analyzer import ClusterAnalyzer
        >>> analyzer = ClusterAnalyzer()
        >>> results = analyzer.run_full_analysis(dataset_path="data/dataset.csv")
        >>> print(results['statistics'])
    """
    
    def __init__(self, config: Optional[ClusterAnalysisConfig] = None,
                 filter_node: Optional[FilterBayesianNode] = None,
                 text_preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize ClusterAnalyzer.
        
        Args:
            config: Configuration object (default: ClusterAnalysisConfig())
            filter_node: FilterBayesianNode for cluster assignment (optional)
            text_preprocessor: TextPreprocessor for encoding texts (optional)
        """
        self.config = config or ClusterAnalysisConfig()
        self.filter_node = filter_node
        self.text_preprocessor = text_preprocessor
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.texts: Optional[List[str]] = None
        self.embeddings: Optional[Any] = None
        self.cluster_assignments: Optional[List[int]] = None
        self.cluster_counts: Optional[Counter] = None
        self.confidences: Optional[List[float]] = None
        
        # Analysis results
        self.statistics: Optional[Dict] = None
        self.orphans: Optional[List[Tuple[int, int]]] = None
        self.problems: Optional[Dict] = None
        self.recommendations: Optional[List[str]] = None
        self.advanced_metrics: Optional[Dict] = None
    
    def load_dataset(self, dataset_path: Optional[str] = None,
                    dataframe: Optional[pd.DataFrame] = None,
                    text_preprocessor: Optional[TextPreprocessor] = None):
        """
        Load dataset from file or DataFrame.
        
        Args:
            dataset_path: Path to CSV file (optional if dataframe provided)
            dataframe: DataFrame already loaded (optional)
            text_preprocessor: TextPreprocessor configured with dataset info (optional)
        
        Raises:
            FileNotFoundError: If dataset_path doesn't exist
            ValueError: If neither dataset_path nor dataframe provided
        """
        if dataframe is not None:
            self.df = dataframe.copy()
        elif dataset_path:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            self.df = pd.read_csv(dataset_path, sep=",", quotechar='"', dtype=str)
        else:
            raise ValueError("Either dataset_path or dataframe must be provided")
        
        # Use TextPreprocessor if provided, otherwise extract texts manually
        if text_preprocessor:
            self.text_preprocessor = text_preprocessor
            # If TextPreprocessor has dataset path, use its method
            if text_preprocessor.path_dataset:
                result = text_preprocessor.analyze_and_prepare_dataset(validate=False)
                self.texts = result['texts']
            else:
                # Extract texts using text_columns from preprocessor
                self._extract_texts_from_dataframe(text_preprocessor.text_columns,
                                                  text_preprocessor.join_with)
        else:
            # Fallback: extract texts using config or default 'text' column
            text_cols = self.config.text_columns or ['text']
            self._extract_texts_from_dataframe(text_cols, self.config.join_with)
    
    def _extract_texts_from_dataframe(self, text_columns: List[str], join_with: str):
        """Extract and combine texts from DataFrame columns."""
        if not text_columns:
            text_columns = ['text']
        
        # Find available columns
        available_cols = [col for col in text_columns if col in self.df.columns]
        if not available_cols:
            # Try to find default 'text' column
            if 'text' in self.df.columns:
                available_cols = ['text']
            else:
                raise ValueError(
                    f"None of the specified text columns found: {text_columns}. "
                    f"Available columns: {list(self.df.columns)}"
                )
        
        # Combine texts from multiple columns
        texts = []
        for idx, row in self.df.iterrows():
            text_parts = []
            for col in available_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    text_parts.append(str(value).strip())
            
            combined_text = join_with.join(text_parts)
            if combined_text.strip():
                texts.append(combined_text)
        
        self.texts = texts
    
    def encode_texts(self, text_preprocessor: Optional[TextPreprocessor] = None):
        """
        Encode texts into embeddings.
        
        Args:
            text_preprocessor: TextPreprocessor to use (uses instance one if None)
        
        Raises:
            ValueError: If texts not loaded
        """
        if self.texts is None or len(self.texts) == 0:
            raise ValueError("Texts must be loaded before encoding. Call load_dataset() first.")
        
        preprocessor = text_preprocessor or self.text_preprocessor
        if preprocessor is None:
            preprocessor = TextPreprocessor(path_dataset=None)
            self.text_preprocessor = preprocessor
        
        self.embeddings = preprocessor.encode_texts(self.texts)
    
    def assign_clusters(self, filter_node: Optional[FilterBayesianNode] = None):
        """
        Assign clusters to embeddings using FilterBayesianNode.
        
        Args:
            filter_node: FilterBayesianNode to use (uses instance one if None)
        
        Raises:
            ValueError: If embeddings not computed
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be computed before assignment. Call encode_texts() first.")
        
        node = filter_node or self.filter_node
        if node is None:
            node = FilterBayesianNode(
                mode="train",
                initial_threshold=0.4,
                min_threshold=0.1,
                adaptive_threshold=True
            )
            self.filter_node = node
        
        self.cluster_assignments = []
        self.confidences = []
        
        for emb in self.embeddings:
            cluster_id, confidence = node.evaluate(emb.unsqueeze(0))
            self.cluster_assignments.append(cluster_id)
            self.confidences.append(confidence)
        
        self.cluster_counts = Counter(self.cluster_assignments)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute basic statistics about cluster distribution.
        
        Returns:
            Dictionary with statistics:
            - num_clusters: Total number of clusters
            - total_texts: Total number of texts
            - ratio_texts_per_cluster: Average texts per cluster
            - cluster_sizes: List of cluster sizes
            - small_clusters: Count of small clusters
            - medium_clusters: Count of medium clusters
            - large_clusters: Count of large clusters
            - avg_size: Average cluster size
            - median_size: Median cluster size
            - std_size: Standard deviation of cluster sizes
        """
        if self.cluster_counts is None:
            raise ValueError("Clusters must be assigned first. Call assign_clusters() first.")
        
        cluster_sizes = list(self.cluster_counts.values())
        num_clusters = len(self.cluster_counts)
        total_texts = len(self.texts) if self.texts else sum(cluster_sizes)
        
        small_clusters = sum(1 for s in cluster_sizes if s < self.config.small_cluster_threshold)
        medium_clusters = sum(1 for s in cluster_sizes 
                             if self.config.small_cluster_threshold <= s < self.config.medium_cluster_threshold)
        large_clusters = sum(1 for s in cluster_sizes if s >= self.config.medium_cluster_threshold)
        
        self.statistics = {
            'num_clusters': num_clusters,
            'total_texts': total_texts,
            'ratio_texts_per_cluster': total_texts / num_clusters if num_clusters > 0 else 0,
            'cluster_sizes': cluster_sizes,
            'small_clusters': small_clusters,
            'medium_clusters': medium_clusters,
            'large_clusters': large_clusters,
            'avg_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_size': np.median(cluster_sizes) if cluster_sizes else 0,
            'std_size': np.std(cluster_sizes) if cluster_sizes else 0,
            'min_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_size': max(cluster_sizes) if cluster_sizes else 0,
        }
        
        return self.statistics
    
    def identify_orphan_clusters(self, min_size: Optional[int] = None, 
                                include_outliers: bool = True) -> Dict[str, Any]:
        """
        Identify orphan clusters (too small to be useful) and outliers.
        
        This method now uses the unified identify_orphan_clusters() function
        from cluster_utils to avoid code duplication.
        
        Args:
            min_size: Minimum size threshold (uses config.orphan_min_size if None)
            include_outliers: Whether to detect outliers (label == -1)
        
        Returns:
            Dictionary with:
            - orphans: List of tuples (cluster_id, size) sorted by size
            - outliers: List of indices with outlier labels (-1)
            - n_orphans: Number of orphan clusters
            - n_outliers: Number of outliers
        """
        if self.cluster_assignments is None:
            raise ValueError("Clusters must be assigned first. Call assign_clusters() first.")
        
        # Import unified function
        from .cluster_utils import identify_orphan_clusters
        
        # Determine threshold
        threshold = min_size if min_size is not None else self.config.orphan_min_size
        
        # Use unified function (returns structured dict)
        result = identify_orphan_clusters(
            self.cluster_assignments,
            min_size=threshold,
            include_outliers=include_outliers,
            return_dict=True
        )
        
        # Store for backward compatibility
        self.orphans = result['orphans']
        
        return result
    
    def detect_problems(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect fragmentation and clustering problems.
        
        Returns:
            Dictionary with detected problems:
            - over_fragmentation: Info about over-fragmentation
            - excess_clusters: Info about excess clusters
            - low_avg_size: Info about low average size
        """
        if self.statistics is None:
            self.compute_statistics()
        
        stats = self.statistics
        problems = {}
        
        # Over-fragmentation: too many small clusters
        small_ratio = stats['small_clusters'] / stats['num_clusters'] if stats['num_clusters'] > 0 else 0
        if small_ratio > self.config.over_fragmentation_ratio:
            problems['over_fragmentation'] = {
                'detected': True,
                'small_clusters': stats['small_clusters'],
                'ratio': small_ratio,
                'message': f"{stats['small_clusters']} clusters have <{self.config.small_cluster_threshold} texts "
                          f"({100*small_ratio:.1f}%)"
            }
        else:
            problems['over_fragmentation'] = {'detected': False}
        
        # Excess clusters: too many clusters relative to texts
        cluster_ratio = stats['num_clusters'] / stats['total_texts'] if stats['total_texts'] > 0 else 0
        if cluster_ratio > self.config.excess_clusters_ratio:
            problems['excess_clusters'] = {
                'detected': True,
                'num_clusters': stats['num_clusters'],
                'total_texts': stats['total_texts'],
                'ratio': cluster_ratio,
                'message': f"{stats['num_clusters']} clusters for {stats['total_texts']} texts"
            }
        else:
            problems['excess_clusters'] = {'detected': False}
        
        # Low average size
        if stats['avg_size'] < self.config.low_avg_size:
            problems['low_avg_size'] = {
                'detected': True,
                'avg_size': stats['avg_size'],
                'threshold': self.config.low_avg_size,
                'message': f"Average of {stats['avg_size']:.1f} texts per cluster"
            }
        else:
            problems['low_avg_size'] = {'detected': False}
        
        self.problems = problems
        return problems
    
    def merge_clusters_adaptively(self, 
                                 max_iterations: Optional[int] = None,
                                 fusion_percentile: Optional[float] = None,
                                 semantic_threshold: Optional[float] = None,
                                 min_clusters_target: Optional[int] = None,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Merge clusters adaptively based on distance/similarity.
        
        This method now uses the unified adaptive_merge_clusters() function
        from cluster_utils to avoid code duplication.
        
        Args:
            max_iterations: Maximum merge iterations (uses config if None)
            fusion_percentile: Percentile for distance threshold (uses config if None)
            semantic_threshold: Minimum similarity for semantic fusion (uses config if None)
            min_clusters_target: Minimum target clusters (uses config if None)
            verbose: Whether to print merge progress
        
        Returns:
            Dictionary with merge statistics:
            - initial_clusters: Number of clusters before merging
            - final_clusters: Number of clusters after merging
            - merges_performed: Number of merge operations
            - iterations: Number of iterations executed
        """
        if self.embeddings is None or self.cluster_assignments is None:
            raise ValueError("Embeddings and cluster assignments required for merging.")
        
        # Import unified function
        from .cluster_utils import adaptive_merge_clusters
        
        # Use unified function with return_stats=True to get statistics
        result = adaptive_merge_clusters(
            self.embeddings,
            self.cluster_assignments,
            max_iterations=max_iterations,
            semantic_threshold=semantic_threshold,
            fusion_percentile=fusion_percentile,
            min_clusters_target=min_clusters_target,
            min_texts_per_cluster=self.config.min_texts_per_cluster,
            return_stats=True,  # Get statistics
            verbose=verbose,
            config=self.config  # Pass config for defaults
        )
        
        # Update internal state with merged labels
        self.cluster_assignments = result['labels'].tolist()
        self.cluster_counts = Counter(self.cluster_assignments)
        
        # Invalidate cached statistics
        self.statistics = None
        self.orphans = None
        self.problems = None
        
        # Return statistics (without labels to match original API)
        return {
            'initial_clusters': result['initial_clusters'],
            'final_clusters': result['final_clusters'],
            'merges_performed': result['merges_performed'],
            'iterations': result['iterations'],
            'clusters_reduced': result['clusters_reduced']
        }
    
    def compute_advanced_metrics(self) -> Dict[str, Any]:
        """
        Compute advanced clustering quality metrics.
        
        This method now uses the unified evaluate_cluster_quality() function
        from cluster_utils to avoid code duplication.
        
        Returns:
            Dictionary with advanced metrics:
            - silhouette_score: Silhouette coefficient (if embeddings available)
            - davies_bouldin_score: Davies-Bouldin index (lower is better)
            - cohesion: Average intra-cluster distance
            - separation: Average inter-cluster distance
            - cohesion_std: Standard deviation of cohesion
            - separation_std: Standard deviation of separation
            - n_clusters: Number of clusters (excluding outliers)
            - n_outliers: Number of outliers (label == -1)
        """
        if self.embeddings is None or self.cluster_assignments is None:
            raise ValueError("Embeddings and cluster assignments required for advanced metrics.")
        
        # Import unified function
        from .cluster_utils import evaluate_cluster_quality
        
        # Use unified function with std included (ClusterAnalyzer format)
        self.advanced_metrics = evaluate_cluster_quality(
            self.embeddings,
            self.cluster_assignments,
            include_std=True,  # ClusterAnalyzer includes std
            metric='cosine',
            return_legacy_format=False  # Use unified format
        )
        
        return self.advanced_metrics
    
    def get_recommended_thresholds(self) -> Dict[str, Any]:
        """
        Generate recommended filter thresholds based on analysis.
        
        Returns:
            Dictionary with recommended thresholds and rationale
        """
        if self.statistics is None:
            self.compute_statistics()
        
        stats = self.statistics
        recommendations = {}
        
        if stats['num_clusters'] > 20:
            recommendations = {
                'initial_threshold': 0.5,
                'min_threshold': 0.15,
                'rationale': f"Too many clusters ({stats['num_clusters']}) for {stats['total_texts']} texts. "
                           f"Target: ~{stats['total_texts']//10}-{stats['total_texts']//15} clusters"
            }
        elif stats['num_clusters'] < 5:
            recommendations = {
                'initial_threshold': 0.3,
                'min_threshold': 0.05,
                'rationale': f"Too few clusters ({stats['num_clusters']}). Lower thresholds to create more clusters"
            }
        else:
            recommendations = {
                'initial_threshold': 0.4,
                'min_threshold': 0.1,
                'rationale': "Current configuration seems reasonable. Adjust based on specific domain needs"
            }
        
        return recommendations
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate human-readable recommendations.
        
        Returns:
            List of recommendation strings
        """
        if self.problems is None:
            self.detect_problems()
        
        recommendations = []
        stats = self.statistics
        problems = self.problems
        
        # Recommendations based on problems
        if problems['over_fragmentation']['detected']:
            recommendations.append(
                f" **  SOBRE-FRAGMENTACIÓN detectada: "
                f"{problems['over_fragmentation']['message']}. "
                f"Recomendación: Aumentar MIN_THRESHOLD a 0.2-0.25"
            )
        
        if problems['excess_clusters']['detected']:
            recommendations.append(
                f" **  EXCESO DE CLUSTERS: {problems['excess_clusters']['message']}. "
                f"Recomendación: Aumentar INITIAL_THRESHOLD o MIN_THRESHOLD a 0.25-0.3"
            )
        
        if problems['low_avg_size']['detected']:
            recommendations.append(
                f" **  PROMEDIO BAJO: {problems['low_avg_size']['message']}. "
                f"Recomendación: Ajustar thresholds para clusters más grandes"
            )
        
        # General recommendations
        if stats['num_clusters'] > 20:
            recommendations.append(
                f" **  Para {stats['total_texts']} textos, considera: "
                f"INITIAL_THRESHOLD: 0.5-0.6 (más alto), "
                f"MIN_THRESHOLD: 0.15-0.2 (más alto). "
                f"Objetivo: ~{stats['total_texts']//10}-{stats['total_texts']//15} clusters"
            )
        elif stats['num_clusters'] < 5:
            recommendations.append(
                f" **  Pocos clusters detectados: "
                f"INITIAL_THRESHOLD: 0.3-0.4 (más bajo), "
                f"MIN_THRESHOLD: 0.05-0.1 (más bajo)"
            )
        else:
            recommendations.append(
                f" **  Configuración actual parece razonable. "
                f"Considera ajustar según el dominio específico"
            )
        
        # Orphan clusters recommendation
        if self.orphans and len(self.orphans) > 0:
            recommendations.append(
                f" **  {len(self.orphans)} clusters huérfanos detectados. "
                f"Recomendación: Fusionar o eliminar estos clusters"
            )
        
        self.recommendations = recommendations
        return recommendations
    
    def analyze_cluster_quality(self, cluster_id: int) -> Dict[str, Any]:
        """
        Analyze quality metrics for a specific cluster.
        
        Args:
            cluster_id: ID of cluster to analyze
        
        Returns:
            Dictionary with cluster-specific metrics
        """
        if self.cluster_assignments is None:
            raise ValueError("Clusters must be assigned first.")
        
        cluster_mask = np.array(self.cluster_assignments) == cluster_id
        cluster_size = cluster_mask.sum()
        
        if cluster_size == 0:
            return {'error': f'Cluster {cluster_id} not found'}
        
        result = {
            'cluster_id': cluster_id,
            'size': cluster_size,
        }
        
        # Average confidence
        if self.confidences:
            cluster_confidences = [self.confidences[i] for i in range(len(self.confidences)) if cluster_mask[i]]
            result['avg_confidence'] = np.mean(cluster_confidences)
            result['min_confidence'] = np.min(cluster_confidences)
            result['max_confidence'] = np.max(cluster_confidences)
        
        # Sample texts
        if self.texts:
            cluster_texts = [self.texts[i] for i in range(len(self.texts)) if cluster_mask[i]]
            result['sample_texts'] = cluster_texts[:5]  # First 5 texts
        
        return result
    
    def visualize_distribution(self, output_path: Optional[str] = None,
                              show_plots: Optional[bool] = None,
                              format: Optional[str] = None) -> Optional[str]:
        """
        Visualize cluster distribution.
        
        Args:
            output_path: Path to save plot (None = auto-generate)
            show_plots: Whether to display plots (None = use config)
            format: Plot format (None = use config)
        
        Returns:
            Path to saved plot file or None
        """
        if self.cluster_counts is None:
            raise ValueError("Clusters must be assigned first. Call assign_clusters() first.")
        
        cluster_sizes = list(self.cluster_counts.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of cluster sizes
        axes[0].hist(cluster_sizes, bins=min(20, len(set(cluster_sizes))),
                    color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Cluster Size (texts)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Cluster Sizes')
        if cluster_sizes:
            mean_size = np.mean(cluster_sizes)
            axes[0].axvline(mean_size, color='red', linestyle='--',
                           label=f'Mean: {mean_size:.1f}')
            axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Top clusters
        top_clusters = sorted(self.cluster_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        cluster_ids = [f"Cluster {cid}" for cid, _ in top_clusters]
        sizes = [size for _, size in top_clusters]
        
        axes[1].barh(cluster_ids, sizes, color='coral', alpha=0.7)
        axes[1].set_xlabel('Number of Texts')
        axes[1].set_title('Top 10 Clusters by Size')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None and self.config.save_plots:
            # Auto-generate path using helper method
            output_path = self.get_default_output_path("fragmentation_analysis.png")
        
        if output_path:
            plot_format = format or self.config.plot_format
            plt.savefig(output_path, format=plot_format, dpi=150, bbox_inches='tight')
        
        show = show_plots if show_plots is not None else self.config.show_plots
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def export_results(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """
        Export analysis results to file.
        
        Args:
            format: Export format: 'json', 'csv', 'yaml'
            output_path: Output file path (None = auto-generate)
        
        Returns:
            Path to exported file
        """
        if self.statistics is None:
            self.compute_statistics()
        
        if self.problems is None:
            self.detect_problems()
        
        if self.recommendations is None:
            self.generate_recommendations()
        
        # Prepare export data
        export_data = {
            'statistics': self.statistics,
            'orphans': self.orphans,
            'problems': self.problems,
            'recommendations': self.recommendations,
        }
        
        if self.advanced_metrics:
            export_data['advanced_metrics'] = self.advanced_metrics
        
        # Generate output path if needed
        if output_path is None:
            output_path = self.get_default_output_path(f"cluster_analysis.{format}")
        
        # Export based on format
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == 'yaml':
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML export")
        elif format == 'csv':
            # Export statistics as CSV
            stats_df = pd.DataFrame([self.statistics])
            stats_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'yaml'")
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis results.
        
        Returns:
            Dictionary with summary information
        """
        if self.statistics is None:
            self.compute_statistics()
        
        if self.problems is None:
            self.detect_problems()
        
        if self.recommendations is None:
            self.generate_recommendations()
        
        return {
            'statistics': self.statistics,
            'orphans_count': len(self.orphans) if self.orphans else 0,
            'problems_detected': sum(1 for p in self.problems.values() if p.get('detected', False)),
            'recommendations_count': len(self.recommendations),
        }
    
    @staticmethod
    def find_dataset(dataset_name: Optional[str] = None,
                    search_paths: Optional[List[str]] = None,
                    project_root: Optional[str] = None) -> Optional[str]:
        """
        Find dataset file by searching common locations.
        
        Args:
            dataset_name: Name of dataset file (e.g., "mathematics_dataset2.csv")
                If None, searches for common dataset names
            search_paths: List of directories to search (optional)
            project_root: Root directory of project (optional, auto-detected if None)
        
        Returns:
            Path to dataset file if found, None otherwise
        """
        if project_root is None:
            # Try to detect project root by looking for common markers
            current_file = os.path.abspath(__file__)
            # Go up from cluster_analyzer.py -> utils/ -> bayesian/ -> nodes/ -> xctopus/ -> src/ -> project root
            potential_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(current_file))))))
            
            # Check if this looks like project root (has datasets/, scripts/, etc.)
            if os.path.exists(os.path.join(potential_root, 'datasets')) or \
               os.path.exists(os.path.join(potential_root, 'scripts')):
                project_root = potential_root
        
        if search_paths is None:
            if project_root:
                search_paths = [
                    os.path.join(project_root, "datasets"),
                    os.path.join(project_root, "data"),
                    os.path.join(project_root, "xctopus", "datasets"),
                ]
            else:
                search_paths = ["datasets", "data"]
        
        # Default dataset names to search if not specified
        if dataset_name is None:
            dataset_names = [
                "mathematics_dataset2.csv",
                "dataset_tests.csv",
                "dataset.csv",
            ]
        else:
            dataset_names = [dataset_name]
        
        # Search for datasets
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for name in dataset_names:
                full_path = os.path.join(search_path, name)
                if os.path.exists(full_path):
                    return full_path
        
        return None
    
    def get_default_output_path(self, filename: str = "fragmentation_analysis.png",
                               output_dir: Optional[str] = None,
                               subdirectory: str = "notebooks") -> str:
        """
        Generate default output path for plots or exports.
        
        Args:
            filename: Name of output file
            output_dir: Base output directory (optional, auto-detected if None)
            subdirectory: Subdirectory within output_dir (default: "notebooks")
        
        Returns:
            Full path to output file
        """
        if output_dir is None:
            # Try to detect project root
            current_file = os.path.abspath(__file__)
            potential_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(current_file))))))
            
            # Check if this looks like project root
            if os.path.exists(os.path.join(potential_root, 'datasets')) or \
               os.path.exists(os.path.join(potential_root, 'scripts')):
                output_dir = os.path.join(potential_root, "xctopus", subdirectory)
            else:
                # Fallback to current directory
                output_dir = subdirectory
        
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    @classmethod
    def run_from_cli(cls, dataset_path: Optional[str] = None,
                    dataset_name: Optional[str] = None,
                    export_format: str = 'json',
                    **kwargs) -> Dict[str, Any]:
        """
        Run analysis from command-line interface.
        
        This method handles CLI-specific logic like finding datasets and
        setting up default paths.
        
        Args:
            dataset_path: Explicit path to dataset (optional)
            dataset_name: Name of dataset to search for (optional)
            export_format: Format for export ('json', 'csv', 'yaml', or None)
            **kwargs: Additional arguments passed to ClusterAnalyzer
        
        Returns:
            Dictionary with analysis results
        """
        # Find dataset if path not provided
        if dataset_path is None:
            dataset_path = cls.find_dataset(dataset_name=dataset_name)
        
        if dataset_path is None:
            raise FileNotFoundError(
                "Dataset not found. Please provide dataset_path or ensure "
                "dataset exists in datasets/ or data/ directory."
            )
        
        # Create analyzer and run
        analyzer = cls(**kwargs)
        return analyzer.run_full_analysis(
            dataset_path=dataset_path,
            export_format=export_format
        )
    
    def run_full_analysis(self, dataset_path: Optional[str] = None,
                         dataframe: Optional[pd.DataFrame] = None,
                         text_preprocessor: Optional[TextPreprocessor] = None,
                         filter_node: Optional[FilterBayesianNode] = None,
                         output_dir: Optional[str] = None,
                         export_format: Optional[str] = 'json') -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            dataset_path: Path to dataset CSV file
            dataframe: DataFrame already loaded
            text_preprocessor: TextPreprocessor configured
            filter_node: FilterBayesianNode for clustering
            output_dir: Directory for output files
            export_format: Format for export ('json', 'csv', 'yaml', or None to skip)
        
        Returns:
            Dictionary with complete analysis results
        """
        print("=" * 70)
        print(" ** CLUSTER FRAGMENTATION ANALYSIS")
        print("=" * 70)
        
        # Load dataset
        self.load_dataset(dataset_path, dataframe, text_preprocessor)
        print(f"\n ** Dataset: {len(self.texts)} texts")
        
        # Encode texts
        self.encode_texts(text_preprocessor)
        
        # Assign clusters
        self.assign_clusters(filter_node)
        
        # Compute statistics
        stats = self.compute_statistics()
        print(f"\n ** Cluster Distribution:")
        print(f"  - Total clusters: {stats['num_clusters']}")
        print(f"  - Ratio texts/cluster: {stats['ratio_texts_per_cluster']:.2f}")
        print(f"\n ** Clusters by Size:")
        print(f"  - Small (<{self.config.small_cluster_threshold} texts): "
              f"{stats['small_clusters']} ({100*stats['small_clusters']/stats['num_clusters']:.1f}%)")
        print(f"  - Medium ({self.config.small_cluster_threshold}-{self.config.medium_cluster_threshold-1} texts): "
              f"{stats['medium_clusters']} ({100*stats['medium_clusters']/stats['num_clusters']:.1f}%)")
        print(f"  - Large ({self.config.medium_cluster_threshold}+ texts): "
              f"{stats['large_clusters']} ({100*stats['large_clusters']/stats['num_clusters']:.1f}%)")
        
        # Identify orphans
        orphans_result = self.identify_orphan_clusters()
        orphans = orphans_result['orphans']
        outliers = orphans_result['outliers']
        
        if orphans:
            print(f"\n ** Orphan Clusters (<{self.config.orphan_min_size} texts): {len(orphans)}")
            for cluster_id, size in orphans[:10]:
                print(f"    - Cluster {cluster_id}: {size} text(s)")
            if len(orphans) > 10:
                print(f"    ... and {len(orphans) - 10} more")
        
        if outliers:
            print(f"\n ** Outliers detected: {len(outliers)} items")
        
        # Detect problems
        problems = self.detect_problems()
        print(f"\n**  Problem Analysis:")
        for problem_name, problem_info in problems.items():
            if problem_info.get('detected', False):
                print(f"**  {problem_name.upper().replace('_', ' ')}: {problem_info.get('message', '')}")
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        print(f"\n!! Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Advanced metrics (optional)
        if self.config.compute_advanced_metrics:
            try:
                advanced = self.compute_advanced_metrics()
                if 'error' not in advanced:
                    print(f"== Advanced Metrics:")
                    if advanced.get('silhouette_score') is not None:
                        print(f"  - Silhouette Score: {advanced['silhouette_score']:.3f}")
                    if advanced.get('cohesion') is not None:
                        print(f"  - Cohesion: {advanced['cohesion']:.3f}")
                    if advanced.get('separation') is not None:
                        print(f"  - Separation: {advanced['separation']:.3f}")
            except Exception as e:
                print(f"\n**  Advanced metrics computation failed: {e}")
        
        # Visualize
        plot_path = None
        if self.config.save_plots:
            plot_path = self.visualize_distribution(output_dir)
            if plot_path:
                print(f"\n== Visualization saved: {plot_path}")
        
        # Export
        export_path = None
        if export_format:
            export_path = self.export_results(export_format, output_dir)
            if export_path:
                print(f"==Results exported: {export_path}")
        
        return {
            'statistics': stats,
            'orphans': orphans,
            'problems': problems,
            'recommendations': recommendations,
            'advanced_metrics': self.advanced_metrics if self.config.compute_advanced_metrics else None,
            'output_files': {
                'plot': plot_path,
                'export': export_path
            }
        }


# Backward compatibility function
def analyze_cluster_distribution(dataset_path: str, filter_node: Optional[FilterBayesianNode] = None, **kwargs):
    """
    Convenience function for backward compatibility.
    
    DEPRECATED: Use ClusterAnalyzer.run_full_analysis() instead.
    
    Args:
        dataset_path: Path to dataset CSV file
        filter_node: FilterBayesianNode for clustering
        **kwargs: Additional arguments passed to ClusterAnalyzer
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = ClusterAnalyzer(**kwargs)
    return analyzer.run_full_analysis(dataset_path=dataset_path, filter_node=filter_node)

