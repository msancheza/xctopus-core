"""
Optimize Step - Cluster Optimization

Encapsulates the cluster optimization functionality from script 06.
Analyzes cluster distribution, merges small clusters, and generates optimization recommendations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from .base import PipelineStep
from xctopus.nodes.bayesian.utils import (
    analyze_cluster_distribution,
    find_similar_cluster_pairs,
    calculate_cluster_centroids,
    determine_node_configuration
)


class OptimizeStep(PipelineStep):
    """
    Step 5: Optimize clusters.
    
    Encapsulates: scripts/deprecated/06_cluster_optimize.py (legacy)
    
    This step:
    1. Analyzes cluster distribution (orphan, small, medium, large)
    2. Identifies clusters that need merging
    3. Merges small clusters with similar neighbors
    4. Generates optimization recommendations
    
    Requires: clustering step to be executed first.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        pipeline.run(step='clustering', epochs=5)
        results = pipeline.run(step='optimize', auto_merge=True)
    """
    
    def get_required_steps(self):
        """
        Optimize step requires clustering to be executed first.
        
        Returns:
            list: ['clustering']
        """
        return ['clustering']
    
    def validate_inputs(self, pipeline, **kwargs):
        """
        Validate that required inputs are available.
        
        Args:
            pipeline: XctopusPipeline instance
            **kwargs: Step-specific parameters
        
        Raises:
            ValueError: If clustering hasn't been executed or knowledge_nodes are missing
        """
        if 'clustering' not in pipeline.results:
            raise ValueError(
                "Clustering step must be executed before optimize. "
                "Execute: pipeline.run(step='clustering')"
            )
        
        if not pipeline.knowledge_nodes:
            raise ValueError(
                "No knowledge nodes found. Clustering step must create nodes first."
            )
    
    def execute(
        self,
        pipeline,
        orphan_threshold: int = 3,
        small_threshold: int = 5,
        medium_threshold: int = 20,
        similarity_threshold: float = 0.7,
        auto_merge: bool = False,
        **kwargs
    ):
        """
        Execute cluster optimization.
        
        Args:
            pipeline: XctopusPipeline instance
            orphan_threshold: Size threshold for orphan clusters (default: 3)
            small_threshold: Size threshold for small clusters (default: 5)
            medium_threshold: Size threshold for medium clusters (default: 20)
            similarity_threshold: Similarity threshold for merging (default: 0.7)
            auto_merge: Automatically merge small clusters (default: False)
            **kwargs: Additional options:
                - max_merge_pairs: Maximum number of pairs to merge (default: 10)
                - export_results: Export results to file (default: False)
                - output_path: Path for exported results (default: None)
                - export_format: Format for export ('json', 'csv') (default: 'json')
        
        Returns:
            dict: Optimization results with keys:
                - 'distribution': Cluster distribution analysis
                - 'recommendations': Optimization recommendations
                - 'merged_clusters': List of merged cluster pairs (if auto_merge=True)
                - 'merge_count': Number of clusters merged
        """
        self.validate_inputs(pipeline)
        
        knowledge_nodes = pipeline.knowledge_nodes
        
        if not knowledge_nodes:
            return {
                'distribution': {},
                'recommendations': [],
                'merged_clusters': [],
                'merge_count': 0
            }
        
        print("[*] Optimizing clusters...")
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for cluster_id, node in knowledge_nodes.items():
            cluster_size = self._get_cluster_size(node, cluster_id)
            if cluster_size > 0:
                cluster_sizes[cluster_id] = cluster_size
        
        if not cluster_sizes:
            print("[WARNING]  No clusters found to optimize")
            return {
                'distribution': {},
                'recommendations': [],
                'merged_clusters': [],
                'merge_count': 0
            }
        
        # Analyze cluster distribution
        print(f"[*] Analyzing cluster distribution...")
        distribution = analyze_cluster_distribution(
            cluster_sizes=cluster_sizes,
            orphan_threshold=orphan_threshold,
            small_threshold=small_threshold,
            medium_threshold=medium_threshold
        )
        
        # Print distribution summary
        self._print_distribution_summary(distribution, orphan_threshold, small_threshold, medium_threshold)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            distribution,
            knowledge_nodes,
            orphan_threshold,
            small_threshold,
            medium_threshold
        )
        
        # Merge clusters if auto_merge is enabled
        merged_clusters = []
        merge_count = 0
        if auto_merge:
            print(f"\n[*] Merging small clusters (similarity_threshold={similarity_threshold})...")
            merged_clusters, merge_count = self._merge_small_clusters(
                knowledge_nodes,
                distribution,
                similarity_threshold,
                kwargs.get('max_merge_pairs', 10)
            )
            print(f"[OK] Merged {merge_count} cluster pairs")
        
        results = {
            'distribution': distribution,
            'recommendations': recommendations,
            'merged_clusters': merged_clusters,
            'merge_count': merge_count
        }
        
        # Export results if requested
        if kwargs.get('export_results', False):
            output_path = kwargs.get('output_path', None)
            export_format = kwargs.get('export_format', 'json')
            self._export_results(results, output_path, export_format)
        
        pipeline.results['optimize'] = results
        
        print(f"\n[OK] Optimization completed")
        
        return results
    
    def _get_cluster_size(self, node, cluster_id):
        """Get cluster size from node's filter memory"""
        if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
            memory = node.filter.memory
            if isinstance(memory, dict):
                return len(memory.get(cluster_id, []))
        return 0
    
    def _print_distribution_summary(self, distribution, orphan_threshold, small_threshold, medium_threshold):
        """Print cluster distribution summary"""
        total_clusters = distribution.get('total_clusters', 0)
        total_embeddings = distribution.get('total_embeddings', 0)
        avg_size = distribution.get('avg_size', 0.0)
        
        orphan_clusters = distribution.get('orphan_clusters', [])
        small_clusters = distribution.get('small_clusters', [])
        medium_clusters = distribution.get('medium_clusters', [])
        large_clusters = distribution.get('large_clusters', [])
        
        print(f"\n[*] Cluster Distribution:")
        print(f"  - Total clusters: {total_clusters}")
        print(f"  - Total embeddings: {total_embeddings}")
        print(f"  - Average size: {avg_size:.1f}")
        print(f"  - Orphan clusters (<{orphan_threshold}): {len(orphan_clusters)}")
        print(f"  - Small clusters ({orphan_threshold}-{small_threshold-1}): {len(small_clusters)}")
        print(f"  - Medium clusters ({small_threshold}-{medium_threshold-1}): {len(medium_clusters)}")
        print(f"  - Large clusters ({medium_threshold}+): {len(large_clusters)}")
    
    def _generate_recommendations(
        self,
        distribution: Dict,
        knowledge_nodes: Dict,
        orphan_threshold: int,
        small_threshold: int,
        medium_threshold: int
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        orphan_clusters = distribution.get('orphan_clusters', [])
        small_clusters = distribution.get('small_clusters', [])
        large_clusters = distribution.get('large_clusters', [])
        
        if orphan_clusters:
            recommendations.append(
                f"Merge {len(orphan_clusters)} orphan clusters immediately "
                f"(clusters with <{orphan_threshold} embeddings)"
            )
        
        if small_clusters:
            recommendations.append(
                f"Consider merging {len(small_clusters)} small clusters "
                f"if they don't add diversity"
            )
        
        if large_clusters:
            for cluster_id, size in sorted(large_clusters, key=lambda x: x[1], reverse=True)[:5]:
                optimal_layers, optimal_rank = determine_node_configuration(cluster_size=size)
                recommendations.append(
                    f"Cluster {cluster_id} ({size} embeddings): "
                    f"Consider updating to {optimal_layers} layers, rank {optimal_rank}"
                )
        
        if not recommendations:
            recommendations.append("No optimization needed - cluster distribution looks good")
        
        return recommendations
    
    def _merge_small_clusters(
        self,
        knowledge_nodes: Dict,
        distribution: Dict,
        similarity_threshold: float,
        max_pairs: int
    ) -> Tuple[List[Tuple[int, int]], int]:
        """
        Merge small clusters with similar neighbors.
        
        Returns:
            Tuple of (merged_pairs, merge_count)
        """
        orphan_clusters = distribution.get('orphan_clusters', [])
        small_clusters = distribution.get('small_clusters', [])
        clusters_to_merge = orphan_clusters + small_clusters
        
        if len(clusters_to_merge) < 2:
            return [], 0
        
        # Calculate centroids for clusters to merge
        centroids = {}
        for cluster_id, _ in clusters_to_merge:
            if cluster_id in knowledge_nodes:
                centroid = self._calculate_cluster_centroid(knowledge_nodes[cluster_id], cluster_id)
                if centroid is not None:
                    centroids[cluster_id] = centroid
        
        if len(centroids) < 2:
            return [], 0
        
        # Find similar pairs
        similar_pairs = find_similar_cluster_pairs(
            centroids=centroids,
            similarity_threshold=similarity_threshold,
            max_pairs=max_pairs
        )
        
        if not similar_pairs:
            return [], 0
        
        # Merge pairs
        merged_pairs = []
        merged_count = 0
        merged_ids = set()
        
        for cluster_id1, cluster_id2, similarity in similar_pairs:
            if cluster_id1 in merged_ids or cluster_id2 in merged_ids:
                continue  # Already merged
            
            if cluster_id1 in knowledge_nodes and cluster_id2 in knowledge_nodes:
                # Merge cluster_id2 into cluster_id1
                node1 = knowledge_nodes[cluster_id1]
                node2 = knowledge_nodes[cluster_id2]
                
                # Transfer embeddings from node2 to node1
                if self._merge_cluster_embeddings(node1, cluster_id1, node2, cluster_id2):
                    merged_pairs.append((cluster_id1, cluster_id2))
                    merged_ids.add(cluster_id2)
                    merged_count += 1
                    
                    # Remove merged cluster
                    del knowledge_nodes[cluster_id2]
        
        return merged_pairs, merged_count
    
    def _calculate_cluster_centroid(self, node, cluster_id):
        """Calculate centroid of a cluster"""
        if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
            memory = node.filter.memory
            if isinstance(memory, dict) and cluster_id in memory:
                embeddings = memory[cluster_id]
                if embeddings:
                    stacked = torch.stack(embeddings)
                    return torch.mean(stacked, dim=0)
        return None
    
    def _merge_cluster_embeddings(self, target_node, target_id, source_node, source_id):
        """Merge embeddings from source cluster into target cluster"""
        try:
            if not (hasattr(target_node, 'filter') and hasattr(target_node.filter, 'memory')):
                return False
            
            if not (hasattr(source_node, 'filter') and hasattr(source_node.filter, 'memory')):
                return False
            
            target_memory = target_node.filter.memory
            source_memory = source_node.filter.memory
            
            if not isinstance(target_memory, dict) or not isinstance(source_memory, dict):
                return False
            
            if source_id not in source_memory:
                return False
            
            source_embeddings = source_memory[source_id]
            if not source_embeddings:
                return False
            
            # Add embeddings to target
            if target_id not in target_memory:
                target_memory[target_id] = []
            
            target_memory[target_id].extend(source_embeddings)
            
            return True
        except Exception as e:
            print(f"  [WARNING]  Error merging clusters {source_id} → {target_id}: {e}")
            return False
    
    def _export_results(self, results: Dict, output_path: Optional[str], format: str):
        """Export optimization results to file"""
        try:
            from datetime import datetime
            import json
            import pandas as pd
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"optimization_results_{timestamp}.{format}"
            
            if format == 'json':
                # Prepare data for JSON serialization
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'distribution': {
                        'total_clusters': results['distribution'].get('total_clusters', 0),
                        'total_embeddings': results['distribution'].get('total_embeddings', 0),
                        'avg_size': float(results['distribution'].get('avg_size', 0.0)),
                        'orphan_clusters_count': len(results['distribution'].get('orphan_clusters', [])),
                        'small_clusters_count': len(results['distribution'].get('small_clusters', [])),
                        'medium_clusters_count': len(results['distribution'].get('medium_clusters', [])),
                        'large_clusters_count': len(results['distribution'].get('large_clusters', []))
                    },
                    'recommendations': results['recommendations'],
                    'merged_clusters': results['merged_clusters'],
                    'merge_count': results['merge_count']
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format == 'csv':
                # Export distribution as CSV
                df_data = []
                distribution = results['distribution']
                for category in ['orphan_clusters', 'small_clusters', 'medium_clusters', 'large_clusters']:
                    for cid, size in distribution.get(category, []):
                        df_data.append({
                            'cluster_id': int(cid),
                            'size': int(size),
                            'category': category.replace('_clusters', '')
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.to_csv(output_path, index=False)
            
            print(f"[OK] Results exported to: {output_path}")
        except Exception as e:
            print(f"[WARNING]  Error exporting results: {e}")

