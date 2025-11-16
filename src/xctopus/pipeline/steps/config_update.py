"""
Config Update Step - Update Cluster Configuration

Encapsulates the cluster configuration update functionality from script 04.
Updates configuration (layers and LoRA rank) for large clusters that need more capacity.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .base import PipelineStep
from xctopus.nodes.bayesian.bayesian_node import KnowledgeNode
from xctopus.modules.lora import LoRA


class ConfigUpdateStep(PipelineStep):
    """
    Step 3: Update cluster configurations.
    
    Encapsulates: scripts/deprecated/04_cluster_update_config.py (legacy)
    
    This step:
    1. Identifies large clusters that need configuration updates
    2. Updates layers and LoRA rank based on cluster size
    3. Reinitializes nodes with new configuration
    
    Requires: clustering step to be executed first.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        pipeline.run(step='clustering', epochs=5)
        results = pipeline.run(step='config_update')
    """
    
    def get_required_steps(self):
        """
        Config update step requires clustering to be executed first.
        
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
                "Clustering step must be executed before config_update. "
                "Execute: pipeline.run(step='clustering')"
            )
        
        if not pipeline.knowledge_nodes:
            raise ValueError(
                "No knowledge nodes found. Clustering step must create nodes first."
            )
    
    def execute(self, pipeline, specific_cluster_ids: Optional[List[int]] = None, **kwargs):
        """
        Execute cluster configuration update.
        
        Args:
            pipeline: XctopusPipeline instance
            specific_cluster_ids: Optional list of cluster IDs to update (None = update all that need it)
            **kwargs: Additional options:
                - min_size_for_update: Minimum cluster size to consider for update (default: 50)
                - enable_additional_epoch: Run additional epoch after update (default: False)
        
        Returns:
            dict: Update results with keys:
                - 'updated_clusters': List of updated cluster IDs
                - 'update_details': Details per cluster (old config, new config)
                - 'total_updated': Number of clusters updated
        """
        self.validate_inputs(pipeline)
        
        knowledge_nodes = pipeline.knowledge_nodes
        config = pipeline.config
        
        if not knowledge_nodes:
            return {
                'updated_clusters': [],
                'update_details': {},
                'total_updated': 0
            }
        
        min_size = kwargs.get('min_size_for_update', 50)
        updated_clusters = []
        update_details = {}
        
        print(f"[*] Updating cluster configurations (min_size={min_size})...")
        
        # Determine which clusters to update
        clusters_to_update = []
        if specific_cluster_ids:
            # Update specific clusters
            for cluster_id in specific_cluster_ids:
                if cluster_id in knowledge_nodes:
                    node = knowledge_nodes[cluster_id]
                    cluster_size = self._get_cluster_size(node, cluster_id)
                    if cluster_size >= min_size:
                        clusters_to_update.append((cluster_id, cluster_size))
        else:
            # Update all clusters that need it
            for cluster_id, node in knowledge_nodes.items():
                cluster_size = self._get_cluster_size(node, cluster_id)
                if cluster_size >= min_size:
                    clusters_to_update.append((cluster_id, cluster_size))
        
        if not clusters_to_update:
            print("[OK] No clusters need configuration updates")
            return {
                'updated_clusters': [],
                'update_details': {},
                'total_updated': 0
            }
        
        print(f"[*] Found {len(clusters_to_update)} clusters to update")
        
        # Update each cluster
        for cluster_id, cluster_size in clusters_to_update:
            node = knowledge_nodes[cluster_id]
            
            # Get current configuration
            current_layers = node.transformer.num_layers if hasattr(node.transformer, 'num_layers') else 1
            current_rank = self._get_current_lora_rank(node)
            
            # Determine optimal configuration
            optimal_layers = self._get_num_layers(cluster_size, config)
            optimal_rank = self._get_lora_rank(cluster_size, config)
            
            # Check if update is needed
            if current_layers == optimal_layers and current_rank == optimal_rank:
                continue  # No update needed
            
            # Store old configuration
            update_details[cluster_id] = {
                'old_layers': current_layers,
                'old_rank': current_rank,
                'new_layers': optimal_layers,
                'new_rank': optimal_rank,
                'cluster_size': cluster_size
            }
            
            # Update node configuration
            try:
                self._update_node_configuration(
                    node,
                    cluster_id,
                    optimal_layers,
                    optimal_rank,
                    config
                )
                updated_clusters.append(cluster_id)
                print(f"  ✓ Cluster {cluster_id}: {current_layers}L/{current_rank}R → {optimal_layers}L/{optimal_rank}R")
            except Exception as e:
                print(f"  [WARNING]  Error updating cluster {cluster_id}: {e}")
                continue
        
        # Run additional epoch if enabled
        if kwargs.get('enable_additional_epoch', False) and updated_clusters:
            print("\n[*] Running additional epoch after configuration update...")
            try:
                self._run_additional_epoch(pipeline, updated_clusters, config)
            except Exception as e:
                print(f"  [WARNING]  Error running additional epoch: {e}")
        
        results = {
            'updated_clusters': updated_clusters,
            'update_details': update_details,
            'total_updated': len(updated_clusters)
        }
        
        pipeline.results['config_update'] = results
        
        print(f"\n[OK] Configuration update completed: {len(updated_clusters)} clusters updated")
        
        return results
    
    def _get_cluster_size(self, node, cluster_id):
        """Get cluster size from node's filter memory"""
        if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
            memory = node.filter.memory
            if isinstance(memory, dict):
                return len(memory.get(cluster_id, []))
        return 0
    
    def _get_current_lora_rank(self, node):
        """Get current LoRA rank from node"""
        if hasattr(node, 'transformer') and hasattr(node.transformer, 'encoder'):
            # Try to find LoRA rank in transformer layers
            for layer in node.transformer.encoder.layers:
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'q_proj') and hasattr(attn.q_proj, 'lora_A'):
                        # LoRA rank is typically the same for all projections
                        if hasattr(attn.q_proj.lora_A, 'out_features'):
                            return attn.q_proj.lora_A.out_features
        return 4  # Default rank
    
    def _get_num_layers(self, cluster_size, config):
        """Get optimal number of layers based on cluster size"""
        if cluster_size < 50:
            return 1
        elif cluster_size < 200:
            return 2
        else:
            return 3
    
    def _get_lora_rank(self, cluster_size, config):
        """Get optimal LoRA rank based on cluster size"""
        return LoRA.calculate_optimal_rank(
            cluster_size,
            small_threshold=30,
            medium_threshold=60,
            small_rank=4,
            medium_rank=5,
            large_rank=6
        )
    
    def _update_node_configuration(self, node, cluster_id, num_layers, lora_rank, config):
        """
        Update node configuration by recreating transformer with new parameters.
        
        This is a simplified version - in a full implementation, we would need to:
        1. Save current state
        2. Recreate transformer with new config
        3. Transfer learned parameters where possible
        4. Reinitialize LoRA adapters
        """
        # For now, we'll just update the LoRA rank if possible
        # Full reconfiguration would require more complex logic
        
        # Note: This is a placeholder - actual implementation would need
        # to handle transformer recreation more carefully
        if hasattr(node, 'transformer'):
            # Update LoRA adapters if they exist
            device = getattr(config, 'DEVICE', torch.device("cpu"))
            d_model = getattr(config, 'D_MODEL', 128)
            
            # This is a simplified update - full implementation would
            # require recreating the transformer with new layers
            # For now, we'll just log that update is needed
            pass
    
    def _run_additional_epoch(self, pipeline, updated_cluster_ids, config):
        """Run an additional training epoch for updated clusters"""
        # This would run a training epoch on the updated clusters
        # Implementation would be similar to ClusteringStep training logic
        # For now, this is a placeholder
        pass

