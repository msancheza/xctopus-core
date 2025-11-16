"""
Clustering Step - Dynamic Clustering Pipeline

Encapsulates the dynamic clustering functionality from scripts 01 + 02.
This is the core step that creates knowledge nodes from embeddings.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from collections import defaultdict

from .base import PipelineStep
from xctopus.nodes.bayesian.core.text_preprocessor import TextPreprocessor
from xctopus.nodes.bayesian.bayesian_filter import FilterBayesianNode
from xctopus.nodes.bayesian.bayesian_node import KnowledgeNode
from xctopus.modules.lora import LoRA


class ClusteringStep(PipelineStep):
    """
    Step 1-2: Dynamic clustering pipeline.
    
    Encapsulates: scripts/deprecated/01_cluster_dynamic_pipeline.py + 02_pipeline_run.py (legacy)
    
    This step:
    1. Loads and encodes texts from dataset
    2. Creates knowledge nodes dynamically
    3. Trains nodes with LoRA
    4. Merges small clusters
    5. Updates configuration for large clusters
    
    This is the core step that all other steps depend on.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        results = pipeline.run(step='clustering', epochs=5)
        nodes = pipeline.get_nodes()
    """
    
    def get_required_steps(self):
        """
        Clustering step doesn't require any previous steps.
        
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
                "dataset_path is required for clustering. "
                "Provide it as argument or set pipeline.dataset_path"
            )
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset file not found: {dataset_path}")
    
    def execute(self, dataset_path, pipeline, epochs=None, **kwargs):
        """
        Execute dynamic clustering pipeline.
        
        Args:
            dataset_path: Path to dataset CSV file
            pipeline: XctopusPipeline instance
            epochs: Number of training epochs (uses config if not provided)
            **kwargs: Additional options:
                - enable_training: Enable LoRA training (default: True)
                - enable_merge: Enable merging of small clusters (default: True)
                - enable_fine_tune_large: Enable fine-tuning of large clusters (default: True)
                - initial_threshold: Initial similarity threshold (default: 0.75)
                - min_threshold: Minimum similarity threshold (default: 0.5)
        
        Returns:
            dict: Clustering results with keys:
                - 'knowledge_nodes': Dictionary of created nodes
                - 'embeddings': Generated embeddings
                - 'cluster_stats': Statistics per cluster
                - 'total_clusters': Total number of clusters created
        """
        self.validate_inputs(pipeline, dataset_path=dataset_path)
        
        # Get configuration
        config = pipeline.config
        num_epochs = epochs or config.NUM_EPOCHS
        
        # Get or create TextPreprocessor
        text_preprocessor = pipeline._get_preprocessor()
        
        # Load dataset and get texts
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Extract texts using pipeline's text_columns
        texts = self._extract_texts(df, pipeline.text_columns, pipeline.join_with)
        
        if not texts:
            raise ValueError("No texts could be extracted from dataset")
        
        print(f"[*] Processing {len(texts)} texts with {num_epochs} epochs")
        
        # Encode texts to embeddings
        print("[*] Encoding texts to embeddings...")
        embeddings = text_preprocessor.encode_texts(texts)
        embeddings = embeddings.to(config.DEVICE if hasattr(config, 'DEVICE') else torch.device("cpu"))
        
        # Initialize filters
        initial_threshold = kwargs.get('initial_threshold', getattr(config, 'INITIAL_THRESHOLD', 0.75))
        min_threshold = kwargs.get('min_threshold', getattr(config, 'MIN_THRESHOLD', 0.5))
        
        # Get max_threshold from config or kwargs, default to 0.9
        max_threshold = kwargs.get('max_threshold', getattr(config, 'MAX_THRESHOLD', 0.9))
        adaptive_threshold = kwargs.get('adaptive_threshold', getattr(config, 'ADAPTIVE_THRESHOLD', True))
        
        local_filter = FilterBayesianNode(
            mode="train",
            initial_threshold=initial_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,  # Make configurable
            threshold_decay=getattr(config, 'THRESHOLD_DECAY', 0.95),
            adaptive_threshold=adaptive_threshold  # Make configurable
        )
        
        # Initialize state
        knowledge_nodes = {}
        optimizers = {}
        criterion = nn.MSELoss()
        device = getattr(config, 'DEVICE', torch.device("cpu"))
        d_model = getattr(config, 'D_MODEL', 128)
        learning_rate = getattr(config, 'LORA_LEARNING_RATE', 1e-3)
        enable_training = kwargs.get('enable_training', getattr(config, 'ENABLE_TRAINING', True))
        
        # Process embeddings through epochs
        print(f"\n[*] Starting clustering with {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            epoch_losses = defaultdict(list)
            
            for i, emb in enumerate(embeddings):
                # Process embedding
                cluster_id, confidence = local_filter.evaluate(emb.unsqueeze(0))
                
                if cluster_id in knowledge_nodes:
                    # Existing cluster
                    node = knowledge_nodes[cluster_id]
                    
                    if enable_training:
                        node.train()
                        
                        # Get or create optimizer
                        if cluster_id not in optimizers:
                            optimizer = node.create_lora_optimizer(learning_rate)
                            if optimizer is not None:
                                optimizers[cluster_id] = optimizer
                            else:
                                optimizers[cluster_id] = torch.optim.Adam(
                                    node.parameters(),
                                    lr=learning_rate
                                )
                        
                        optimizer = optimizers[cluster_id]
                        optimizer.zero_grad()
                    
                    # Forward pass
                    emb_input = emb.unsqueeze(0).unsqueeze(0)
                    refined, feedback = node(emb_input)
                    
                    # Training
                    if enable_training and cluster_id in optimizers:
                        if node.input_proj:
                            target = node.input_proj(emb_input).mean(dim=1)
                        else:
                            target = emb_input.mean(dim=1)
                        
                        loss = criterion(refined, target)
                        loss.backward()
                        optimizer.step()
                else:
                    # New cluster - create KnowledgeNode
                    cluster_size = 1
                    if hasattr(local_filter, 'memory') and isinstance(local_filter.memory, dict):
                        cluster_size = len(local_filter.memory.get(cluster_id, []))
                        if cluster_size == 0:
                            cluster_size = 1
                    
                    # Get adaptive configuration
                    num_layers = self._get_num_layers(cluster_size, config)
                    lora_rank = self._get_lora_rank(cluster_size, config)
                    
                    # Create KnowledgeNode
                    node = KnowledgeNode(
                        src_vocab_size=getattr(config, 'SRC_VOCAB_SIZE', 5000),
                        tgt_vocab_size=getattr(config, 'TGT_VOCAB_SIZE', 5000),
                        d_model=d_model,
                        num_heads=getattr(config, 'NUM_HEADS', 4),
                        num_layers=num_layers,
                        d_ff=getattr(config, 'D_FF', 512),
                        max_seq_length=getattr(config, 'MAX_SEQ_LENGTH', 1),
                        dropout=getattr(config, 'DROPOUT', 0.1),
                        embedding_dim=None,
                        use_lora=True,
                        lora_r=lora_rank,
                        lora_alpha=getattr(config, 'LORA_ALPHA', 1.0)
                    )
                    node.filter = local_filter
                    
                    # Create input projection
                    input_proj = nn.Linear(emb.shape[-1], d_model).to(device)
                    node.input_proj = input_proj.to(device)
                    
                    # Freeze base parameters
                    node.transformer.freeze_base_parameters()
                    node = node.to(device)
                    
                    knowledge_nodes[cluster_id] = node
                    
                    # Create optimizer
                    if enable_training:
                        optimizer = node.create_lora_optimizer(learning_rate)
                        if optimizer is not None:
                            optimizers[cluster_id] = optimizer
                    
                    # Initial forward
                    emb_input = emb.unsqueeze(0).unsqueeze(0)
                    refined, feedback = node(emb_input)
                
                # Track losses for statistics
                if enable_training and i % 10 == 0:
                    cluster_id, _ = local_filter.evaluate(emb.unsqueeze(0))
                    if cluster_id in knowledge_nodes:
                        node = knowledge_nodes[cluster_id]
                        node.eval()
                        with torch.no_grad():
                            emb_input = emb.unsqueeze(0).unsqueeze(0).to(device)
                            refined, _ = node(emb_input)
                            if node.input_proj:
                                target = node.input_proj(emb_input).mean(dim=1)
                            else:
                                target = emb_input.mean(dim=1)
                            loss = criterion(refined, target)
                            epoch_losses[cluster_id].append(loss.item())
                        node.train()
            
            # Print epoch statistics
            if epoch_losses:
                all_losses = [l for losses in epoch_losses.values() for l in losses]
                avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
                print(f"  [*] Average loss: {avg_loss:.6f}")
            
            # Merge small clusters after each epoch (except last)
            if kwargs.get('enable_merge', True) and epoch < num_epochs - 1:
                nodes_before = len(knowledge_nodes)
                self._merge_small_clusters(
                    knowledge_nodes,
                    optimizers,
                    config,
                    local_filter
                )
                nodes_after = len(knowledge_nodes)
                if nodes_before > nodes_after:
                    print(f"  [*] Merged clusters: {nodes_before} -> {nodes_after} nodes")
        
        # Final merge
        if kwargs.get('enable_merge', True):
            nodes_before = len(knowledge_nodes)
            self._merge_small_clusters(
                knowledge_nodes,
                optimizers,
                config,
                local_filter
            )
            nodes_after = len(knowledge_nodes)
            if nodes_before > nodes_after:
                print(f"  [*] Final merge: {nodes_before} -> {nodes_after} nodes")
        
        # Compute cluster statistics
        cluster_stats = []
        for cluster_id, node in knowledge_nodes.items():
            cluster_size = 0
            if hasattr(node.filter, 'memory') and isinstance(node.filter.memory, dict):
                cluster_size = len(node.filter.memory.get(cluster_id, []))
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'layers': node.transformer.num_layers if hasattr(node.transformer, 'num_layers') else 0
            })
        
        # Store in pipeline state
        pipeline.knowledge_nodes = knowledge_nodes
        pipeline.embeddings = embeddings
        
        # Store results
        results = {
            'knowledge_nodes': knowledge_nodes,
            'embeddings': embeddings,
            'cluster_stats': cluster_stats,
            'total_clusters': len(knowledge_nodes),
            'total_texts': len(texts)
        }
        
        pipeline.results['clustering'] = results
        
        print(f"\n[OK] Clustering completed: {len(knowledge_nodes)} clusters created")
        
        return results
    
    def _extract_texts(self, df, text_columns, join_with):
        """Extract and combine text columns from DataFrame"""
        if not text_columns:
            text_columns = ['text']
        
        available_cols = [col for col in text_columns if col in df.columns]
        if not available_cols:
            raise ValueError(f"None of the specified text columns found: {text_columns}")
        
        texts = []
        for _, row in df.iterrows():
            text_parts = [str(row[col]) for col in available_cols if pd.notna(row.get(col))]
            if text_parts:
                combined = join_with.join(text_parts)
                if combined.strip():
                    texts.append(combined)
        
        return texts
    
    def _get_num_layers(self, cluster_size, config):
        """Get adaptive number of layers based on cluster size"""
        if cluster_size < 50:
            return 1
        elif cluster_size < 200:
            return 2
        else:
            return 3
    
    def _get_lora_rank(self, cluster_size, config):
        """Get adaptive LoRA rank based on cluster size"""
        return LoRA.calculate_optimal_rank(
            cluster_size,
            small_threshold=30,
            medium_threshold=60,
            small_rank=4,
            medium_rank=5,
            large_rank=6
        )
    
    def _merge_small_clusters(self, knowledge_nodes, optimizers, config, filter_node):
        """Merge small clusters with their closest semantic neighbors"""
        min_size = getattr(config, 'MIN_CLUSTER_SIZE', 8)
        similarity_threshold = getattr(config, 'MERGE_SIMILARITY_THRESHOLD', 0.7)
        
        # Identify small clusters
        small_clusters = []
        for cluster_id, node in knowledge_nodes.items():
            if hasattr(node.filter, 'memory') and isinstance(node.filter.memory, dict):
                cluster_size = len(node.filter.memory.get(cluster_id, []))
                if cluster_size < min_size:
                    small_clusters.append((cluster_id, cluster_size))
        
        if not small_clusters:
            return
        
        # Merge each small cluster with closest neighbor
        merged_count = 0
        for small_id, small_size in small_clusters:
            if small_id not in knowledge_nodes:
                continue
            
            small_node = knowledge_nodes[small_id]
            if not hasattr(small_node.filter, 'memory') or small_id not in small_node.filter.memory:
                continue
            
            small_embeddings = small_node.filter.memory[small_id]
            if not small_embeddings:
                continue
            
            # Calculate centroid
            small_centroid = torch.mean(torch.stack(small_embeddings), dim=0)
            
            # Find closest cluster
            best_match_id = None
            best_similarity = -1
            
            for other_id, other_node in knowledge_nodes.items():
                if other_id == small_id:
                    continue
                
                if not hasattr(other_node.filter, 'memory') or other_id not in other_node.filter.memory:
                    continue
                
                other_embeddings = other_node.filter.memory[other_id]
                if not other_embeddings:
                    continue
                
                other_centroid = torch.mean(torch.stack(other_embeddings), dim=0)
                similarity = torch.nn.functional.cosine_similarity(
                    small_centroid.view(-1),
                    other_centroid.view(-1),
                    dim=0
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = other_id
            
            # Merge if similarity is sufficient
            if best_match_id and best_similarity > similarity_threshold:
                target_node = knowledge_nodes[best_match_id]
                
                if hasattr(target_node.filter, 'memory'):
                    if best_match_id not in target_node.filter.memory:
                        target_node.filter.memory[best_match_id] = []
                    target_node.filter.memory[best_match_id].extend(small_embeddings)
                
                # Remove small cluster
                del knowledge_nodes[small_id]
                if small_id in optimizers:
                    del optimizers[small_id]
                
                merged_count += 1
        
        if merged_count > 0:
            print(f"  ✓ Merged {merged_count} small clusters")

