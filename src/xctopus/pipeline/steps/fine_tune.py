"""
Fine-Tune Step - Fine-tune Large Clusters

Encapsulates the fine-tuning functionality from script 05.
Performs focused LoRA fine-tuning on large clusters to improve specialization.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional, List

from .base import PipelineStep
from xctopus.nodes.bayesian.utils import (
    identify_large_clusters,
    fine_tune_cluster_with_lora
)


class FineTuneStep(PipelineStep):
    """
    Step 4: Fine-tune large clusters.
    
    Encapsulates: scripts/deprecated/05_cluster_fine_tune_large.py (legacy)
    
    This step:
    1. Identifies large clusters (>= min_size)
    2. Performs focused LoRA fine-tuning on each large cluster
    3. Tracks improvement metrics (loss reduction)
    4. Exports results (optional)
    
    Requires: clustering step to be executed first.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        pipeline.run(step='clustering', epochs=5)
        results = pipeline.run(step='fine_tune', min_size=50, num_epochs=3)
    """
    
    def get_required_steps(self):
        """
        Fine-tune step requires clustering to be executed first.
        
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
                "Clustering step must be executed before fine_tune. "
                "Execute: pipeline.run(step='clustering')"
            )
        
        if not pipeline.knowledge_nodes:
            raise ValueError(
                "No knowledge nodes found. Clustering step must create nodes first."
            )
    
    def execute(
        self,
        pipeline,
        min_size: int = 50,
        num_epochs: int = 3,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """
        Execute fine-tuning on large clusters.
        
        Args:
            pipeline: XctopusPipeline instance
            min_size: Minimum cluster size for fine-tuning (default: 50)
            num_epochs: Number of additional training epochs (default: 3)
            learning_rate: Learning rate for fine-tuning (None = use config)
            **kwargs: Additional options:
                - export_results: Export results to file (default: False)
                - output_path: Path for exported results (default: None)
                - export_format: Format for export ('json', 'csv') (default: 'json')
                - verbose: Enable verbose output (default: False)
        
        Returns:
            dict: Fine-tuning results with keys:
                - 'fine_tuned_clusters': List of cluster IDs that were fine-tuned
                - 'results': Dictionary of results per cluster (improvement, loss, etc.)
                - 'summary': Summary statistics (avg improvement, etc.)
                - 'total_clusters': Number of clusters fine-tuned
        """
        self.validate_inputs(pipeline)
        
        knowledge_nodes = pipeline.knowledge_nodes
        config = pipeline.config
        
        if not knowledge_nodes:
            return {
                'fine_tuned_clusters': [],
                'results': {},
                'summary': {},
                'total_clusters': 0
            }
        
        # Validate parameters
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if learning_rate is not None and (learning_rate <= 0 or learning_rate > 1):
            raise ValueError("learning_rate must be in (0, 1]")
        
        # Use config learning rate if not provided
        if learning_rate is None:
            learning_rate = getattr(config, 'LORA_LEARNING_RATE', 1e-3)
        
        print(f"[*] Fine-tuning large clusters (min_size={min_size}, epochs={num_epochs})...")
        
        # Identify large clusters
        try:
            large_clusters = identify_large_clusters(knowledge_nodes, min_size=min_size)
        except Exception as e:
            print(f"[ERROR] Error identifying large clusters: {e}")
            return {
                'fine_tuned_clusters': [],
                'results': {},
                'summary': {},
                'total_clusters': 0
            }
        
        if not large_clusters:
            print(f"[WARNING]  No clusters with {min_size}+ embeddings for fine-tuning")
            return {
                'fine_tuned_clusters': [],
                'results': {},
                'summary': {},
                'total_clusters': 0
            }
        
        print(f"[*] Large clusters identified: {len(large_clusters)}")
        for cid, size, _ in large_clusters:
            print(f"  - Cluster {cid}: {size} embeddings")
        
        # Fine-tune each cluster
        print(f"\n[*] Starting fine-tuning ({num_epochs} additional epochs)...")
        print("-" * 70)
        
        fine_tune_results = {}
        device = getattr(config, 'DEVICE', torch.device("cpu"))
        criterion = nn.MSELoss()
        
        # Get or create optimizers (stored in pipeline state if available)
        optimizers = getattr(pipeline, '_optimizers', {})
        
        for cluster_id, cluster_size, node in large_clusters:
            try:
                start_time = time.time()
                print(f"\n📌 Cluster {cluster_id} ({cluster_size} embeddings):")
                
                # Extract embeddings safely
                embeddings = self._extract_embeddings_safely(node, cluster_id)
                if not embeddings:
                    print(f"  [WARNING]  No embeddings available")
                    continue
                
                if kwargs.get('verbose', False):
                    print(f"  [*] Embeddings extracted: {len(embeddings)}")
                
                # Get or create optimizer
                optimizer = optimizers.get(cluster_id, None)
                
                # Fine-tune using unified function
                try:
                    results = fine_tune_cluster_with_lora(
                        node=node,
                        embeddings=embeddings,
                        optimizer=optimizer,
                        criterion=criterion,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                        device=device
                    )
                except Exception as e:
                    print(f"  [ERROR] Error during fine-tuning: {e}")
                    if kwargs.get('verbose', False):
                        import traceback
                        traceback.print_exc()
                    continue
                
                if 'error' in results:
                    print(f"  [ERROR] Error in fine-tuning: {results['error']}")
                    continue
                
                # Store optimizer if created
                if cluster_id not in optimizers and 'error' not in results:
                    if 'optimizer' in results:
                        optimizers[cluster_id] = results['optimizer']
                
                # Calculate timing
                training_time = time.time() - start_time
                time_per_epoch = training_time / num_epochs if num_epochs > 0 else 0
                embeddings_per_second = len(embeddings) / training_time if training_time > 0 else 0
                
                # Store results
                fine_tune_results[cluster_id] = {
                    'size': cluster_size,
                    'initial_loss': results.get('initial_loss', 0.0),
                    'final_loss': results.get('final_loss', 0.0),
                    'improvement': results.get('improvement', 0.0),
                    'training_time': training_time,
                    'time_per_epoch': time_per_epoch,
                    'embeddings_per_second': embeddings_per_second
                }
                
                # Print summary
                improvement_pct = (
                    (results.get('improvement', 0.0) / results.get('initial_loss', 1.0) * 100)
                    if results.get('initial_loss', 0.0) > 0 else 0.0
                )
                print(f"  [OK] Improvement: {results.get('improvement', 0.0):.6f} ({improvement_pct:.2f}%)")
                print(f"     Loss: {results.get('initial_loss', 0.0):.6f} → {results.get('final_loss', 0.0):.6f}")
                
            except Exception as e:
                print(f"  [ERROR] Error processing cluster {cluster_id}: {e}")
                if kwargs.get('verbose', False):
                    import traceback
                    traceback.print_exc()
                continue
        
        # Store optimizers in pipeline state
        pipeline._optimizers = optimizers
        
        # Calculate summary statistics
        improvements = [r.get('improvement', 0.0) for r in fine_tune_results.values()]
        summary = {
            'avg_improvement': float(sum(improvements) / len(improvements)) if improvements else 0.0,
            'max_improvement': float(max(improvements)) if improvements else 0.0,
            'min_improvement': float(min(improvements)) if improvements else 0.0,
            'total_embeddings': sum(r.get('size', 0) for r in fine_tune_results.values()),
            'total_clusters': len(fine_tune_results)
        }
        
        results = {
            'fine_tuned_clusters': list(fine_tune_results.keys()),
            'results': fine_tune_results,
            'summary': summary,
            'total_clusters': len(fine_tune_results)
        }
        
        # Export results if requested
        if kwargs.get('export_results', False):
            output_path = kwargs.get('output_path', None)
            export_format = kwargs.get('export_format', 'json')
            self._export_results(results, output_path, export_format)
        
        pipeline.results['fine_tune'] = results
        
        print(f"\n[OK] Fine-tuning completed: {len(fine_tune_results)} clusters fine-tuned")
        print(f"   Average improvement: {summary['avg_improvement']:.6f}")
        
        return results
    
    def _extract_embeddings_safely(self, node, cluster_id):
        """Safely extract embeddings from a KnowledgeNode"""
        try:
            if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
                memory = node.filter.memory
                if isinstance(memory, dict):
                    return memory.get(cluster_id, [])
                elif isinstance(memory, list):
                    return memory
            return []
        except Exception:
            return []
    
    def _export_results(self, results: Dict, output_path: Optional[str], format: str):
        """Export fine-tuning results to file"""
        try:
            from datetime import datetime
            import json
            import pandas as pd
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"fine_tune_results_{timestamp}.{format}"
            
            if format == 'json':
                # Prepare data for JSON serialization
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_clusters': results['total_clusters'],
                    'results': {}
                }
                
                for cluster_id, r in results['results'].items():
                    initial_loss = r.get('initial_loss', 0.0)
                    improvement_pct = (
                        (r.get('improvement', 0.0) / initial_loss * 100)
                        if initial_loss > 0 else 0.0
                    )
                    
                    export_data['results'][str(cluster_id)] = {
                        'cluster_id': cluster_id,
                        'size': r.get('size', 0),
                        'initial_loss': float(r.get('initial_loss', 0.0)),
                        'final_loss': float(r.get('final_loss', 0.0)),
                        'improvement': float(r.get('improvement', 0.0)),
                        'improvement_pct': float(improvement_pct),
                        'training_time': float(r.get('training_time', 0.0)),
                        'time_per_epoch': float(r.get('time_per_epoch', 0.0)),
                        'embeddings_per_second': float(r.get('embeddings_per_second', 0.0))
                    }
                
                export_data['summary'] = results['summary']
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format == 'csv':
                df_data = []
                for cluster_id, r in results['results'].items():
                    initial_loss = r.get('initial_loss', 0.0)
                    improvement_pct = (
                        (r.get('improvement', 0.0) / initial_loss * 100)
                        if initial_loss > 0 else 0.0
                    )
                    
                    df_data.append({
                        'cluster_id': cluster_id,
                        'size': r.get('size', 0),
                        'initial_loss': initial_loss,
                        'final_loss': r.get('final_loss', 0.0),
                        'improvement': r.get('improvement', 0.0),
                        'improvement_pct': improvement_pct,
                        'training_time': r.get('training_time', 0.0),
                        'time_per_epoch': r.get('time_per_epoch', 0.0),
                        'embeddings_per_second': r.get('embeddings_per_second', 0.0)
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.to_csv(output_path, index=False)
            
            print(f"[OK] Results exported to: {output_path}")
        except Exception as e:
            print(f"[WARNING]  Error exporting results: {e}")

