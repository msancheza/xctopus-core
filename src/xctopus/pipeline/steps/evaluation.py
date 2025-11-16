"""
Evaluation Step - Learning Performance Evaluation

Encapsulates the learning evaluation functionality from scripts 08 and 09.
Evaluates learning performance, LoRA contributions, and cluster quality.
"""

import torch
from typing import Dict, Any, Optional, List

from .base import PipelineStep
from xctopus.nodes.bayesian.utils import (
    LearningEvaluator,
    LoRAAuditor
)
from xctopus.nodes.bayesian.utils.cluster_utils import (
    evaluate_cluster_quality,
    extract_embeddings_from_nodes
)


class EvaluationStep(PipelineStep):
    """
    Step 7: Learning performance evaluation.
    
    Encapsulates: scripts/deprecated/08_learning_evaluate.py + 09_performance_evaluate.py (legacy)
    
    This step:
    1. Evaluates node performance with test embeddings
    2. Compares before/after training
    3. Evaluates LoRA contributions
    4. Evaluates cluster quality metrics
    5. Generates comprehensive learning report
    
    Requires: clustering step to be executed first.
    
    Modes:
    - 'learning': Focus on learning evaluation (default)
    - 'performance': Focus on performance metrics
    - 'full': Complete evaluation (learning + performance)
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        pipeline.run(step='clustering', epochs=5)
        results = pipeline.run(step='evaluation', mode='full')
    """
    
    def get_required_steps(self):
        """
        Evaluation step requires clustering to be executed first.
        
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
                "Clustering step must be executed before evaluation. "
                "Execute: pipeline.run(step='clustering')"
            )
        
        if not pipeline.knowledge_nodes:
            raise ValueError(
                "No knowledge nodes found. Clustering step must create nodes first."
            )
    
    def execute(
        self,
        pipeline,
        mode: str = 'learning',
        test_texts: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        train_epochs: int = 5,
        **kwargs
    ):
        """
        Execute learning performance evaluation.
        
        Args:
            pipeline: XctopusPipeline instance
            mode: Evaluation mode ('learning', 'performance', 'full') (default: 'learning')
            test_texts: Optional list of test texts (if None, uses dataset)
            dataset_path: Optional path to dataset for generating test texts
            train_epochs: Number of training epochs for before/after comparison (default: 5)
            **kwargs: Additional options:
                - use_existing_optimizers: Use existing optimizers from pipeline (default: True)
                - include_visualization: Include visualization in report (default: False)
                - verbose: Enable verbose output (default: True)
        
        Returns:
            dict: Evaluation results with keys:
                - 'node_performance': Performance metrics per node
                - 'before_after_comparison': Before/after training comparison
                - 'lora_contributions': LoRA contribution analysis
                - 'cluster_quality': Cluster quality metrics
                - 'learning_report': Complete learning report (if mode='learning' or 'full')
                - 'performance_metrics': Performance metrics (if mode='performance' or 'full')
        """
        self.validate_inputs(pipeline)
        
        knowledge_nodes = pipeline.knowledge_nodes
        config = pipeline.config
        
        if not knowledge_nodes:
            return {
                'node_performance': {},
                'before_after_comparison': {},
                'lora_contributions': {},
                'cluster_quality': {},
                'learning_report': {},
                'performance_metrics': {}
            }
        
        # Validate mode
        valid_modes = ['learning', 'performance', 'full']
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Valid modes: {valid_modes}"
            )
        
        print(f"[*] Running evaluation (mode: {mode})...")
        
        # Get or create TextPreprocessor
        text_preprocessor = pipeline._get_preprocessor()
        
        # Get test texts
        if test_texts is None:
            # Try to get from dataset
            if dataset_path or pipeline.dataset_path:
                test_texts = self._extract_test_texts(
                    dataset_path or pipeline.dataset_path,
                    pipeline,
                    kwargs.get('num_test_texts', 50)
                )
            else:
                # Use sample texts if no dataset available
                test_texts = [
                    "Sample text for testing learning performance",
                    "Another test text to evaluate node responses"
                ]
        
        if not test_texts:
            print("[WARNING]  No test texts available for evaluation")
            return {
                'node_performance': {},
                'before_after_comparison': {},
                'lora_contributions': {},
                'cluster_quality': {},
                'learning_report': {},
                'performance_metrics': {}
            }
        
        print(f"📝 Using {len(test_texts)} test texts for evaluation")
        
        # Initialize evaluators
        device = getattr(config, 'DEVICE', torch.device("cpu"))
        d_model = getattr(config, 'D_MODEL', 128)
        verbose = kwargs.get('verbose', True)
        learning_rate = getattr(config, 'LORA_LEARNING_RATE', 1e-4)
        
        learning_evaluator = LearningEvaluator(device=device, d_model=d_model, verbose=verbose)
        lora_auditor = LoRAAuditor(device=device, verbose=verbose)
        
        results = {
            'node_performance': {},
            'before_after_comparison': {},
            'lora_contributions': {},
            'cluster_quality': {},
            'learning_report': {},
            'performance_metrics': {}
        }
        
        # Learning evaluation (if mode is 'learning' or 'full')
        if mode in ['learning', 'full']:
            print("\n📚 Running learning evaluation...")
            
            try:
                # Prepare pipeline result structure for LearningEvaluator
                pipeline_result = {
                    'knowledge_nodes': knowledge_nodes,
                    'texts': test_texts,
                    'embeddings': pipeline.embeddings if hasattr(pipeline, 'embeddings') else None
                }
                
                # Generate learning report
                learning_report = learning_evaluator.generate_learning_report(
                    pipeline_result,
                    test_texts=test_texts,
                    train_epochs=train_epochs,
                    use_existing_optimizers=kwargs.get('use_existing_optimizers', True),
                    learning_rate=learning_rate
                )
                
                results['learning_report'] = learning_report
                
                # Compare before/after training
                try:
                    before_after = learning_evaluator.compare_before_after_training(
                        pipeline_result,
                        test_texts=test_texts,
                        train_epochs=train_epochs,
                        use_existing_optimizers=kwargs.get('use_existing_optimizers', True),
                        learning_rate=learning_rate
                    )
                    results['before_after_comparison'] = before_after
                except Exception as e:
                    if verbose:
                        print(f"  [WARNING]  Error in before/after comparison: {e}")
            
            except Exception as e:
                print(f"[ERROR] Error during learning evaluation: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Performance evaluation (if mode is 'performance' or 'full')
        if mode in ['performance', 'full']:
            print("\n⚡ Running performance evaluation...")
            
            try:
                # Evaluate node performance
                node_performance = {}
                test_embeddings = text_preprocessor.encode_texts(test_texts[:10])  # Use first 10 for performance
                
                for cluster_id, node in knowledge_nodes.items():
                    try:
                        # Get embeddings for this cluster
                        cluster_embeddings = self._get_cluster_embeddings(node, cluster_id)
                        if cluster_embeddings:
                            # Use cluster embeddings for evaluation
                            performance = learning_evaluator.evaluate_node_performance(
                                node,
                                cluster_embeddings[:5],  # Use first 5 for performance
                                node_id=str(cluster_id)
                            )
                            node_performance[cluster_id] = performance
                    except Exception as e:
                        if verbose:
                            print(f"  [WARNING]  Error evaluating node {cluster_id}: {e}")
                
                results['node_performance'] = node_performance
                
                # Evaluate LoRA contributions
                lora_contributions = {}
                for cluster_id, node in list(knowledge_nodes.items())[:5]:  # Limit to first 5 for performance
                    try:
                        # Capture initial LoRA state
                        initial_state = lora_auditor.capture_state(node)
                        
                        # Get a test embedding
                        if test_embeddings and len(test_embeddings) > 0:
                            test_emb = test_embeddings[0]
                            contribution = lora_auditor.compute_contribution(node, test_emb, initial_state)
                            lora_contributions[cluster_id] = contribution
                    except Exception as e:
                        if verbose:
                            print(f"  [WARNING]  Error evaluating LoRA contribution for {cluster_id}: {e}")
                
                results['lora_contributions'] = lora_contributions
                
                # Evaluate cluster quality
                try:
                    embeddings, labels, _ = extract_embeddings_from_nodes(knowledge_nodes)
                    if embeddings is not None and len(embeddings) > 0:
                        quality_metrics = evaluate_cluster_quality(embeddings, labels)
                        results['cluster_quality'] = quality_metrics
                        results['performance_metrics'] = {
                            'silhouette_score': quality_metrics.get('silhouette_score', 0.0),
                            'davies_bouldin_score': quality_metrics.get('davies_bouldin_score', 0.0),
                            'calinski_harabasz_score': quality_metrics.get('calinski_harabasz_score', 0.0)
                        }
                except Exception as e:
                    if verbose:
                        print(f"  [WARNING]  Error evaluating cluster quality: {e}")
            
            except Exception as e:
                print(f"[ERROR] Error during performance evaluation: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        pipeline.results['evaluation'] = results
        
        print("\n[OK] Evaluation completed")
        
        return results
    
    def _extract_test_texts(self, dataset_path: str, pipeline, num_texts: int = 50):
        """Extract test texts from dataset"""
        try:
            import pandas as pd
            import random
            
            df = pd.read_csv(dataset_path)
            
            # Use pipeline's text columns
            text_columns = pipeline.text_columns if hasattr(pipeline, 'text_columns') else ['text']
            join_with = pipeline.join_with if hasattr(pipeline, 'join_with') else '\n'
            
            # Extract texts
            texts = []
            for _, row in df.iterrows():
                text_parts = [
                    str(row[col]) for col in text_columns
                    if col in df.columns and pd.notna(row.get(col))
                ]
                if text_parts:
                    combined = join_with.join(text_parts)
                    if combined.strip():
                        texts.append(combined)
            
            # Sample random texts if we have more than needed
            if len(texts) > num_texts:
                texts = random.sample(texts, num_texts)
            
            return texts
        except Exception as e:
            print(f"[WARNING]  Error extracting test texts: {e}")
            return []
    
    def _get_cluster_embeddings(self, node, cluster_id):
        """Get embeddings from a cluster node"""
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

