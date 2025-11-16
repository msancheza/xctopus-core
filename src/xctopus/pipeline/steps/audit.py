"""
Audit Step - Learning Effectiveness Audit

Encapsulates the learning audit functionality from script 07.
Evaluates how KnowledgeNodes respond to new texts and analyzes learning effectiveness.
"""

import torch
from typing import Dict, Any, Optional, List

from .base import PipelineStep
from xctopus.nodes.bayesian.utils import LearningAuditor


class AuditStep(PipelineStep):
    """
    Step 6: Learning effectiveness audit.
    
    Encapsulates: scripts/deprecated/07_learning_audit.py (legacy)
    
    This step:
    1. Evaluates how KnowledgeNodes respond to new texts
    2. Analyzes cluster quality and memory distribution
    3. Compares outputs before/after training
    4. Generates comprehensive audit report
    
    Requires: clustering step to be executed first.
    
    Example:
        pipeline = XctopusPipeline('data.csv')
        pipeline.run(step='clustering', epochs=5)
        results = pipeline.run(step='audit', test_texts=['text1', 'text2'])
    """
    
    def get_required_steps(self):
        """
        Audit step requires clustering to be executed first.
        
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
                "Clustering step must be executed before audit. "
                "Execute: pipeline.run(step='clustering')"
            )
        
        if not pipeline.knowledge_nodes:
            raise ValueError(
                "No knowledge nodes found. Clustering step must create nodes first."
            )
    
    def execute(
        self,
        pipeline,
        test_texts: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        **kwargs
    ):
        """
        Execute learning effectiveness audit.
        
        Args:
            pipeline: XctopusPipeline instance
            test_texts: Optional list of test texts for audit (if None, uses dataset)
            dataset_path: Optional path to dataset for generating test texts
            **kwargs: Additional options:
                - include_visualization: Include visualization in report (default: True)
                - num_iterations: Number of iterations for effectiveness test (default: 3)
                - verbose: Enable verbose output (default: True)
        
        Returns:
            dict: Audit results with keys:
                - 'node_responses': Response metrics per node
                - 'cluster_quality': Cluster quality analysis
                - 'before_after_comparison': Comparison results
                - 'learning_effectiveness': Effectiveness test results
                - 'report': Complete audit report
        """
        self.validate_inputs(pipeline)
        
        knowledge_nodes = pipeline.knowledge_nodes
        config = pipeline.config
        
        if not knowledge_nodes:
            return {
                'node_responses': {},
                'cluster_quality': {},
                'before_after_comparison': {},
                'learning_effectiveness': {},
                'report': {}
            }
        
        print("[*] Running learning effectiveness audit...")
        
        # Get or create TextPreprocessor
        text_preprocessor = pipeline._get_preprocessor()
        
        # Get test texts
        if test_texts is None:
            # Try to get from dataset
            if dataset_path or pipeline.dataset_path:
                test_texts = self._extract_test_texts(
                    dataset_path or pipeline.dataset_path,
                    pipeline,
                    kwargs.get('num_test_texts', 10)
                )
            else:
                # Use sample texts if no dataset available
                test_texts = [
                    "Sample text for testing learning effectiveness",
                    "Another test text to evaluate node responses"
                ]
        
        if not test_texts:
            print("[WARNING]  No test texts available for audit")
            return {
                'node_responses': {},
                'cluster_quality': {},
                'before_after_comparison': {},
                'learning_effectiveness': {},
                'report': {}
            }
        
        print(f"📝 Using {len(test_texts)} test texts for audit")
        
        # Get filter node from first knowledge node
        filter_node = None
        if knowledge_nodes:
            first_node = next(iter(knowledge_nodes.values()))
            if hasattr(first_node, 'filter'):
                filter_node = first_node.filter
        
        if filter_node is None:
            print("[WARNING]  No filter node available for audit")
            return {
                'node_responses': {},
                'cluster_quality': {},
                'before_after_comparison': {},
                'learning_effectiveness': {},
                'report': {}
            }
        
        # Initialize LearningAuditor
        device = getattr(config, 'DEVICE', torch.device("cpu"))
        d_model = getattr(config, 'D_MODEL', 128)
        verbose = kwargs.get('verbose', True)
        
        auditor = LearningAuditor(device=device, d_model=d_model, verbose=verbose)
        
        # Generate comprehensive audit report
        try:
            print("\n[*] Generating audit report...")
            report = auditor.generate_audit_report(
                knowledge_nodes,
                test_texts,
                text_preprocessor,
                filter_node,
                include_visualization=kwargs.get('include_visualization', True)
            )
            
            # Extract individual components
            node_responses = {}
            for cluster_id, node in knowledge_nodes.items():
                try:
                    # Get embeddings for test texts
                    test_embeddings = text_preprocessor.encode_texts(test_texts[:1])  # Use first text
                    if len(test_embeddings) > 0:
                        response = auditor.evaluate_node_response(
                            node,
                            test_embeddings[0],
                            str(cluster_id)
                        )
                        node_responses[cluster_id] = response
                except Exception as e:
                    if verbose:
                        print(f"  [WARNING]  Error evaluating node {cluster_id}: {e}")
            
            # Analyze cluster quality
            try:
                cluster_quality = auditor.analyze_cluster_quality(knowledge_nodes)
            except Exception as e:
                if verbose:
                    print(f"  [WARNING]  Error analyzing cluster quality: {e}")
                cluster_quality = {}
            
            # Compare before/after (if applicable)
            before_after_comparison = {}
            try:
                test_embeddings = text_preprocessor.encode_texts(test_texts)
                before_after_comparison = auditor.compare_before_after(
                    knowledge_nodes,
                    test_texts,
                    text_preprocessor
                )
            except Exception as e:
                if verbose:
                    print(f"  [WARNING]  Error in before/after comparison: {e}")
            
            # Test learning effectiveness
            learning_effectiveness = {}
            try:
                num_iterations = kwargs.get('num_iterations', 3)
                learning_effectiveness = auditor.test_learning_effectiveness(
                    knowledge_nodes,
                    test_texts,
                    text_preprocessor,
                    filter_node,
                    num_iterations=num_iterations
                )
            except Exception as e:
                if verbose:
                    print(f"  [WARNING]  Error testing learning effectiveness: {e}")
            
            results = {
                'node_responses': node_responses,
                'cluster_quality': cluster_quality,
                'before_after_comparison': before_after_comparison,
                'learning_effectiveness': learning_effectiveness,
                'report': report
            }
            
            pipeline.results['audit'] = results
            
            print("\n[OK] Audit completed")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Error during audit: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return {
                'node_responses': {},
                'cluster_quality': {},
                'before_after_comparison': {},
                'learning_effectiveness': {},
                'report': {}
            }
    
    def _extract_test_texts(self, dataset_path: str, pipeline, num_texts: int = 10):
        """Extract test texts from dataset"""
        try:
            import pandas as pd
            
            df = pd.read_csv(dataset_path)
            
            # Use pipeline's text columns
            text_columns = pipeline.text_columns if hasattr(pipeline, 'text_columns') else ['text']
            join_with = pipeline.join_with if hasattr(pipeline, 'join_with') else '\n'
            
            # Extract texts
            texts = []
            for _, row in df.head(num_texts).iterrows():
                text_parts = [
                    str(row[col]) for col in text_columns
                    if col in df.columns and pd.notna(row.get(col))
                ]
                if text_parts:
                    combined = join_with.join(text_parts)
                    if combined.strip():
                        texts.append(combined)
            
            return texts
        except Exception as e:
            print(f"[WARNING]  Error extracting test texts: {e}")
            return []

