"""
Tests for ClusteringStep (Phase 5)

Tests the ClusteringStep implementation and its integration with XctopusPipeline.
Note: These tests may require torch and other dependencies, so they may skip
if dependencies are not available.
"""

import unittest
import os
from xctopus.pipeline import XctopusPipeline
from xctopus.pipeline.steps.clustering import ClusteringStep
from xctopus.pipeline.steps import get_step


class TestClusteringStep(unittest.TestCase):
    """Test cases for ClusteringStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_clustering_step_creation(self):
        """Test that ClusteringStep can be instantiated"""
        step = ClusteringStep()
        self.assertIsNotNone(step)
    
    def test_get_required_steps(self):
        """Test that ClusteringStep has no required steps"""
        step = ClusteringStep()
        required = step.get_required_steps()
        self.assertEqual(required, [])
    
    def test_validate_inputs_missing_dataset(self):
        """Test validation fails when dataset_path is missing"""
        step = ClusteringStep()
        pipeline = XctopusPipeline()
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        
        self.assertIn("dataset_path is required", str(context.exception))
    
    def test_validate_inputs_valid(self):
        """Test validation passes with valid dataset"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        step = ClusteringStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        # Should not raise
        step.validate_inputs(pipeline, dataset_path=self.csv_path)
    
    def test_get_step_clustering(self):
        """Test that clustering step can be retrieved from registry"""
        step = get_step('clustering')
        self.assertIsInstance(step, ClusteringStep)


class TestClusteringStepIntegration(unittest.TestCase):
    """Test cases for ClusteringStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_clustering_basic(self):
        """Test running clustering step through pipeline.run()"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Run clustering step with minimal epochs for testing
        try:
            results = pipeline.run(
                step='clustering',
                epochs=1,  # Minimal epochs for faster testing
                enable_training=False,  # Disable training for faster tests
                enable_merge=False  # Disable merging for faster tests
            )
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn('knowledge_nodes', results)
            self.assertIn('embeddings', results)
            self.assertIn('total_clusters', results)
            
            # Verify state stored in pipeline
            self.assertGreater(len(pipeline.knowledge_nodes), 0)
            self.assertIsNotNone(pipeline.embeddings)
            self.assertIn('clustering', pipeline.results)
            
        except Exception as e:
            # If execution fails due to missing dependencies, skip test
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_pipeline_run_clustering_with_epochs(self):
        """Test clustering with custom number of epochs"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            results = pipeline.run(
                step='clustering',
                epochs=2,
                enable_training=False,
                enable_merge=False
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn('total_clusters', results)
            
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_clustering_stores_knowledge_nodes(self):
        """Test that clustering stores knowledge_nodes in pipeline state"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # Initially, knowledge_nodes should be empty
            self.assertEqual(len(pipeline.knowledge_nodes), 0)
            
            # Run clustering
            results = pipeline.run(
                step='clustering',
                epochs=1,
                enable_training=False,
                enable_merge=False
            )
            
            # After clustering, knowledge_nodes should be populated
            self.assertGreater(len(pipeline.knowledge_nodes), 0)
            self.assertEqual(
                len(pipeline.knowledge_nodes),
                results['total_clusters']
            )
            
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_clustering_stores_embeddings(self):
        """Test that clustering stores embeddings in pipeline state"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # Initially, embeddings should be None
            self.assertIsNone(pipeline.embeddings)
            
            # Run clustering
            results = pipeline.run(
                step='clustering',
                epochs=1,
                enable_training=False,
                enable_merge=False
            )
            
            # After clustering, embeddings should be stored
            self.assertIsNotNone(pipeline.embeddings)
            self.assertEqual(
                len(pipeline.embeddings),
                results['total_texts']
            )
            
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

