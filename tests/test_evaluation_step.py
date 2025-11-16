"""
Tests for EvaluationStep (Phase 7, Sprint 7.3)
"""

import unittest
import os
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps.evaluation import EvaluationStep
from xctopus.pipeline.steps import get_step


class TestEvaluationStep(unittest.TestCase):
    """Test cases for EvaluationStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_required_steps(self):
        """Test that evaluation requires clustering"""
        step = EvaluationStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering'])
    
    def test_validate_inputs_missing_clustering(self):
        """Test that validate_inputs raises error if clustering not executed"""
        step = EvaluationStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_validate_inputs_no_nodes(self):
        """Test that validate_inputs raises error if no knowledge nodes"""
        step = EvaluationStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}  # Fake clustering result
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("knowledge nodes", str(context.exception).lower())
    
    def test_get_step_evaluation(self):
        """Test that evaluation step can be retrieved from registry"""
        step = get_step('evaluation')
        self.assertIsInstance(step, EvaluationStep)
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError"""
        step = EvaluationStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}
        pipeline.knowledge_nodes = {0: None}  # Fake node
        
        with self.assertRaises(ValueError) as context:
            step.execute(pipeline, mode='invalid_mode')
        self.assertIn("Invalid mode", str(context.exception))


class TestEvaluationStepIntegration(unittest.TestCase):
    """Test cases for EvaluationStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_evaluation_requires_clustering(self):
        """Test that evaluation requires clustering to be executed first"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Try to run evaluation without clustering
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='evaluation')
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_pipeline_run_evaluation_after_clustering(self):
        """Test that evaluation works after clustering"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # First run clustering
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            
            # Then run evaluation with different modes
            for mode in ['learning', 'performance', 'full']:
                test_texts = ['Test text 1', 'Test text 2']
                result = pipeline.run(step='evaluation', mode=mode, test_texts=test_texts)
                
                # Verify results structure
                self.assertIsInstance(result, dict)
                self.assertIn('node_performance', result)
                self.assertIn('cluster_quality', result)
                self.assertIn('evaluation', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

