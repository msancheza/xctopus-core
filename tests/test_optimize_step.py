"""
Tests for OptimizeStep (Phase 7, Sprint 7.1)
"""

import unittest
import os
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps.optimize import OptimizeStep
from xctopus.pipeline.steps import get_step


class TestOptimizeStep(unittest.TestCase):
    """Test cases for OptimizeStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_required_steps(self):
        """Test that optimize requires clustering"""
        step = OptimizeStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering'])
    
    def test_validate_inputs_missing_clustering(self):
        """Test that validate_inputs raises error if clustering not executed"""
        step = OptimizeStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_validate_inputs_no_nodes(self):
        """Test that validate_inputs raises error if no knowledge nodes"""
        step = OptimizeStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}  # Fake clustering result
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("knowledge nodes", str(context.exception).lower())
    
    def test_get_step_optimize(self):
        """Test that optimize step can be retrieved from registry"""
        step = get_step('optimize')
        self.assertIsInstance(step, OptimizeStep)


class TestOptimizeStepIntegration(unittest.TestCase):
    """Test cases for OptimizeStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_optimize_requires_clustering(self):
        """Test that optimize requires clustering to be executed first"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Try to run optimize without clustering
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='optimize')
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_pipeline_run_optimize_after_clustering(self):
        """Test that optimize works after clustering"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # First run clustering
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            
            # Then run optimize
            result = pipeline.run(step='optimize', auto_merge=False)
            
            # Verify results structure
            self.assertIsInstance(result, dict)
            self.assertIn('distribution', result)
            self.assertIn('recommendations', result)
            self.assertIn('merge_count', result)
            self.assertIn('optimize', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

