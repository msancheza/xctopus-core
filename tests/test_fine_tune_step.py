"""
Tests for FineTuneStep (Phase 7, Sprint 7.2)
"""

import unittest
import os
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps.fine_tune import FineTuneStep
from xctopus.pipeline.steps import get_step


class TestFineTuneStep(unittest.TestCase):
    """Test cases for FineTuneStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_required_steps(self):
        """Test that fine_tune requires clustering"""
        step = FineTuneStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering'])
    
    def test_validate_inputs_missing_clustering(self):
        """Test that validate_inputs raises error if clustering not executed"""
        step = FineTuneStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_validate_inputs_no_nodes(self):
        """Test that validate_inputs raises error if no knowledge nodes"""
        step = FineTuneStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}  # Fake clustering result
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("knowledge nodes", str(context.exception).lower())
    
    def test_get_step_fine_tune(self):
        """Test that fine_tune step can be retrieved from registry"""
        step = get_step('fine_tune')
        self.assertIsInstance(step, FineTuneStep)


class TestFineTuneStepIntegration(unittest.TestCase):
    """Test cases for FineTuneStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_fine_tune_requires_clustering(self):
        """Test that fine_tune requires clustering to be executed first"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Try to run fine_tune without clustering
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='fine_tune')
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_pipeline_run_fine_tune_after_clustering(self):
        """Test that fine_tune works after clustering"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # First run clustering
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            
            # Then run fine_tune
            result = pipeline.run(step='fine_tune', min_size=1, num_epochs=1)
            
            # Verify results structure
            self.assertIsInstance(result, dict)
            self.assertIn('fine_tuned_clusters', result)
            self.assertIn('results', result)
            self.assertIn('summary', result)
            self.assertIn('fine_tune', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

