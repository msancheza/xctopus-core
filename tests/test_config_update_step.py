"""
Tests for ConfigUpdateStep (Phase 7, Sprint 7.1)
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps.config_update import ConfigUpdateStep
from xctopus.pipeline.steps import get_step


class TestConfigUpdateStep(unittest.TestCase):
    """Test cases for ConfigUpdateStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_required_steps(self):
        """Test that config_update requires clustering"""
        step = ConfigUpdateStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering'])
    
    def test_validate_inputs_missing_clustering(self):
        """Test that validate_inputs raises error if clustering not executed"""
        step = ConfigUpdateStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_validate_inputs_no_nodes(self):
        """Test that validate_inputs raises error if no knowledge nodes"""
        step = ConfigUpdateStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}  # Fake clustering result
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("knowledge nodes", str(context.exception).lower())
    
    def test_get_step_config_update(self):
        """Test that config_update step can be retrieved from registry"""
        step = get_step('config_update')
        self.assertIsInstance(step, ConfigUpdateStep)


class TestConfigUpdateStepIntegration(unittest.TestCase):
    """Test cases for ConfigUpdateStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_config_update_requires_clustering(self):
        """Test that config_update requires clustering to be executed first"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Try to run config_update without clustering
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='config_update')
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_pipeline_run_config_update_after_clustering(self):
        """Test that config_update works after clustering"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # First run clustering
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            
            # Then run config_update
            result = pipeline.run(step='config_update')
            
            # Verify results structure
            self.assertIsInstance(result, dict)
            self.assertIn('updated_clusters', result)
            self.assertIn('total_updated', result)
            self.assertIn('config_update', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

