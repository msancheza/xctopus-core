"""
Tests for AuditStep (Phase 7, Sprint 7.2)
"""

import unittest
import os
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps.audit import AuditStep
from xctopus.pipeline.steps import get_step


class TestAuditStep(unittest.TestCase):
    """Test cases for AuditStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_required_steps(self):
        """Test that audit requires clustering"""
        step = AuditStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering'])
    
    def test_validate_inputs_missing_clustering(self):
        """Test that validate_inputs raises error if clustering not executed"""
        step = AuditStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_validate_inputs_no_nodes(self):
        """Test that validate_inputs raises error if no knowledge nodes"""
        step = AuditStep()
        pipeline = XctopusPipeline(self.csv_path)
        pipeline.results['clustering'] = {}  # Fake clustering result
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        self.assertIn("knowledge nodes", str(context.exception).lower())
    
    def test_get_step_audit(self):
        """Test that audit step can be retrieved from registry"""
        step = get_step('audit')
        self.assertIsInstance(step, AuditStep)


class TestAuditStepIntegration(unittest.TestCase):
    """Test cases for AuditStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_audit_requires_clustering(self):
        """Test that audit requires clustering to be executed first"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Try to run audit without clustering
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='audit')
        self.assertIn("clustering", str(context.exception).lower())
    
    def test_pipeline_run_audit_after_clustering(self):
        """Test that audit works after clustering"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        try:
            # First run clustering
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            
            # Then run audit with test texts
            test_texts = ['Test text 1', 'Test text 2']
            result = pipeline.run(step='audit', test_texts=test_texts)
            
            # Verify results structure
            self.assertIsInstance(result, dict)
            self.assertIn('node_responses', result)
            self.assertIn('cluster_quality', result)
            self.assertIn('report', result)
            self.assertIn('audit', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

