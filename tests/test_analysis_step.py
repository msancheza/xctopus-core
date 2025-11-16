"""
Tests for AnalysisStep (Phase 4)

Tests the AnalysisStep implementation and its integration with XctopusPipeline.
"""

import unittest
import os
from xctopus.pipeline import XctopusPipeline
from xctopus.pipeline.steps.analysis import AnalysisStep
from xctopus.pipeline.steps import get_step


class TestAnalysisStep(unittest.TestCase):
    """Test cases for AnalysisStep"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_analysis_step_creation(self):
        """Test that AnalysisStep can be instantiated"""
        step = AnalysisStep()
        self.assertIsNotNone(step)
    
    def test_get_required_steps(self):
        """Test that AnalysisStep has no required steps"""
        step = AnalysisStep()
        required = step.get_required_steps()
        self.assertEqual(required, [])
    
    def test_validate_inputs_missing_dataset(self):
        """Test validation fails when dataset_path is missing"""
        step = AnalysisStep()
        pipeline = XctopusPipeline()
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline)
        
        self.assertIn("dataset_path is required", str(context.exception))
    
    def test_validate_inputs_invalid_file(self):
        """Test validation fails when dataset file doesn't exist"""
        step = AnalysisStep()
        pipeline = XctopusPipeline()
        
        with self.assertRaises(ValueError) as context:
            step.validate_inputs(pipeline, dataset_path='nonexistent.csv')
        
        self.assertIn("not found", str(context.exception))
    
    def test_validate_inputs_valid(self):
        """Test validation passes with valid dataset"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        step = AnalysisStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        # Should not raise
        step.validate_inputs(pipeline, dataset_path=self.csv_path)
    
    def test_get_step_analysis(self):
        """Test that analysis step can be retrieved from registry"""
        step = get_step('analysis')
        self.assertIsInstance(step, AnalysisStep)
    
    def test_execute_analysis_basic(self):
        """Test basic execution of analysis step"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        step = AnalysisStep()
        pipeline = XctopusPipeline(self.csv_path)
        
        # Execute analysis (may take time, so we'll just verify it doesn't crash)
        # In a real scenario, we'd mock ClusterAnalyzer for faster tests
        try:
            results = step.execute(
                dataset_path=self.csv_path,
                pipeline=pipeline,
                save_plots=False,  # Skip plots for faster tests
                compute_advanced_metrics=False  # Skip advanced metrics for faster tests
            )
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn('statistics', results)
            self.assertIn('num_clusters', results['statistics'])
            
            # Verify results stored in pipeline
            self.assertIn('analysis', pipeline.results)
            self.assertEqual(pipeline.results['analysis'], results)
        except Exception as e:
            # If execution fails due to missing dependencies (torch, etc.),
            # that's okay for now - we're just testing the structure
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


class TestAnalysisStepIntegration(unittest.TestCase):
    """Test cases for AnalysisStep integration with XctopusPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_pipeline_run_analysis(self):
        """Test running analysis step through pipeline.run()"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Run analysis step
        try:
            results = pipeline.run(
                step='analysis',
                save_plots=False,
                compute_advanced_metrics=False
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertIn('statistics', results)
            
            # Verify results stored in pipeline
            self.assertIn('analysis', pipeline.results)
        except Exception as e:
            # If execution fails due to missing dependencies, skip test
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_pipeline_run_analysis_with_dataset_path(self):
        """Test running analysis with explicit dataset_path"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline()  # No dataset_path in init
        
        # Run analysis with explicit dataset_path
        try:
            results = pipeline.run(
                step='analysis',
                dataset_path=self.csv_path,
                save_plots=False,
                compute_advanced_metrics=False
            )
            
            self.assertIsInstance(results, dict)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_pipeline_run_analysis_no_step(self):
        """Test that run() without step raises NotImplementedError"""
        pipeline = XctopusPipeline()
        
        with self.assertRaises(NotImplementedError):
            pipeline.run()  # Full pipeline not yet implemented
    
    def test_pipeline_run_invalid_step(self):
        """Test that run() with invalid step raises ValueError"""
        pipeline = XctopusPipeline()
        
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='nonexistent_step')
        
        self.assertIn("not found", str(context.exception))


if __name__ == '__main__':
    unittest.main()

