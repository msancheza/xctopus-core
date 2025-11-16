"""
Tests for run_full_pipeline() method (Phase 6, Sprint 6.2)

Tests full pipeline execution with skip flags and correct execution order.
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline


class TestRunFullPipeline(unittest.TestCase):
    """Test cases for run_full_pipeline() method"""
    
    def setUp(self):
        """Create a temporary dataset for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create a simple test dataset
        df = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2', 'Sample text 3'],
            'id': [1, 2, 3]
        })
        df.to_csv(self.dataset_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_run_full_pipeline_missing_dataset(self):
        """Test that run_full_pipeline() raises ValueError when dataset_path is missing"""
        pipeline = XctopusPipeline()  # No dataset_path
        
        with self.assertRaises(ValueError) as context:
            pipeline.run_full_pipeline()
        self.assertIn("dataset_path is required", str(context.exception))
    
    def test_run_full_pipeline_with_dataset_path(self):
        """Test that run_full_pipeline() accepts dataset_path parameter"""
        pipeline = XctopusPipeline()  # No dataset_path in init
        
        # Should work with dataset_path in run_full_pipeline()
        # This will fail at clustering if dependencies are missing, but should not fail at validation
        try:
            result = pipeline.run_full_pipeline(
                dataset_path=self.dataset_path,
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1  # Minimal epochs for testing
            )
            # Should have clustering results
            self.assertIn('clustering', result)
            self.assertIn('summary', result)
        except Exception as e:
            # If it fails due to missing dependencies (torch, etc.), that's okay
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_full_pipeline_skip_analysis(self):
        """Test that skip_analysis flag works correctly"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            result = pipeline.run_full_pipeline(
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1
            )
            
            # Should not have analysis results
            self.assertNotIn('analysis', result)
            # Should have clustering results
            self.assertIn('clustering', result)
            # Summary should reflect skipped steps
            self.assertIn('analysis', result['summary']['skipped_steps'])
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_full_pipeline_with_analysis(self):
        """Test that run_full_pipeline() includes analysis when not skipped"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            result = pipeline.run_full_pipeline(
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1,
                save_plots=False,  # Skip plots for faster tests
                compute_advanced_metrics=False  # Skip advanced metrics for faster tests
            )
            
            # Should have analysis results
            self.assertIn('analysis', result)
            # Should have clustering results
            self.assertIn('clustering', result)
            # Summary should include analysis in executed steps
            self.assertIn('analysis', result['summary']['executed_steps'])
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_full_pipeline_summary(self):
        """Test that run_full_pipeline() returns a proper summary"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            result = pipeline.run_full_pipeline(
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1
            )
            
            # Check summary structure
            self.assertIn('summary', result)
            summary = result['summary']
            self.assertIn('executed_steps', summary)
            self.assertIn('total_steps', summary)
            self.assertIn('skipped_steps', summary)
            self.assertIn('knowledge_nodes_count', summary)
            self.assertIn('clusters_created', summary)
            
            # Should have at least clustering executed
            self.assertIn('clustering', summary['executed_steps'])
            self.assertGreaterEqual(summary['total_steps'], 1)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_full_pipeline_clustering_required(self):
        """Test that clustering step is always executed (cannot be skipped)"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            result = pipeline.run_full_pipeline(
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1
            )
            
            # Clustering should always be present
            self.assertIn('clustering', result)
            self.assertIn('clustering', result['summary']['executed_steps'])
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_without_step_calls_full_pipeline(self):
        """Test that pipeline.run() without step calls run_full_pipeline()"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            # This should call run_full_pipeline() internally
            result = pipeline.run(
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True,
                epochs=1
            )
            
            # Should have summary (indicates full pipeline was run)
            self.assertIn('summary', result)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

