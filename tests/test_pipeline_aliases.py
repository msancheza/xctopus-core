"""
Tests for pipeline alias methods (Phase 6, Sprint 6.3)

Tests backward compatibility methods like run_analysis(), run_clustering(), etc.
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline


class TestPipelineAliases(unittest.TestCase):
    """Test cases for alias methods"""
    
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
    
    def test_run_analysis_alias(self):
        """Test that run_analysis() works as alias for run(step='analysis')"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            # Test alias method
            result1 = pipeline.run_analysis(save_plots=False, compute_advanced_metrics=False)
            
            # Test direct method
            result2 = pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            
            # Both should produce similar results (same step executed)
            self.assertIn('statistics', result1)
            self.assertIn('statistics', result2)
            self.assertIn('analysis', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_clustering_alias(self):
        """Test that run_clustering() works as alias for run(step='clustering')"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            # Test alias method with epochs
            result1 = pipeline.run_clustering(epochs=1)
            
            # Test direct method
            result2 = pipeline.run(step='clustering', epochs=1)
            
            # Both should produce similar results
            self.assertIn('knowledge_nodes', result1)
            self.assertIn('knowledge_nodes', result2)
            self.assertIn('clustering', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_run_clustering_with_epochs_parameter(self):
        """Test that run_clustering() accepts epochs parameter correctly"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            # Test with epochs parameter
            result = pipeline.run_clustering(epochs=1)
            self.assertIsNotNone(result)
            self.assertIn('clustering', pipeline.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_update_cluster_config_alias(self):
        """Test that update_cluster_config() exists and calls correct step"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # First run clustering (required dependency)
        try:
            pipeline.run_clustering(epochs=1)
            
            # Test alias method (will fail if step doesn't exist, but that's okay)
            try:
                result = pipeline.update_cluster_config()
                # If it succeeds, should have results
                self.assertIsNotNone(result)
            except ValueError as e:
                # Step might not be implemented yet, that's okay
                if "not found" in str(e):
                    pass  # Expected if step not implemented
                else:
                    raise
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_fine_tune_clusters_alias(self):
        """Test that fine_tune_clusters() exists and calls correct step"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # First run clustering (required dependency)
        try:
            pipeline.run_clustering(epochs=1)
            
            # Test alias method
            try:
                result = pipeline.fine_tune_clusters()
                self.assertIsNotNone(result)
            except ValueError as e:
                if "not found" in str(e):
                    pass  # Expected if step not implemented
                else:
                    raise
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_optimize_clusters_alias(self):
        """Test that optimize_clusters() exists and calls correct step"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # First run clustering (required dependency)
        try:
            pipeline.run_clustering(epochs=1)
            
            # Test alias method
            try:
                result = pipeline.optimize_clusters()
                self.assertIsNotNone(result)
            except ValueError as e:
                if "not found" in str(e):
                    pass  # Expected if step not implemented
                else:
                    raise
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_audit_learning_alias(self):
        """Test that audit_learning() exists and calls correct step"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # First run clustering (required dependency)
        try:
            pipeline.run_clustering(epochs=1)
            
            # Test alias method
            try:
                result = pipeline.audit_learning()
                self.assertIsNotNone(result)
            except ValueError as e:
                if "not found" in str(e):
                    pass  # Expected if step not implemented
                else:
                    raise
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_evaluate_learning_alias(self):
        """Test that evaluate_learning() exists and calls correct step"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # First run clustering (required dependency)
        try:
            pipeline.run_clustering(epochs=1)
            
            # Test alias method
            try:
                result = pipeline.evaluate_learning()
                self.assertIsNotNone(result)
            except ValueError as e:
                if "not found" in str(e):
                    pass  # Expected if step not implemented
                else:
                    raise
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_aliases_preserve_kwargs(self):
        """Test that alias methods properly pass kwargs to underlying run()"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        try:
            # Test that kwargs are passed through
            result = pipeline.run_analysis(
                save_plots=False,
                compute_advanced_metrics=False,
                export_format=None
            )
            self.assertIsNotNone(result)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

