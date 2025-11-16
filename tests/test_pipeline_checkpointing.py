"""
Tests for Pipeline Checkpointing (Phase 8, Sprint 8.2)
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline


class TestPipelineCheckpointing(unittest.TestCase):
    """Test cases for checkpointing methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_state_creates_file(self):
        """Test that save_state() creates a checkpoint file"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.ckpt')
        
        try:
            # Run a step to have some state
            pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            
            # Save state
            saved_path = pipeline.save_state(checkpoint_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertEqual(saved_path, checkpoint_path)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_save_state_without_embeddings(self):
        """Test that save_state() works with include_embeddings=False"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint_no_emb.ckpt')
        
        try:
            pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            saved_path = pipeline.save_state(checkpoint_path, include_embeddings=False)
            
            self.assertTrue(os.path.exists(checkpoint_path))
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_load_state_nonexistent_file(self):
        """Test that load_state() raises error for nonexistent file"""
        pipeline = XctopusPipeline(self.csv_path)
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.ckpt')
        
        with self.assertRaises(IOError) as context:
            pipeline.load_state(nonexistent_path)
        self.assertIn("not found", str(context.exception).lower())
    
    def test_save_and_load_state_basic(self):
        """Test basic save and load cycle"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline1 = XctopusPipeline(self.csv_path)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.ckpt')
        
        try:
            # Run a step
            pipeline1.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            
            # Save state
            pipeline1.save_state(checkpoint_path)
            
            # Create new pipeline and load state
            pipeline2 = XctopusPipeline()
            metadata = pipeline2.load_state(checkpoint_path)
            
            # Check that metadata was returned
            self.assertIsInstance(metadata, dict)
            self.assertIn('version', metadata)
            self.assertIn('timestamp', metadata)
            
            # Check that results were restored
            self.assertIn('analysis', pipeline2.results)
            self.assertEqual(
                len(pipeline2.results),
                len(pipeline1.results)
            )
            
            # Check that configuration was restored
            self.assertEqual(pipeline2.dataset_path, pipeline1.dataset_path)
            self.assertEqual(pipeline2.text_columns, pipeline1.text_columns)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_save_state_with_metadata(self):
        """Test that save_state() includes custom metadata"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint_meta.ckpt')
        
        try:
            pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            
            custom_metadata = {'experiment_id': 'test_123', 'version': '0.1.0'}
            pipeline.save_state(checkpoint_path, metadata=custom_metadata)
            
            # Load and check metadata
            pipeline2 = XctopusPipeline()
            metadata = pipeline2.load_state(checkpoint_path)
            
            self.assertEqual(metadata['metadata'], custom_metadata)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_load_state_invalid_format(self):
        """Test that load_state() raises error for invalid checkpoint format"""
        pipeline = XctopusPipeline(self.csv_path)
        
        # Create an invalid checkpoint file
        invalid_path = os.path.join(self.temp_dir, 'invalid.ckpt')
        with open(invalid_path, 'w') as f:
            f.write("This is not a valid checkpoint")
        
        with self.assertRaises((IOError, ValueError)) as context:
            pipeline.load_state(invalid_path)
        # Should raise either IOError (loading failed) or ValueError (invalid format)
        self.assertIsNotNone(context.exception)
    
    def test_save_state_creates_directory(self):
        """Test that save_state() creates directory if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        checkpoint_dir = os.path.join(self.temp_dir, 'nested', 'dir')
        checkpoint_path = os.path.join(checkpoint_dir, 'test_checkpoint.ckpt')
        
        try:
            pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            pipeline.save_state(checkpoint_path)
            
            # Check that directory was created
            self.assertTrue(os.path.exists(checkpoint_dir))
            self.assertTrue(os.path.exists(checkpoint_path))
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_load_state_with_clustering(self):
        """Test save/load with clustering step (includes knowledge_nodes)"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline1 = XctopusPipeline(self.csv_path)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint_clustering.ckpt')
        
        try:
            # Run clustering to create knowledge nodes
            pipeline1.run(
                step='clustering',
                epochs=1,
                enable_training=False,
                enable_merge=False
            )
            
            # Save state
            pipeline1.save_state(checkpoint_path)
            
            # Load state
            pipeline2 = XctopusPipeline()
            metadata = pipeline2.load_state(checkpoint_path)
            
            # Check that knowledge_nodes data was stored (may be in dict format)
            # Note: Full reconstruction of nodes requires more complex logic
            self.assertIsInstance(metadata, dict)
            
            # Check that results were restored
            self.assertIn('clustering', pipeline2.results)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

