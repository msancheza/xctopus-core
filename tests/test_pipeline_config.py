"""
Tests for PipelineConfig (Phase 0)

Tests basic functionality of PipelineConfig class.
"""

import unittest
from xctopus.pipeline.config import PipelineConfig


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig"""
    
    def test_config_initialization(self):
        """Test that PipelineConfig can be instantiated with defaults"""
        config = PipelineConfig()
        
        # Check that config object exists
        self.assertIsNotNone(config)
        
        # Check default values
        self.assertIsNone(config.TEXT_COLUMNS)
        self.assertEqual(config.JOIN_WITH, "\n")
        self.assertIsNone(config.LABEL_COLUMN)
        self.assertIsNone(config.ID_COLUMN)
        self.assertFalse(config.DROP_EMPTY)
        
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.MAX_LENGTH, 512)
        self.assertTrue(config.NORMALIZE_EMBEDDINGS)
        
        self.assertEqual(config.NUM_EPOCHS, 5)
        self.assertEqual(config.LEARNING_RATE, 0.001)
        self.assertTrue(config.ENABLE_FINE_TUNE_LARGE)
        self.assertTrue(config.ENABLE_AUTO_UPDATE)
        
        self.assertEqual(config.MIN_CLUSTER_SIZE, 8)
        self.assertEqual(config.ORPHAN_THRESHOLD, 3)
        self.assertEqual(config.MERGE_SIMILARITY_THRESHOLD, 0.7)
        
        self.assertTrue(config.ENABLE_AUDIT)
        self.assertTrue(config.ENABLE_PERFORMANCE_EVAL)
        
        self.assertTrue(config.AUTO_DETECT_TEXT_COLUMNS)
        self.assertIsNone(config.SUGGESTED_TEXT_COLUMNS)
        self.assertTrue(config.VALIDATE_COLUMNS)
        self.assertFalse(config.STRICT_MODE)
    
    def test_config_to_dict(self):
        """Test that to_dict() returns all configuration attributes"""
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        # Check that dict is not empty
        self.assertIsInstance(config_dict, dict)
        self.assertGreater(len(config_dict), 0)
        
        # Check that all expected keys are present
        expected_keys = [
            'TEXT_COLUMNS', 'JOIN_WITH', 'LABEL_COLUMN', 'ID_COLUMN', 'DROP_EMPTY',
            'EMBEDDING_MODEL', 'MAX_LENGTH', 'NORMALIZE_EMBEDDINGS',
            'NUM_EPOCHS', 'LEARNING_RATE', 'ENABLE_FINE_TUNE_LARGE', 'ENABLE_AUTO_UPDATE',
            'MIN_CLUSTER_SIZE', 'ORPHAN_THRESHOLD', 'MERGE_SIMILARITY_THRESHOLD',
            'ENABLE_AUDIT', 'ENABLE_PERFORMANCE_EVAL',
            'AUTO_DETECT_TEXT_COLUMNS', 'SUGGESTED_TEXT_COLUMNS', 'VALIDATE_COLUMNS', 'STRICT_MODE'
        ]
        
        for key in expected_keys:
            self.assertIn(key, config_dict, f"Key '{key}' missing from config dict")
    
    def test_config_update(self):
        """Test that update() method modifies configuration correctly"""
        config = PipelineConfig()
        
        # Update some values
        config.update(
            NUM_EPOCHS=10,
            LEARNING_RATE=0.0001,
            TEXT_COLUMNS=["title", "abstract"]
        )
        
        # Check that values were updated
        self.assertEqual(config.NUM_EPOCHS, 10)
        self.assertEqual(config.LEARNING_RATE, 0.0001)
        self.assertEqual(config.TEXT_COLUMNS, ["title", "abstract"])
        
        # Check that other values remain unchanged
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.MAX_LENGTH, 512)
    
    def test_config_update_invalid_attribute(self):
        """Test that update() raises error for invalid attributes"""
        config = PipelineConfig()
        
        with self.assertRaises(AttributeError) as context:
            config.update(INVALID_ATTRIBUTE=123)
        
        self.assertIn("has no attribute 'INVALID_ATTRIBUTE'", str(context.exception))
    
    def test_config_independence(self):
        """Test that multiple config instances are independent"""
        config1 = PipelineConfig()
        config2 = PipelineConfig()
        
        # Modify config1
        config1.update(NUM_EPOCHS=10)
        
        # config2 should still have default value
        self.assertEqual(config1.NUM_EPOCHS, 10)
        self.assertEqual(config2.NUM_EPOCHS, 5)


if __name__ == '__main__':
    unittest.main()

