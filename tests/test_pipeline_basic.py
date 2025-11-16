"""
Tests for XctopusPipeline basic functionality (Phase 3)

Tests initialization, configuration loading, auto-detection, and basic methods.
"""

import unittest
import os
import tempfile
from xctopus.pipeline import XctopusPipeline, PipelineConfig


class TestXctopusPipelineInitialization(unittest.TestCase):
    """Test cases for XctopusPipeline initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
        self.yaml_path = os.path.join(self.test_dir, 'test_config_yaml.yaml')
    
    def test_init_with_defaults(self):
        """Test initialization with default configuration"""
        pipeline = XctopusPipeline()
        
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline.config, PipelineConfig)
        self.assertIsNone(pipeline.dataset_path)
        self.assertEqual(pipeline.text_columns, ['text'])  # Default fallback
        self.assertEqual(pipeline.knowledge_nodes, {})
        self.assertEqual(pipeline.results, {})
        self.assertIsNone(pipeline.embeddings)
    
    def test_init_with_dataset_path(self):
        """Test initialization with dataset path"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        self.assertEqual(pipeline.dataset_path, self.csv_path)
        self.assertIsNotNone(pipeline.text_columns)
        # Should auto-detect 'title' and 'abstract'
        self.assertGreater(len(pipeline.text_columns), 0)
    
    def test_init_with_config_object(self):
        """Test initialization with PipelineConfig object"""
        config = PipelineConfig()
        config.NUM_EPOCHS = 10
        config.LEARNING_RATE = 0.0001
        
        pipeline = XctopusPipeline(config=config)
        
        self.assertEqual(pipeline.config.NUM_EPOCHS, 10)
        self.assertEqual(pipeline.config.LEARNING_RATE, 0.0001)
    
    def test_init_with_yaml_config(self):
        """Test initialization with YAML config file"""
        if not os.path.exists(self.yaml_path):
            self.skipTest(f"Test YAML file not found: {self.yaml_path}")
        
        pipeline = XctopusPipeline(config=self.yaml_path)
        
        # Check that config was loaded from YAML
        self.assertEqual(pipeline.config.NUM_EPOCHS, 10)
        self.assertEqual(pipeline.config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
    
    def test_init_with_kwargs_override(self):
        """Test that kwargs override config values"""
        config = PipelineConfig()
        config.NUM_EPOCHS = 5
        
        pipeline = XctopusPipeline(
            config=config,
            text_columns=['title', 'abstract'],
            model_name='custom-model'
        )
        
        # kwargs should override config
        self.assertEqual(pipeline.text_columns, ['title', 'abstract'])
        self.assertEqual(pipeline.model_name, 'custom-model')
        # Config value should remain unchanged
        self.assertEqual(pipeline.config.NUM_EPOCHS, 5)
    
    def test_init_auto_detect_columns(self):
        """Test auto-detection of text columns"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(
            self.csv_path,
            text_columns=None,  # Trigger auto-detection
            auto_detect_text_columns=True
        )
        
        # Should detect 'title' and 'abstract'
        self.assertIn('title', pipeline.text_columns)
        self.assertIn('abstract', pipeline.text_columns)
    
    def test_init_manual_text_columns(self):
        """Test manual specification of text columns"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(
            self.csv_path,
            text_columns=['title', 'notes']
        )
        
        self.assertEqual(pipeline.text_columns, ['title', 'notes'])
    
    def test_init_column_validation_warning(self):
        """Test column validation in warning mode"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Should not raise error, only warning
        with self.assertWarns(UserWarning):
            pipeline = XctopusPipeline(
                self.csv_path,
                text_columns=['title', 'nonexistent'],
                strict_mode=False
            )
    
    def test_init_column_validation_strict(self):
        """Test column validation in strict mode"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Should raise error in strict mode
        with self.assertRaises(ValueError):
            XctopusPipeline(
                self.csv_path,
                text_columns=['title', 'nonexistent'],
                strict_mode=True
            )


class TestXctopusPipelineMethods(unittest.TestCase):
    """Test cases for XctopusPipeline methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_nodes(self):
        """Test get_nodes() method"""
        pipeline = XctopusPipeline()
        
        nodes = pipeline.get_nodes()
        self.assertIsInstance(nodes, dict)
        self.assertEqual(nodes, {})
        
        # Modify nodes and check
        pipeline.knowledge_nodes = {'node1': 'test'}
        self.assertEqual(pipeline.get_nodes(), {'node1': 'test'})
    
    def test_get_results(self):
        """Test get_results() method"""
        pipeline = XctopusPipeline()
        
        results = pipeline.get_results()
        self.assertIsInstance(results, dict)
        self.assertEqual(results, {})
        
        # Modify results and check
        pipeline.results = {'analysis': {'clusters': 5}}
        self.assertEqual(pipeline.get_results(), {'analysis': {'clusters': 5}})
    
    def test_get_config(self):
        """Test get_config() method"""
        config = PipelineConfig()
        config.NUM_EPOCHS = 10
        
        pipeline = XctopusPipeline(config=config)
        
        retrieved_config = pipeline.get_config()
        self.assertIsInstance(retrieved_config, PipelineConfig)
        self.assertEqual(retrieved_config.NUM_EPOCHS, 10)
    
    def test_get_preprocessor_lazy(self):
        """Test that _get_preprocessor() uses lazy initialization"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Preprocessor should be None initially
        self.assertIsNone(pipeline._preprocessor)
        
        # Get preprocessor (lazy initialization)
        preprocessor = pipeline._get_preprocessor()
        
        # Should now be initialized
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(pipeline._preprocessor)
        
        # Should return same instance on subsequent calls
        preprocessor2 = pipeline._get_preprocessor()
        self.assertIs(preprocessor, preprocessor2)
    
    def test_get_step_lazy(self):
        """Test that _get_step() uses lazy initialization"""
        from xctopus.pipeline.steps.base import PipelineStep
        
        # Register a test step
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        from xctopus.pipeline.steps import register_step
        step_name = f"test_step_{id(TestStep)}"
        register_step(step_name, TestStep)
        
        try:
            pipeline = XctopusPipeline()
            
            # Step should not be in cache initially
            self.assertNotIn(step_name, pipeline._steps)
            
            # Get step (lazy initialization)
            step = pipeline._get_step(step_name)
            
            # Should now be in cache
            self.assertIn(step_name, pipeline._steps)
            
            # Should return same instance on subsequent calls
            step2 = pipeline._get_step(step_name)
            self.assertIs(step, step2)
        finally:
            # Cleanup
            from xctopus.pipeline.steps import unregister_step
            try:
                unregister_step(step_name)
            except:
                pass


if __name__ == '__main__':
    unittest.main()

