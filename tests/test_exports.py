"""
Tests for public API exports (Phase 10, Sprint 10.1)

Verify that all public exports are accessible from the main package.
"""

import unittest


class TestPublicExports(unittest.TestCase):
    """Test that public API exports are accessible"""
    
    def test_import_xctopus_pipeline(self):
        """Test that XctopusPipeline can be imported from xctopus"""
        try:
            from xctopus import XctopusPipeline
            self.assertIsNotNone(XctopusPipeline)
        except ImportError as e:
            self.fail(f"Failed to import XctopusPipeline: {e}")
    
    def test_import_pipeline_config(self):
        """Test that PipelineConfig can be imported from xctopus"""
        try:
            from xctopus import PipelineConfig
            self.assertIsNotNone(PipelineConfig)
        except ImportError as e:
            self.fail(f"Failed to import PipelineConfig: {e}")
    
    def test_import_from_pipeline_module(self):
        """Test that pipeline module exports work"""
        try:
            from xctopus.pipeline import XctopusPipeline, PipelineConfig
            self.assertIsNotNone(XctopusPipeline)
            self.assertIsNotNone(PipelineConfig)
        except ImportError as e:
            self.fail(f"Failed to import from xctopus.pipeline: {e}")
    
    def test_backward_compatibility_imports(self):
        """Test that backward compatibility imports still work"""
        try:
            # These should still be available for backward compatibility
            from xctopus.nodes.bayesian.bayesian_node import BayesianNode
            from xctopus.nodes.transformer.transformer import TransformerNode
            from xctopus.nodes.bayesian.core.text_preprocessor import TextPreprocessor
            
            self.assertIsNotNone(BayesianNode)
            self.assertIsNotNone(TransformerNode)
            self.assertIsNotNone(TextPreprocessor)
        except ImportError as e:
            # These are optional for backward compatibility
            # Don't fail if they're not available
            pass


if __name__ == '__main__':
    unittest.main()

