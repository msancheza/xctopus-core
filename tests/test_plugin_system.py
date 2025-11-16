"""
Tests for Plugin System (Phase 8, Sprint 8.3)

Tests the complete plugin system including custom step registration,
execution, and integration with pipeline graph.
"""

import unittest
import os
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps import (
    register_step,
    get_step,
    list_all_steps,
    unregister_step
)
from xctopus.pipeline.steps.base import PipelineStep


class TestPluginSystem(unittest.TestCase):
    """Test cases for plugin system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def tearDown(self):
        """Clean up registered steps"""
        # Try to unregister any test steps
        test_steps = ['test_simple', 'test_dependent', 'test_stats']
        for step_name in test_steps:
            try:
                unregister_step(step_name)
            except ValueError:
                pass  # Step not registered, that's okay
    
    def test_register_and_execute_custom_step(self):
        """Test registering and executing a custom step"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Define custom step
        class SimpleTestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"result": "test", "custom": True}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        # Register step
        register_step("test_simple", SimpleTestStep)
        
        # Verify it's registered
        all_steps = list_all_steps()
        self.assertIn("test_simple", all_steps)
        
        # Execute step
        pipeline = XctopusPipeline(self.csv_path)
        result = pipeline.run(step='test_simple')
        
        self.assertEqual(result['result'], "test")
        self.assertTrue(result['custom'])
        self.assertIn('test_simple', pipeline.results)
    
    def test_custom_step_with_dependencies(self):
        """Test custom step that requires other steps"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        # Define custom step with dependency
        class DependentTestStep(PipelineStep):
            def get_required_steps(self):
                return ['clustering']
            
            def execute(self, pipeline, **kwargs):
                nodes = pipeline.knowledge_nodes
                return {
                    "node_count": len(nodes),
                    "has_nodes": len(nodes) > 0
                }
            
            def validate_inputs(self, pipeline, **kwargs):
                if 'clustering' not in pipeline.results:
                    raise ValueError("Clustering required")
        
        # Register step
        register_step("test_dependent", DependentTestStep)
        
        # Try to run without dependency (should fail)
        pipeline = XctopusPipeline(self.csv_path)
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='test_dependent')
        self.assertIn("clustering", str(context.exception).lower())
        
        # Run with dependency (should work)
        try:
            pipeline.run(step='clustering', epochs=1, enable_training=False, enable_merge=False)
            result = pipeline.run(step='test_dependent')
            self.assertIn('node_count', result)
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_custom_step_in_graph(self):
        """Test that custom steps appear in pipeline graph"""
        # Define and register custom step
        class GraphTestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"in_graph": True}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        register_step("test_stats", GraphTestStep)
        
        try:
            pipeline = XctopusPipeline(self.csv_path)
            graph = pipeline.get_graph()
            
            # Check that custom step is in graph
            step_names = [node['name'] for node in graph['nodes']]
            self.assertIn('test_stats', step_names)
            
            # Check that it's marked as custom
            test_node = next((n for n in graph['nodes'] if n['name'] == 'test_stats'), None)
            self.assertIsNotNone(test_node)
            self.assertEqual(test_node['type'], 'custom')
        finally:
            try:
                unregister_step("test_stats")
            except ValueError:
                pass
    
    def test_custom_step_preserves_results(self):
        """Test that custom step results are preserved in pipeline"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        class ResultsTestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                result = {"data": [1, 2, 3], "count": 3}
                pipeline.results['test_results'] = result
                return result
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        register_step("test_results", ResultsTestStep)
        
        try:
            pipeline = XctopusPipeline(self.csv_path)
            result = pipeline.run(step='test_results')
            
            # Check that result is in pipeline.results
            self.assertIn('test_results', pipeline.results)
            self.assertEqual(pipeline.results['test_results'], result)
        finally:
            try:
                unregister_step("test_results")
            except ValueError:
                pass
    
    def test_unregister_custom_step(self):
        """Test unregistering a custom step"""
        class UnregisterTestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        # Register
        register_step("test_unregister", UnregisterTestStep)
        self.assertIn("test_unregister", list_all_steps())
        
        # Unregister
        unregister_step("test_unregister")
        self.assertNotIn("test_unregister", list_all_steps())
        
        # Try to unregister again (should fail)
        with self.assertRaises(ValueError):
            unregister_step("test_unregister")
    
    def test_cannot_unregister_builtin_step(self):
        """Test that built-in steps cannot be unregistered"""
        with self.assertRaises(ValueError) as context:
            unregister_step("clustering")
        self.assertIn("built-in", str(context.exception).lower())
    
    def test_custom_step_with_kwargs(self):
        """Test that custom steps receive kwargs correctly"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        class KwargsTestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {
                    "received_kwargs": kwargs,
                    "custom_param": kwargs.get('custom_param', None)
                }
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        register_step("test_kwargs", KwargsTestStep)
        
        try:
            pipeline = XctopusPipeline(self.csv_path)
            result = pipeline.run(
                step='test_kwargs',
                custom_param='test_value',
                another_param=123
            )
            
            self.assertEqual(result['custom_param'], 'test_value')
            self.assertIn('another_param', result['received_kwargs'])
        finally:
            try:
                unregister_step("test_kwargs")
            except ValueError:
                pass
    
    def test_multiple_custom_steps(self):
        """Test registering and using multiple custom steps"""
        class Step1(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"step": 1}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        class Step2(PipelineStep):
            def get_required_steps(self):
                return ['test_multi_1']
            def execute(self, pipeline, **kwargs):
                return {"step": 2, "previous": pipeline.results.get('test_multi_1')}
            def validate_inputs(self, pipeline, **kwargs):
                if 'test_multi_1' not in pipeline.results:
                    raise ValueError("Step1 required")
        
        register_step("test_multi_1", Step1)
        register_step("test_multi_2", Step2)
        
        try:
            pipeline = XctopusPipeline(self.csv_path)
            
            # Execute step 1
            result1 = pipeline.run(step='test_multi_1')
            self.assertEqual(result1['step'], 1)
            
            # Execute step 2 (depends on step 1)
            result2 = pipeline.run(step='test_multi_2')
            self.assertEqual(result2['step'], 2)
            self.assertIsNotNone(result2['previous'])
        finally:
            try:
                unregister_step("test_multi_1")
                unregister_step("test_multi_2")
            except ValueError:
                pass


if __name__ == '__main__':
    unittest.main()

