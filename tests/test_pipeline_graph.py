"""
Tests for Pipeline Graph functionality (Phase 8, Sprint 8.1)
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.steps import register_step
from xctopus.pipeline.steps.base import PipelineStep


class TestPipelineGraph(unittest.TestCase):
    """Test cases for pipeline graph methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.test_dir, 'test_dataset_sample.csv')
    
    def test_get_graph_structure(self):
        """Test that get_graph() returns correct structure"""
        pipeline = XctopusPipeline(self.csv_path)
        graph = pipeline.get_graph()
        
        # Check structure
        self.assertIsInstance(graph, dict)
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        self.assertIn('executed', graph)
        self.assertIn('available', graph)
        
        # Check types
        self.assertIsInstance(graph['nodes'], list)
        self.assertIsInstance(graph['edges'], list)
        self.assertIsInstance(graph['executed'], list)
        self.assertIsInstance(graph['available'], list)
    
    def test_get_graph_nodes(self):
        """Test that graph nodes have correct structure"""
        pipeline = XctopusPipeline(self.csv_path)
        graph = pipeline.get_graph()
        
        # Check that we have nodes
        self.assertGreater(len(graph['nodes']), 0)
        
        # Check node structure
        for node in graph['nodes']:
            self.assertIn('name', node)
            self.assertIn('type', node)
            self.assertIn('executed', node)
            self.assertIn('dependencies', node)
            self.assertIn(node['type'], ['builtin', 'custom'])
            self.assertIsInstance(node['dependencies'], list)
    
    def test_get_graph_executed_status(self):
        """Test that executed steps are marked correctly"""
        if not os.path.exists(self.csv_path):
            self.skipTest(f"Test CSV file not found: {self.csv_path}")
        
        pipeline = XctopusPipeline(self.csv_path)
        
        # Initially no steps executed
        graph = pipeline.get_graph()
        self.assertEqual(len(graph['executed']), 0)
        
        # Execute a step
        try:
            pipeline.run(step='analysis', save_plots=False, compute_advanced_metrics=False)
            
            # Check that analysis is marked as executed
            graph = pipeline.get_graph()
            self.assertIn('analysis', graph['executed'])
            
            # Check that node is marked as executed
            analysis_node = next((n for n in graph['nodes'] if n['name'] == 'analysis'), None)
            self.assertIsNotNone(analysis_node)
            self.assertTrue(analysis_node['executed'])
        except Exception as e:
            if "torch" in str(e).lower() or "sentence" in str(e).lower():
                self.skipTest(f"Execution requires dependencies: {e}")
            else:
                raise
    
    def test_get_graph_dependencies(self):
        """Test that graph correctly shows dependencies"""
        pipeline = XctopusPipeline(self.csv_path)
        graph = pipeline.get_graph()
        
        # Find clustering step (should have no dependencies)
        clustering_node = next((n for n in graph['nodes'] if n['name'] == 'clustering'), None)
        if clustering_node:
            self.assertEqual(clustering_node['dependencies'], [])
        
        # Find steps that depend on clustering
        steps_with_deps = [n for n in graph['nodes'] if 'clustering' in n['dependencies']]
        if steps_with_deps:
            # Check that edges exist
            clustering_edges = [e for e in graph['edges'] if e['from'] == 'clustering']
            self.assertGreater(len(clustering_edges), 0)
    
    def test_export_graph_mermaid(self):
        """Test that export_graph_mermaid() generates valid Mermaid code"""
        pipeline = XctopusPipeline(self.csv_path)
        mermaid_code = pipeline.export_graph_mermaid()
        
        # Check that it's a string
        self.assertIsInstance(mermaid_code, str)
        
        # Check that it contains Mermaid syntax
        self.assertIn('graph TD', mermaid_code)
        
        # Check that it contains step names
        graph = pipeline.get_graph()
        for step_name in graph['available'][:3]:  # Check first 3
            display_name = step_name.replace('_', ' ').title()
            # Mermaid code should contain the step name in some form
            self.assertTrue(
                step_name in mermaid_code or display_name in mermaid_code,
                f"Step {step_name} not found in Mermaid code"
            )
    
    def test_export_graph_mermaid_to_file(self):
        """Test that export_graph_mermaid() can save to file"""
        pipeline = XctopusPipeline(self.csv_path)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            temp_path = f.name
        
        try:
            mermaid_code = pipeline.export_graph_mermaid(output_path=temp_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Check that file contains Mermaid code
            with open(temp_path, 'r') as f:
                file_content = f.read()
                self.assertEqual(file_content, mermaid_code)
                self.assertIn('graph TD', file_content)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_visualize_graph_fallback(self):
        """Test that visualize_graph() falls back to Mermaid if graphviz not available"""
        pipeline = XctopusPipeline(self.csv_path)
        
        # This should work even without graphviz
        result = pipeline.visualize_graph()
        
        # Should return None or a path
        self.assertTrue(result is None or isinstance(result, str))
    
    def test_get_graph_with_custom_step(self):
        """Test that get_graph() includes custom registered steps"""
        pipeline = XctopusPipeline(self.csv_path)
        
        # Register a custom step
        class CustomTestStep(PipelineStep):
            def get_required_steps(self):
                return ['clustering']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'custom'}
        
        register_step('custom_test', CustomTestStep)
        
        try:
            graph = pipeline.get_graph()
            
            # Check that custom step is in graph
            custom_node = next((n for n in graph['nodes'] if n['name'] == 'custom_test'), None)
            self.assertIsNotNone(custom_node)
            self.assertEqual(custom_node['type'], 'custom')
            self.assertIn('clustering', custom_node['dependencies'])
        finally:
            # Clean up
            from xctopus.pipeline.steps import unregister_step
            try:
                unregister_step('custom_test')
            except ValueError:
                pass


if __name__ == '__main__':
    unittest.main()

