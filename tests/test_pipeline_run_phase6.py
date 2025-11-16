"""
Tests for XctopusPipeline.run() improvements (Phase 6, Sprint 6.1)

Tests robust dependency validation, error handling, and step execution.
"""

import unittest
import os
import tempfile
import pandas as pd
from xctopus.pipeline.pipeline import XctopusPipeline
from xctopus.pipeline.config import PipelineConfig
from xctopus.pipeline.steps.base import PipelineStep
from xctopus.pipeline.steps import register_step


class TestPipelineRunValidation(unittest.TestCase):
    """Test cases for run() method validation and error handling"""
    
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
    
    def test_run_with_invalid_step_name(self):
        """Test that run() raises ValueError for invalid step names"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # Empty string
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='')
        self.assertIn("non-empty string", str(context.exception))
        
        # Whitespace only
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='   ')
        self.assertIn("non-empty string", str(context.exception))
        
        # Non-existent step
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='nonexistent_step')
        self.assertIn("not found", str(context.exception))
        self.assertIn("Available steps", str(context.exception))
    
    def test_run_with_none_step(self):
        """Test that run() without step raises NotImplementedError"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        with self.assertRaises(NotImplementedError) as context:
            pipeline.run()
        self.assertIn("Full pipeline execution", str(context.exception))
        self.assertIn("Phase 6", str(context.exception))
    
    def test_run_analysis_success(self):
        """Test that run() executes analysis step successfully"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # Should not raise any errors
        result = pipeline.run(step='analysis')
        
        # Check that result is stored
        self.assertIn('analysis', pipeline.results)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
    
    def test_run_with_missing_dataset_path(self):
        """Test that run() raises ValueError when dataset_path is missing"""
        pipeline = XctopusPipeline()  # No dataset_path
        
        with self.assertRaises(ValueError) as context:
            pipeline.run(step='analysis')
        self.assertIn("dataset_path is required", str(context.exception))
    
    def test_run_with_custom_dataset_path(self):
        """Test that run() accepts custom dataset_path parameter"""
        pipeline = XctopusPipeline()  # No dataset_path in init
        
        # Should work with dataset_path in run()
        result = pipeline.run(step='analysis', dataset_path=self.dataset_path)
        self.assertIsNotNone(result)


class TestDependencyValidation(unittest.TestCase):
    """Test cases for dependency validation"""
    
    def setUp(self):
        """Create a temporary dataset for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        df = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'id': [1, 2]
        })
        df.to_csv(self.dataset_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_dependency_validation_no_dependencies(self):
        """Test that steps without dependencies can run"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # Analysis and clustering have no dependencies
        result1 = pipeline.run(step='analysis')
        self.assertIsNotNone(result1)
        
        # Should not raise dependency error
        result2 = pipeline.run(step='clustering', epochs=1)
        self.assertIsNotNone(result2)
    
    def test_dependency_validation_missing_dependency(self):
        """Test that run() detects missing dependencies"""
        
        # Create a step that requires 'clustering'
        class DependentStep(PipelineStep):
            def get_required_steps(self):
                return ['clustering']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'dependent'}
        
        # Register the step
        register_step('dependent', DependentStep)
        
        try:
            pipeline = XctopusPipeline(self.dataset_path)
            
            # Try to run dependent step without running clustering first
            with self.assertRaises(ValueError) as context:
                pipeline.run(step='dependent')
            
            self.assertIn("requires", str(context.exception))
            self.assertIn("clustering", str(context.exception))
            
            # Now run clustering first
            pipeline.run(step='clustering', epochs=1)
            
            # Now dependent step should work
            result = pipeline.run(step='dependent')
            self.assertIsNotNone(result)
        finally:
            # Clean up: unregister the step
            from xctopus.pipeline.steps import unregister_step
            try:
                unregister_step('dependent')
            except ValueError:
                pass  # Already unregistered
    
    def test_dependency_validation_recursive(self):
        """Test that dependency validation works recursively"""
        
        # Create a chain of dependencies: step3 -> step2 -> step1
        class Step1(PipelineStep):
            def get_required_steps(self):
                return []
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'step1'}
        
        class Step2(PipelineStep):
            def get_required_steps(self):
                return ['step1']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'step2'}
        
        class Step3(PipelineStep):
            def get_required_steps(self):
                return ['step2']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'step3'}
        
        # Register steps
        register_step('step1', Step1)
        register_step('step2', Step2)
        register_step('step3', Step3)
        
        try:
            pipeline = XctopusPipeline(self.dataset_path)
            
            # Try to run step3 without step2 or step1
            with self.assertRaises(ValueError) as context:
                pipeline.run(step='step3')
            self.assertIn("step2", str(context.exception))
            
            # Try to run step2 without step1
            with self.assertRaises(ValueError) as context:
                pipeline.run(step='step2')
            self.assertIn("step1", str(context.exception))
            
            # Run in correct order
            pipeline.run(step='step1')
            pipeline.run(step='step2')
            result = pipeline.run(step='step3')
            self.assertEqual(result['result'], 'step3')
        finally:
            # Clean up
            from xctopus.pipeline.steps import unregister_step
            for step_name in ['step1', 'step2', 'step3']:
                try:
                    unregister_step(step_name)
                except ValueError:
                    pass
    
    def test_dependency_validation_circular_dependency(self):
        """Test that circular dependencies are detected"""
        
        class StepA(PipelineStep):
            def get_required_steps(self):
                return ['step_b']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'a'}
        
        class StepB(PipelineStep):
            def get_required_steps(self):
                return ['step_a']
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def execute(self, pipeline, **kwargs):
                return {'result': 'b'}
        
        # Register steps
        register_step('step_a', StepA)
        register_step('step_b', StepB)
        
        try:
            pipeline = XctopusPipeline(self.dataset_path)
            
            # Try to run step_a (which requires step_b, which requires step_a)
            # This should detect the circular dependency
            with self.assertRaises(ValueError) as context:
                pipeline.run(step='step_a')
            self.assertIn("Circular dependency", str(context.exception))
        finally:
            # Clean up
            from xctopus.pipeline.steps import unregister_step
            for step_name in ['step_a', 'step_b']:
                try:
                    unregister_step(step_name)
                except ValueError:
                    pass


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in run()"""
    
    def setUp(self):
        """Create a temporary dataset for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        df = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'id': [1, 2]
        })
        df.to_csv(self.dataset_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_run_with_nonexistent_dataset(self):
        """Test that run() handles nonexistent dataset files"""
        pipeline = XctopusPipeline('nonexistent_file.csv')
        
        with self.assertRaises((ValueError, RuntimeError)) as context:
            pipeline.run(step='analysis')
        # Should raise an error about file not found
        self.assertIsNotNone(context.exception)
    
    def test_run_preserves_results_on_error(self):
        """Test that results are preserved even if a step fails"""
        pipeline = XctopusPipeline(self.dataset_path)
        
        # Run analysis successfully
        result1 = pipeline.run(step='analysis')
        self.assertIn('analysis', pipeline.results)
        
        # Try to run a step that will fail (with invalid parameters)
        # This should not clear previous results
        try:
            pipeline.run(step='clustering', epochs=-1)  # Invalid epochs
        except Exception:
            pass  # Expected to fail
        
        # Previous results should still be there
        self.assertIn('analysis', pipeline.results)


if __name__ == '__main__':
    unittest.main()

