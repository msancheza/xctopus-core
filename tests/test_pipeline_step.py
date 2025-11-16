"""
Tests for PipelineStep base class (Phase 1)

Tests basic functionality of PipelineStep abstract class and step registration system.
"""

import unittest
from xctopus.pipeline.steps.base import PipelineStep
from xctopus.pipeline.steps import (
    register_step,
    get_step,
    list_all_steps,
    unregister_step
)


class TestPipelineStep(unittest.TestCase):
    """Test cases for PipelineStep base class"""
    
    def test_pipeline_step_is_abstract(self):
        """Test that PipelineStep cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            PipelineStep()  # Should fail because it's abstract
    
    def test_concrete_step_implementation(self):
        """Test that a concrete step can be created"""
        
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"result": "test"}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step = TestStep()
        self.assertIsInstance(step, PipelineStep)
        self.assertIsInstance(step, TestStep)
    
    def test_step_execute_required(self):
        """Test that execute() must be implemented"""
        
        class IncompleteStep(PipelineStep):
            def validate_inputs(self, pipeline, **kwargs):
                pass
            # Missing execute() method
        
        with self.assertRaises(TypeError):
            IncompleteStep()  # Should fail because execute() is missing
    
    def test_step_validate_inputs_required(self):
        """Test that validate_inputs() must be implemented"""
        
        class IncompleteStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            # Missing validate_inputs() method
        
        with self.assertRaises(TypeError):
            IncompleteStep()  # Should fail because validate_inputs() is missing
    
    def test_get_required_steps_default(self):
        """Test that get_required_steps() returns empty list by default"""
        
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step = TestStep()
        required = step.get_required_steps()
        self.assertEqual(required, [])
        self.assertIsInstance(required, list)
    
    def test_get_required_steps_override(self):
        """Test that get_required_steps() can be overridden"""
        
        class DependentStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
            
            def get_required_steps(self):
                return ['clustering', 'analysis']
        
        step = DependentStep()
        required = step.get_required_steps()
        self.assertEqual(required, ['clustering', 'analysis'])
    
    def test_get_step_name(self):
        """Test that get_step_name() returns correct name"""
        
        class AnalysisStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step = AnalysisStep()
        name = step.get_step_name()
        self.assertEqual(name, 'analysis')
    
    def test_step_with_config(self):
        """Test that step can be initialized with config"""
        from xctopus.pipeline.config import PipelineConfig
        
        config = PipelineConfig()
        
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step = TestStep(config=config)
        self.assertEqual(step.config, config)


class TestStepRegistration(unittest.TestCase):
    """Test cases for step registration system"""
    
    def setUp(self):
        """Clear registry before each test"""
        # Note: In a real scenario, we'd need to clear the registry
        # For now, we'll use unique names for each test
    
    def test_register_step(self):
        """Test that a custom step can be registered"""
        
        class CustomStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"custom": True}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        # Register with unique name
        step_name = f"test_custom_{id(CustomStep)}"
        register_step(step_name, CustomStep)
        
        # Verify it's registered
        all_steps = list_all_steps()
        self.assertIn(step_name, all_steps)
    
    def test_register_step_invalid_class(self):
        """Test that registering non-PipelineStep class fails"""
        
        class NotAStep:
            pass
        
        with self.assertRaises(ValueError) as context:
            register_step("invalid", NotAStep)
        
        self.assertIn("must inherit from PipelineStep", str(context.exception))
    
    def test_register_step_duplicate(self):
        """Test that registering duplicate name fails"""
        
        class Step1(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        class Step2(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step_name = f"test_duplicate_{id(Step1)}"
        register_step(step_name, Step1)
        
        # Try to register again
        with self.assertRaises(ValueError) as context:
            register_step(step_name, Step2)
        
        self.assertIn("already registered", str(context.exception))
    
    def test_get_step_registered(self):
        """Test that registered step can be retrieved"""
        
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"test": True}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step_name = f"test_get_{id(TestStep)}"
        register_step(step_name, TestStep)
        
        # Retrieve step
        step = get_step(step_name)
        self.assertIsInstance(step, TestStep)
        self.assertIsInstance(step, PipelineStep)
    
    def test_get_step_not_found(self):
        """Test that getting non-existent step raises error"""
        with self.assertRaises(ValueError) as context:
            get_step("nonexistent_step_12345")
        
        self.assertIn("not found", str(context.exception))
    
    def test_list_all_steps(self):
        """Test that list_all_steps() returns all steps"""
        
        class Step1(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        class Step2(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        name1 = f"test_list1_{id(Step1)}"
        name2 = f"test_list2_{id(Step2)}"
        
        register_step(name1, Step1)
        register_step(name2, Step2)
        
        all_steps = list_all_steps()
        self.assertIn(name1, all_steps)
        self.assertIn(name2, all_steps)
        self.assertIsInstance(all_steps, dict)
    
    def test_unregister_step(self):
        """Test that a registered step can be unregistered"""
        
        class TestStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {}
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        step_name = f"test_unregister_{id(TestStep)}"
        register_step(step_name, TestStep)
        
        # Verify registered
        self.assertIn(step_name, list_all_steps())
        
        # Unregister
        unregister_step(step_name)
        
        # Verify unregistered
        self.assertNotIn(step_name, list_all_steps())
    
    def test_unregister_step_not_registered(self):
        """Test that unregistering non-existent step fails"""
        with self.assertRaises(ValueError) as context:
            unregister_step("nonexistent_unregister_12345")
        
        self.assertIn("not registered", str(context.exception))


if __name__ == '__main__':
    unittest.main()

