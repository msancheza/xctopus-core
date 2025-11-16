"""
Pipeline Steps Module

This module provides the base PipelineStep class and the step registration system.
Steps can be built-in or registered as plugins.

The registration system allows extending Xctopus without modifying core code,
similar to frameworks like Airflow, Prefect, MLFlow Pipelines, PyTorch Lightning.
"""

from .base import PipelineStep

# Registry for custom steps (plugins)
_STEP_REGISTRY: dict = {}

# Built-in steps (populated as we implement them)
_BUILTIN_STEPS: dict = {
    'analysis': 'xctopus.pipeline.steps.analysis.AnalysisStep',
    'clustering': 'xctopus.pipeline.steps.clustering.ClusteringStep',
    'config_update': 'xctopus.pipeline.steps.config_update.ConfigUpdateStep',
    'optimize': 'xctopus.pipeline.steps.optimize.OptimizeStep',
    'fine_tune': 'xctopus.pipeline.steps.fine_tune.FineTuneStep',
    'audit': 'xctopus.pipeline.steps.audit.AuditStep',
    'evaluation': 'xctopus.pipeline.steps.evaluation.EvaluationStep',
}


def register_step(name: str, step_class: type):
    """
    Register a custom pipeline step.
    
    This allows extending Xctopus without modifying core code.
    Custom steps can be registered and used just like built-in steps.
    
    Args:
        name: Unique name for the step (e.g., "my_custom_step")
        step_class: Class that inherits from PipelineStep
    
    Raises:
        ValueError: If step_class doesn't inherit from PipelineStep
        ValueError: If name is already registered (built-in or custom)
    
    Example:
        class MyCustomStep(PipelineStep):
            def execute(self, pipeline, **kwargs):
                return {"result": "custom"}
            
            def validate_inputs(self, pipeline, **kwargs):
                pass
        
        register_step("my_custom", MyCustomStep)
    """
    if not issubclass(step_class, PipelineStep):
        raise ValueError(
            f"step_class must inherit from PipelineStep, "
            f"got {step_class.__name__}"
        )
    
    if name in _STEP_REGISTRY:
        raise ValueError(
            f"Custom step '{name}' is already registered. "
            f"Use unregister_step() first to replace it."
        )
    
    if name in _BUILTIN_STEPS:
        raise ValueError(
            f"Step name '{name}' is reserved for a built-in step. "
            f"Choose a different name."
        )
    
    _STEP_REGISTRY[name] = step_class


def get_step(name: str) -> PipelineStep:
    """
    Get a step instance by name (built-in or registered).
    
    Args:
        name: Name of the step to retrieve
    
    Returns:
        PipelineStep: Instance of the requested step
    
    Raises:
        ValueError: If step name is not found
    
    Example:
        step = get_step("analysis")
        results = step.execute(pipeline, **kwargs)
    """
    # First check custom registry
    if name in _STEP_REGISTRY:
        step_class = _STEP_REGISTRY[name]
        return step_class()
    
    # Then check built-in steps
    if name in _BUILTIN_STEPS:
        module_path = _BUILTIN_STEPS[name]
        # Dynamic import of built-in step
        module_path_parts = module_path.rsplit('.', 1)
        if len(module_path_parts) == 2:
            module_name, class_name = module_path_parts
            module = __import__(module_name, fromlist=[class_name])
            step_class = getattr(module, class_name)
            return step_class()
        else:
            raise ValueError(f"Invalid module path format: {module_path}")
    
    # Step not found
    available = list(_BUILTIN_STEPS.keys()) + list(_STEP_REGISTRY.keys())
    raise ValueError(
        f"Step '{name}' not found. "
        f"Available steps: {available}"
    )


def list_all_steps() -> dict:
    """
    List all available steps (built-in + registered).
    
    Returns:
        dict: Dictionary mapping step names to their classes
            Format: {'step_name': step_class, ...}
    
    Example:
        all_steps = list_all_steps()
        print(f"Available steps: {list(all_steps.keys())}")
    """
    all_steps = {}
    
    # Add built-in steps (as placeholders for now)
    for name in _BUILTIN_STEPS.keys():
        all_steps[name] = _BUILTIN_STEPS[name]  # Module path for now
    
    # Add registered custom steps
    all_steps.update(_STEP_REGISTRY)
    
    return all_steps


def unregister_step(name: str):
    """
    Unregister a custom step.
    
    Only custom (registered) steps can be unregistered.
    Built-in steps cannot be unregistered.
    
    Args:
        name: Name of the step to unregister
    
    Raises:
        ValueError: If step is built-in or not registered
    
    Example:
        unregister_step("my_custom_step")
    """
    if name in _BUILTIN_STEPS:
        raise ValueError(
            f"Cannot unregister built-in step '{name}'. "
            f"Only custom registered steps can be unregistered."
        )
    
    if name not in _STEP_REGISTRY:
        raise ValueError(
            f"Step '{name}' is not registered. "
            f"Cannot unregister."
        )
    
    del _STEP_REGISTRY[name]


# Export public API
__all__ = [
    'PipelineStep',
    'register_step',
    'get_step',
    'list_all_steps',
    'unregister_step',
]

