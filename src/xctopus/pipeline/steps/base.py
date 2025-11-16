"""
Pipeline Step Base Class

Provides the abstract base class PipelineStep that all pipeline steps must inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class PipelineStep(ABC):
    """
    Abstract base class for all pipeline steps.
    
    Each step implements its own logic but follows the same interface.
    Steps can be built-in or registered as plugins.
    
    All steps must implement:
    - execute(): The main execution logic
    - validate_inputs(): Validation of required inputs
    
    Steps can optionally override:
    - get_required_steps(): List of steps that must run before this one
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline step.
        
        Args:
            config: Optional configuration object (PipelineConfig or similar)
        """
        self.config = config
    
    @abstractmethod
    def execute(self, pipeline, **kwargs):
        """
        Execute the pipeline step.
        
        This is the main method that contains the step's logic.
        It must be implemented by all concrete step classes.
        
        Args:
            pipeline: Instance of XctopusPipeline (provides access to state)
            **kwargs: Step-specific parameters
        
        Returns:
            dict: Results of the step execution
        
        Raises:
            ValueError: If required inputs are missing or invalid
            RuntimeError: If execution fails
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, pipeline, **kwargs):
        """
        Validate that required inputs are available.
        
        This method should check that:
        - Required pipeline state exists (e.g., knowledge_nodes for optimize step)
        - Required parameters are provided
        - Dependencies are satisfied
        
        Args:
            pipeline: Instance of XctopusPipeline
            **kwargs: Step-specific parameters
        
        Raises:
            ValueError: If validation fails (missing inputs, invalid state, etc.)
        """
        pass
    
    def get_required_steps(self) -> List[str]:
        """
        Return list of steps that must execute before this step.
        
        This method defines the dependency graph of the pipeline.
        Steps that depend on others should override this method.
        
        Returns:
            list: List of step names that must run first (e.g., ['clustering'])
        
        Example:
            def get_required_steps(self):
                return ['clustering']  # This step requires clustering to run first
        """
        return []  # By default, no steps required
    
    def get_step_name(self) -> str:
        """
        Get the name of this step.
        
        By default, returns the class name in lowercase with 'step' removed.
        Can be overridden for custom naming.
        
        Returns:
            str: Step name (e.g., 'analysis', 'clustering')
        """
        class_name = self.__class__.__name__
        # Remove 'Step' suffix if present
        if class_name.endswith('Step'):
            return class_name[:-4].lower()
        return class_name.lower()

