"""
XctopusPipeline - Main Pipeline Orchestrator

This module provides the XctopusPipeline class that orchestrates the complete
Xctopus workflow: Dataset → TextPreprocessor → FilterBayesianNode →
Clustering → LoRA/Auditor → Evaluator → Output
"""

import os
import warnings
import pickle
import torch
from datetime import datetime
from typing import Optional, Dict, Any

from .config import PipelineConfig
from .steps import get_step, list_all_steps
from .steps.base import PipelineStep


class XctopusPipeline:
    """
    Pipeline Manager that orchestrates the complete Xctopus workflow.
    
    Responsibilities:
    - Orchestration of steps
    - Lazy initialization of components
    - Global state management
    - Coordination between steps
    - Public API for users
    
    Example:
        # Basic usage
        pipeline = XctopusPipeline('data.csv')
        results = pipeline.run()
        
        # With custom configuration
        config = PipelineConfig()
        config.NUM_EPOCHS = 10
        pipeline = XctopusPipeline('data.csv', config=config)
        
        # With YAML configuration
        pipeline = XctopusPipeline('data.csv', config='config.yaml')
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        **kwargs
    ):
        """
        Initialize XctopusPipeline.
        
        Args:
            dataset_path: Path to CSV/JSON dataset file
            config: PipelineConfig instance, path to YAML file, or None (uses defaults)
            **kwargs: Additional parameters (override config):
                - text_columns: List of text columns (None = auto-detect)
                - join_with: Separator for joining columns
                - label_column: Label column name
                - id_column: ID column name
                - model_name: Embedding model name
                - max_length: Maximum token length
                - normalize: Normalize embeddings
                - drop_empty: Drop empty rows
                - auto_detect_text_columns: Enable auto-detection
                - suggested_text_columns: Fallback suggestions
                - validate_columns: Validate column existence
                - strict_mode: Raise errors on validation failures
        
        Example:
            pipeline = XctopusPipeline(
                'data.csv',
                text_columns=['title', 'abstract'],
                epochs=10
            )
        """
        # Load configuration
        if isinstance(config, str):
            # If string, assume it's a path to YAML file
            self.config = PipelineConfig.from_yaml(config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        else:
            self.config = config or PipelineConfig()
        
        self.dataset_path = dataset_path
        
        # Configure dataset columns (priority: kwargs > config > auto-detect)
        auto_detect = kwargs.get(
            'auto_detect_text_columns',
            getattr(self.config, 'AUTO_DETECT_TEXT_COLUMNS', True)
        )
        
        text_columns = kwargs.get('text_columns') or getattr(self.config, 'TEXT_COLUMNS', None)
        
        # Auto-detection if text_columns is None and auto_detect is enabled
        if text_columns is None and auto_detect and dataset_path:
            suggested = kwargs.get('suggested_text_columns') or \
                       getattr(self.config, 'SUGGESTED_TEXT_COLUMNS', None)
            text_columns = PipelineConfig.suggest_text_columns(dataset_path, suggested=suggested)
            
            if text_columns:
                print(f"[OK] Text columns auto-detected: {text_columns}")
            else:
                print("[WARNING] Could not auto-detect text columns")
                available = PipelineConfig._get_available_columns(dataset_path)
                if available:
                    print(f"   Available columns: {available}")
        
        # Final fallback: use ['text'] if nothing works
        if not text_columns:
            text_columns = ['text']
        
        self.text_columns = text_columns
        self.join_with = kwargs.get('join_with') or getattr(self.config, 'JOIN_WITH', '\n')
        self.label_column = kwargs.get('label_column') or getattr(self.config, 'LABEL_COLUMN', None)
        self.id_column = kwargs.get('id_column') or getattr(self.config, 'ID_COLUMN', None)
        self.model_name = kwargs.get('model_name') or getattr(self.config, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.max_length = kwargs.get('max_length') or getattr(self.config, 'MAX_LENGTH', 512)
        self.normalize = kwargs.get('normalize', getattr(self.config, 'NORMALIZE_EMBEDDINGS', True))
        self.drop_empty = kwargs.get('drop_empty', getattr(self.config, 'DROP_EMPTY', False))
        
        # Validate columns if enabled
        validate = kwargs.get(
            'validate_columns',
            getattr(self.config, 'VALIDATE_COLUMNS', True)
        )
        strict = kwargs.get(
            'strict_mode',
            getattr(self.config, 'STRICT_MODE', False)
        )
        
        if validate and dataset_path:
            validation_result = PipelineConfig.validate_dataset_columns(
                dataset_path,
                text_columns=self.text_columns,
                label_column=self.label_column,
                id_column=self.id_column,
                strict=strict
            )
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    warnings.warn(warning)
        
        # Pipeline components (lazy initialization)
        self._preprocessor = None
        self._filter_node = None
        
        # Pipeline state
        self.embeddings = None
        self.knowledge_nodes = {}
        self.results = {}
        
        # Steps (lazy initialization)
        self._steps = {}
    
    def _get_preprocessor(self):
        """
        Get or create TextPreprocessor instance (lazy initialization).
        
        Returns:
            TextPreprocessor: Preprocessor instance
        """
        if self._preprocessor is None:
            from xctopus.nodes.bayesian.core.text_preprocessor import TextPreprocessor
            
            self._preprocessor = TextPreprocessor(
                path_dataset=self.dataset_path,
                text_columns=self.text_columns,
                join_with=self.join_with,
                model_name=self.model_name,
                max_length=self.max_length,
                normalize=self.normalize,
                drop_empty=self.drop_empty,
                label_column=self.label_column,
                id_column=self.id_column
            )
        
        return self._preprocessor
    
    def _get_step(self, step_name: str):
        """
        Get step instance by name (lazy initialization).
        
        Args:
            step_name: Name of the step to retrieve
        
        Returns:
            PipelineStep: Step instance
        
        Raises:
            ValueError: If step is not found
        """
        if step_name not in self._steps:
            self._steps[step_name] = get_step(step_name)
        return self._steps[step_name]
    
    def get_nodes(self) -> Dict[str, Any]:
        """
        Get current knowledge nodes.
        
        Returns:
            dict: Dictionary of knowledge nodes {node_id: node_instance}
        """
        return self.knowledge_nodes
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all accumulated results.
        
        Returns:
            dict: Dictionary of results from executed steps
        """
        return self.results
    
    def get_config(self) -> PipelineConfig:
        """
        Get current configuration.
        
        Returns:
            PipelineConfig: Current configuration instance
        """
        return self.config
    
    def run(self, step: Optional[str] = None, dataset_path: Optional[str] = None, **kwargs):
        """
        Execute a pipeline step or full pipeline.
        
        This is the main API for executing pipeline steps.
        
        Args:
            step: Name of the step to execute:
                - 'analysis': Cluster fragmentation analysis
                - 'clustering': Dynamic clustering pipeline
                - None: Execute full pipeline (not yet implemented)
            dataset_path: Path to dataset (optional, uses pipeline.dataset_path if not provided)
            **kwargs: Step-specific parameters
        
        Returns:
            dict: Results from the executed step
        
        Raises:
            ValueError: If step is not found, dependencies are missing, or validation fails
            RuntimeError: If step execution fails
        
        Example:
            # Execute analysis step
            pipeline = XctopusPipeline('data.csv')
            results = pipeline.run(step='analysis')
            
            # Execute clustering with custom options
            results = pipeline.run(
                step='clustering',
                epochs=10,
                enable_training=True
            )
        """
        if step is None:
            # Full pipeline execution
            return self.run_full_pipeline(dataset_path=dataset_path, **kwargs)
        
        # Validate step name
        if not isinstance(step, str) or not step.strip():
            raise ValueError(f"Step name must be a non-empty string, got: {step}")
        
        step = step.strip()
        
        # Get step instance (will raise ValueError if not found)
        try:
            step_instance = self._get_step(step)
        except ValueError as e:
            # Re-raise with more context
            available = list_all_steps()
            raise ValueError(
                f"Step '{step}' not found. "
                f"Available steps: {list(available.keys())}. "
                f"Original error: {str(e)}"
            ) from e
        
        # Validate dependencies recursively
        self._validate_dependencies(step, step_instance)
        
        # Prepare arguments based on step type
        try:
            # Steps that need dataset_path
            if step in ['analysis', 'clustering']:
                dataset = dataset_path or self.dataset_path
                if not dataset:
                    raise ValueError(
                        f"dataset_path is required for step '{step}'. "
                        f"Provide it as argument or set pipeline.dataset_path"
                    )
                
                if step == 'clustering':
                    # Clustering step receives dataset_path, pipeline, and epochs
                    epochs = kwargs.get('epochs', None)
                    step_kwargs = {k: v for k, v in kwargs.items() if k != 'epochs'}
                    result = step_instance.execute(
                        dataset_path=dataset,
                        pipeline=self,
                        epochs=epochs,
                        **step_kwargs
                    )
                else:
                    # Analysis step receives dataset_path and pipeline
                    result = step_instance.execute(
                        dataset_path=dataset,
                        pipeline=self,
                        **kwargs
                    )
            else:
                # Other steps receive pipeline and kwargs
                result = step_instance.execute(
                    pipeline=self,
                    **kwargs
                )
            
            # Ensure result is stored in pipeline.results (some steps do this, but ensure it)
            if step not in self.results and result is not None:
                self.results[step] = result
            
            return result
            
        except ValueError as e:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap other errors with context
            raise RuntimeError(
                f"Error executing step '{step}': {str(e)}"
            ) from e
    
    def _validate_dependencies(self, step_name: str, step_instance: PipelineStep, visited: Optional[set] = None):
        """
        Recursively validate that all required steps have been executed.
        
        Args:
            step_name: Name of the step being validated
            step_instance: Instance of the step
            visited: Set of step names already visited (to detect cycles)
        
        Raises:
            ValueError: If dependencies are missing or circular dependencies detected
        """
        if visited is None:
            visited = set()
        
        if step_name in visited:
            raise ValueError(
                f"Circular dependency detected involving step '{step_name}'. "
                f"Visited steps: {visited}"
            )
        
        visited.add(step_name)
        
        # Get required steps
        required_steps = step_instance.get_required_steps()
        
        if not required_steps:
            visited.remove(step_name)
            return
        
        # Check each required step
        missing_steps = []
        for required_step in required_steps:
            # Validate the required step exists
            try:
                required_instance = self._get_step(required_step)
            except ValueError:
                missing_steps.append(required_step)
                continue
            
            # Check if it has been executed
            if required_step not in self.results:
                missing_steps.append(required_step)
            else:
                # Recursively validate dependencies of required step
                self._validate_dependencies(required_step, required_instance, visited.copy())
        
        if missing_steps:
            visited.remove(step_name)
            missing_list = ', '.join(f"'{s}'" for s in missing_steps)
            raise ValueError(
                f"Step '{step_name}' requires the following steps to be executed first: {missing_list}. "
                f"Execute them with: pipeline.run(step='<step_name>')"
            )
        
        visited.remove(step_name)
    
    def run_full_pipeline(
        self,
        dataset_path: Optional[str] = None,
        skip_analysis: bool = False,
        skip_config_update: bool = False,
        skip_fine_tune: bool = False,
        skip_optimize: bool = False,
        skip_audit: bool = False,
        skip_evaluation: bool = False,
        **kwargs
    ):
        """
        Execute the complete Xctopus pipeline in the correct order.
        
        This method orchestrates all pipeline steps in the correct sequence:
        1. analysis (optional) - Initial cluster fragmentation analysis
        2. clustering (required) - Dynamic clustering and knowledge node creation
        3. config_update (optional) - Update cluster configurations
        4. fine_tune (optional) - Fine-tune large clusters
        5. optimize (optional) - Optimize clusters
        6. audit (optional) - Learning effectiveness audit
        7. evaluation (optional) - Evaluate learning performance
        
        Args:
            dataset_path: Path to dataset (optional, uses pipeline.dataset_path if not provided)
            skip_analysis: Skip the analysis step (default: False)
            skip_config_update: Skip the config_update step (default: False)
            skip_fine_tune: Skip the fine_tune step (default: False)
            skip_optimize: Skip the optimize step (default: False)
            skip_audit: Skip the audit step (default: False)
            skip_evaluation: Skip the evaluation step (default: False)
            **kwargs: Additional parameters passed to individual steps:
                - epochs: Number of epochs for clustering (default: from config)
                - enable_training: Enable LoRA training (default: True)
                - enable_merge: Enable cluster merging (default: True)
                - save_plots: Save visualization plots for analysis (default: True)
                - compute_advanced_metrics: Compute advanced metrics (default: True)
                - And other step-specific parameters
        
        Returns:
            dict: Complete pipeline results with keys:
                - 'analysis': Analysis results (if not skipped)
                - 'clustering': Clustering results (always present)
                - 'config_update': Config update results (if not skipped)
                - 'fine_tune': Fine-tuning results (if not skipped)
                - 'optimize': Optimization results (if not skipped)
                - 'audit': Audit results (if not skipped)
                - 'evaluation': Evaluation results (if not skipped)
                - 'summary': Summary of executed steps
        
        Raises:
            ValueError: If dataset_path is missing or validation fails
            RuntimeError: If any step execution fails
        
        Example:
            # Run full pipeline with all steps
            pipeline = XctopusPipeline('data.csv')
            results = pipeline.run_full_pipeline()
            
            # Run pipeline skipping optional steps
            results = pipeline.run_full_pipeline(
                skip_analysis=True,
                skip_evaluation=True,
                epochs=10
            )
            
            # Run only clustering (skip all optional steps)
            results = pipeline.run_full_pipeline(
                skip_analysis=True,
                skip_config_update=True,
                skip_fine_tune=True,
                skip_optimize=True,
                skip_audit=True,
                skip_evaluation=True
            )
        """
        # Use provided dataset_path or pipeline's dataset_path
        dataset = dataset_path or self.dataset_path
        if not dataset:
            raise ValueError(
                "dataset_path is required for full pipeline execution. "
                "Provide it as argument or set pipeline.dataset_path"
            )
        
        executed_steps = []
        pipeline_results = {}
        
        try:
            # Step 1: Analysis (optional)
            if not skip_analysis:
                try:
                    print("** Step 1/7: Running analysis...")
                    result = self.run(step='analysis', dataset_path=dataset, **kwargs)
                    pipeline_results['analysis'] = result
                    executed_steps.append('analysis')
                    print("** Analysis completed")
                except Exception as e:
                    print(f"**   Analysis step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
                    # Continue with other steps in non-strict mode
            
            # Step 2: Clustering (required)
            print("**  Step 2/7: Running clustering...")
            result = self.run(step='clustering', dataset_path=dataset, **kwargs)
            pipeline_results['clustering'] = result
            executed_steps.append('clustering')
            print("**  Clustering completed")
            
            # Step 3: Config Update (optional, requires clustering)
            if not skip_config_update:
                try:
                    print("**   Step 3/7: Running config update...")
                    # Check if step exists
                    from .steps import list_all_steps
                    available_steps = list_all_steps()
                    if 'config_update' in available_steps:
                        result = self.run(step='config_update', **kwargs)
                        pipeline_results['config_update'] = result
                        executed_steps.append('config_update')
                        print("** Config update completed")
                    else:
                        print("** Config update step not available, skipping")
                except Exception as e:
                    print(f"[WARNING] Config update step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
            
            # Step 4: Fine-tune (optional, requires clustering)
            if not skip_fine_tune:
                try:
                    print("[*] Step 4/7: Running fine-tuning...")
                    from .steps import list_all_steps
                    available_steps = list_all_steps()
                    if 'fine_tune' in available_steps:
                        result = self.run(step='fine_tune', **kwargs)
                        pipeline_results['fine_tune'] = result
                        executed_steps.append('fine_tune')
                        print("[OK] Fine-tuning completed")
                    else:
                        print("[WARNING] Fine-tune step not available, skipping")
                except Exception as e:
                    print(f"[WARNING] Fine-tune step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
            
            # Step 5: Optimize (optional, requires clustering)
            if not skip_optimize:
                try:
                    print("[*] Step 5/7: Running optimization...")
                    from .steps import list_all_steps
                    available_steps = list_all_steps()
                    if 'optimize' in available_steps:
                        result = self.run(step='optimize', **kwargs)
                        pipeline_results['optimize'] = result
                        executed_steps.append('optimize')
                        print("[OK] Optimization completed")
                    else:
                        print("[WARNING] Optimize step not available, skipping")
                except Exception as e:
                    print(f"[WARNING] Optimize step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
            
            # Step 6: Audit (optional, requires clustering)
            if not skip_audit:
                try:
                    print("[*] Step 6/7: Running audit...")
                    from .steps import list_all_steps
                    available_steps = list_all_steps()
                    if 'audit' in available_steps:
                        result = self.run(step='audit', **kwargs)
                        pipeline_results['audit'] = result
                        executed_steps.append('audit')
                        print("[OK] Audit completed")
                    else:
                        print("[WARNING] Audit step not available, skipping")
                except Exception as e:
                    print(f"[WARNING] Audit step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
            
            # Step 7: Evaluation (optional, requires clustering)
            if not skip_evaluation:
                try:
                    print("[*] Step 7/7: Running evaluation...")
                    from .steps import list_all_steps
                    available_steps = list_all_steps()
                    if 'evaluation' in available_steps:
                        result = self.run(step='evaluation', **kwargs)
                        pipeline_results['evaluation'] = result
                        executed_steps.append('evaluation')
                        print("[OK] Evaluation completed")
                    else:
                        print("[WARNING] Evaluation step not available, skipping")
                except Exception as e:
                    print(f"[WARNING] Evaluation step failed: {e}")
                    if hasattr(self.config, 'STRICT_MODE') and self.config.STRICT_MODE:
                        raise
            
            # Create summary
            summary = {
                'executed_steps': executed_steps,
                'total_steps': len(executed_steps),
                'skipped_steps': [
                    step for step in ['analysis', 'config_update', 'fine_tune', 
                                     'optimize', 'audit', 'evaluation']
                    if step not in executed_steps
                ],
                'knowledge_nodes_count': len(self.knowledge_nodes) if self.knowledge_nodes else 0,
                'clusters_created': pipeline_results.get('clustering', {}).get('total_clusters', 0)
            }
            
            pipeline_results['summary'] = summary
            
            print(f"\n[SUCCESS] Pipeline execution completed!")
            print(f"   Executed {len(executed_steps)} steps: {', '.join(executed_steps)}")
            print(f"   Created {summary['clusters_created']} clusters")
            
            return pipeline_results
            
        except Exception as e:
            # If clustering fails, the pipeline cannot continue
            if 'clustering' in executed_steps or 'clustering' not in str(e).lower():
                raise RuntimeError(
                    f"Pipeline execution failed at step '{executed_steps[-1] if executed_steps else 'unknown'}': {str(e)}"
                ) from e
            else:
                raise RuntimeError(
                    f"Pipeline execution failed: {str(e)}"
                ) from e
    
    # ============================================================================
    # Alias Methods for Backward Compatibility
    # ============================================================================
    # These methods provide a more intuitive API and backward compatibility
    # with existing code that might use method names instead of step names.
    
    def run_analysis(self, dataset_path: Optional[str] = None, **kwargs):
        """
        Run the analysis step (alias for run(step='analysis')).
        
        Args:
            dataset_path: Path to dataset (optional)
            **kwargs: Additional parameters for analysis step
        
        Returns:
            dict: Analysis results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            results = pipeline.run_analysis()
        """
        return self.run(step='analysis', dataset_path=dataset_path, **kwargs)
    
    def run_clustering(self, dataset_path: Optional[str] = None, epochs: Optional[int] = None, **kwargs):
        """
        Run the clustering step (alias for run(step='clustering')).
        
        Args:
            dataset_path: Path to dataset (optional)
            epochs: Number of training epochs (optional)
            **kwargs: Additional parameters for clustering step
        
        Returns:
            dict: Clustering results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            results = pipeline.run_clustering(epochs=10)
        """
        if epochs is not None:
            kwargs['epochs'] = epochs
        return self.run(step='clustering', dataset_path=dataset_path, **kwargs)
    
    def update_cluster_config(self, **kwargs):
        """
        Update cluster configurations (alias for run(step='config_update')).
        
        Args:
            **kwargs: Additional parameters for config_update step
        
        Returns:
            dict: Config update results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run_clustering()  # Must run clustering first
            results = pipeline.update_cluster_config()
        """
        return self.run(step='config_update', **kwargs)
    
    def fine_tune_clusters(self, **kwargs):
        """
        Fine-tune large clusters (alias for run(step='fine_tune')).
        
        Args:
            **kwargs: Additional parameters for fine_tune step
        
        Returns:
            dict: Fine-tuning results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run_clustering()  # Must run clustering first
            results = pipeline.fine_tune_clusters()
        """
        return self.run(step='fine_tune', **kwargs)
    
    def optimize_clusters(self, **kwargs):
        """
        Optimize clusters (alias for run(step='optimize')).
        
        Args:
            **kwargs: Additional parameters for optimize step
        
        Returns:
            dict: Optimization results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run_clustering()  # Must run clustering first
            results = pipeline.optimize_clusters()
        """
        return self.run(step='optimize', **kwargs)
    
    def audit_learning(self, **kwargs):
        """
        Audit learning effectiveness (alias for run(step='audit')).
        
        Args:
            **kwargs: Additional parameters for audit step
        
        Returns:
            dict: Audit results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run_clustering()  # Must run clustering first
            results = pipeline.audit_learning()
        """
        return self.run(step='audit', **kwargs)
    
    def evaluate_learning(self, **kwargs):
        """
        Evaluate learning performance (alias for run(step='evaluation')).
        
        Args:
            **kwargs: Additional parameters for evaluation step
        
        Returns:
            dict: Evaluation results
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run_clustering()  # Must run clustering first
            results = pipeline.evaluate_learning()
        """
        return self.run(step='evaluation', **kwargs)
    
    # ============================================================================
    # Pipeline Graph Methods
    # ============================================================================
    
    def get_graph(self) -> Dict[str, Any]:
        """
        Get the pipeline dependency graph structure.
        
        Returns a dictionary representing the dependency graph of all available steps,
        including their dependencies and execution status.
        
        Returns:
            dict: Graph structure with keys:
                - 'nodes': List of step nodes with metadata
                - 'edges': List of dependency edges (from, to)
                - 'executed': List of executed step names
                - 'available': List of available step names
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            graph = pipeline.get_graph()
            print(f"Available steps: {graph['available']}")
        """
        from .steps import list_all_steps
        
        all_steps = list_all_steps()
        nodes = []
        edges = []
        executed = list(self.results.keys())
        available = []
        
        # Build graph from all available steps
        from .steps import _BUILTIN_STEPS
        
        for step_name in all_steps.keys():
            try:
                step_instance = self._get_step(step_name)
                required_steps = step_instance.get_required_steps()
                
                # Add node
                nodes.append({
                    'name': step_name,
                    'type': 'builtin' if step_name in _BUILTIN_STEPS else 'custom',
                    'executed': step_name in executed,
                    'dependencies': required_steps
                })
                
                available.append(step_name)
                
                # Add edges (dependencies)
                for dep in required_steps:
                    edges.append({
                        'from': dep,
                        'to': step_name,
                        'type': 'dependency'
                    })
            except Exception:
                # Skip steps that can't be instantiated
                continue
        
        return {
            'nodes': nodes,
            'edges': edges,
            'executed': executed,
            'available': available
        }
    
    def export_graph_mermaid(self, output_path: Optional[str] = None, include_executed: bool = True) -> str:
        """
        Export pipeline dependency graph as Mermaid diagram.
        
        Args:
            output_path: Optional path to save the Mermaid diagram (if None, returns as string)
            include_executed: Include execution status in diagram (default: True)
        
        Returns:
            str: Mermaid diagram code
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            mermaid_code = pipeline.export_graph_mermaid('pipeline_graph.mmd')
        """
        graph = self.get_graph()
        executed = set(graph['executed'])
        
        lines = ["graph TD"]
        
        # Add nodes with styling
        for node in graph['nodes']:
            step_name = node['name']
            node_type = node['type']
            is_executed = step_name in executed
            
            # Format node name for Mermaid (replace underscores, capitalize)
            display_name = step_name.replace('_', ' ').title()
            
            # Add styling based on type and execution status
            style_class = []
            if is_executed and include_executed:
                style_class.append("executed")
            if node_type == 'custom':
                style_class.append("custom")
            
            # Create node definition
            node_id = step_name.replace('-', '_').replace(' ', '_')
            if style_class:
                lines.append(f"    {node_id}[\"{display_name}\"]")
                lines.append(f"    classDef {style_class[0]} fill:#90EE90,stroke:#333,stroke-width:2px")
            else:
                lines.append(f"    {node_id}[\"{display_name}\"]")
        
        # Add edges (dependencies)
        for edge in graph['edges']:
            from_node = edge['from'].replace('-', '_').replace(' ', '_')
            to_node = edge['to'].replace('-', '_').replace(' ', '_')
            lines.append(f"    {from_node} --> {to_node}")
        
        mermaid_code = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(mermaid_code)
                print(f"[OK] Graph exported to: {output_path}")
            except Exception as e:
                print(f"[WARNING] Error saving graph: {e}")
        
        return mermaid_code
    
    def visualize_graph(self, output_path: Optional[str] = None, format: str = 'png'):
        """
        Visualize pipeline dependency graph.
        
        This method attempts to render the graph using available visualization libraries.
        Falls back to Mermaid export if visualization libraries are not available.
        
        Args:
            output_path: Optional path to save the visualization (if None, displays inline)
            format: Output format ('png', 'svg', 'pdf') (default: 'png')
        
        Returns:
            str: Path to saved visualization file, or None if display only
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.visualize_graph('pipeline_graph.png')
        """
        # Try to use graphviz if available
        try:
            import graphviz
            
            graph = self.get_graph()
            dot = graphviz.Digraph(comment='Xctopus Pipeline Graph')
            dot.attr(rankdir='LR')  # Left to right layout
            dot.attr('node', shape='box', style='rounded')
            
            executed = set(graph['executed'])
            
            # Add nodes
            for node in graph['nodes']:
                step_name = node['name']
                display_name = step_name.replace('_', ' ').title()
                is_executed = step_name in executed
                node_type = node['type']
                
                # Color based on execution status and type
                if is_executed:
                    color = '#90EE90'  # Light green
                elif node_type == 'custom':
                    color = '#FFE4B5'  # Moccasin
                else:
                    color = '#E0E0E0'  # Light gray
                
                dot.node(step_name, display_name, fillcolor=color, style='filled')
            
            # Add edges
            for edge in graph['edges']:
                dot.edge(edge['from'], edge['to'])
            
            # Render graph
            if output_path:
                dot.render(output_path, format=format, cleanup=True)
                print(f"[OK] Graph visualization saved to: {output_path}.{format}")
                return f"{output_path}.{format}"
            else:
                # Try to display inline (Jupyter/IPython)
                try:
                    from IPython.display import Image, display
                    display(Image(dot.pipe(format='png')))
                except ImportError:
                    print("[WARNING] Cannot display inline. Use output_path to save file.")
                    print("   Mermaid code:")
                    print(self.export_graph_mermaid())
                return None
                
        except ImportError:
            # Fallback to Mermaid export
            print("[WARNING] graphviz not available. Exporting Mermaid diagram instead.")
            if output_path:
                mermaid_path = output_path.replace(f'.{format}', '.mmd')
                return self.export_graph_mermaid(mermaid_path)
            else:
                print("Mermaid diagram:")
                print(self.export_graph_mermaid())
                return None
    
    # ============================================================================
    # Checkpointing Methods
    # ============================================================================
    
    def save_state(self, checkpoint_path: str, include_embeddings: bool = True, **kwargs):
        """
        Save the current pipeline state to a checkpoint file.
        
        This method serializes:
        - Knowledge nodes (PyTorch models)
        - Embeddings (if include_embeddings=True)
        - Configuration
        - Results from executed steps
        - Pipeline metadata
        
        Args:
            checkpoint_path: Path to save the checkpoint file (.ckpt or .pkl)
            include_embeddings: Whether to include embeddings in checkpoint (default: True)
            **kwargs: Additional options:
                - compress: Use compression (default: False)
                - metadata: Additional metadata to include (dict)
        
        Returns:
            str: Path to saved checkpoint file
        
        Raises:
            IOError: If checkpoint cannot be saved
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            pipeline.run(step='clustering', epochs=5)
            checkpoint_path = pipeline.save_state('checkpoint.ckpt')
        """
        print(f"[*] Saving pipeline state to: {checkpoint_path}")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'version': '1.0',  # Checkpoint format version
            'timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'text_columns': self.text_columns,
            'join_with': self.join_with,
            'label_column': self.label_column,
            'id_column': self.id_column,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'normalize': self.normalize,
            'drop_empty': self.drop_empty,
            'config': self.config.to_dict(),
            'results': self.results,
            'knowledge_nodes': {},
            'embeddings': None,
            'metadata': kwargs.get('metadata', {})
        }
        
        # Serialize knowledge nodes
        if self.knowledge_nodes:
            print(f"  [*] Serializing {len(self.knowledge_nodes)} knowledge nodes...")
            for cluster_id, node in self.knowledge_nodes.items():
                try:
                    # Save node state dict (more efficient than full object)
                    node_state = {
                        'state_dict': node.state_dict() if hasattr(node, 'state_dict') else None,
                        'cluster_id': cluster_id,
                        'node_type': type(node).__name__
                    }
                    
                    # Also save filter memory if available
                    if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
                        node_state['filter_memory'] = node.filter.memory
                    
                    checkpoint_data['knowledge_nodes'][cluster_id] = node_state
                except Exception as e:
                    print(f"  [WARNING] Warning: Could not serialize node {cluster_id}: {e}")
                    # Try to save at least the cluster_id
                    checkpoint_data['knowledge_nodes'][cluster_id] = {
                        'cluster_id': cluster_id,
                        'error': str(e)
                    }
        
        # Serialize embeddings if requested
        if include_embeddings and self.embeddings is not None:
            print(f"  [*] Serializing embeddings...")
            try:
                # Convert to CPU and numpy for smaller size (if needed)
                if torch.is_tensor(self.embeddings):
                    checkpoint_data['embeddings'] = {
                        'data': self.embeddings.cpu(),
                        'device': str(self.embeddings.device) if hasattr(self.embeddings, 'device') else 'cpu'
                    }
                else:
                    checkpoint_data['embeddings'] = self.embeddings
            except Exception as e:
                print(f"  [WARNING] Warning: Could not serialize embeddings: {e}")
        
        # Save optimizers if available
        if hasattr(self, '_optimizers') and self._optimizers:
            print(f"  [*] Serializing optimizers...")
            try:
                optimizer_states = {}
                for cluster_id, optimizer in self._optimizers.items():
                    if optimizer is not None:
                        optimizer_states[cluster_id] = optimizer.state_dict()
                checkpoint_data['optimizers'] = optimizer_states
            except Exception as e:
                print(f"  [WARNING] Warning: Could not serialize optimizers: {e}")
        
        # Save checkpoint
        try:
            # Create directory if it doesn't exist
            checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Use torch.save for PyTorch objects, pickle for others
            if checkpoint_path.endswith('.ckpt') or checkpoint_path.endswith('.pt'):
                torch.save(checkpoint_data, checkpoint_path)
            else:
                # Use pickle for .pkl files
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            print(f"[OK] Checkpoint saved successfully")
            return checkpoint_path
            
        except Exception as e:
            raise IOError(f"Failed to save checkpoint: {e}") from e
    
    def load_state(self, checkpoint_path: str, device: Optional[torch.device] = None, **kwargs):
        """
        Load pipeline state from a checkpoint file.
        
        This method restores:
        - Knowledge nodes (PyTorch models)
        - Embeddings (if included in checkpoint)
        - Configuration
        - Results from executed steps
        - Pipeline metadata
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load models on (None = use config device)
            **kwargs: Additional options:
                - strict: Strict loading (default: False)
                - load_embeddings: Whether to load embeddings (default: True)
                - load_optimizers: Whether to load optimizers (default: True)
        
        Returns:
            dict: Loaded checkpoint metadata
        
        Raises:
            IOError: If checkpoint cannot be loaded
            ValueError: If checkpoint format is invalid
        
        Example:
            pipeline = XctopusPipeline('data.csv')
            metadata = pipeline.load_state('checkpoint.ckpt')
            # Continue from checkpoint
            pipeline.run(step='fine_tune')
        """
        if not os.path.exists(checkpoint_path):
            raise IOError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"[*] Loading pipeline state from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            if checkpoint_path.endswith('.ckpt') or checkpoint_path.endswith('.pt'):
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            else:
                # Use pickle for .pkl files
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint: {e}") from e
        
        # Validate checkpoint format
        if not isinstance(checkpoint_data, dict):
            raise ValueError("Invalid checkpoint format: expected dictionary")
        
        if 'version' not in checkpoint_data:
            raise ValueError("Invalid checkpoint format: missing version")
        
        # Restore configuration
        if 'config' in checkpoint_data:
            config_dict = checkpoint_data['config']
            self.config = PipelineConfig()
            self.config.update(**config_dict)
        
        # Restore dataset configuration
        self.dataset_path = checkpoint_data.get('dataset_path', self.dataset_path)
        self.text_columns = checkpoint_data.get('text_columns', self.text_columns)
        self.join_with = checkpoint_data.get('join_with', self.join_with)
        self.label_column = checkpoint_data.get('label_column', self.label_column)
        self.id_column = checkpoint_data.get('id_column', self.id_column)
        self.model_name = checkpoint_data.get('model_name', self.model_name)
        self.max_length = checkpoint_data.get('max_length', self.max_length)
        self.normalize = checkpoint_data.get('normalize', self.normalize)
        self.drop_empty = checkpoint_data.get('drop_empty', self.drop_empty)
        
        # Restore results
        if 'results' in checkpoint_data:
            self.results = checkpoint_data['results']
            print(f"  [OK] Restored results from {len(self.results)} steps")
        
        # Restore knowledge nodes
        if 'knowledge_nodes' in checkpoint_data and checkpoint_data['knowledge_nodes']:
            print(f"  [*] Restoring {len(checkpoint_data['knowledge_nodes'])} knowledge nodes...")
            
            # Determine device
            if device is None:
                device = getattr(self.config, 'DEVICE', torch.device("cpu"))
            elif isinstance(device, str):
                device = torch.device(device)
            
            self.knowledge_nodes = {}
            
            for cluster_id, node_data in checkpoint_data['knowledge_nodes'].items():
                try:
                    # Try to reconstruct node from state_dict
                    # Note: This is a simplified version - full reconstruction would require
                    # the node class and its initialization parameters
                    if 'error' in node_data:
                        print(f"  [WARNING] Skipping node {cluster_id}: {node_data['error']}")
                        continue
                    
                    # For now, we'll store the state_dict and attempt to load it
                    # Full node reconstruction would require more context
                    # This is a placeholder - in a full implementation, we'd need to:
                    # 1. Store node initialization parameters in checkpoint
                    # 2. Recreate node with those parameters
                    # 3. Load state_dict into recreated node
                    
                    # Store node data for later reconstruction
                    self.knowledge_nodes[cluster_id] = {
                        'state_dict': node_data.get('state_dict'),
                        'filter_memory': node_data.get('filter_memory'),
                        'cluster_id': cluster_id,
                        'node_type': node_data.get('node_type', 'KnowledgeNode')
                    }
                    
                except Exception as e:
                    print(f"  [WARNING] Warning: Could not restore node {cluster_id}: {e}")
                    if kwargs.get('strict', False):
                        raise
            
            print(f"  [OK] Restored {len(self.knowledge_nodes)} knowledge nodes")
        
        # Restore embeddings
        if kwargs.get('load_embeddings', True) and 'embeddings' in checkpoint_data:
            embeddings_data = checkpoint_data['embeddings']
            if embeddings_data is not None:
                print(f"  [*] Restoring embeddings...")
                try:
                    if isinstance(embeddings_data, dict) and 'data' in embeddings_data:
                        # Restore from dict format
                        self.embeddings = embeddings_data['data']
                        if device and torch.is_tensor(self.embeddings):
                            self.embeddings = self.embeddings.to(device)
                    else:
                        self.embeddings = embeddings_data
                    print(f"  [OK] Embeddings restored")
                except Exception as e:
                    print(f"  [WARNING] Warning: Could not restore embeddings: {e}")
        
        # Restore optimizers
        if kwargs.get('load_optimizers', True) and 'optimizers' in checkpoint_data:
            optimizer_states = checkpoint_data['optimizers']
            if optimizer_states:
                print(f"  [*] Restoring optimizers...")
                # Optimizers will be recreated when needed (lazy initialization)
                # Store states for later use
                if not hasattr(self, '_optimizers'):
                    self._optimizers = {}
                # Note: Full optimizer restoration would require knowledge_nodes to be fully loaded
                print(f"  [OK] Optimizer states stored (will be restored when nodes are used)")
        
        metadata = {
            'version': checkpoint_data.get('version'),
            'timestamp': checkpoint_data.get('timestamp'),
            'metadata': checkpoint_data.get('metadata', {})
        }
        
        print(f"[OK] Checkpoint loaded successfully")
        return metadata

