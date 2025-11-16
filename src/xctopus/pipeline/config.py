"""
Pipeline Configuration Module

Provides PipelineConfig class for managing pipeline configuration.
Supports YAML loading, auto-detection of text columns, and column validation.
"""

import os
import yaml
import pandas as pd
import torch
from typing import List, Optional, Dict, Any


class PipelineConfig:
    """
    Configuration class for XctopusPipeline.
    
    This class holds all configuration parameters needed by the pipeline.
    In Phase 0, we implement only basic attributes with sensible defaults.
    YAML support will be added in Phase 2.
    
    Attributes:
        # Dataset configuration
        TEXT_COLUMNS: List of column names to use for text extraction
        JOIN_WITH: Separator for joining multiple text columns
        LABEL_COLUMN: Column name containing labels (None for unsupervised)
        ID_COLUMN: Column name containing unique identifiers
        DROP_EMPTY: Whether to drop rows with empty text
        
        # Embeddings configuration
        EMBEDDING_MODEL: Name of the SentenceTransformer model
        MAX_LENGTH: Maximum token length
        NORMALIZE_EMBEDDINGS: Whether to normalize embeddings to unit norm
        
        # Pipeline configuration
        NUM_EPOCHS: Number of training epochs
        LEARNING_RATE: Learning rate for training
        ENABLE_FINE_TUNE_LARGE: Whether to enable fine-tuning of large clusters
        ENABLE_AUTO_UPDATE: Whether to enable automatic config updates
        
        # Clustering configuration
        MIN_CLUSTER_SIZE: Minimum cluster size
        ORPHAN_THRESHOLD: Size threshold for orphan clusters
        MERGE_SIMILARITY_THRESHOLD: Similarity threshold for merging clusters
        
        # Evaluation configuration
        ENABLE_AUDIT: Whether to enable learning audit
        ENABLE_PERFORMANCE_EVAL: Whether to enable performance evaluation
        
        # Auto-detection configuration (Phase 2)
        AUTO_DETECT_TEXT_COLUMNS: Whether to auto-detect text columns
        SUGGESTED_TEXT_COLUMNS: Fallback suggestions for text columns
        VALIDATE_COLUMNS: Whether to validate column existence
        STRICT_MODE: Whether to raise errors on validation failures
    """
    
    def __init__(self):
        """Initialize PipelineConfig with default values."""
        
        # Dataset configuration
        self.TEXT_COLUMNS = None  # None = auto-detect (Phase 2)
        self.JOIN_WITH = "\n"
        self.LABEL_COLUMN = None  # None = unsupervised learning
        self.ID_COLUMN = None
        self.DROP_EMPTY = False
        
        # Embeddings configuration
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_LENGTH = 512
        self.NORMALIZE_EMBEDDINGS = True
        
        # Pipeline configuration
        self.NUM_EPOCHS = 5
        self.LEARNING_RATE = 0.001
        self.ENABLE_FINE_TUNE_LARGE = True
        self.ENABLE_AUTO_UPDATE = True
        
        # Clustering configuration
        self.MIN_CLUSTER_SIZE = 8
        self.ORPHAN_THRESHOLD = 3
        self.MERGE_SIMILARITY_THRESHOLD = 0.7
        
        # Evaluation configuration
        self.ENABLE_AUDIT = True
        self.ENABLE_PERFORMANCE_EVAL = True
        
        # Auto-detection configuration (Phase 2 - placeholders)
        self.AUTO_DETECT_TEXT_COLUMNS = True
        self.SUGGESTED_TEXT_COLUMNS = None
        self.VALIDATE_COLUMNS = True
        self.STRICT_MODE = False
        
        # Clustering-specific configuration (for ClusteringStep)
        self.D_MODEL = 128
        self.DEVICE = torch.device("cpu")
        self.LORA_LEARNING_RATE = 1e-3
        self.ENABLE_TRAINING = True
        self.INITIAL_THRESHOLD = 0.75
        self.MIN_THRESHOLD = 0.5
        self.THRESHOLD_DECAY = 0.95
        self.MIN_CLUSTER_SIZE = 8
        self.MERGE_SIMILARITY_THRESHOLD = 0.7
        self.SRC_VOCAB_SIZE = 5000
        self.TGT_VOCAB_SIZE = 5000
        self.D_FF = 512
        self.NUM_HEADS = 4
        self.MAX_SEQ_LENGTH = 1
        self.DROPOUT = 0.1
        self.LORA_ALPHA = 1.0
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Dictionary representation of configuration
        """
        return {
            # Dataset
            'TEXT_COLUMNS': self.TEXT_COLUMNS,
            'JOIN_WITH': self.JOIN_WITH,
            'LABEL_COLUMN': self.LABEL_COLUMN,
            'ID_COLUMN': self.ID_COLUMN,
            'DROP_EMPTY': self.DROP_EMPTY,
            
            # Embeddings
            'EMBEDDING_MODEL': self.EMBEDDING_MODEL,
            'MAX_LENGTH': self.MAX_LENGTH,
            'NORMALIZE_EMBEDDINGS': self.NORMALIZE_EMBEDDINGS,
            
            # Pipeline
            'NUM_EPOCHS': self.NUM_EPOCHS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'ENABLE_FINE_TUNE_LARGE': self.ENABLE_FINE_TUNE_LARGE,
            'ENABLE_AUTO_UPDATE': self.ENABLE_AUTO_UPDATE,
            
            # Clustering
            'MIN_CLUSTER_SIZE': self.MIN_CLUSTER_SIZE,
            'ORPHAN_THRESHOLD': self.ORPHAN_THRESHOLD,
            'MERGE_SIMILARITY_THRESHOLD': self.MERGE_SIMILARITY_THRESHOLD,
            
            # Evaluation
            'ENABLE_AUDIT': self.ENABLE_AUDIT,
            'ENABLE_PERFORMANCE_EVAL': self.ENABLE_PERFORMANCE_EVAL,
            
            # Auto-detection
            'AUTO_DETECT_TEXT_COLUMNS': self.AUTO_DETECT_TEXT_COLUMNS,
            'SUGGESTED_TEXT_COLUMNS': self.SUGGESTED_TEXT_COLUMNS,
            'VALIDATE_COLUMNS': self.VALIDATE_COLUMNS,
            'STRICT_MODE': self.STRICT_MODE,
            
            # Clustering-specific
            'D_MODEL': self.D_MODEL,
            'DEVICE': str(self.DEVICE),  # Convert device to string for serialization
            'LORA_LEARNING_RATE': self.LORA_LEARNING_RATE,
            'ENABLE_TRAINING': self.ENABLE_TRAINING,
            'INITIAL_THRESHOLD': self.INITIAL_THRESHOLD,
            'MIN_THRESHOLD': self.MIN_THRESHOLD,
            'THRESHOLD_DECAY': self.THRESHOLD_DECAY,
            'SRC_VOCAB_SIZE': self.SRC_VOCAB_SIZE,
            'TGT_VOCAB_SIZE': self.TGT_VOCAB_SIZE,
            'D_FF': self.D_FF,
            'NUM_HEADS': self.NUM_HEADS,
            'MAX_SEQ_LENGTH': self.MAX_SEQ_LENGTH,
            'DROPOUT': self.DROPOUT,
            'LORA_ALPHA': self.LORA_ALPHA,
        }
    
    def update(self, **kwargs):
        """
        Update configuration attributes.
        
        Args:
            **kwargs: Attribute names and values to update
            
        Example:
            config = PipelineConfig()
            config.update(NUM_EPOCHS=10, LEARNING_RATE=0.0001)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"PipelineConfig has no attribute '{key}'. "
                    f"Available attributes: {list(self.to_dict().keys())}"
                )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            PipelineConfig: Configuration instance loaded from YAML
        
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid
        
        Example:
            config = PipelineConfig.from_yaml('config.yaml')
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        config = cls()
        
        # Map YAML structure to config attributes
        # YAML structure matches the design document
        if 'dataset' in yaml_data:
            dataset = yaml_data['dataset']
            config.TEXT_COLUMNS = dataset.get('text_columns')
            config.JOIN_WITH = dataset.get('join_with', config.JOIN_WITH)
            config.LABEL_COLUMN = dataset.get('label_column')
            config.ID_COLUMN = dataset.get('id_column')
            config.DROP_EMPTY = dataset.get('drop_empty', config.DROP_EMPTY)
            config.AUTO_DETECT_TEXT_COLUMNS = dataset.get('auto_detect_text_columns', config.AUTO_DETECT_TEXT_COLUMNS)
            config.SUGGESTED_TEXT_COLUMNS = dataset.get('suggested_text_columns')
            config.VALIDATE_COLUMNS = dataset.get('validate_columns', config.VALIDATE_COLUMNS)
            config.STRICT_MODE = dataset.get('strict_mode', config.STRICT_MODE)
        
        if 'embeddings' in yaml_data:
            embeddings = yaml_data['embeddings']
            config.EMBEDDING_MODEL = embeddings.get('model_name', config.EMBEDDING_MODEL)
            config.MAX_LENGTH = embeddings.get('max_length', config.MAX_LENGTH)
            config.NORMALIZE_EMBEDDINGS = embeddings.get('normalize', config.NORMALIZE_EMBEDDINGS)
        
        if 'pipeline' in yaml_data:
            pipeline = yaml_data['pipeline']
            config.NUM_EPOCHS = pipeline.get('epochs', config.NUM_EPOCHS)
            config.LEARNING_RATE = pipeline.get('learning_rate', config.LEARNING_RATE)
            config.ENABLE_FINE_TUNE_LARGE = pipeline.get('enable_fine_tune', config.ENABLE_FINE_TUNE_LARGE)
            config.ENABLE_AUTO_UPDATE = pipeline.get('enable_auto_update', config.ENABLE_AUTO_UPDATE)
        
        if 'clustering' in yaml_data:
            clustering = yaml_data['clustering']
            config.MIN_CLUSTER_SIZE = clustering.get('min_cluster_size', config.MIN_CLUSTER_SIZE)
            config.ORPHAN_THRESHOLD = clustering.get('orphan_threshold', config.ORPHAN_THRESHOLD)
        
        if 'evaluation' in yaml_data:
            evaluation = yaml_data['evaluation']
            config.ENABLE_AUDIT = evaluation.get('enable_audit', config.ENABLE_AUDIT)
            config.ENABLE_PERFORMANCE_EVAL = evaluation.get('enable_performance', config.ENABLE_PERFORMANCE_EVAL)
        
        return config
    
    @staticmethod
    def detect_text_columns(dataset_path: str) -> List[str]:
        """
        Automatically detect text columns in a dataset.
        
        Strategy:
        1. Look for columns with common names: ['text', 'content', 'body', 'abstract', 
           'title', 'description', 'summary', 'notes', 'comment']
        2. Analyze data types: columns with dtype 'object' (strings)
        3. Validate content: columns with >80% non-null values and valid text
        4. Return list ordered by priority (most common first)
        
        Args:
            dataset_path: Path to CSV dataset file
        
        Returns:
            list: List of detected column names, or [] if none found
        
        Example:
            columns = PipelineConfig.detect_text_columns('data.csv')
            print(f"Detected columns: {columns}")
        """
        try:
            df = pd.read_csv(dataset_path, nrows=1000)  # Sample for quick analysis
        except Exception as e:
            raise ValueError(f"Error reading dataset: {str(e)}")
        
        # Common text column names
        common_text_names = [
            'text', 'content', 'body', 'abstract', 'title', 
            'description', 'summary', 'notes', 'comment', 'message',
            'article', 'post', 'review', 'document'
        ]
        
        detected = []
        
        # 1. Search by name
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in common_text_names):
                if df[col].dtype == 'object':  # String type
                    # Validate that it has text content
                    non_null_ratio = df[col].notna().sum() / len(df)
                    if non_null_ratio > 0.8:  # 80% non-null
                        detected.append((col, 'name_match', non_null_ratio))
        
        # 2. Search by data type (fallback)
        if not detected:
            for col in df.columns:
                if df[col].dtype == 'object':
                    non_null_ratio = df[col].notna().sum() / len(df)
                    avg_length = df[col].astype(str).str.len().mean()
                    if non_null_ratio > 0.8 and avg_length > 10:  # Significant text
                        detected.append((col, 'type_match', non_null_ratio))
        
        # Sort by priority (non-null ratio)
        detected.sort(key=lambda x: x[2], reverse=True)
        
        return [col[0] for col in detected]
    
    @staticmethod
    def suggest_text_columns(dataset_path: str, suggested: Optional[List[str]] = None) -> List[str]:
        """
        Suggest text columns using auto-detection + manual suggestions.
        
        Args:
            dataset_path: Path to dataset
            suggested: List of manual suggestions (fallback)
        
        Returns:
            list: Suggested column names
        """
        # Try auto-detection first
        detected = PipelineConfig.detect_text_columns(dataset_path)
        
        if detected:
            return detected
        
        # Fallback to manual suggestions
        if suggested:
            try:
                df = pd.read_csv(dataset_path, nrows=1)
                # Validate that suggestions exist in dataset
                valid_suggestions = [col for col in suggested if col in df.columns]
                if valid_suggestions:
                    return valid_suggestions
            except Exception:
                pass
        
        # Last fallback: 'text' column or first object column
        try:
            df = pd.read_csv(dataset_path, nrows=1)
            if 'text' in df.columns:
                return ['text']
            
            object_cols = [col for col in df.columns if df[col].dtype == 'object']
            if object_cols:
                return [object_cols[0]]
        except Exception:
            pass
        
        return []  # No text columns found
    
    @staticmethod
    def _get_available_columns(dataset_path: str) -> List[str]:
        """
        Get list of available columns in the dataset.
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            list: List of column names
        """
        try:
            df = pd.read_csv(dataset_path, nrows=1)
            return list(df.columns)
        except Exception:
            return []
    
    @staticmethod
    def validate_dataset_columns(
        dataset_path: str,
        text_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
        id_column: Optional[str] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that specified columns exist in the dataset.
        
        Args:
            dataset_path: Path to dataset
            text_columns: List of text column names
            label_column: Label column name
            id_column: ID column name
            strict: If True, raise errors. If False, only warnings.
        
        Returns:
            dict: Validation result with keys:
                - 'valid': bool - Whether validation passed
                - 'warnings': list - List of warning messages
                - 'errors': list - List of error messages
                - 'available_columns': list - List of available columns
        
        Raises:
            ValueError: If strict=True and validation fails
        """
        try:
            df = pd.read_csv(dataset_path, nrows=1)  # Only read headers
        except Exception as e:
            error_msg = f"Error reading dataset: {str(e)}"
            if strict:
                raise ValueError(error_msg)
            return {
                'valid': False,
                'warnings': [],
                'errors': [error_msg],
                'available_columns': []
            }
        
        available_cols = set(df.columns)
        warnings = []
        errors = []
        
        # Validate text_columns
        if text_columns:
            missing = [col for col in text_columns if col not in available_cols]
            if missing:
                msg = f"Text columns not found: {missing}"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        
        # Validate label_column
        if label_column and label_column not in available_cols:
            msg = f"Label column not found: {label_column}"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        
        # Validate id_column
        if id_column and id_column not in available_cols:
            msg = f"ID column not found: {id_column}"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        
        if errors:
            raise ValueError("Validation errors:\n" + "\n".join(errors))
        
        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'available_columns': list(available_cols)
        }

