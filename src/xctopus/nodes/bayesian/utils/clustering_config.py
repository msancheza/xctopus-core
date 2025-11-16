"""
Configuration for Dynamic Clustering Pipeline.

This module provides DynamicClusteringConfig class for configuring
the dynamic clustering pipeline with customizable parameters.
"""

import torch

# Try to import HDBSCAN (optional)
HDBSCAN_AVAILABLE = False
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class DynamicClusteringConfig:
    """
    Configuration class for dynamic clustering pipeline.
    
    All parameters can be customized when instantiating the class.
    If a parameter is not provided, it will use the default value.
    
    Example:
        >>> # Use all defaults
        >>> config = DynamicClusteringConfig()
        
        >>> # Customize specific parameters
        >>> config = DynamicClusteringConfig(
        ...     EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2',
        ...     MIN_CLUSTER_SIZE=5,
        ...     NUM_EPOCHS=5
        ... )
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with customizable parameters.
        
        Args:
            **kwargs: Any configuration parameter to override defaults.
                     See class attributes for available parameters.
        """
        # Embeddings configuration
        self.EMBEDDING_MODEL = kwargs.get(
            'EMBEDDING_MODEL',
            'distiluse-base-multilingual-cased-v1'
        )
        # Options: 'sentence-transformers/all-MiniLM-L6-v2' (general)
        #          'allenai/scibert_scivocab_uncased' (sciences)
        #          'bert-base-uncased' (general)
        #          'distiluse-base-multilingual-cased-v1' (multilingual)
        self.NORMALIZE_EMBEDDINGS = kwargs.get('NORMALIZE_EMBEDDINGS', True)
        
        # Preprocessing
        self.MIN_CLUSTER_SIZE = kwargs.get('MIN_CLUSTER_SIZE', 3)
        self.MIN_SAMPLES = kwargs.get('MIN_SAMPLES', 2)
        self.CLUSTER_SELECTION_METHOD = kwargs.get('CLUSTER_SELECTION_METHOD', 'leaf')
        self.CLUSTER_SELECTION_EPSILON = kwargs.get('CLUSTER_SELECTION_EPSILON', 0.0)
        
        # Initial clustering
        self.USE_HDBSCAN = kwargs.get('USE_HDBSCAN', HDBSCAN_AVAILABLE)
        self.LINKAGE_METHOD = kwargs.get('LINKAGE_METHOD', 'average')
        self.METRIC = kwargs.get('METRIC', 'cosine')
        
        # Evaluation
        self.MIN_SILHOUETTE = kwargs.get('MIN_SILHOUETTE', 0.3)
        self.MAX_DB_INDEX = kwargs.get('MAX_DB_INDEX', 2.0)
        
        # Adaptive fusion
        self.FUSION_PERCENTILE = kwargs.get('FUSION_PERCENTILE', 25)
        self.MIN_CLUSTER_DISTANCE = kwargs.get('MIN_CLUSTER_DISTANCE', None)
        self.MAX_FUSION_ITERATIONS = kwargs.get('MAX_FUSION_ITERATIONS', 10)
        self.MIN_CLUSTERS_TARGET = kwargs.get('MIN_CLUSTERS_TARGET', 5)
        self.SEMANTIC_FUSION_THRESHOLD = kwargs.get('SEMANTIC_FUSION_THRESHOLD', 0.85)
        
        # Fine-tuning
        self.ADAPTIVE_THRESHOLD = kwargs.get('ADAPTIVE_THRESHOLD', True)
        self.MIN_TEXTS_PER_CLUSTER = kwargs.get('MIN_TEXTS_PER_CLUSTER', 5)
        
        # Dynamic stopping criteria
        self.MIN_TEXTS_PER_NODE = kwargs.get(
            'MIN_TEXTS_PER_NODE',
            {
                'mathematics': 20,
                'aerospace': 10,
                'default': 5
            }
        )
        self.USE_DOMAIN_SPECIFIC_MIN = kwargs.get('USE_DOMAIN_SPECIFIC_MIN', False)
        
        # Auto-update configuration
        self.ENABLE_AUTO_UPDATE_CONFIG = kwargs.get('ENABLE_AUTO_UPDATE_CONFIG', True)
        self.ENABLE_ADDITIONAL_EPOCH_AFTER_UPDATE = kwargs.get(
            'ENABLE_ADDITIONAL_EPOCH_AFTER_UPDATE',
            True
        )
        
        # Decision logging
        self.LOG_FUSION_DECISIONS = kwargs.get('LOG_FUSION_DECISIONS', True)
        self.MIN_FUSION_SIMILARITY = kwargs.get('MIN_FUSION_SIMILARITY', 0.5)
        
        # Minimum threshold for creating KnowledgeNodes
        self.MIN_CLUSTER_SIZE_FOR_NODE = kwargs.get('MIN_CLUSTER_SIZE_FOR_NODE', 50)
        
        # Architecture
        self.D_MODEL = kwargs.get('D_MODEL', 128)
        self.DEVICE = kwargs.get('DEVICE', torch.device("cpu"))
        self.SRC_VOCAB_SIZE = kwargs.get('SRC_VOCAB_SIZE', 5000)
        self.TGT_VOCAB_SIZE = kwargs.get('TGT_VOCAB_SIZE', 5000)
        self.D_FF = kwargs.get('D_FF', 512)
        self.NUM_HEADS = kwargs.get('NUM_HEADS', 4)
        self.MAX_SEQ_LENGTH = kwargs.get('MAX_SEQ_LENGTH', 1)
        self.DROPOUT = kwargs.get('DROPOUT', 0.1)
        
        # LoRA
        self.USE_LORA = kwargs.get('USE_LORA', True)
        self.LORA_R = kwargs.get('LORA_R', 4)
        self.LORA_ALPHA = kwargs.get('LORA_ALPHA', 1.0)
        self.LORA_LEARNING_RATE = kwargs.get('LORA_LEARNING_RATE', 1e-3)
        
        # Training
        self.NUM_EPOCHS = kwargs.get('NUM_EPOCHS', 3)
        self.ENABLE_TRAINING = kwargs.get('ENABLE_TRAINING', True)
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'EMBEDDING_MODEL': self.EMBEDDING_MODEL,
            'NORMALIZE_EMBEDDINGS': self.NORMALIZE_EMBEDDINGS,
            'MIN_CLUSTER_SIZE': self.MIN_CLUSTER_SIZE,
            'MIN_SAMPLES': self.MIN_SAMPLES,
            'CLUSTER_SELECTION_METHOD': self.CLUSTER_SELECTION_METHOD,
            'CLUSTER_SELECTION_EPSILON': self.CLUSTER_SELECTION_EPSILON,
            'USE_HDBSCAN': self.USE_HDBSCAN,
            'LINKAGE_METHOD': self.LINKAGE_METHOD,
            'METRIC': self.METRIC,
            'MIN_SILHOUETTE': self.MIN_SILHOUETTE,
            'MAX_DB_INDEX': self.MAX_DB_INDEX,
            'FUSION_PERCENTILE': self.FUSION_PERCENTILE,
            'MIN_CLUSTER_DISTANCE': self.MIN_CLUSTER_DISTANCE,
            'MAX_FUSION_ITERATIONS': self.MAX_FUSION_ITERATIONS,
            'MIN_CLUSTERS_TARGET': self.MIN_CLUSTERS_TARGET,
            'SEMANTIC_FUSION_THRESHOLD': self.SEMANTIC_FUSION_THRESHOLD,
            'ADAPTIVE_THRESHOLD': self.ADAPTIVE_THRESHOLD,
            'MIN_TEXTS_PER_CLUSTER': self.MIN_TEXTS_PER_CLUSTER,
            'MIN_TEXTS_PER_NODE': self.MIN_TEXTS_PER_NODE,
            'USE_DOMAIN_SPECIFIC_MIN': self.USE_DOMAIN_SPECIFIC_MIN,
            'ENABLE_AUTO_UPDATE_CONFIG': self.ENABLE_AUTO_UPDATE_CONFIG,
            'ENABLE_ADDITIONAL_EPOCH_AFTER_UPDATE': self.ENABLE_ADDITIONAL_EPOCH_AFTER_UPDATE,
            'LOG_FUSION_DECISIONS': self.LOG_FUSION_DECISIONS,
            'MIN_FUSION_SIMILARITY': self.MIN_FUSION_SIMILARITY,
            'MIN_CLUSTER_SIZE_FOR_NODE': self.MIN_CLUSTER_SIZE_FOR_NODE,
            'D_MODEL': self.D_MODEL,
            'DEVICE': str(self.DEVICE),
            'SRC_VOCAB_SIZE': self.SRC_VOCAB_SIZE,
            'TGT_VOCAB_SIZE': self.TGT_VOCAB_SIZE,
            'D_FF': self.D_FF,
            'NUM_HEADS': self.NUM_HEADS,
            'MAX_SEQ_LENGTH': self.MAX_SEQ_LENGTH,
            'DROPOUT': self.DROPOUT,
            'USE_LORA': self.USE_LORA,
            'LORA_R': self.LORA_R,
            'LORA_ALPHA': self.LORA_ALPHA,
            'LORA_LEARNING_RATE': self.LORA_LEARNING_RATE,
            'NUM_EPOCHS': self.NUM_EPOCHS,
            'ENABLE_TRAINING': self.ENABLE_TRAINING,
        }
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

