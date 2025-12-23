"""
KnowledgeNode for Clustering Layer.

Node core: contains Transformer (standby), LoRA Adapter and LocalFilter.
Processes embeddings and maintains statistics (centroid, mass, variance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from .settings import (
    S_MIN,
    EMBEDDING_DIM,
    DEVICE,
    DTYPE,
    MODEL_BASE_NAME,
    LORA_RANK_DEFAULT,
)

logger = logging.getLogger(__name__)


class LocalFilter:
    """
    Local filter for embedding membership validation.
    
    Protects the semantic purity of the KnowledgeNode by validating that
    new embeddings are sufficiently similar to the centroid.
    """
    
    def __init__(self, threshold: float = S_MIN):
        """
        Initialize the local filter.
        
        Args:
            threshold: Minimum similarity threshold (default: S_MIN from settings)
        """
        self.threshold = threshold
        logger.debug(f"LocalFilter initialized with threshold={threshold:.3f}")
    
    def accepts(self, embedding: torch.Tensor, centroid: Optional[torch.Tensor]) -> bool:
        """
        Validate if an embedding belongs to the node.
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
            centroid: Node centroid [EMBEDDING_DIM] or None if it's the first embedding
        
        Returns:
            bool: True if embedding exceeds threshold, False otherwise
        """
        # If no centroid (first embedding), always accept
        if centroid is None:
            logger.debug("First embedding, automatically accepted")
            return True
        
        # Ensure device and dtype
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        centroid = centroid.to(device=DEVICE, dtype=DTYPE)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            centroid.unsqueeze(0),
            dim=1
        ).item()
        
        accepted = similarity >= self.threshold
        
        if not accepted:
            logger.debug(
                f"Embedding rejected by LocalFilter: "
                f"similarity={similarity:.3f} < threshold={self.threshold:.3f}"
            )
        
        return accepted


class KnowledgeNode:
    """
    Knowledge Node for Clustering Layer.
    
    Contains:
    - Transformer (standby for Layer 1 - structure only)
    - LoRA Adapter (initialized but not trained)
    - LocalFilter (membership validation)
    - Internal state: centroid, mass, variance (M2)
    
    For Layer 1 (Clustering), the focus is on organic organization
    through statistics, not on Transformer training.
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        initial_centroid: Optional[torch.Tensor] = None,
        initial_mass: int = 0,
        initial_variance: float = 0.0,
    ):
        """
        Initialize a KnowledgeNode.
        
        Args:
            node_id: Unique identifier (default: generated UUID)
            initial_centroid: Initial centroid (default: None)
            initial_mass: Initial mass (default: 0)
            initial_variance: Initial M2 variance (default: 0.0)
                Note: Real variance is calculated as M2 / (mass - 1) in get_signature()
        """
        # Unique identifier
        self.node_id = node_id or str(uuid4())
        
        # ========================================================================
        # Transformer (Standby for Layer 1)
        # ========================================================================
        # In Layer 1, Transformer is on standby - structure only
        # Not trained, but initialized for future phases
        # For now, we only store the reference to the base model
        self.model_base_name = MODEL_BASE_NAME
        self.transformer_ready = False  # Flag to indicate it's on standby
        logger.debug(f"Transformer on standby: {self.model_base_name}")
        
        # ========================================================================
        # LoRA Adapter (Initialized but not trained)
        # ========================================================================
        # For Layer 1, LoRA is initialized but not trained
        # We only store parameters for future phases
        self.lora_rank = LORA_RANK_DEFAULT
        self.lora_ready = False  # Flag to indicate it's not trained
        logger.debug(f"LoRA Adapter initialized (rank={self.lora_rank}, not trained)")
        
        # ========================================================================
        # LocalFilter (Membership Validation)
        # ========================================================================
        self.local_filter = LocalFilter(threshold=S_MIN)
        
        # ========================================================================
        # Internal State: Statistics
        # ========================================================================
        # Centroid: average of embeddings
        if initial_centroid is not None:
            self.centroid = initial_centroid.to(device=DEVICE, dtype=DTYPE)
        else:
            self.centroid = None
        
        # Mass: number of embeddings
        self.mass = initial_mass
        
        # Variance: M2 in Welford's algorithm (sum of squares of differences)
        # Note: Real variance is calculated as M2 / (mass - 1) in get_signature()
        # M2 is initialized with initial_variance (which should be real variance * (mass - 1))
        # if loaded from Repository, or 0.0 if it's a new node
        self.variance = initial_variance
        
        # Note: We don't store embeddings in memory (Repository handles persistence)
        # This avoids duplication and memory issues with large nodes
        
        logger.info(
            f"KnowledgeNode '{self.node_id}' initialized "
            f"(mass={self.mass}, M2={self.variance:.4f})"
        )
    
    def process(self, embedding: torch.Tensor) -> bool:
        """
        Process an embedding and update node statistics.
        
        Flow:
        1. Input validation
        2. LocalFilter validates membership
        3. If rejected → returns False
        4. Updates centroid (Welford's algorithm)
        5. Increments mass
        6. Updates variance M2 (Welford's algorithm)
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
        
        Returns:
            bool: True if embedding was accepted and processed, False if rejected
        
        Raises:
            ValueError: If embedding is empty or has incorrect shape
        """
        # ========================================================================
        # Input Validation
        # ========================================================================
        
        if embedding is None or embedding.numel() == 0:
            logger.warning("Empty embedding received in process()")
            raise ValueError("Embedding cannot be None or empty")
        
        # Ensure it's 1D
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
        
        if embedding.dim() != 1:
            raise ValueError(f"Embedding must be 1D, received shape: {embedding.shape}")
        
        if embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must have dimension {EMBEDDING_DIM}, "
                f"received: {embedding.shape[0]}"
            )
        
        # Ensure device and dtype
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        
        # ========================================================================
        # LocalFilter: Membership Validation
        # ========================================================================
        
        if not self.local_filter.accepts(embedding, self.centroid):
            logger.debug(f"Embedding rejected by LocalFilter in node '{self.node_id}'")
            return False
        
        # ========================================================================
        # Statistics Update (Welford's Algorithm)
        # ========================================================================
        # We use Welford's algorithm for:
        # 1. Centroid: Avoids overflow in FP16 (doesn't multiply centroid * mass)
        # 2. Variance: Calculates real variance (second statistical moment)
        # ========================================================================
        
        # If it's the first embedding, initialize centroid
        if self.centroid is None:
            self.centroid = embedding.clone()
            self.mass = 1
            self.variance = 0.0  # M2 = 0 for the first embedding
            logger.debug(f"First embedding in node '{self.node_id}', centroid initialized")
        else:
            # Increment mass first (necessary for calculation)
            self.mass += 1
            
            # ========================================================================
            # Welford's Algorithm for Centroid
            # ========================================================================
            # Formula: C_n = C_{n-1} + (x_n - C_{n-1}) / n
            # Advantage: Avoids multiplying centroid * mass (overflow risk in FP16)
            # ========================================================================
            delta = embedding - self.centroid
            self.centroid += delta / self.mass
            
            # ========================================================================
            # Welford's Algorithm for Variance (M2)
            # ========================================================================
            # M2_n = M2_{n-1} + (x_n - C_{n-1}) · (x_n - C_n)
            # Where:
            #   - delta = x_n - C_{n-1} (difference from previous centroid)
            #   - delta2 = x_n - C_n (difference from new centroid)
            #   - M2 is the accumulated sum of squares of differences
            # ========================================================================
            delta2 = embedding - self.centroid
            # Dot product: delta · delta2 (sum of squared differences)
            self.variance += torch.dot(delta, delta2).item()
        
        logger.debug(
            f"Embedding processed in node '{self.node_id}': "
            f"mass={self.mass}, M2={self.variance:.4f}"
        )
        
        return True
    
    def get_signature(self) -> Dict[str, Any]:
        """
        Return the statistical signature of the node.
        
        Calculates real variance from M2 (sum of squares):
        - Real variance = M2 / (mass - 1) if mass > 1
        - Real variance = 0.0 if mass <= 1
        
        Returns:
            Dict with keys: node_id, centroid, mass, variance (real variance)
        
        Raises:
            ValueError: If node has no centroid (hasn't processed embeddings yet)
        """
        if self.centroid is None:
            raise ValueError(
                f"Node '{self.node_id}' has no centroid "
                f"(hasn't processed embeddings yet)"
            )
        
        # Calculate real variance from M2
        # Variance = M2 / (n - 1) for sample (Bessel's correction)
        if self.mass > 1:
            real_variance = self.variance / (self.mass - 1)
        else:
            real_variance = 0.0
        
        return {
            "node_id": self.node_id,
            "centroid": self.centroid.clone(),  # Clone to avoid mutations
            "mass": self.mass,
            "variance": real_variance,  # Calculated real variance
        }
