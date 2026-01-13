"""
KnowledgeNode for Clustering Layer.

Refactored: 2025-12-27
Changed from "execution unit" to "data/metadata unit"

Node core: contains LocalFilter and maintains statistics (centroid, mass, variance).
Transformer/LoRA weights are stored in Repository, not in the object (prevents OOM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .repository import KNRepository

from .settings import (
    S_MIN,
    EMBEDDING_DIM,
    DEVICE,
    DTYPE,
    # MODEL_BASE_NAME and LORA_RANK_DEFAULT removed 2025-12-27
    # These were only used for standby Transformer/LoRA references, not needed in Layer 1
)

logger = logging.getLogger(__name__)


class LocalFilter:
    """
    Local filter for embedding membership validation.
    
    Protects the semantic purity of the KnowledgeNode by validating that
    new embeddings are sufficiently similar to the centroid.
    
    Added: 2025-12-27
    Extended with calculate_affinity() for Layer 2 inference ranking.
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
    
    def calculate_affinity(
        self,
        embedding: torch.Tensor,
        centroid: torch.Tensor,
        mass: int,
        variance: Optional[float],
        epsilon: float = 1e-6
    ) -> float:
        """
        Calculate Bayesian affinity considering variance penalty.
        
        Added: 2025-12-27
        Purpose: Return affinity score for Layer 2 inference ranking.
        
        Formula: Affinity = (Mass × Similarity) / (Variance + ε)
        
        Why:
        - High variance = low reliability (semantic uncertainty)
        - Penalizes nodes that "think they know" but are uncertain
        - Protects Transformer from noisy information in Layer 2
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
            centroid: Node centroid [EMBEDDING_DIM]
            mass: Node mass (number of embeddings)
            variance: Node variance (semantic coherence), can be None
            epsilon: Small value to avoid division by zero (default: 1e-6)
        
        Returns:
            float: Affinity score (higher = better match)
        
        Raises:
            ValueError: If embedding or centroid is None or empty
        """
        # ========================================================================
        # Input Validation
        # ========================================================================
        if embedding is None or embedding.numel() == 0:
            raise ValueError("Embedding cannot be None or empty")
        
        if centroid is None or centroid.numel() == 0:
            raise ValueError("Centroid cannot be None or empty")
        
        # Ensure device and dtype
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        centroid = centroid.to(device=DEVICE, dtype=DTYPE)
        
        # ========================================================================
        # Calculate Cosine Similarity
        # ========================================================================
        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            centroid.unsqueeze(0),
            dim=1
        ).item()
        
        # ========================================================================
        # Handle NULL Variance (2025-12-27)
        # ========================================================================
        # If variance is None, treat as 0.0 (no variance penalty)
        # This can happen if node has mass <= 1 (not enough data points)
        if variance is None:
            variance = 0.0
            logger.debug("Variance is None, treating as 0.0 for affinity calculation")
        
        # ========================================================================
        # Validate Mass (2025-12-27)
        # ========================================================================
        # Mass should be >= 0, but if it's 0, affinity should be 0
        if mass <= 0:
            logger.debug(f"Mass is {mass}, returning 0.0 affinity")
            return 0.0
        
        # ========================================================================
        # Calculate Affinity with Variance Penalty
        # ========================================================================
        # Formula: Affinity = (Mass × Similarity) / (Variance + ε)
        # - Higher mass → higher affinity (more data points)
        # - Higher similarity → higher affinity (better match)
        # - Higher variance → lower affinity (penalty for uncertainty)
        # - epsilon prevents division by zero
        denominator = variance + epsilon
        
        # Validation: Ensure denominator is positive (should always be true with epsilon)
        if denominator <= 0:
            logger.warning(
                f"Invalid denominator in affinity calculation: {denominator} "
                f"(variance={variance}, epsilon={epsilon})"
            )
            denominator = epsilon  # Fallback to epsilon
        
        affinity = (mass * similarity) / denominator
        
        logger.debug(
            f"Affinity calculated: {affinity:.4f} "
            f"(mass={mass}, similarity={similarity:.3f}, variance={variance:.4f})"
        )
        
        return affinity


class KnowledgeNode:
    """
    Knowledge Node for Clustering Layer.
    
    Refactored: 2025-12-27
    Changed from "execution unit" to "data/metadata unit"
    
    Contains:
    - LocalFilter (membership validation)
    - Internal state: centroid, mass, variance (M2)
    
    Removed (2025-12-27):
    - Transformer references (not needed in Layer 1)
    - LoRA references (weights stored in Repository, not in object)
    
    For Layer 1 (Clustering), the focus is on organic organization
    through statistics. Transformer/LoRA will be handled in Layer 2.
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
        # Removed: Transformer and LoRA References (2025-12-27)
        # ========================================================================
        # Removed attributes:
        #   - model_base_name: Was only a reference, not used in Layer 1
        #   - transformer_ready: Flag not needed (Transformer not loaded in Layer 1)
        #   - lora_rank: Parameter not used (LoRA not initialized in Layer 1)
        #   - lora_ready: Flag not needed (LoRA not trained in Layer 1)
        # 
        # Reason: KnowledgeNode is now a "data/metadata unit" not an "execution unit"
        # Transformer/LoRA will be handled in Layer 2, weights stored in Repository
        # This reduces memory footprint and aligns with new architecture
        # ========================================================================
        
        # ========================================================================
        # LocalFilter (Membership Validation)
        # ========================================================================
        self.local_filter = LocalFilter(threshold=S_MIN)
        
        # ========================================================================
        # PEFT/LoRA Metadata (2025-12-27)
        # ========================================================================
        # Metadata only - weights are stored in Repository, not loaded in RAM
        # This prevents OOM when multiple KnowledgeNodes are in memory
        # Weights are loaded on-demand via lazy loading when needed
        self.peft_available: bool = False
        self.peft_format: Optional[str] = None  # 'safetensors' or 'bin'
        self.peft_size: Optional[int] = None  # Size in bytes
        self._training_status: Optional[str] = None  # 'TRAINING', 'COMPLETED', 'FAILED', or None
        
        # ========================================================================
        # Phase 2: Training State (2025-01-XX)
        # ========================================================================
        # Local flag for quick access (synchronized with _training_status)
        self.is_training: bool = False
        # Buffer for embeddings received during training (temporary storage)
        self.training_buffer: list = []  # List of torch.Tensor embeddings
        self.training_buffer_source_ids: list = []  # List of source_id strings
        # ========================================================================
        
        # ========================================================================
        # Internal State: Statistics
        # ========================================================================
        # Centroid: average of embeddings
        # Validation: Handle None centroid (first embedding case)
        if initial_centroid is not None:
            # Ensure centroid is on correct device and dtype
            self.centroid = initial_centroid.to(device=DEVICE, dtype=DTYPE)
        else:
            self.centroid = None
        
        # Mass: number of embeddings
        # Validation: Ensure mass is non-negative
        if initial_mass < 0:
            logger.warning(f"Invalid initial_mass: {initial_mass}, setting to 0")
            initial_mass = 0
        self.mass = initial_mass
        
        # Variance: M2 in Welford's algorithm (sum of squares of differences)
        # Note: Real variance is calculated as M2 / (mass - 1) in get_signature()
        # M2 is initialized with initial_variance (which should be real variance * (mass - 1))
        # if loaded from Repository, or 0.0 if it's a new node
        # Validation: Ensure variance is non-negative
        if initial_variance < 0:
            logger.warning(f"Invalid initial_variance: {initial_variance}, setting to 0.0")
            initial_variance = 0.0
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
            # Validation: Ensure mass > 0 to prevent division by zero (2025-12-27)
            # At this point, mass should always be >= 1 (incremented above),
            # but validation prevents potential issues if state is corrupted
            if self.mass <= 0:
                logger.error(f"Invalid mass value in node '{self.node_id}': {self.mass}")
                raise ValueError(f"Mass must be > 0 for centroid update, got {self.mass}")
            
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
        # Validation: Ensure mass > 1 to prevent division by zero (2025-12-27)
        if self.mass > 1:
            # Safe division: mass > 1 guarantees (mass - 1) > 0
            real_variance = self.variance / (self.mass - 1)
        else:
            # For mass <= 1, variance is 0 (not enough data points)
            real_variance = 0.0
        
        return {
            "node_id": self.node_id,
            "centroid": self.centroid.clone(),  # Clone to avoid mutations
            "mass": self.mass,
            "variance": real_variance,  # Calculated real variance
        }
    
    def calculate_affinity(self, embedding: torch.Tensor) -> float:
        """
        Calculate Bayesian affinity for this node.
        
        Added: 2025-12-27
        Purpose: Layer 2 can use this to rank nodes for inference.
        
        Uses LocalFilter.calculate_affinity() with node's current state.
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
        
        Returns:
            float: Affinity score (higher = better match)
        
        Raises:
            ValueError: If node has no centroid (hasn't processed embeddings yet)
        """
        if self.centroid is None:
            raise ValueError(
                f"Node '{self.node_id}' has no centroid "
                f"(hasn't processed embeddings yet)"
            )
        
        # Get real variance from signature (handles mass <= 1 case)
        signature = self.get_signature()
        variance = signature['variance']  # Can be 0.0 if mass <= 1
        
        # Calculate affinity using LocalFilter
        return self.local_filter.calculate_affinity(
            embedding=embedding,
            centroid=self.centroid,
            mass=self.mass,
            variance=variance  # Can be 0.0, not None (get_signature() ensures this)
        )
    
    @classmethod
    def from_repository(
        cls,
        repository: 'KNRepository',
        node_id: str
    ) -> 'KnowledgeNode':
        """
        Create KnowledgeNode instance from Repository data.
        
        Added: 2025-12-27
        Purpose: Load node with signature and PEFT metadata from Repository.
        
        This factory method enables lazy loading of KnowledgeNodes from SQLite,
        loading only metadata (not PEFT weights) to prevent OOM.
        
        Args:
            repository: KNRepository instance
            node_id: Node ID to load
        
        Returns:
            KnowledgeNode instance with data from Repository
        
        Raises:
            ValueError: If node doesn't exist in Repository
        """
        # ========================================================================
        # Load Signature from Repository (2025-12-27)
        # ========================================================================
        signature = repository.get_signature(node_id)
        if not signature:
            raise ValueError(f"KN {node_id} does not exist in Repository")
        
        # ========================================================================
        # Convert Real Variance to M2 (for Welford's Algorithm) (2025-12-27)
        # ========================================================================
        # Repository stores real variance (M2 / (mass - 1))
        # KnowledgeNode stores M2 (sum of squares) internally
        # Conversion: M2 = variance * (mass - 1) if mass > 1, else 0.0
        # ========================================================================
        mass = signature["mass"]
        real_variance = signature["variance"]
        
        # Handle NULL variance (shouldn't happen, but safety check)
        if real_variance is None:
            logger.warning(f"Variance is None for node {node_id}, treating as 0.0")
            real_variance = 0.0
        
        # Convert to M2
        if mass > 1:
            initial_m2 = real_variance * (mass - 1)
        else:
            initial_m2 = 0.0
        
        # ========================================================================
        # Create KnowledgeNode Instance (2025-12-27)
        # ========================================================================
        kn = cls(
            node_id=signature["node_id"],
            initial_centroid=signature["centroid"],
            initial_mass=mass,
            initial_variance=initial_m2
        )
        
        # ========================================================================
        # Load PEFT Metadata (Lazy Loading - Not Weights) (2025-12-27)
        # ========================================================================
        # Only load metadata flags, not actual weights (prevents OOM)
        # Weights are loaded on-demand via get_peft_from_repository() when needed
        # ========================================================================
        kn.peft_available = repository.has_peft_weights(node_id)
        
        if kn.peft_available:
            # Load PEFT metadata (format, size) without loading weights
            peft_data = repository.get_peft_weights(node_id)
            if peft_data:
                # Handle NULL values safely
                kn.peft_format = peft_data.get('format')  # Can be None
                kn.peft_size = peft_data.get('size')  # Can be None
                
                # Validate format is not None if weights exist
                if kn.peft_format is None:
                    logger.warning(
                        f"PEFT weights exist for {node_id} but format is None, "
                        f"setting to 'unknown'"
                    )
                    kn.peft_format = 'unknown'
            else:
                # Inconsistent state: has_peft_weights() returned True but get_peft_weights() returned None
                logger.warning(
                    f"Inconsistent PEFT state for {node_id}: "
                    f"has_peft_weights=True but get_peft_weights() returned None"
                )
                kn.peft_available = False
                kn.peft_format = None
                kn.peft_size = None
        
        # ========================================================================
        # Load Training Status (2025-12-27)
        # ========================================================================
        # Load training status for Layer 2 decision making
        # Can be None (not training), 'TRAINING', 'COMPLETED', or 'FAILED'
        # ========================================================================
        kn._training_status = repository.get_peft_training_status(node_id)
        # Note: _training_status can be None (not training), which is valid
        
        logger.debug(
            f"KnowledgeNode '{node_id}' loaded from Repository "
            f"(mass={mass}, peft_available={kn.peft_available}, "
            f"training_status={kn._training_status})"
        )
        
        return kn
    
    def save_peft_to_repository(
        self,
        repository: 'KNRepository',
        weights_bytes: bytes,
        format: str = "safetensors",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save PEFT weights to Repository.
        
        Added: 2025-12-27
        Purpose: Layer 2 can save trained adapter weights.
        
        This method saves weights to Repository and updates local metadata
        to reflect the new state (prevents need to reload from Repository).
        
        Args:
            repository: KNRepository instance
            weights_bytes: Serialized weights file (bytes)
            format: 'safetensors' or 'bin' (default: 'safetensors')
            config: Optional PEFT configuration dict
        
        Raises:
            ValueError: If weights_bytes is None or empty
            ValueError: If format is invalid
        """
        # ========================================================================
        # Input Validation (2025-12-27)
        # ========================================================================
        if weights_bytes is None:
            raise ValueError("weights_bytes cannot be None")
        
        if not isinstance(weights_bytes, bytes):
            raise ValueError(f"weights_bytes must be bytes, got {type(weights_bytes)}")
        
        if len(weights_bytes) == 0:
            raise ValueError("weights_bytes cannot be empty")
        
        # Validate format
        valid_formats = ['safetensors', 'bin']
        if format not in valid_formats:
            raise ValueError(
                f"Invalid format: {format}. Must be one of {valid_formats}"
            )
        
        # ========================================================================
        # Save to Repository (2025-12-27)
        # ========================================================================
        # Reuse Repository method (no code duplication)
        repository.save_peft_weights(
            self.node_id, weights_bytes, format, config
        )
        
        # ========================================================================
        # Update Local Metadata (2025-12-27)
        # ========================================================================
        # Update local flags to reflect new state (avoids need to reload from Repository)
        self.peft_available = True
        self.peft_format = format
        self.peft_size = len(weights_bytes)
        
        logger.debug(
            f"PEFT weights saved to Repository for node '{self.node_id}' "
            f"(format={format}, size={self.peft_size} bytes)"
        )
    
    def get_peft_from_repository(
        self,
        repository: 'KNRepository'
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve PEFT weights from Repository (lazy loading).
        
        Added: 2025-12-27
        Purpose: Load weights only when needed (prevents OOM).
        
        This method implements lazy loading: weights are not loaded in RAM
        until explicitly requested. This is critical for Layer 2 when multiple
        KnowledgeNodes are in memory simultaneously.
        
        Args:
            repository: KNRepository instance
        
        Returns:
            PEFT data dict with keys: weights_bytes, format, size, checksum, config, trained_timestamp
            Returns None if PEFT weights are not available
        
        Note:
            This method loads actual weights into RAM. Use only when needed for inference/training.
        """
        # ========================================================================
        # Check Availability (2025-12-27)
        # ========================================================================
        # If local metadata says PEFT is not available, return None immediately
        # (avoids unnecessary Repository call)
        if not self.peft_available:
            logger.debug(
                f"PEFT weights not available for node '{self.node_id}' "
                f"(peft_available=False)"
            )
            return None
        
        # ========================================================================
        # Load from Repository (2025-12-27)
        # ========================================================================
        # Reuse Repository method (no code duplication)
        # This loads actual weights into RAM (lazy loading)
        peft_data = repository.get_peft_weights(self.node_id)
        
        # ========================================================================
        # Handle NULL/Inconsistent State (2025-12-27)
        # ========================================================================
        # If Repository returns None but local metadata says available,
        # update local state to reflect reality
        if peft_data is None:
            logger.warning(
                f"Inconsistent PEFT state for node '{self.node_id}': "
                f"peft_available=True but Repository returned None. "
                f"Updating local state."
            )
            self.peft_available = False
            self.peft_format = None
            self.peft_size = None
            return None
        
        logger.debug(
            f"PEFT weights loaded from Repository for node '{self.node_id}' "
            f"(size={peft_data.get('size', 'unknown')} bytes)"
        )
        
        return peft_data
    
    @property
    def is_under_training(self) -> bool:
        """
        Check if node is currently being trained.
        
        Added: 2025-12-27
        Purpose: Layer 2 can decide to use Transformer base as fallback.
        
        This property enables intelligent fallback decisions:
        - If node is under training, Layer 2 can use Transformer base instead
        - Prevents using incomplete/partially trained weights
        
        Returns:
            True if training status is 'TRAINING', False otherwise
        """
        # Handle NULL training status (None = not training)
        return self._training_status == 'TRAINING'
    
    @property
    def peft_ready(self) -> bool:
        """
        Check if PEFT weights are ready for inference.
        
        Added: 2025-12-27
        Purpose: Verify node has trained weights and is not currently training.
        
        This property combines PEFT availability and training status:
        - PEFT must be available (weights exist)
        - Node must not be under training (weights are complete)
        
        Returns:
            True if PEFT available and not under training, False otherwise
        """
        # PEFT is ready if:
        # 1. PEFT weights are available (peft_available == True)
        # 2. Node is not currently being trained (is_under_training == False)
        return self.peft_available and not self.is_under_training
