"""
FilterBayesian for Clustering Layer.

Implements the 4 Golden Rules for routing embeddings to Knowledge Nodes:
1. Critical Mass: Favors large nodes (log1p(mass))
2. Semantic Purity: Rejects if similarity < S_MIN
3. Singularity: Returns NEW_BUFFER if no match
4. Stability: Reads signatures from Repository (no persistent internal state)
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Any, Optional

from .settings import S_MIN, LAMBDA_FACTOR, DEVICE, DTYPE

logger = logging.getLogger(__name__)


class FilterBayesian:
    """
    Bayesian Filter for routing embeddings to Knowledge Nodes.
    
    Implements the 4 Golden Rules:
    1. Critical Mass: Favors large nodes (log1p(mass) * LAMBDA_FACTOR)
    2. Semantic Purity: Rejects if similarity < S_MIN
    3. Singularity: Returns NEW_BUFFER if no match
    4. Stability: Reads signatures from Repository (no persistent internal state)
    
    Optimization:
    - Vectorized calculation: 1 vs N nodes in a single operation
    - Uses tensors on DEVICE and DTYPE from settings
    """
    
    def __init__(self):
        """
        Initialize the filter without persistent internal state.
        
        Only maintains temporary cache of signatures for efficient calculations.
        """
        self.ids: List[str] = []
        self.centroids: Optional[torch.Tensor] = None
        self.masses: Optional[torch.Tensor] = None
        logger.debug("FilterBayesian initialized")

    def refresh_signatures(self, signatures: List[Dict[str, Any]]) -> None:
        """
        Transform Repository metadata into tensors ready 
        for parallel matrix calculation.
        
        Rule 4 (Statistical Stability): The filter does not decide based on its 
        own state, but on the statistical signature that the Repository provides.
        
        Args:
            signatures: List of signatures with keys: node_id, centroid, mass, variance
        """
        if not signatures:
            self.ids = []
            self.centroids = None
            self.masses = None
            logger.debug("Empty signatures, clearing state")
            return

        self.ids = [sig["node_id"] for sig in signatures]
        
        # Stack centroids into a single matrix [N, EMBEDDING_DIM]
        # Directly on the device and precision from settings
        self.centroids = torch.stack([sig["centroid"] for sig in signatures]).to(
            device=DEVICE, dtype=DTYPE
        )
        
        # Mass vector [N]
        self.masses = torch.tensor(
            [sig["mass"] for sig in signatures], 
            device=DEVICE, dtype=DTYPE
        )
        
        logger.debug(f"Signatures updated: {len(self.ids)} active nodes")

    def route(self, embedding: torch.Tensor) -> Tuple[str, float]:
        """
        Optimized matrix routing using the 4 Golden Rules.
        
        Calculates similarity of 1 vs N nodes in a single vectorized operation.
        
        Args:
            embedding: Embedding tensor [384] in FP16
        
        Returns:
            Tuple[str, float]: 
                - If there's a match: (node_id, routing_score)
                - If no match: ("NEW_BUFFER", 0.0)
        
        Raises:
            ValueError: If embedding is empty or has incorrect shape
        
        Implements:
            - Rule 1: Critical Mass (log1p(mass) * LAMBDA_FACTOR)
            - Rule 2: Semantic Purity (S_MIN threshold)
            - Rule 3: Singularity (NEW_BUFFER if no match)
            - Rule 4: Stability (uses signatures from Repository)
        """
        # ========================================================================
        # Input Validation
        # ========================================================================
        
        if embedding is None or embedding.numel() == 0:
            logger.warning("Empty embedding received, returning NEW_BUFFER")
            return "NEW_BUFFER", 0.0
        
        # Ensure it's 1D
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
        
        if embedding.dim() != 1:
            raise ValueError(f"Embedding must be 1D, received shape: {embedding.shape}")
        
        # Ensure device and dtype
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        
        # ========================================================================
        # Rule 4: Statistical Stability
        # Check if signatures are available
        # ========================================================================
        
        if self.centroids is None or len(self.ids) == 0:
            logger.debug("No signatures available, returning NEW_BUFFER")
            return "NEW_BUFFER", 0.0

        # ========================================================================
        # Vectorized Calculation: Cosine Similarity
        # Parallel matrix operation: 1 vs N in a single pass
        # ========================================================================
        
        # embedding: [384] -> [1, 384]
        # centroids: [N, 384]
        # sims: [N]
        sims = F.cosine_similarity(embedding.unsqueeze(0), self.centroids, dim=1)

        # ========================================================================
        # Rule 2: Semantic Purity (Internal Cohesion)
        # Only consider nodes that exceed the semantic threshold
        # ========================================================================
        
        valid_mask = sims >= S_MIN
        
        if not torch.any(valid_mask):
            max_sim = sims.max().item()
            logger.debug(
                f"Embedding does not exceed S_MIN={S_MIN:.3f} "
                f"(max_sim={max_sim:.3f}), returning NEW_BUFFER"
            )
            # Rule 3: Singularity - return NEW_BUFFER
            return "NEW_BUFFER", 0.0

        # ========================================================================
        # Rule 1: Critical Mass (Biased Routing)
        # score = sim + log1p(mass) * LAMBDA_FACTOR
        # Favors large nodes when similarities are close
        # ========================================================================
        
        scores = sims + torch.log1p(self.masses) * LAMBDA_FACTOR

        # Penalize invalid nodes so they're not selected by argmax
        scores[~valid_mask] = -100.0 

        # ========================================================================
        # Best Match Selection
        # ========================================================================
        
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        best_sim = sims[best_idx].item()
        best_mass = self.masses[best_idx].item()
        best_node_id = self.ids[best_idx]
        
        logger.debug(
            f"Successful routing: {best_node_id} "
            f"(sim={best_sim:.3f}, mass={best_mass:.0f}, score={best_score:.4f})"
        )
        
        return best_node_id, best_score
