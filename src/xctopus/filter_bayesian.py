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

from .settings import S_MIN, LAMBDA_FACTOR, THRESH_DECAY, THRESH_MIN_LOG, DEVICE, DTYPE

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

    def partial_update(self, node_id: str, new_centroid: torch.Tensor, new_mass: int) -> None:
        """
        Incrementally update a single node's state in memory.
        
        This avoids full re-initialization of tensors (O(1) vs O(N)).
        
        Args:
            node_id: The ID of the node to update
            new_centroid: New centroid tensor [384]
            new_mass: New mass value
        """
        if node_id not in self.ids:
            # If node doesn't exist in current state, we can't update it cheaply
            # This shouldn't happen in normal flow if signatures are refreshed periodically
            logger.warning(f"Attempted partial update on unknown node {node_id}")
            return

        idx = self.ids.index(node_id)
        
        # Ensure correct device/dtype
        new_centroid = new_centroid.to(device=DEVICE, dtype=DTYPE)
        
        # In-place updates
        self.centroids[idx] = new_centroid
        self.masses[idx] = float(new_mass)
        
        logger.debug(f"Partial update for {node_id}: mass {new_mass}")

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
            - Rule 2: Semantic Purity (Dynamic S_MIN threshold)
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
        # Rule 2: Semantic Purity (Dynamic Threshold)
        # S_EFF = S_MIN - (THRESH_DECAY / log1p(mass))
        # Smaller nodes are more permissive, larger nodes are stricter
        # ========================================================================
        
        # Calculate dynamic thresholds for all nodes
        # Note: self.masses contains raw mass values (float)
        
        mass_log = torch.log1p(self.masses)
        
        # Avoid division by zero if mass is 0 (shouldn't happen but safety first)
        # log1p(0) = 0. log1p(1) = 0.69. 
        # Clamp to THRESH_MIN_LOG to ensure numerical stability
        mass_log_safe = torch.clamp(mass_log, min=THRESH_MIN_LOG)
        
        # Calculate individual thresholds using THRESH_DECAY from settings
        dynamic_thresholds = S_MIN - (THRESH_DECAY / mass_log_safe)
        
        # Dynamic threshold behavior:
        # - Small nodes (mass=6): log1p(6)~1.94 -> threshold ≈ S_MIN - 0.077 ≈ 0.52
        # - Medium nodes (mass=20): log1p(20)~3.04 -> threshold ≈ S_MIN - 0.049 ≈ 0.55
        # - Large nodes (mass=100): log1p(100)~4.6 -> threshold ≈ S_MIN - 0.033 ≈ 0.57
        # As mass grows, threshold converges to S_MIN (more restrictive, maintains semantic purity) 
        # S_MIN - big_number (small mass) is much lower. 
        # So small nodes get LOWER threshold (more permissive). Correct.
        
        valid_mask = sims >= dynamic_thresholds
        
        if not torch.any(valid_mask):
            max_sim = sims.max().item()
            # Find what the threshold would have been for the best match
            best_idx = torch.argmax(sims)
            req_thresh = dynamic_thresholds[best_idx].item()
            
            logger.debug(
                f"No match > dynamic threshold (max_sim={max_sim:.3f} vs req={req_thresh:.3f}), "
                "returning NEW_BUFFER"
            )
            # Rule 3: Singularity - return NEW_BUFFER
            return "NEW_BUFFER", 0.0

        # ========================================================================
        # Rule 1: Critical Mass (Biased Routing)
        # score = sim + log1p(mass) * LAMBDA_FACTOR
        # Favors large nodes when similarities are close
        # ========================================================================
        
        scores = sims + mass_log * LAMBDA_FACTOR

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
        best_thresh = dynamic_thresholds[best_idx].item()
        
        logger.debug(
            f"Successful routing: {best_node_id} "
            f"(sim={best_sim:.3f} >= {best_thresh:.3f}, mass={best_mass:.0f}, score={best_score:.4f})"
        )
        
        return best_node_id, best_score
