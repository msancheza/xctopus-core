"""
Orchestrator for Clustering Layer.

System brain: decides to create nodes or send embeddings to existing ones.
Integrates FilterBayesian, KnowledgeNode and Repository.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional, List
from uuid import uuid4

from .repository import KNRepository
from .knowledgenode import KnowledgeNode
from .filter_bayesian import FilterBayesian
from .settings import (
    BUFFER_THRESHOLD,
    DEVICE,
    DTYPE,
    EMBEDDING_DIM,
    S_MIN,
    REFRESH_INTERVAL,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for Clustering Layer.
    
    Responsibilities:
    - Execute FilterBayesian decisions
    - Manage active KnowledgeNodes
    - Create and promote buffers to KnowledgeNodes
    - Update signatures in Repository
    
    Flow:
    1. Receives decision from FilterBayesian: (node_id, score) or ("NEW_BUFFER", 0.0)
    2. If EXISTING_KN: processes embedding in existing node
    3. If NEW_BUFFER: creates buffer, checks threshold, promotes if necessary
    """
    
    def __init__(self, repository: KNRepository, filter_bayesian: FilterBayesian):
        """
        Initialize the Orchestrator.
        
        Args:
            repository: KNRepository instance for persistence
            filter_bayesian: FilterBayesian instance for intelligent refreshes
        """
        if repository is None:
            raise ValueError("Repository cannot be None")
        if filter_bayesian is None:
            raise ValueError("FilterBayesian cannot be None")
        
        self.repository = repository
        self.filter = filter_bayesian
        self.active_nodes: Dict[str, KnowledgeNode] = {}  # {node_id: KnowledgeNode}
        
        # In-memory counters for progress display (without periodic validation)
        self.kn_count: int = 0  # Total Knowledge Nodes created
        self.buffer_count: int = 0  # Total active buffers
        
        # Load existing nodes from Repository (warmup)
        self._warmup()
        
        logger.info("Orchestrator initialized")
    
    def _warmup(self) -> None:
        """
        Load existing nodes from Repository to active_nodes.
        
        Allows the system to continue from where it left off if restarted.
        """
        signatures = self.repository.get_all_signatures()
        
        if not signatures:
            logger.debug("No existing nodes in Repository, starting from scratch")
            # Initialize counters to 0 (already at 0, but explicit)
            self.kn_count = 0
            active_buffers = self.repository.get_all_active_buffers()
            self.buffer_count = len(active_buffers)
            return
        
        logger.info(f"Loading {len(signatures)} existing nodes from Repository...")
        
        for signature in signatures:
            # Convert real variance to M2 (sum of squares)
            # M2 = variance * (mass - 1) if mass > 1, else 0.0
            if signature["mass"] > 1:
                initial_m2 = signature["variance"] * (signature["mass"] - 1)
            else:
                initial_m2 = 0.0
            
            kn = KnowledgeNode(
                node_id=signature["node_id"],
                initial_centroid=signature["centroid"],
                initial_mass=signature["mass"],
                initial_variance=initial_m2,  # M2, not real variance
            )
            self.active_nodes[signature["node_id"]] = kn
        
        logger.info(f"{len(self.active_nodes)} nodes loaded into active_nodes")
        
        # Initialize counters from database (warmup)
        self.kn_count = len(self.active_nodes)
        active_buffers = self.repository.get_all_active_buffers()
        self.buffer_count = len(active_buffers)
        
        logger.debug(f"Counters initialized: KNs={self.kn_count}, Buffers={self.buffer_count}")
    
    def _find_similar_buffer(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Search for an existing buffer with sufficient similarity to the embedding.
        
        OPTIMIZED: Uses vectorized calculation (1 vs N in one operation).
        Similar to FilterBayesian.route() pattern for better performance.
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
        
        Returns:
            buffer_id if one is found with similarity > S_MIN, None otherwise
        """
        # Get all centroids in a single operation
        # This avoids N individual database queries
        buffer_ids, centroids = self.repository.get_all_buffer_centroids()
        
        if len(buffer_ids) == 0:
            return None
        
        # ========================================================================
        # Vectorized Calculation: Cosine Similarity
        # Parallel matrix operation: 1 vs N in a single pass
        # ========================================================================
        
        # embedding: [EMBEDDING_DIM] -> [1, EMBEDDING_DIM]
        # centroids: [N, EMBEDDING_DIM]
        # sims: [N]
        sims = F.cosine_similarity(embedding.unsqueeze(0), centroids, dim=1)
        
        # ========================================================================
        # Find best match that exceeds S_MIN
        # ========================================================================
        
        valid_mask = sims >= S_MIN
        
        if not torch.any(valid_mask):
            return None
        
        # Penalize invalid buffers so they're not selected by argmax
        scores = sims.clone()
        scores[~valid_mask] = -100.0
        
        # Get best match
        best_idx = torch.argmax(scores).item()
        best_similarity = sims[best_idx].item()
        best_buffer_id = buffer_ids[best_idx]
        
        logger.debug(
            f"Similar buffer found: {best_buffer_id} "
            f"(similarity={best_similarity:.3f})"
        )
        
        return best_buffer_id
    
    def process_decision(
        self, 
        decision: Tuple[str, float], 
        embedding: torch.Tensor
    ) -> str:
        """
        Process a decision from FilterBayesian.
        
        Args:
            decision: Tuple (node_id or "NEW_BUFFER", score)
            embedding: Embedding tensor [EMBEDDING_DIM]
        
        Returns:
            str: ID of node or buffer where embedding was processed
        
        Raises:
            ValueError: If embedding is invalid or decision is incorrect
        """
        # ========================================================================
        # Input Validation
        # ========================================================================
        
        if embedding is None or embedding.numel() == 0:
            raise ValueError("Embedding cannot be None or empty")
        
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
        
        if embedding.dim() != 1 or embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be 1D with dimension {EMBEDDING_DIM}, "
                f"received: {embedding.shape}"
            )
        
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        
        node_id_or_buffer, score = decision
        
        # ========================================================================
        # Case 1: NEW_BUFFER (Rule 3: Singularity)
        # ========================================================================
        
        if node_id_or_buffer == "NEW_BUFFER":
            logger.debug(f"Decision: NEW_BUFFER (score={score:.4f})")
            buffer_id = self.create_buffer(embedding)
            
            # Check if buffer reaches threshold
            buffer_size = self.repository.get_buffer_size(buffer_id)
            if buffer_size >= BUFFER_THRESHOLD:
                logger.info(
                    f"Buffer {buffer_id} reached threshold ({buffer_size} >= {BUFFER_THRESHOLD}), "
                    f"promoting to KnowledgeNode"
                )
                new_node_id = self.promote_buffer_to_kn(buffer_id)
                return new_node_id
            else:
                logger.debug(f"Buffer {buffer_id} has {buffer_size} embeddings (threshold: {BUFFER_THRESHOLD})")
                return buffer_id
        
        # ========================================================================
        # Case 2: EXISTING_KN (Rule 1 and 2: Critical Mass and Semantic Purity)
        # ========================================================================
        
        node_id = node_id_or_buffer
        logger.debug(f"Decision: EXISTING_KN '{node_id}' (score={score:.4f})")
        
        # Get KnowledgeNode from active nodes dict
        if node_id not in self.active_nodes:
            logger.warning(
                f"Node '{node_id}' is not in active nodes. "
                f"Loading from Repository or creating new node."
            )
            # Try to load from Repository
            signature = self.repository.get_signature(node_id)
            if signature:
                # Convert real variance to M2 (sum of squares)
                # M2 = variance * (mass - 1) if mass > 1, else 0.0
                if signature["mass"] > 1:
                    initial_m2 = signature["variance"] * (signature["mass"] - 1)
                else:
                    initial_m2 = 0.0
                
                # Create node from existing signature
                kn = KnowledgeNode(
                    node_id=signature["node_id"],
                    initial_centroid=signature["centroid"],
                    initial_mass=signature["mass"],
                    initial_variance=initial_m2,  # M2, not real variance
                )
                self.active_nodes[node_id] = kn
                logger.debug(f"Node '{node_id}' loaded from Repository")
            else:
                # If it doesn't exist in Repository, create new node
                logger.warning(f"Node '{node_id}' doesn't exist in Repository, creating new node")
                kn = KnowledgeNode(node_id=node_id)
                self.active_nodes[node_id] = kn
        
        kn = self.active_nodes[node_id]
        
        # Process embedding in the node
        # LocalFilter validates membership (may reject)
        accepted = kn.process(embedding)
        
        if not accepted:
            # If LocalFilter rejects, create new buffer
            logger.debug(
                f"Embedding rejected by LocalFilter in node '{node_id}', "
                f"creating new buffer"
            )
            buffer_id = self.create_buffer(embedding)
            
            # Check threshold
            buffer_size = self.repository.get_buffer_size(buffer_id)
            if buffer_size >= BUFFER_THRESHOLD:
                logger.info(
                    f"Buffer {buffer_id} reached threshold ({buffer_size} >= {BUFFER_THRESHOLD}), "
                    f"promoting to KnowledgeNode"
                )
                new_node_id = self.promote_buffer_to_kn(buffer_id)
                return new_node_id
            else:
                return buffer_id
        
        # Update signature in Repository
        signature = kn.get_signature()
        self.repository.update_node_stats(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"],
        )
        
        # Intelligent refresh: update FilterBayesian every REFRESH_INTERVAL embeddings
        if signature["mass"] % REFRESH_INTERVAL == 0:
            new_signatures = self.repository.get_all_signatures()
            self.filter.refresh_signatures(new_signatures)
            logger.debug(
                f"Signatures refreshed in FilterBayesian "
                f"(node '{node_id}' reached mass={signature['mass']})"
            )
        
        logger.debug(
            f"Embedding processed in node '{node_id}': "
            f"mass={signature['mass']}, variance={signature['variance']:.4f}"
        )
        
        return node_id
    
    def create_buffer(self, embedding: torch.Tensor) -> str:
        """
        Create a new temporary buffer or add to an existing similar one.
        
        First tries to find an existing buffer with similarity > S_MIN.
        If not found, creates a new buffer.
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
        
        Returns:
            str: Buffer ID (existing or new)
        """
        # Try to find similar buffer
        similar_buffer_id = self._find_similar_buffer(embedding)
        
        if similar_buffer_id:
            # Add to existing buffer
            self.repository.add_to_buffer(similar_buffer_id, embedding)
            logger.debug(f"Embedding added to existing buffer: {similar_buffer_id}")
            return similar_buffer_id
        
        # No similar buffer found, create a new one
        buffer_id = f"buffer_{uuid4()}"
        
        # Create buffer in Repository
        self.repository.create_buffer(buffer_id)
        
        # Add embedding to buffer
        self.repository.add_to_buffer(buffer_id, embedding)
        
        # Update buffer counter
        self.buffer_count += 1
        
        logger.debug(f"New buffer created: {buffer_id} (Total buffers: {self.buffer_count})")
        
        return buffer_id
    
    def promote_buffer_to_kn(self, buffer_id: str) -> str:
        """
        Promote a buffer to KnowledgeNode.
        
        Flow:
        1. Get embeddings from buffer
        2. Calculate initial centroid (average of embeddings)
        3. Calculate initial mass (len(embeddings))
        4. Calculate initial variance
        5. Create KnowledgeNode with these values
        6. Process all buffer embeddings in the new node
        7. Create signature in Repository
        8. Add node to active nodes dict
        9. Delete buffer
        
        Args:
            buffer_id: ID of buffer to promote
        
        Returns:
            str: ID of the new KnowledgeNode created
        
        Raises:
            ValueError: If buffer is empty or doesn't exist
        """
        logger.info(f"Promoting buffer '{buffer_id}' to KnowledgeNode")
        
        # Get embeddings from buffer
        embeddings = self.repository.get_buffer_embeddings(buffer_id)
        
        if not embeddings:
            raise ValueError(f"Buffer '{buffer_id}' is empty, cannot promote")
        
        # ========================================================================
        # Calculate Initial Statistics
        # ========================================================================
        
        # Stack embeddings: [N, EMBEDDING_DIM]
        embeddings_tensor = torch.stack(embeddings).to(device=DEVICE, dtype=DTYPE)
        
        # Initial centroid: average of embeddings
        initial_centroid = embeddings_tensor.mean(dim=0)
        
        # Initial mass: number of embeddings
        initial_mass = len(embeddings)
        
        # Initial variance: Calculate M2 (sum of squares) using Welford's algorithm
        # We need to calculate M2 so the node can continue with incremental algorithm
        # M2 = sum of (x_i - mean) · (x_i - updated_mean)
        initial_m2 = 0.0
        
        if initial_mass > 1:
            # Calculate M2 incrementally (simulating Welford's algorithm)
            running_mean = embeddings_tensor[0].clone()
            for i in range(1, initial_mass):
                delta = embeddings_tensor[i] - running_mean
                running_mean += delta / (i + 1)
                delta2 = embeddings_tensor[i] - running_mean
                initial_m2 += torch.dot(delta, delta2).item()
        
        # initial_variance is now M2 (not real variance)
        initial_variance = initial_m2
        
        # ========================================================================
        # Create KnowledgeNode
        # ========================================================================
        
        # Generate unique node_id
        node_id = f"kn_{uuid4()}"
        
        # Create node with initial statistics
        kn = KnowledgeNode(
            node_id=node_id,
            initial_centroid=initial_centroid,
            initial_mass=initial_mass,
            initial_variance=initial_variance,
        )
        
        # Process all buffer embeddings in the new node
        # (This will update statistics incrementally)
        for emb in embeddings:
            kn.process(emb)
        
        # ========================================================================
        # Persist in Repository
        # ========================================================================
        
        # Get final signature (after processing all embeddings)
        signature = kn.get_signature()
        
        # Create signature in Repository
        self.repository.save_new_kn(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"],
        )
        
        # Add node to active nodes dict
        self.active_nodes[node_id] = kn
        
        # Delete buffer
        self.repository.delete_buffer(buffer_id)
        
        # Update counters: new KN created, buffer deleted
        self.kn_count += 1
        self.buffer_count -= 1
        
        logger.info(
            f"Buffer '{buffer_id}' promoted to KnowledgeNode '{node_id}': "
            f"mass={signature['mass']}, variance={signature['variance']:.4f} "
            f"(Total KNs: {self.kn_count}, Buffers: {self.buffer_count})"
        )
        
        return node_id
    
    def get_active_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Get an active node from the dict.
        
        Args:
            node_id: Node ID
        
        Returns:
            KnowledgeNode or None if it doesn't exist
        """
        return self.active_nodes.get(node_id)
    
    def get_counts(self) -> Dict[str, int]:
        """
        Get current counters for Knowledge Nodes and Buffers.
        
        Returns:
            Dict with 'kn_count' and 'buffer_count'
        """
        return {
            "kn_count": self.kn_count,
            "buffer_count": self.buffer_count
        }
    
    def get_active_node_count(self) -> int:
        """
        Return the number of active nodes.
        
        Returns:
            int: Number of active nodes
        """
        return len(self.active_nodes)
