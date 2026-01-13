"""
Orchestrator for Clustering Layer.

System brain: decides to create nodes or send embeddings to existing ones.
Integrates FilterBayesian, KnowledgeNode and Repository.
"""

import torch
import torch.nn.functional as F
import logging
import warnings
from typing import Dict, Tuple, Optional, List, Any
from uuid import uuid4
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future

# Suppress transformers internal warnings about torch.tensor()
# This is an internal transformers warning, not from our code
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", category=UserWarning)

from .repository import KNRepository
from .knowledgenode import KnowledgeNode
from .filter_bayesian import FilterBayesian
from .post_processing import PostProcessor
from .transformer_base import TransformerBase
from .settings import (
    BUFFER_THRESHOLD,
    DEVICE,
    DTYPE,
    EMBEDDING_DIM,
    S_MIN,
    REFRESH_INTERVAL,
    TRAINING_THRESHOLD,
)
from .data_manager import DataManager

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
    
    def __init__(
        self, 
        repository: KNRepository, 
        filter_bayesian: FilterBayesian,
        dataset_paths: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            repository: KNRepository instance for persistence
            filter_bayesian: FilterBayesian instance for intelligent refreshes
            dataset_paths: Optional dictionary mapping dataset names to file paths
                Example: {
                    'arxiv': '/path/to/arxiv_data.csv',
                    '20newsgroups': '/path/to/20newsgroups.csv'
                }
                If None, DataManager will be initialized but won't have datasets
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
        
        # ========================================================================
        # Phase 2: DataManager (2025-01-XX)
        # ========================================================================
        # Bridge between Repository and original datasets for training
        # Translates source_ids (pointers) to original texts
        # ========================================================================
        self.data_manager = DataManager(
            dataset_paths=dataset_paths or {},
            use_sqlite_index=True,
            sqlite_index_dir=None
        )
        logger.debug(f"DataManager initialized with {len(dataset_paths or {})} dataset paths")
        
        # ========================================================================
        # Phase 2: Training Futures Tracking (2025-01-XX)
        # ========================================================================
        # Track training tasks for monitoring (optional)
        # ========================================================================
        self._training_futures: Dict[str, Future] = {}
        
        # ========================================================================
        # LRU Cache for PEFT Weights (2025-12-27)
        # ========================================================================
        # Prevents multiple BLOB reads from SQLite when multiple prompts need same KN
        # This is critical for Layer 2 performance when multiple inference requests
        # need the same adapter weights simultaneously
        # ========================================================================
        self._peft_cache: OrderedDict[str, bytes] = OrderedDict()
        self._peft_cache_max_size = 100 * 1024 * 1024  # 100 MB (configurable)
        self._peft_cache_current_size = 0  # Current size in bytes
        logger.debug("PEFT LRU cache initialized (max 100MB)")
        
        # ========================================================================
        # Async Updates for Layer 2 (2025-12-27)
        # ========================================================================
        # Thread pool for non-blocking database updates and training tasks
        # Only used in Layer 2 (inference) - Layer 1 keeps sync updates
        # This allows Layer 2 to update statistics without blocking user response
        # Phase 2: Also used for asynchronous training tasks
        # ========================================================================
        self._update_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xctopus-update")
        logger.debug("Async update executor initialized (max 2 workers)")
         
        # Layer 3 components (Lazy Init or passed in?)
        # For simplicity, we initialize them here or expect them injected.
        # Given the task, we'll initialize them here to complete the connection.
        self.post_processor = PostProcessor()
        # TransformerBase is a singleton, so instantiation is cheap
        self.transformer = TransformerBase()
        
        # Load existing nodes from Repository (warmup)
        self._warmup()
        
        logger.debug("Orchestrator initialized")
    
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
        
        # ========================================================================
        # Load KnowledgeNodes using from_repository() (2025-12-27)
        # ========================================================================
        # Changed from direct KnowledgeNode() creation to from_repository()
        # This ensures PEFT metadata and training status are loaded correctly
        # ========================================================================
        for signature in signatures:
            node_id = signature["node_id"]
            try:
                # Use factory method to load from Repository (includes PEFT metadata)
                kn = KnowledgeNode.from_repository(self.repository, node_id)
                self.active_nodes[node_id] = kn
            except ValueError as e:
                # Handle case where node doesn't exist (shouldn't happen, but safety check)
                logger.warning(
                    f"Failed to load node '{node_id}' from Repository: {e}. "
                    f"Skipping this node."
                )
                continue
        
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
    
        return best_buffer_id

    def process_embedding(
        self, 
        embedding: torch.Tensor, 
        source_id: Optional[str] = None
    ) -> str:
        """
        Reactive Flow (Layer 2 + Layer 3):
        1. Route embedding (FilterBayesian)
        2. If NEW_BUFFER -> Create buffer
        3. If KN -> Load LoRA -> Transformer -> Feedback -> Apply Feedback
        
        Phase 2: Also saves source_id pointer and checks training threshold.
        
        Args:
            embedding: Input embedding tensor
            source_id: Optional source ID (pointer to original dataset)
                If provided, will be saved to Repository after successful processing
            
        Returns:
            Status string or Node ID
        """
        # 1. Routing
        node_id, score = self.filter.route(embedding)
        
        if node_id == "NEW_BUFFER":
            logger.debug(f"Routing decision: NEW_BUFFER (score={score:.4f})")
            return self._handle_new_buffer(embedding)
            
        logger.debug(f"Routing decision: {node_id} (score={score:.4f})")
        
        # 2. Layer 3 Execution
        try:
            # Load LoRA weights (uses LRU cache)
            lora_weights = self.repository.get_lora_weights(node_id)
            
            # Run Transformer (Singleton + Safe Adapter)
            # Note: We assume the embedding can be used for generation or similar.
            # If lora_weights is None, it runs with base model (acceptable fallback or error?)
            # For now, we allow fallback to base model if no weights (cold start node).
            output = self.transformer.forward_with_node(embedding, lora_weights)
            
            # 3. Post-Processing & Feedback
            feedback = self.post_processor.evaluate(output)
            
            # 4. Reactive Update
            if feedback.status == "OK":
                self.repository.apply_feedback(node_id, feedback)
                logger.debug(f"Feedback applied to {node_id}")
            else:
                logger.debug(f"PostProcessor rejected response from {node_id}. Redirecting to buffer.")
                return self._handle_new_buffer(embedding)

            # 4. Layer 1 Update (Process Validation & Welford)
            return self._process_kn_update(node_id, embedding, source_id)
            
        except Exception as e:
            logger.error(f"Error in reactive flow for {node_id}: {e}", exc_info=True)
            # Fallback: preserve data in buffer
            return self._handle_new_buffer(embedding)

    def process_decision(
        self, 
        decision: Tuple[str, float], 
        embedding: torch.Tensor,
        source_id: Optional[str] = None
    ) -> str:
        """
        Process a decision from FilterBayesian.
        
        Phase 2: Also accepts source_id for data provenance.
        
        Args:
            decision: Tuple (node_id or "NEW_BUFFER", score)
            embedding: Embedding tensor [EMBEDDING_DIM]
            source_id: Optional source ID (pointer to original dataset)
                If provided, will be saved to Repository after successful processing
        
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
            # ========================================================================
            # Try to load from Repository using from_repository() (2025-12-27)
            # ========================================================================
            # Changed from direct KnowledgeNode() creation to from_repository()
            # This ensures PEFT metadata and training status are loaded correctly
            # ========================================================================
            try:
                # Use factory method to load from Repository (includes PEFT metadata)
                kn = KnowledgeNode.from_repository(self.repository, node_id)
                self.active_nodes[node_id] = kn
                logger.debug(f"Node '{node_id}' loaded from Repository")
            except ValueError:
                # If it doesn't exist in Repository, create new node
                # This can happen if FilterBayesian references a node that was deleted
                logger.warning(
                    f"Node '{node_id}' doesn't exist in Repository, creating new node"
                )
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
        
        # ========================================================================
        # Phase 2: Save Source ID Pointer (2025-01-XX)
        # ========================================================================
        # Save pointer to original dataset for training data provenance
        # Only save if source_id is provided and embedding was accepted
        # ========================================================================
        if source_id is not None:
            try:
                self.repository.save_pointer(node_id, source_id)
                logger.debug(f"Saved pointer for node '{node_id}': source_id='{source_id}'")
            except Exception as e:
                # Don't interrupt flow if pointer save fails
                logger.debug(f"Failed to save pointer for node '{node_id}': {e}")
        
        # Update signature in Repository
        signature = kn.get_signature()
        self.repository.update_node_stats(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"],
        )
        
        # ========================================================================
        # Phase 2: Training Threshold Check (2025-01-XX)
        # ========================================================================
        # Check if node has reached training threshold and should be queued for training
        # ========================================================================
        self._check_and_trigger_training(node_id, signature)
        
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
        
        # ========================================================================
        # Load KnowledgeNode using from_repository() (2025-12-27)
        # ========================================================================
        # Changed from keeping the directly created KnowledgeNode to loading
        # from Repository using from_repository(). This ensures consistency:
        # - PEFT metadata is loaded correctly (even if None)
        # - Training status is loaded correctly (even if None)
        # - Node state matches Repository state exactly
        # ========================================================================
        try:
            # Reload from Repository to ensure metadata is synchronized
            kn_loaded = KnowledgeNode.from_repository(self.repository, node_id)
            self.active_nodes[node_id] = kn_loaded
        except ValueError as e:
            # This should never happen since we just saved the node
            logger.error(
                f"Failed to load node '{node_id}' from Repository after saving: {e}. "
                f"Using directly created node (metadata may be incomplete)."
            )
            # Fallback: use the directly created node (metadata may be incomplete)
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
    
    def _get_peft_weights_cached(self, node_id: str) -> Optional[bytes]:
        """
        Get PEFT weights with LRU cache.
        
        Added: 2025-12-27
        Purpose: Avoid multiple BLOB reads from SQLite for same node.
        
        This method implements an LRU (Least Recently Used) cache for PEFT weights.
        When multiple prompts need the same adapter weights simultaneously, this
        prevents redundant SQLite BLOB reads, significantly improving performance.
        
        Args:
            node_id: Node ID
        
        Returns:
            weights_bytes or None if not available
        
        Note:
            Cache has a maximum size of 100MB. When full, least recently used
            weights are evicted to make room for new entries.
        """
        # ========================================================================
        # Check Cache First (2025-12-27)
        # ========================================================================
        # If weights are in cache, move to end (most recently used) and return
        # ========================================================================
        if node_id in self._peft_cache:
            # Move to end (most recently used) - LRU update
            weights = self._peft_cache.pop(node_id)
            self._peft_cache[node_id] = weights
            logger.debug(f"PEFT weights for '{node_id}' retrieved from cache")
            return weights
        
        # ========================================================================
        # Cache Miss: Load from Repository (2025-12-27)
        # ========================================================================
        # Reuse Repository method (no code duplication)
        # ========================================================================
        peft_data = self.repository.get_peft_weights(node_id)
        
        # Handle NULL/None case
        if not peft_data:
            logger.debug(f"PEFT weights not available for '{node_id}' in Repository")
            return None
        
        # Extract weights bytes
        weights = peft_data.get('weights_bytes')
        
        # Handle NULL weights_bytes (shouldn't happen, but safety check)
        if weights is None:
            logger.warning(
                f"PEFT data exists for '{node_id}' but weights_bytes is None"
            )
            return None
        
        if not isinstance(weights, bytes):
            logger.warning(
                f"PEFT weights for '{node_id}' is not bytes, got {type(weights)}"
            )
            return None
        
        weights_size = len(weights)
        
        # ========================================================================
        # Evict if Cache is Full (2025-12-27)
        # ========================================================================
        # Remove least recently used items (first items in OrderedDict)
        # until there's enough space for new entry
        # ========================================================================
        while (
            self._peft_cache_current_size + weights_size > self._peft_cache_max_size
            and self._peft_cache
        ):
            # Remove least recently used (first item in OrderedDict)
            evicted_id, evicted_weights = self._peft_cache.popitem(last=False)
            evicted_size = len(evicted_weights)
            self._peft_cache_current_size -= evicted_size
            
            logger.debug(
                f"Evicted '{evicted_id}' from PEFT cache "
                f"({evicted_size} bytes freed, "
                f"cache size: {self._peft_cache_current_size} bytes)"
            )
        
        # ========================================================================
        # Add to Cache (2025-12-27)
        # ========================================================================
        # Add new entry at end (most recently used)
        # ========================================================================
        # Check if entry fits after eviction
        if self._peft_cache_current_size + weights_size <= self._peft_cache_max_size:
            self._peft_cache[node_id] = weights
            self._peft_cache_current_size += weights_size
            
            logger.debug(
                f"PEFT weights for '{node_id}' loaded and cached "
                f"({weights_size} bytes, "
                f"cache size: {self._peft_cache_current_size}/{self._peft_cache_max_size} bytes)"
            )
        else:
            # Entry is too large for cache, don't cache it
            logger.warning(
                f"PEFT weights for '{node_id}' ({weights_size} bytes) "
                f"too large for cache (max: {self._peft_cache_max_size} bytes), "
                f"not caching"
            )
        
        return weights
    
    def clear_peft_cache(self) -> None:
        """
        Clear the PEFT weights cache.
        
        Added: 2025-12-27
        Purpose: Allow manual cache clearing if needed (e.g., memory pressure).
        
        This method can be called to free memory if the cache grows too large
        or if weights need to be reloaded from Repository.
        """
        cache_size = self._peft_cache_current_size
        cache_count = len(self._peft_cache)
        
        self._peft_cache.clear()
        self._peft_cache_current_size = 0
        
        logger.debug(
            f"PEFT cache cleared ({cache_count} entries, {cache_size} bytes freed)"
        )
    
    def update_stats_async(
        self,
        node_id: str,
        centroid: torch.Tensor,
        mass: int,
        variance: float
    ) -> None:
        """
        Update node statistics asynchronously (non-blocking).
        
        Added: 2025-12-27
        Purpose: Layer 2 can update stats without blocking user response.
        
        This method submits the update to a thread pool and returns immediately.
        The actual database update happens in a background thread, allowing
        Layer 2 inference to respond quickly to users while statistics are
        updated asynchronously.
        
        Note: This is for Layer 2 (inference). Layer 1 should use sync updates
        for consistency in batch processing.
        
        Args:
            node_id: Node ID
            centroid: Updated centroid tensor
            mass: Updated mass
            variance: Updated variance
        
        Raises:
            ValueError: If node_id is None or empty
            ValueError: If centroid is None
            ValueError: If mass < 0 or variance < 0
        """
        # ========================================================================
        # Input Validation (2025-12-27)
        # ========================================================================
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, got {type(node_id)}")
        
        if centroid is None:
            raise ValueError("centroid cannot be None")
        
        if mass < 0:
            raise ValueError(f"mass must be >= 0, got {mass}")
        
        if variance < 0:
            raise ValueError(f"variance must be >= 0, got {variance}")
        
        # ========================================================================
        # Submit to Thread Pool (2025-12-27)
        # ========================================================================
        # Reuse Repository method (no code duplication)
        # Don't wait for completion - returns immediately
        # ========================================================================
        future = self._update_executor.submit(
            self.repository.update_node_stats,
            node_id,
            centroid,
            mass,
            variance
        )
        
        # Note: We don't wait for the future to complete
        # Errors in the background thread will be logged by Repository
        logger.debug(f"Async update queued for node '{node_id}' (mass={mass}, variance={variance:.4f})")
    
    def update_stats_sync(
        self,
        node_id: str,
        centroid: torch.Tensor,
        mass: int,
        variance: float
    ) -> None:
        """
        Update node statistics synchronously (blocking).
        
        Added: 2025-12-27
        Purpose: Layer 1 uses this for consistency (batch processing).
        
        This method updates statistics immediately and waits for completion.
        Layer 1 (clustering) uses this to ensure data consistency during
        batch processing, where synchronous updates are preferred.
        
        Args:
            node_id: Node ID
            centroid: Updated centroid tensor
            mass: Updated mass
            variance: Updated variance
        
        Raises:
            ValueError: If node_id is None or empty
            ValueError: If centroid is None
            ValueError: If mass < 0 or variance < 0
        """
        # ========================================================================
        # Input Validation (2025-12-27)
        # ========================================================================
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, got {type(node_id)}")
        
        if centroid is None:
            raise ValueError("centroid cannot be None")
        
        if mass < 0:
            raise ValueError(f"mass must be >= 0, got {mass}")
        
        if variance < 0:
            raise ValueError(f"variance must be >= 0, got {variance}")
        
        # ========================================================================
        # Update Synchronously (2025-12-27)
        # ========================================================================
        # Reuse Repository method (no code duplication)
        # Blocks until update completes
        # ========================================================================
        self.repository.update_node_stats(node_id, centroid, mass, variance)
        logger.debug(f"Sync update completed for node '{node_id}' (mass={mass}, variance={variance:.4f})")
    
    def shutdown(self) -> None:
        """
        Shutdown the Orchestrator and clean up resources.
        
        Added: 2025-12-27
        Purpose: Clean shutdown of async executor and other resources.
        
        This method should be called when the Orchestrator is no longer needed
        to ensure proper cleanup of thread pools and other resources.
        """
        # Shutdown async executor
        if hasattr(self, '_update_executor'):
            self._update_executor.shutdown(wait=True)
            logger.debug("Async update executor shut down")

    def _handle_new_buffer(self, embedding: torch.Tensor) -> str:
        """Helper to create buffer and check promotion."""
        buffer_id = self.create_buffer(embedding)
        
        # Check threshold
        buffer_size = self.repository.get_buffer_size(buffer_id)
        if buffer_size >= BUFFER_THRESHOLD:
            logger.info(
                f"Buffer {buffer_id} reached threshold ({buffer_size} >= {BUFFER_THRESHOLD}), "
                f"promoting to KnowledgeNode"
            )
            return self.promote_buffer_to_kn(buffer_id)
        return buffer_id

    def _process_kn_update(
        self, 
        node_id: str, 
        embedding: torch.Tensor,
        source_id: Optional[str] = None
    ) -> str:
        """
        Helper to process embedding in an existing KN (Layer 1 Logic).
        
        Phase 2: Also saves source_id pointer and checks training threshold.
        
        Args:
            node_id: Knowledge Node ID
            embedding: Embedding tensor
            source_id: Optional source ID (pointer to original dataset)
        
        Returns:
            Node ID where embedding was processed
        """
        # Load Node
        if node_id not in self.active_nodes:
            try:
                kn = KnowledgeNode.from_repository(self.repository, node_id)
                self.active_nodes[node_id] = kn
            except ValueError:
                kn = KnowledgeNode(node_id=node_id)
                self.active_nodes[node_id] = kn
        
        kn = self.active_nodes[node_id]
        
        # ========================================================================
        # Phase 2: Training Buffer Check (2025-01-XX)
        # ========================================================================
        # If node is currently training, store embedding in buffer instead of processing
        # ========================================================================
        if kn.is_training:
            # Store in training buffer (will be processed after training completes)
            kn.training_buffer.append(embedding)
            if source_id is not None:
                kn.training_buffer_source_ids.append(source_id)
            logger.debug(
                f"Node '{node_id}' is training, embedding stored in buffer "
                f"(buffer size: {len(kn.training_buffer)})"
            )
            return node_id
        
        # Local Filter Check (Semantic Purity)
        accepted = kn.process(embedding)
        
        if not accepted:
            logger.debug(f"Embedding rejected by LocalFilter in {node_id}. Redirecting to buffer.")
            return self._handle_new_buffer(embedding)
        
        # ========================================================================
        # Phase 2: Save Source ID Pointer (2025-01-XX)
        # ========================================================================
        # Save pointer to original dataset for training data provenance
        # Only save if source_id is provided and embedding was accepted
        # ========================================================================
        if source_id is not None:
            try:
                self.repository.save_pointer(node_id, source_id)
                logger.debug(f"Saved pointer for node '{node_id}': source_id='{source_id}'")
            except Exception as e:
                # Don't interrupt flow if pointer save fails
                logger.debug(f"Failed to save pointer for node '{node_id}': {e}")
        
        # Update Repository
        signature = kn.get_signature()
        self.repository.update_node_stats(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"]
        )
        
        # ========================================================================
        # Phase 2: Training Threshold Check (2025-01-XX)
        # ========================================================================
        # Check if node has reached training threshold and should be queued for training
        # ========================================================================
        self._check_and_trigger_training(node_id, signature)
        
        return node_id
    
    def _check_and_trigger_training(self, node_id: str, signature: Dict[str, Any]) -> None:
        """
        Check if node should be queued for training and trigger if conditions are met.
        
        Phase 2: Training Trigger Logic
        
        Conditions:
        1. Node mass >= TRAINING_THRESHOLD
        2. Node is not already trained (repository.is_trained(node_id) == False)
        3. Node is not currently training (kn.is_training == False)
        
        Args:
            node_id: Knowledge Node ID
            signature: Node signature dictionary (from kn.get_signature())
        """
        try:
            # Validate signature
            if signature is None or "mass" not in signature:
                logger.debug(f"Invalid signature for node '{node_id}', skipping training check")
                return
            
            mass = signature.get("mass", 0)
            
            # Condition 1: Check mass threshold
            if mass < TRAINING_THRESHOLD:
                return  # Not enough mass yet
            
            # Condition 2: Check if already trained
            if self.repository.is_trained(node_id):
                logger.debug(f"Node '{node_id}' already trained, skipping training trigger")
                return
            
            # Condition 3: Check if currently training
            kn = self.active_nodes.get(node_id)
            if kn is None:
                logger.debug(f"Node '{node_id}' not in active_nodes, skipping training trigger")
                return
            
            if kn.is_training:
                logger.debug(f"Node '{node_id}' is already training, skipping duplicate trigger")
                return
            
            # All conditions met - queue training task
            logger.debug(
                f"Training threshold reached for node '{node_id}': "
                f"mass={mass} >= {TRAINING_THRESHOLD}"
            )
            self._queue_training_task(node_id)
            
        except Exception as e:
            # Don't interrupt flow if training check fails
            logger.debug(f"Error checking training threshold for node '{node_id}': {e}")
    
    def _queue_training_task(self, node_id: str) -> None:
        """
        Queue training task for a Knowledge Node.
        
        Phase 2: Asynchronous Training Queue
        
        This method:
        1. Marks node as training (is_training = True)
        2. Updates peft_training_status in Repository
        3. Submits training task to ThreadPoolExecutor
        
        Args:
            node_id: Knowledge Node ID to train
        """
        try:
            # Validate node exists
            kn = self.active_nodes.get(node_id)
            if kn is None:
                logger.debug(f"Node '{node_id}' not found in active_nodes, cannot queue training")
                return
            
            # Mark as training
            kn.is_training = True
            
            # Update training status in Repository
            try:
                self.repository.set_peft_training_status(node_id, 'TRAINING')
                logger.debug(f"Node '{node_id}' marked as TRAINING in Repository")
            except Exception as e:
                logger.debug(f"Failed to update training status for node '{node_id}': {e}")
                # Continue anyway - flag is set locally
            
            # Submit training task to ThreadPoolExecutor
            future = self._update_executor.submit(self._async_train_kn, node_id)
            
            # Track future for monitoring (optional)
            self._training_futures[node_id] = future
            
            logger.debug(f"Training task queued for node '{node_id}'")
            
        except Exception as e:
            # Clean up on error
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = False
            logger.debug(f"Error queueing training task for node '{node_id}': {e}")
    
    def _async_train_kn(self, node_id: str) -> None:
        """
        Asynchronously train a Knowledge Node adapter.
        
        Phase 2: Training Execution
        
        Flow:
        1. Get source_ids from Repository
        2. Translate source_ids to texts using DataManager
        3. Train adapter using TransformerBase
        4. Save weights to Repository
        5. Mark as trained
        6. Process training buffer (embeddings received during training)
        
        Args:
            node_id: Knowledge Node ID to train
        """
        try:
            logger.debug(f"Starting training for node '{node_id}'")
            
            # 1. Get source_ids from Repository
            source_ids = self.repository.get_training_pointers(node_id)
            
            if not source_ids:
                logger.debug(f"No source_ids found for node '{node_id}', cannot train")
                # Mark as not training
                kn = self.active_nodes.get(node_id)
                if kn:
                    kn.is_training = False
                self.repository.set_peft_training_status(node_id, None)
                return
            
            logger.debug(f"Retrieved {len(source_ids)} source_ids for node '{node_id}'")
            
            # 2. Translate source_ids to texts using DataManager
            texts = self.data_manager.get_texts_from_pointers(source_ids)
            
            if not texts:
                logger.debug(f"No texts found for node '{node_id}', cannot train")
                # Mark as not training
                kn = self.active_nodes.get(node_id)
                if kn:
                    kn.is_training = False
                self.repository.set_peft_training_status(node_id, None)
                return
            
            logger.debug(f"Retrieved {len(texts)} texts for node '{node_id}'")
            
            # 3. Train adapter using TransformerBase
            # Note: This method doesn't exist yet in TransformerBase, will be implemented later
            # For now, we'll check if it exists and handle gracefully
            if not hasattr(self.transformer, 'train_kn_adapter'):
                logger.debug(
                    f"TransformerBase.train_kn_adapter() not implemented yet, "
                    f"skipping training for node '{node_id}'"
                )
                # Mark as not training
                kn = self.active_nodes.get(node_id)
                if kn:
                    kn.is_training = False
                self.repository.set_peft_training_status(node_id, None)
                return
            
            weights_bytes = self.transformer.train_kn_adapter(node_id, texts)
            
            if weights_bytes is None:
                logger.debug(f"Training returned None weights for node '{node_id}'")
                # Mark as failed
                kn = self.active_nodes.get(node_id)
                if kn:
                    kn.is_training = False
                self.repository.set_peft_training_status(node_id, 'FAILED')
                return
            
            # 4. Save weights to Repository
            self.repository.save_peft_weights(
                node_id,
                weights_bytes,
                format="safetensors"
            )
            logger.debug(f"PEFT weights saved for node '{node_id}'")
            
            # 5. Mark as trained
            self.repository.mark_as_trained(node_id)
            self.repository.set_peft_training_status(node_id, 'COMPLETED')
            logger.debug(f"Node '{node_id}' marked as trained")
            
            # 6. Process training buffer (embeddings received during training)
            kn = self.active_nodes.get(node_id)
            if kn and kn.training_buffer:
                logger.debug(
                    f"Processing {len(kn.training_buffer)} buffered embeddings "
                    f"for node '{node_id}'"
                )
                
                # Process each buffered embedding
                for i, (emb, src_id) in enumerate(zip(
                    kn.training_buffer,
                    kn.training_buffer_source_ids if kn.training_buffer_source_ids else [None] * len(kn.training_buffer)
                )):
                    try:
                        # Process embedding
                        accepted = kn.process(emb)
                        if accepted and src_id is not None:
                            # Save pointer
                            try:
                                self.repository.save_pointer(node_id, src_id)
                            except Exception:
                                pass  # Don't interrupt buffer processing
                    except Exception as e:
                        logger.debug(f"Error processing buffered embedding {i} for node '{node_id}': {e}")
                
                # Update statistics after processing buffer
                signature = kn.get_signature()
                self.repository.update_node_stats(
                    node_id=signature["node_id"],
                    centroid=signature["centroid"],
                    mass=signature["mass"],
                    variance=signature["variance"]
                )
                
                # Check if new threshold reached (recursive training trigger)
                self._check_and_trigger_training(node_id, signature)
                
                # Clear buffer
                kn.training_buffer.clear()
                kn.training_buffer_source_ids.clear()
                logger.debug(f"Training buffer cleared for node '{node_id}'")
            
            # 7. Clear training flag
            if kn:
                kn.is_training = False
            
            # Remove from futures tracking
            self._training_futures.pop(node_id, None)
            
            logger.debug(f"Training completed successfully for node '{node_id}'")
            
        except Exception as e:
            # Error handling
            logger.debug(f"Training failed for node '{node_id}': {e}", exc_info=True)
            
            # Mark as failed
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = False
                # Clear buffer on error (embeddings will be reprocessed on next routing)
                kn.training_buffer.clear()
                kn.training_buffer_source_ids.clear()
            
            try:
                self.repository.set_peft_training_status(node_id, 'FAILED')
            except Exception:
                pass  # Don't fail on status update
            
            # Remove from futures tracking
            self._training_futures.pop(node_id, None)