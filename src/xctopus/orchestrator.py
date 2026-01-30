"""
Orchestrator for Clustering Layer.

System brain: decides to create nodes or send embeddings to existing ones.
Integrates FilterBayesian, KnowledgeNode and Repository.
"""

import torch
import torch.nn.functional as F
import logging
import warnings
import threading
import sys
from typing import Dict, Tuple, Optional, List, Any, Callable
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
    # REFRESH_INTERVAL,  # DEPRECATED: No longer used (signatures updated immediately)
    TRAINING_THRESHOLD,
    MAX_CONCURRENT_TRAINING,
    TRAINING_DELTA_MULTIPLIER,
    TRAINING_DELTA_TIMEOUT_DAYS,
    MAX_TRAINING_TEXTS,
    TRAINING_BATCH_SIZE,
    # Inheritance Constants
    INHERITANCE_ENABLED,
    TITAN_MIN_MASS,
    TITAN_MAX_VARIANCE,
    TITAN_SIMILARITY_THRESHOLD,
    PROGRESSIVE_ADOPTION_THRESHOLD,
)
from .data_manager import DataManager
from .evaluation import Evaluator

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
        # Fase 8: Pool de Re-evaluación (2025-01-XX)
        # ========================================================================
        # Cola para re-evaluar nodos después de que sus padres fallan o se entrenan
        # Evita race conditions procesando en batch
        # ========================================================================
        self.re_evaluation_queue: List[str] = []  # Queue of node_ids to re-evaluate
        self.re_evaluation_lock = threading.Lock()  # Thread-safe access to queue
        
        # ========================================================================
        # Phase 2: DataManager (2025-01-XX)
        # ========================================================================
        # Bridge between Repository and original datasets for training
        # Translates source_ids (pointers) to original texts
        # ========================================================================
        self.data_manager = DataManager(
            dataset_paths=dataset_paths or {},
            use_sqlite_index=True,
            sqlite_index_dir=None,
            repository=self.repository
        )
        logger.debug(f"DataManager initialized with {len(dataset_paths or {})} dataset paths")
        
        # ========================================================================
        # Phase 2: Training Futures Tracking (2025-01-XX)
        # ========================================================================
        # Track training tasks for monitoring (optional)
        # ========================================================================
        self._training_futures: Dict[str, Future] = {}
        
        # ========================================================================
        # Cache for Pending Training Nodes (2025-01-XX)
        # ========================================================================
        # Cache to avoid expensive get_pending_training_nodes() calls
        # Updated when nodes are marked for training
        # ========================================================================
        self._pending_training_cache: set = set()  # Set of node_ids marked for training
        self._pending_cache_lock = threading.Lock()  # Thread-safe access
        
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
        # CRITICAL: Use MAX_CONCURRENT_TRAINING to limit concurrent training tasks
        # This prevents multiple threads from accessing TransformerBase singleton simultaneously
        # ========================================================================
        self._update_executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_TRAINING, 
            thread_name_prefix="xctopus-update"
        )
        logger.debug(f"Async update executor initialized (max {MAX_CONCURRENT_TRAINING} workers)")
        
        # ========================================================================
        # CRITICAL: Lock for TransformerBase singleton (2025-01-14)
        # ========================================================================
        # TransformerBase singleton is NOT thread-safe for concurrent training.
        # This lock ensures only one training task accesses the singleton at a time.
        # ========================================================================
        self._training_lock = threading.Lock()
        logger.debug("Training lock initialized for TransformerBase singleton protection")
         
        # Layer 3 components (Lazy Init or passed in?)
        # For simplicity, we initialize them here or expect them injected.
        # Given the task, we'll initialize them here to complete the connection.
        self.post_processor = PostProcessor()
        # TransformerBase is a singleton, so instantiation is cheap
        self.transformer = TransformerBase()
        
        # ========================================================================
        # Training Session Management (2026-01-26)
        # ========================================================================
        # Generate a short, unique session ID for this run (e.g., 'sA9x2')
        # This decouples source_id prefixes from brittle filenames.
        import random
        import string
        session_chars = string.ascii_letters + string.digits
        self.session_id = 's' + ''.join(random.choice(session_chars) for _ in range(4))
        
        # Register session if we have dataset information
        if dataset_paths:
            # For now, register all provided datasets to this session
            # This allows DataManager to resolve the session_id to any of these files
            for name, path in dataset_paths.items():
                self.repository.register_session(self.session_id, name, path)
            logger.info(f"Training session initialized: {self.session_id} ({len(dataset_paths)} datasets registered)")
        else:
            logger.debug(f"Training session initialized without pre-registered datasets: {self.session_id}")

        # Phase 3: Evaluator
        self.evaluator = Evaluator(self.repository, self.transformer, self.data_manager)
        
        # Load existing nodes from Repository (warmup)
        self._warmup()
        
        logger.debug(f"Orchestrator initialized | session_id={self.session_id}")
    
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
        
        # ========================================================================
        # CRITICAL: Initialize FilterBayesian with loaded signatures
        # ========================================================================
        # This was missing! The filter needs the signatures to route embeddings
        # ========================================================================
        self.filter.refresh_signatures(signatures)
        logger.debug(f"FilterBayesian initialized with {len(signatures)} signatures")
        
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
        # Session Management: Use session_id as prefix if no prefix is present
        if source_id and ':' not in source_id:
            source_id = f"{self.session_id}:{source_id}"
        elif not source_id:
            # Fallback for missing source_id
            source_id = f"{self.session_id}:{self.kn_count + self.buffer_count}"

        # 1. Routing
        node_id, score = self.filter.route(embedding)
        
        if node_id == "NEW_BUFFER":
            logger.debug(f"Routing decision: NEW_BUFFER (score={score:.4f})")
            return self._handle_new_buffer(embedding, source_id=source_id)
            
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
            
            # Validate feedback is not None
            if feedback is None:
                logger.warning(f"PostProcessor returned None for node {node_id}, redirecting to buffer")
                return self._handle_new_buffer(embedding, source_id=source_id)
            
            # 4. Reactive Update
            if feedback.status == "OK":
                # Apply feedback (will handle non-existent nodes internally)
                self.repository.apply_feedback(node_id, feedback)
                logger.debug(f"Feedback applied to {node_id}")
            else:
                logger.debug(f"PostProcessor rejected response from {node_id}. Redirecting to buffer.")
                return self._handle_new_buffer(embedding, source_id=source_id)

            # 5. Layer 1 Update (Process Validation & Welford)
            return self._process_kn_update(node_id, embedding, source_id)
            
        except Exception as e:
            logger.error(f"Error in reactive flow for {node_id}: {e}", exc_info=True)
            # Fallback: preserve data in buffer
            return self._handle_new_buffer(embedding, source_id=source_id)


    def process_batch(
        self, 
        embeddings: torch.Tensor, 
        source_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Process a batch of embeddings efficiently (High Performance).
        
        Solves the "Atomic Inference" bottleneck by:
        1. Routing all embeddings at once (Vectorized Routing).
        2. Grouping embeddings by Node ID.
        3. Loading LoRA weights ONCE per node.
        4. Running Transformer inference in BATCH mode.
        
        Args:
            embeddings: Batch of embeddings [BATCH_SIZE, 384]
            source_ids: Optional list of source IDs (one per embedding)
            
        Returns:
            List of result statuses/IDs corresponding to inputs
        """
        if embeddings is None or embeddings.numel() == 0:
            return []
            
        # Ensure 2D
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            
        batch_size = embeddings.shape[0]
        
        # Session Management: Ensure source_ids have session prefix
        actual_source_ids = []
        if source_ids:
            for sid in source_ids:
                if sid and ':' not in sid:
                    actual_source_ids.append(f"{self.session_id}:{sid}")
                else:
                    actual_source_ids.append(sid)
        else:
            # Fallback numeric IDs using sequential count
            start_count = self.kn_count + self.buffer_count
            actual_source_ids = [f"{self.session_id}:{start_count + i}" for i in range(batch_size)]
        
        source_ids = actual_source_ids
            
        # 1. Vectorized Routing
        # returns list of (node_id, score)
        routing_decisions = self.filter.route_batch(embeddings)
        
        # 2. Grouping by Node ID
        # node_groups = { 'kn_1': [index_0, index_5], 'kn_2': [index_1], ... }
        node_groups: Dict[str, List[int]] = {}
        for i, (node_id, _) in enumerate(routing_decisions):
            if node_id not in node_groups:
                node_groups[node_id] = []
            node_groups[node_id].append(i)
            
        final_results = [None] * batch_size
        
        # 3. Process each group
        for node_id, indices in node_groups.items():
            # Extract batch for this node
            # [GROUP_SIZE, 384]
            group_embeddings = embeddings[indices]
            group_source_ids = [source_ids[i] for i in indices]
            
            # Case A: NEW_BUFFER
            if node_id == "NEW_BUFFER":
                # Buffers are fast, process sequentially or we could optimize later
                for i, idx in enumerate(indices):
                    # We use the single-item method for buffers as it handles creation logic
                    src_id = group_source_ids[i] if i < len(group_source_ids) else None
                    res = self._handle_new_buffer(group_embeddings[i], source_id=src_id)
                    final_results[idx] = res
                continue
                
            # Case B: Knowledge Node
            try:
                # Load LoRA weights ONCE for the whole group
                # This solves the "Sequential Loading" bottleneck
                lora_weights = self.repository.get_lora_weights(node_id)
                
                # Run Transformer in BATCH mode
                # This solves the "Atomic Inference" bottleneck
                # Returns list of dicts
                trans_results = self.transformer.forward_batch_with_node(
                    group_embeddings, 
                    lora_weights
                )
                
                # ========================================================================
                
                # Check results and apply feedback
                for i, idx in enumerate(indices):
                    emb = group_embeddings[i]
                    src_id = group_source_ids[i]
                    res = trans_results[i]
                    
                    # Post-Processing & Feedback
                    feedback = self.post_processor.evaluate(res)
                    
                    # Validate feedback is not None
                    if feedback is None:
                        logger.warning(f"PostProcessor returned None for node {node_id} in batch, redirecting to buffer")
                        final_results[idx] = self._handle_new_buffer(emb, source_id=src_id)
                        continue
                    
                    if feedback.status == "OK":
                        # Apply feedback (will handle non-existent nodes internally)
                        self.repository.apply_feedback(node_id, feedback)
                        # Layer 1 Update (Process Validation & Welford)
                        # We do this sequentially as it involves DB state logic that is tricky to batch cleanly
                        # but it's much faster than inference, so not a major bottleneck
                        status = self._process_kn_update(node_id, emb, src_id)
                        final_results[idx] = status
                    else:
                        logger.debug(f"PostProcessor rejected response from {node_id}. Redirecting to buffer.")
                        # Fallback to buffer
                        status = self._handle_new_buffer(emb, source_id=src_id)
                        final_results[idx] = status
                        
            except Exception as e:
                logger.error(f"Error in batch flow for node {node_id}: {e}", exc_info=True)
                # Fallback for the whole group
                for i, idx in enumerate(indices):
                    src_id = group_source_ids[i] if i < len(group_source_ids) else None
                    final_results[idx] = self._handle_new_buffer(group_embeddings[i], source_id=src_id)
                    
        return final_results
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
            buffer_id = self.create_buffer(embedding, source_id=source_id)
            
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
            buffer_id = self.create_buffer(embedding, source_id=source_id)
            
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
        try:
            signature = kn.get_signature()
        except ValueError as e:
            # Node has no centroid (hasn't processed embeddings yet)
            # This shouldn't happen if process() was successful, but handle gracefully
            logger.warning(f"Node '{node_id}' has no signature: {e}, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Validate signature is not None
        if signature is None:
            logger.warning(f"Node '{node_id}' returned None signature, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Validate signature has required keys
        if not all(key in signature for key in ["node_id", "centroid", "mass", "variance"]):
            logger.warning(f"Node '{node_id}' signature missing required keys, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Update node stats (will handle non-existent nodes internally)
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
        
        # Note: FilterBayesian is now refreshed immediately after each embedding
        # is accepted (see above). This ensures centroids evolve in real-time.
        
        logger.debug(
            f"Embedding processed in node '{node_id}': "
            f"mass={signature['mass']}, variance={signature['variance']:.4f}"
        )
        
        return node_id
    
    def create_buffer(self, embedding: torch.Tensor, source_id: Optional[str] = None) -> str:
        """
        Create a new temporary buffer or add to an existing similar one.
        
        First tries to find an existing buffer with similarity > S_MIN.
        If not found, creates a new buffer.
        
        Phase 2: Also saves source_id if provided (for training data provenance).
        
        Args:
            embedding: Embedding tensor [EMBEDDING_DIM]
            source_id: Optional source ID (pointer to original dataset)
                If provided, will be saved for later transfer to KnowledgeNode
        
        Returns:
            str: Buffer ID (existing or new)
        """
        # Try to find similar buffer
        similar_buffer_id = self._find_similar_buffer(embedding)
        
        if similar_buffer_id:
            # Add to existing buffer
            self.repository.add_to_buffer(similar_buffer_id, embedding, source_id=source_id)
            logger.debug(f"Embedding added to existing buffer: {similar_buffer_id}")
            return similar_buffer_id
        
        # No similar buffer found, create a new one
        buffer_id = f"buffer_{uuid4()}"
        
        # Create buffer in Repository
        self.repository.create_buffer(buffer_id)
        
        # Add embedding to buffer (with source_id if provided)
        self.repository.add_to_buffer(buffer_id, embedding, source_id=source_id)
        
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

        # ========================================================================
        # Titan Selection (Inheritance) (2025-01-XX)
        # ========================================================================
        # Identify best parent (Titan) for this new node
        # ========================================================================
        parent_node_id = None
        parent_similarity = None
        inheritance_type = "ROOT"  # Default
        
        if INHERITANCE_ENABLED:
            parent_node_id, parent_similarity = self._identify_best_parent(initial_centroid)
            if parent_node_id:
                inheritance_type = "HERITAGE"
                logger.info(
                    f"Inheritance selected: Node '{node_id}' will inherit from Titan '{parent_node_id}' "
                    f"(sim={parent_similarity:.3f}, type={inheritance_type})"
                )
            else:
                logger.debug(f"No suitable Titan found for '{node_id}', born as orphan (ROOT).")
        
        # Process all buffer embeddings in the new node
        # (This will update statistics incrementally)
        for emb in embeddings:
            kn.process(emb)
        
        # ========================================================================
        # Persist in Repository
        # ========================================================================
        
        # Get final signature (after processing all embeddings)
        try:
            signature = kn.get_signature()
        except ValueError as e:
            logger.error(f"Error getting signature for promoted node '{node_id}': {e}")
            raise ValueError(f"Cannot promote buffer {buffer_id}: node has no valid signature") from e
        
        # Validate signature is not None and has required keys
        if signature is None or not all(key in signature for key in ["node_id", "centroid", "mass", "variance"]):
            logger.error(f"Invalid signature for promoted node '{node_id}'")
            raise ValueError(f"Cannot promote buffer {buffer_id}: invalid signature")
        
        # Create signature in Repository
        self.repository.save_new_kn(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"],
            parent_node_id=parent_node_id,
            parent_similarity=parent_similarity,
            inheritance_type=inheritance_type
        )
        
        # Force commit immediately to ensure node is available for subsequent operations
        # This prevents "node does not exist" errors when saving pointers in the same batch
        self.repository.conn.commit()
        
        # ========================================================================
        # IMPORTANT: Transfer source_ids from buffer to KnowledgeNode (2025-01-25)
        # ========================================================================
        # This block MUST be AFTER save_new_kn() and commit() - DO NOT MOVE IT!
        # 
        # WHY THIS IS CRITICAL:
        # - save_pointer() requires the node to exist in the database
        # - If called before save_new_kn(), it fails with "KN does not exist"
        # - Without source_ids, training cannot retrieve original texts
        # - This was causing training to be skipped entirely (0 pointers found)
        #
        # DO NOT optimize by moving this earlier - it will break training!
        # ========================================================================
        buffer_source_ids = self.repository.get_buffer_source_ids(buffer_id)
        if buffer_source_ids:
            transferred_count = 0
            for source_id in buffer_source_ids:
                try:
                    self.repository.save_pointer(node_id, source_id)
                    transferred_count += 1
                    logger.debug(f"Transferred source_id '{source_id}' from buffer '{buffer_id}' to node '{node_id}'")
                except Exception as e:
                    logger.debug(f"Failed to transfer source_id '{source_id}' from buffer '{buffer_id}': {e}")
            logger.debug(f"Transferred {transferred_count}/{len(buffer_source_ids)} source_ids from buffer '{buffer_id}' to node '{node_id}'")
        else:
            logger.debug(f"No source_ids to transfer from buffer '{buffer_id}' to node '{node_id}'")
        
        # ========================================================================
        # Load KnowledgeNode using from_repository() (2025-12-27)
        # ========================================================================
        # Changed from keeping the directly created KnowledgeNode to loading
        # from Repository using from_repository(). This ensures consistency:
        # - PEFT metadata is loaded correctly (even if None)
        # - Training status is loaded correctly (even if None)

        try:
            kn = KnowledgeNode.from_repository(self.repository, node_id)
        except Exception as e:
            logger.error(f"Error loading new KN {node_id} from repo: {e}")
            # Fallback to current kn if loading fails
        
        # Add to active nodes
        self.active_nodes[node_id] = kn
        
        # Update counter
        self.kn_count += 1
        
        # Delete buffer
        self.repository.delete_buffer(buffer_id)
        self.buffer_count -= 1
        
        logger.info(
            f"Buffer promoted to KN: {node_id} "
            f"(mass={signature['mass']}, variance={signature['variance']:.4f})"
        )
        
        # Update FilterBayesian immediately
        self.filter.partial_update(
            node_id=signature["node_id"],
            new_centroid=signature["centroid"],
            new_mass=signature["mass"]
        )
        
        # ========================================================================
        # Phase 2: Training Threshold Check (2026-01-26)
        # ========================================================================
        # Check if new node already reached training threshold (mass >= 10)
        # This prevents nodes from being "trapped" without training flags
        # ========================================================================
        self._check_and_trigger_training(node_id, signature)
        
        return node_id

    def check_quarantine_status(self, node_id: str) -> str:
        """
        Check quarantine status for a node.
        
        Added: Fase 4 (2025-01-XX)
        
        Args:
            node_id: Node ID to check
        
        Returns:
            - 'IN_QUARANTINE': Node is currently in quarantine
            - 'QUARANTINE_EXPIRED': Quarantine period has expired, node can be re-evaluated
            - 'FORCED_ROOT': Node is permanently marked as ROOT
            - 'NO_QUARANTINE': Node is not in quarantine
        """
        # Fase 9: Validaciones de entrada
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            logger.warning(f" check_quarantine_status: node_id inválido: {node_id}")
            return 'NO_QUARANTINE'
        
        try:
            quarantine_info = self.repository.get_quarantine_info(node_id)
            if not quarantine_info:
                return 'NO_QUARANTINE'
            
            # Fase 9: Manejo robusto de NULLs
            forced_root = bool(quarantine_info.get('forced_root') or False)
            if forced_root:
                logger.debug(f"Nodo {node_id} está marcado como ROOT permanente")
                return 'FORCED_ROOT'
            
            # Fase 9: Validación de rango para quarantine_count
            quarantine_count = quarantine_info.get('quarantine_count')
            if quarantine_count is None:
                quarantine_count = 0
            else:
                try:
                    quarantine_count = int(quarantine_count)
                    if quarantine_count < 0:
                        logger.warning(f" quarantine_count negativo para {node_id}: {quarantine_count}, corrigiendo a 0")
                        quarantine_count = 0
                except (ValueError, TypeError):
                    logger.warning(f" quarantine_count inválido para {node_id}: {quarantine_count}, usando 0")
                    quarantine_count = 0
            
            if quarantine_count == 0:
                return 'NO_QUARANTINE'
            
            last_exit = quarantine_info.get('last_quarantine_exit')
            if last_exit is None:
                # Node is currently in quarantine (entered but hasn't exited)
                logger.debug(f"Nodo {node_id} está en cuarentena (count={quarantine_count})")
                return 'IN_QUARANTINE'
            
            # Parse last_exit datetime
            from datetime import datetime
            try:
                if isinstance(last_exit, str):
                    exit_time = datetime.fromisoformat(last_exit.replace('Z', '+00:00'))
                else:
                    exit_time = last_exit
            except Exception as e:
                logger.warning(f" Error parseando last_quarantine_exit para {node_id}: {e}")
                # If we can't parse, assume expired
                return 'QUARANTINE_EXPIRED'
            
            # Calculate time since exit
            now = datetime.utcnow()
            if exit_time.tzinfo is None:
                # Assume UTC if no timezone
                from datetime import timezone
                exit_time = exit_time.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            
            time_diff = (now - exit_time).total_seconds() / 3600.0  # Hours
            
            # Determine quarantine duration based on count
            if quarantine_count == 1:
                quarantine_duration = 48.0  # 48 hours
            elif quarantine_count == 2:
                quarantine_duration = 168.0  # 1 week (168 hours)
            else:
                # Should not reach here (forced_root should be True for count >= 3)
                quarantine_duration = 168.0
            
            if time_diff < quarantine_duration:
                logger.debug(
                    f"Nodo {node_id} en cuarentena hasta {quarantine_duration}h "
                    f"(transcurrido: {time_diff:.1f}h)"
                )
                return 'IN_QUARANTINE'
            else:
                logger.info(
                    f"Nodo {node_id} cuarentena expirada ({time_diff:.1f}h > {quarantine_duration}h), "
                    f"puede ser re-evaluado"
                )
                return 'QUARANTINE_EXPIRED'
                
        except Exception as e:
            logger.error(f"Error verificando cuarentena para {node_id}: {e}", exc_info=True)
            # On error, assume no quarantine (conservative)
            return 'NO_QUARANTINE'
    
    def should_increment_quarantine(
        self, 
        node_id: str, 
        failure_type: str = "TRAINING_ERROR",
        training_progress: Optional[float] = None
    ) -> bool:
        """
        Determine if quarantine count should be incremented based on simplified limits table.
        
        Added: Fase 4/5 (2025-01-XX)
        
        Simplified Logic:
        - Increment if: (time_since_quarantine < 48h) AND (failure_type == HERITAGE) AND (progress < 50% OR NULL)
        - Otherwise: Don't increment (new event or late failure)
        
        Args:
            node_id: Node ID
            failure_type: Type of failure ('PARENT_FAILED', 'LIFE_INSURANCE', 'TRAINING_ERROR', 'SYSTEM_ERROR')
            training_progress: Training progress (0.0-1.0) or None
        
        Returns:
            True if quarantine should be incremented, False otherwise
        """
        # Fase 9: Validaciones de entrada
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            logger.warning(f" should_increment_quarantine: node_id inválido: {node_id}")
            return False
        
        # Fase 9: Validación de failure_type
        valid_failure_types = ('PARENT_FAILED', 'LIFE_INSURANCE', 'TRAINING_ERROR', 'SYSTEM_ERROR')
        if failure_type not in valid_failure_types:
            logger.warning(f" should_increment_quarantine: failure_type inválido: {failure_type}, usando TRAINING_ERROR")
            failure_type = "TRAINING_ERROR"
        
        # Fase 9: Validación de training_progress
        if training_progress is not None:
            try:
                training_progress = float(training_progress)
                if not (0.0 <= training_progress <= 1.0):
                    logger.warning(
                        f" should_increment_quarantine: training_progress fuera de rango [0.0, 1.0]: "
                        f"{training_progress}, usando None"
                    )
                    training_progress = None
            except (ValueError, TypeError):
                logger.warning(f" should_increment_quarantine: training_progress inválido: {training_progress}, usando None")
                training_progress = None
        
        try:
            quarantine_info = self.repository.get_quarantine_info(node_id)
            if not quarantine_info:
                # First failure, don't increment yet (will be handled separately)
                return False
            
            last_exit = quarantine_info.get('last_quarantine_exit')
            if last_exit is None:
                # Currently in quarantine, don't increment again
                return False
            
            # Parse last_exit datetime
            from datetime import datetime
            try:
                if isinstance(last_exit, str):
                    exit_time = datetime.fromisoformat(last_exit.replace('Z', '+00:00'))
                else:
                    exit_time = last_exit
            except Exception as e:
                logger.warning(f" Error parseando last_quarantine_exit para {node_id}: {e}")
                # If we can't parse, assume it's a new event
                return False
            
            # Calculate time since exit
            now = datetime.utcnow()
            if exit_time.tzinfo is None:
                from datetime import timezone
                exit_time = exit_time.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            
            time_diff_hours = (now - exit_time).total_seconds() / 3600.0
            
            # Check conditions from simplified table
            time_condition = time_diff_hours < 48.0
            failure_condition = failure_type in ('PARENT_FAILED', 'LIFE_INSURANCE')
            progress_condition = training_progress is None or training_progress < 0.5
            
            should_increment = time_condition and failure_condition and progress_condition
            
            logger.debug(
                f"🔍 Evaluando reincidencia para {node_id}: "
                f"tiempo={time_diff_hours:.1f}h, tipo={failure_type}, progreso={training_progress}, "
                f"incrementar={should_increment}"
            )
            
            if should_increment:
                logger.info(f"Nodo {node_id} cumple condiciones de reincidencia")
            else:
                logger.info(
                    f"Nodo {node_id} no cumple condiciones "
                    f"(tiempo={time_diff_hours:.1f}h, tipo={failure_type}, progreso={training_progress})"
                )
            
            return should_increment
            
        except Exception as e:
            logger.error(f"Error evaluando reincidencia para {node_id}: {e}", exc_info=True)
            # On error, don't increment (conservative)
            return False

    def _calculate_affinity(
        self, 
        titan_id: str, 
        child_centroid: torch.Tensor,
        use_progressive_threshold: bool = False
    ) -> Optional[Tuple[str, float]]:
        """
        Helper to check affinity between a specific titan and a child.
        
        Fase 3: Also verifies that titan is trained and not FAILED.
        Fase 7: Uses PROGRESSIVE_ADOPTION_THRESHOLD (0.60) if use_progressive_threshold=True,
                otherwise uses TITAN_SIMILARITY_THRESHOLD (0.70) for Late Adoption.
        
        Args:
            titan_id: Titan node ID
            child_centroid: Child node centroid tensor
            use_progressive_threshold: If True, use 0.60 threshold (Progressive Adoption),
                                      otherwise use 0.70 (Late Adoption)
        """
        titan_sig = self.repository.get_signature(titan_id)
        if not titan_sig:
            return None
        
        # Fase 3: Verify that titan is trained
        if not self.repository.is_trained(titan_id):
            logger.debug(f"🔍 Titan {titan_id} excluido en _calculate_affinity: no está entrenado")
            return None
        
        # Fase 3: Verify that titan is not FAILED
        try:
            training_status = self.repository.get_peft_training_status(titan_id)
            if training_status == 'FAILED':
                logger.debug(f"🔍 Titan {titan_id} excluido en _calculate_affinity: está marcado como FAILED")
                return None
        except Exception as e:
            logger.debug(f" Error verificando estado de entrenamiento para {titan_id}: {e}")
            # If we can't check, be conservative and exclude
            return None
            
        titan_centroid = titan_sig['centroid']
        variance = titan_sig['variance']
        mass = titan_sig['mass']
        
        # Calculate cosine similarity
        try:
            similarity = torch.nn.functional.cosine_similarity(
                child_centroid.unsqueeze(0), 
                titan_centroid.unsqueeze(0)
            ).item()
        except Exception as e:
            logger.error(f"Error calculando similitud para {titan_id}: {e}", exc_info=True)
            return None
        
        # Fase 7: Use appropriate threshold
        threshold = PROGRESSIVE_ADOPTION_THRESHOLD if use_progressive_threshold else TITAN_SIMILARITY_THRESHOLD
        
        # Fase 9: Validación de rango para similarity
        if not isinstance(similarity, (int, float)):
            logger.warning(f" Similitud no es numérica para {titan_id}: {similarity} (tipo: {type(similarity)})")
            return None
        
        if not (0.0 <= similarity <= 1.0):
            logger.warning(f" Similitud fuera de rango [0.0, 1.0] para {titan_id}: {similarity}")
            # Clamp to valid range for safety
            similarity = max(0.0, min(1.0, float(similarity)))
        
        if similarity >= threshold:
             return titan_id, similarity
        return None

    def add_to_re_evaluation_queue(self, node_id: str) -> None:
        """
        Add a node to the re-evaluation queue.
        
        This is called when a node's parent fails training, so the node
        can be re-evaluated to find a new parent.
        
        Added: 2025-01-XX (Fase 8: Pool de Re-evaluación)
        
        Args:
            node_id: ID of the node to add to the queue
        """
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            logger.warning(f" Invalid node_id for re-evaluation queue: {node_id}")
            return
        
        # Validate node exists
        try:
            sig = self.repository.get_signature(node_id)
            if not sig:
                logger.debug(f"Node {node_id} not found, skipping re-evaluation queue")
                return
        except Exception as e:
            logger.warning(f" Error validating node {node_id} for re-evaluation queue: {e}")
            return
        
        # Thread-safe add to queue (prevent duplicates)
        with self.re_evaluation_lock:
            if node_id not in self.re_evaluation_queue:
                self.re_evaluation_queue.append(node_id)
                logger.debug(f"Agregando {node_id} a cola de re-evaluación")
    
    def process_re_evaluation_queue(self) -> int:
        """
        Process the re-evaluation queue to find new parents for orphaned nodes.
        
        This method is called after each successful node training to allow
        orphaned nodes (whose parents failed) to find new parents.
        
        Added: 2025-01-XX (Fase 8: Pool de Re-evaluación)
        
        Returns:
            Number of nodes successfully re-evaluated and adopted
        """
        re_evaluated_count = 0
        
        # Thread-safe get queue snapshot
        with self.re_evaluation_lock:
            if not self.re_evaluation_queue:
                return 0
            # Get snapshot and clear queue
            queue_snapshot = list(self.re_evaluation_queue)
            self.re_evaluation_queue.clear()
        
        if not queue_snapshot:
            return 0
        
        logger.debug(f"Procesando cola de re-evaluación: {len(queue_snapshot)} nodos pendientes")
        
        # Load all signatures and trained node IDs once (optimization)
        all_signatures = self.repository.get_all_signatures()
        trained_node_ids = set(self.repository.get_trained_node_ids())
        
        # Get all FAILED node IDs to exclude them
        failed_node_ids = set()
        try:
            cursor = self.repository.conn.cursor()
            cursor.execute(
                "SELECT node_id FROM nodes WHERE status = 'KN' AND peft_training_status = 'FAILED'"
            )
            failed_node_ids = {row[0] for row in cursor.fetchall()}
            cursor.close()
        except Exception as e:
            logger.warning(f" Error obteniendo nodos FAILED para re-evaluación: {e}")
        
        # Filter to only trained Titans (same logic as Late Adoption Scan)
        trained_titans = [
            sig for sig in all_signatures
            if sig.get('mass', 0) >= 20  # TITAN_MIN_MASS
            and (sig.get('variance') is None or sig.get('variance', 1.0) <= 0.5)  # TITAN_MAX_VARIANCE
            and sig['node_id'] in trained_node_ids
            and sig['node_id'] not in failed_node_ids  # Exclude FAILED nodes
        ]
        
        # Process each node in queue
        for node_id in queue_snapshot:
            try:
                # Validate node still exists and is not training
                sig = self.repository.get_signature(node_id)
                if not sig:
                    logger.debug(f"Node {node_id} no longer exists, skipping re-evaluation")
                    continue
                
                # Check if node is currently training
                training_status = self.repository.get_peft_training_status(node_id)
                if training_status == 'TRAINING':
                    logger.debug(f"Node {node_id} is currently training, skipping re-evaluation")
                    # Re-add to queue for later processing
                    with self.re_evaluation_lock:
                        if node_id not in self.re_evaluation_queue:
                            self.re_evaluation_queue.append(node_id)
                    continue
                
                # Check quarantine status
                quarantine_status = self.check_quarantine_status(node_id)
                if quarantine_status == 'IN_QUARANTINE':
                    logger.debug(f"Node {node_id} is in quarantine, skipping re-evaluation")
                    continue
                
                # Get node centroid
                centroid = sig.get('centroid')
                if centroid is None:
                    logger.debug(f"Node {node_id} has no centroid, skipping re-evaluation")
                    continue
                
                # Find best parent from trained Titans (Late Adoption logic)
                child_centroid_tensor = centroid.to(device=DEVICE, dtype=DTYPE)
                best_parent_id = None
                best_similarity = 0.0
                
                for titan_sig in trained_titans:
                    if titan_sig['node_id'] == node_id:
                        continue  # Skip self
                    
                    titan_centroid = titan_sig['centroid'].to(device=DEVICE, dtype=DTYPE)
                    similarity = F.cosine_similarity(
                        child_centroid_tensor.unsqueeze(0),
                        titan_centroid.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    # Use TITAN_SIMILARITY_THRESHOLD (0.70) for Late Adoption
                    if similarity >= 0.70 and similarity > best_similarity:
                        best_similarity = similarity
                        best_parent_id = titan_sig['node_id']
                
                # If found a parent, update node
                if best_parent_id and best_parent_id != node_id:
                    try:
                        self.repository.update_parent_info(node_id, best_parent_id, best_similarity)
                        logger.info(
                            f"Re-evaluación exitosa: Node '{node_id}' adoptado por Titan '{best_parent_id}' "
                            f"(sim={best_similarity:.3f})"
                        )
                        re_evaluated_count += 1
                    except Exception as e:
                        logger.warning(f" Error actualizando padre para {node_id} durante re-evaluación: {e}")
                else:
                    logger.debug(f"Re-evaluación: Node '{node_id}' no encontró padre adecuado (mejor sim={best_similarity:.3f})")
                    
            except Exception as e:
                logger.warning(
                    f" Error procesando {node_id} en cola de re-evaluación: {e}",
                    exc_info=True
                )
                # Continue with next node
                continue
        
        return re_evaluated_count
    
    def _identify_best_parent(self, child_centroid: torch.Tensor, exclude_node_id: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
        """
        Identify the best 'Titan' parent for a new node.
        
        Criteria for Parent (Titan):
        1. Mature: Mass >= TITAN_MIN_MASS
        2. Stable: Variance <= TITAN_MAX_VARIANCE
        3. Relevant: Similarity >= TITAN_SIMILARITY_THRESHOLD
        
        Selection:
        - Max Affinity Score = (Mass * Similarity) / (1 + Variance)
        
        Args:
            child_centroid: Centroid of the new child node
            exclude_node_id: Optional node ID to exclude from candidates (prevents self-adoption)
            
        Returns:
            Tuple (parent_node_id, similarity) or (None, None)
        """
        # Get all signatures (candidates)
        signatures = self.repository.get_all_signatures()
        
        if not signatures:
            return None, None
            
        # Filter candidates (Maturity & Stability) and exclude self if provided
        # Fase 3: Also exclude FAILED nodes and nodes that are not trained
        titans = []
        for sig in signatures:
            # Basic criteria: Mass and Variance
            if sig['mass'] < TITAN_MIN_MASS:
                continue
            if sig['variance'] is not None and sig['variance'] > TITAN_MAX_VARIANCE:
                continue
            # Exclude self if provided
            if exclude_node_id is not None and sig['node_id'] == exclude_node_id:
                continue
            # Fase 3: Exclude FAILED nodes
            node_id = sig['node_id']
            if not self.repository.is_trained(node_id):
                logger.debug(f"🔍 Candidato {node_id} excluido: no está entrenado")
                continue
            # Check if node is FAILED (Fase 3)
            try:
                training_status = self.repository.get_peft_training_status(node_id)
                if training_status == 'FAILED':
                    logger.debug(f"🔍 Candidato {node_id} excluido: está marcado como FAILED")
                    continue
            except Exception as e:
                logger.debug(f" Error verificando estado de entrenamiento para {node_id}: {e}")
                # If we can't check, be conservative and exclude
                continue
            titans.append(sig)
        
        if not titans:
            logger.debug("No nodes qualify as Titans (mass/variance criteria)")
            return None, None
            
        # Prepare tensors for vectorized similarity
        titan_centroids = torch.stack([sig['centroid'] for sig in titans]).to(device=DEVICE, dtype=DTYPE)
        child_centroid = child_centroid.to(device=DEVICE, dtype=DTYPE)
        
        # Calculate Cosine Similarity
        # [N_titans]
        sims = F.cosine_similarity(child_centroid.unsqueeze(0), titan_centroids, dim=1)
        
        # Filter by Similarity Threshold
        valid_mask = sims >= TITAN_SIMILARITY_THRESHOLD
        
        # Indices of valid titans
        valid_indices = torch.nonzero(valid_mask).squeeze()
        
        if valid_indices.numel() == 0:
            return None, None
            
        # Handle single result case (scalar vs tensor)
        if valid_indices.ndim == 0:
            valid_indices = valid_indices.unsqueeze(0)
            
        best_score = -1.0
        best_titan_idx = -1
        best_similarity = 0.0
        
        # Calculate Affinity Score for valid candidates
        for idx in valid_indices:
            idx = idx.item()
            titan = titans[idx]
            sim = sims[idx].item()
            mass = titan['mass']
            variance = titan['variance'] if titan['variance'] is not None else 0.0
            
            # Affinity Score: Heavily favors high mass + high similarity 
            # Penalizes high variance
            affinity = (mass * sim) / (1.0 + variance)
            
            if affinity > best_score:
                best_score = affinity
                best_titan_idx = idx
                best_similarity = sim
                
        if best_titan_idx != -1:
            best_parent = titans[best_titan_idx]
            return best_parent['node_id'], best_similarity
            
        return None, None
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
        
        # ========================================================================
        # CRITICAL: Update FilterBayesian with new node (2025-01-14)
        # ========================================================================
        # When a new node is created, FilterBayesian must be updated so it can
        # route future embeddings to this new node. Without this, the new node
        # will never receive embeddings because FilterBayesian doesn't know it exists.
        # ========================================================================
        # Refresh all signatures to include the new node
        # This is O(N) but only happens when promoting buffers (infrequent)
        all_signatures = self.repository.get_all_signatures()
        self.filter.refresh_signatures(all_signatures)
        logger.debug(
            f"FilterBayesian refreshed with new node '{node_id}' "
            f"(total signatures: {len(all_signatures)})"
        )
        
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
    
    def has_active_training(self) -> bool:
        """
        Check if there are any active training tasks.
        
        Returns:
            True if any node is currently training, False otherwise
        """
        # Check if there are any training futures that are not done
        if self._training_futures:
            for future in self._training_futures.values():
                if not future.done():
                    return True
        
        # Also check active nodes for is_training flag (faster check)
        for kn in self.active_nodes.values():
            if kn.is_training:
                return True
        
        return False
    
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
    
    def run_deferred_training(self, progress_interval: int = 1) -> Dict[str, Any]:
        """
        Execute deferred training for all nodes marked as needing training.
        
        Added: 2025-01-14
        Purpose: Run training for all nodes marked with needs_training flag.
        This method runs after processing is complete (Layer 2: Specialization).
        It trains all nodes sequentially, one at a time.
        
        Flow:
        1. Get all nodes with needs_training = 1
        2. Order by dependencies (parents first)
        3. For each node:
           a. Activate training buffer
           b. Re-fetch all current source_ids
           c. Get texts
           d. Train adapter
           e. Save weights
           f. Process training buffer
           g. Clear flags
        
        Args:
            progress_interval: How often to log progress (default: every node)
        
        Returns:
            Dictionary with training statistics:
            - total_pending: Total nodes pending training
            - trained: Number of nodes successfully trained
            - failed: Number of nodes that failed training
        """
        # ========================================================================
        # Phase 0: Cleanup Stale Training States (2025-01-XX)
        # ========================================================================
        # Clean up any nodes that were left in 'TRAINING' state due to interruptions
        # This ensures consistent state when restarting after Ctrl+C or crashes
        # ========================================================================
        cleaned_count = self.repository.cleanup_stale_training_states()
        if cleaned_count > 0:
            logger.info(
                f"🧹 Cleaned stale training states | count={cleaned_count} | "
                f"from_previous_session=True"
            )
        
        # ========================================================================
        # Phase 0.1: Sanitize Orphans (2025-01-XX) - Fase 2
        # ========================================================================
        # Clean nodes that point to parents that failed in training
        # This ensures database consistency before starting training
        # ========================================================================
        sanitized = self.repository.sanitize_orphans()
        if sanitized > 0:
            logger.info(
                f"🛡️  Sanitización completada | affected={sanitized} | "
                f"phase=initial_cleanup"
            )
        
        pending = self.repository.get_pending_training_nodes()
        
        # Update cache with current pending nodes
        with self._pending_cache_lock:
            self._pending_training_cache = set(pending)
        
        if not pending:
            logger.info("No nodes pending training")
            print("   No hay nodos pendientes de entrenamiento")
            sys.stdout.flush()
            return {
                "total_pending": 0,
                "trained": 0,
                "failed": 0
            }
        
        # ========================================================================
        # OPTIMIZATION: Cache all signatures once to avoid repeated DB reads
        # ========================================================================
        # Load all signatures for pending nodes in a single batch to minimize DB locks
        signatures_cache = {}
        for node_id in pending:
            sig = self.repository.get_signature(node_id)
            if sig:
                signatures_cache[node_id] = sig
        
        # ========================================================================
        # Phase 4: Late Adoption Scan (NUEVO) - OPTIMIZED
        # ========================================================================
        # Antes de ordenar, dar una oportunidad a los ROOT de encontrar padre.
        # Esto desbloquea el "Inheritance Unlocking".
        # OPTIMIZATION: Load all signatures ONCE and check is_trained in batch
        # ========================================================================
        scan_count = 0
        # Load all signatures once for Late Adoption Scan
        all_signatures_for_scan = self.repository.get_all_signatures()
        
        # OPTIMIZATION: Get all trained node IDs in a single query
        trained_node_ids = set(self.repository.get_trained_node_ids())
        
        # Fase 3: Get all FAILED node IDs to exclude them
        # Query to get all FAILED node IDs efficiently
        failed_node_ids = set()
        try:
            cursor = self.repository.conn.cursor()
            cursor.execute(
                "SELECT node_id FROM nodes WHERE status = 'KN' AND peft_training_status = 'FAILED'"
            )
            failed_node_ids = {row[0] for row in cursor.fetchall()}
            cursor.close()
        except Exception as e:
            logger.warning(f" Error obteniendo nodos FAILED para filtrado: {e}")
        
        # Filter to only trained Titans for Late Adoption (using cached trained IDs)
        # Fase 3: Also exclude FAILED nodes
        trained_titans_for_scan = [
            sig for sig in all_signatures_for_scan
            if sig.get('mass', 0) >= 20  # TITAN_MIN_MASS
            and (sig.get('variance') is None or sig.get('variance', 1.0) <= 0.5)  # TITAN_MAX_VARIANCE
            and sig['node_id'] in trained_node_ids  # Use cached set instead of DB query
            and sig['node_id'] not in failed_node_ids  # Fase 3: Exclude FAILED nodes
        ]
        
        if failed_node_ids:
            logger.debug(f"🔍 Late Adoption: Excluyendo {len(failed_node_ids)} nodos FAILED de candidatos")
        
        for node_id in pending:
            sig = signatures_cache.get(node_id)
            # Solo escanear si es huérfano (ROOT) y no tiene padre asignado
            if sig and not sig.get('parent_node_id'):
                centroid = sig.get('centroid')
                if centroid is not None:
                    # Use cached trained titans instead of calling _identify_best_parent
                    # (which would call get_all_signatures() again)
                    best_parent_id = None
                    best_similarity = 0.0
                    
                    # Find best parent from cached trained titans
                    child_centroid_tensor = centroid.to(device=DEVICE, dtype=DTYPE)
                    for titan_sig in trained_titans_for_scan:
                        if titan_sig['node_id'] == node_id:
                            continue  # Skip self
                        titan_centroid = titan_sig['centroid'].to(device=DEVICE, dtype=DTYPE)
                        similarity = F.cosine_similarity(
                            child_centroid_tensor.unsqueeze(0),
                            titan_centroid.unsqueeze(0),
                            dim=1
                        ).item()
                        
                        # Check similarity threshold (TITAN_SIMILARITY_THRESHOLD = 0.75)
                        if similarity >= 0.75 and similarity > best_similarity:
                            best_similarity = similarity
                            best_parent_id = titan_sig['node_id']
                    
                    # CRITICAL: Validar que el padre esté entrenado para Adopción Tardía
                    if best_parent_id and best_parent_id != node_id:
                        # ¡Adopción Tardía Éxitosa!
                        try:
                            self.repository.update_parent_info(node_id, best_parent_id, best_similarity)
                            # Update cache after successful adoption
                            sig['parent_node_id'] = best_parent_id
                            sig['parent_similarity'] = best_similarity
                            sig['inheritance_type'] = 'HERITAGE'
                            logger.info(
                                f"Late Adoption: Node '{node_id}' adopted by Titan '{best_parent_id}' "
                                f"(sim={best_similarity:.3f})"
                            )
                            scan_count += 1
                        except Exception as e:
                            logger.error(f"Failed to update late adoption for {node_id}: {e}")
        
        if scan_count > 0:
            logger.info(
                f"👨‍👧 Late Adoption Scan completado | nodes_adopted={scan_count} | "
                f"phase=pre_training"
            )

        # ========================================================================
        # Ordenar por dependencias (Titanes Reales Primero)
        # ========================================================================
        # Separar en 4 grupos explícitos:
        # 1. Titanes reales (mass >= 20, variance <= 0.5) - ÚNICOS que pueden ser padres
        # 2. ROOT no-Titanes (no cumplen criterios) - NO pueden ser padres
        # 3. HERITAGE listos (con padre entrenado) - Pueden entrenarse ahora
        # 4. HERITAGE esperando (sin padre entrenado) - Deben esperar
        
        real_titans = []  # Titanes reales (pueden ser padres)
        root_non_titans = []  # ROOT que NO son Titanes (no pueden ser padres)
        children = []  # HERITAGE (con padre asignado)
        
        # OPTIMIZATION: Get all trained node IDs once to avoid multiple DB queries
        trained_node_ids = set(self.repository.get_trained_node_ids())
        
        for node_id in pending:
            sig = signatures_cache.get(node_id)
            
            # Separar HERITAGE de ROOT
            if sig and sig.get('parent_node_id'):
                children.append(node_id)
            else:
                # Es ROOT - verificar si es Titan real
                mass = sig.get('mass', 0) if sig else 0
                variance = sig.get('variance') if sig else None
                
                # Criterios de Titan: mass >= 20 AND variance <= 0.5
                is_real_titan = (
                    mass >= TITAN_MIN_MASS and
                    (variance is None or variance <= TITAN_MAX_VARIANCE)
                )
                
                if is_real_titan:
                    real_titans.append(node_id)
                else:
                    root_non_titans.append(node_id)
        
        # Orden 1: Titanes reales (TODOS primero, sin importar tamaño)
        # Lo importante es tener más Titanes entrenados, no el orden entre ellos
        real_titans_sorted = sorted(
            real_titans,
            key=lambda nid: signatures_cache[nid]['mass'],
            reverse=False  # Pequeños primero (rápido desbloqueo)
        )
        
        # Orden 2: ROOT no-Titanes (después de Titanes, no desbloquean herencia)
        root_non_titans_sorted = sorted(
            root_non_titans,
            key=lambda nid: signatures_cache[nid]['mass'],
            reverse=True  # Grandes primero (más impacto)
        )
        
        # Separar children: los que tienen padres entrenados vs los que esperan
        children_ready = []
        children_waiting = []
        
        for child_id in children:
            sig = signatures_cache.get(child_id)
            parent_id = sig.get('parent_node_id') if sig else None
            
            if parent_id and parent_id in trained_node_ids:  # Use cached set instead of DB query
                children_ready.append(child_id)
            else:
                children_waiting.append(child_id)
        
        # Orden 3: HERITAGE listos (pueden entrenarse si padre está listo)
        children_ready_sorted = sorted(
            children_ready,
            key=lambda nid: signatures_cache[nid]['mass'],
            reverse=True  # Grandes primero
        )
        
        # Orden 4: HERITAGE esperando (deben esperar a que padre se entrene)
        # Sin ordenar - esperan padre
        
        # Orden final: Titanes reales → ROOT no-Titanes → HERITAGE listos → HERITAGE esperando
        ordered_pending = (
            real_titans_sorted +           # PRIORIDAD 1: Todos los Titanes primero
            root_non_titans_sorted +       # PRIORIDAD 2: ROOT no-Titanes
            children_ready_sorted +        # PRIORIDAD 3: HERITAGE listos
            children_waiting                # PRIORIDAD 4: HERITAGE esperando
        )
        
        if children_waiting or root_non_titans_sorted:
            logger.info(
                f"Training order: {len(real_titans_sorted)} Real Titans, "
                f"{len(root_non_titans_sorted)} ROOT (non-Titans), "
                f"{len(children_ready_sorted)} Children (ready), "
                f"{len(children_waiting)} Children (waiting for parent)"
            )
        
        logger.info(
            f"📊 Starting deferred training | total_nodes={len(ordered_pending)} | "
            f"real_titans={len(real_titans_sorted)} | root_non_titans={len(root_non_titans_sorted)} | "
            f"children_ready={len(children_ready_sorted)} | children_waiting={len(children_waiting)}"
        )
        
        trained = 0
        failed = 0
        skipped = 0  # Track skipped nodes (no source_ids/texts)
        current_node_index = 0  # Track current node index for cleanup
        
        try:
            total_nodes = len(ordered_pending)
            print(f"   Entrenando {total_nodes} nodos pendientes...")
            sys.stdout.flush()
            
            for i, node_id in enumerate(ordered_pending, 1):
                current_node_index = i  # Update current index
                logger.info(f"Training node {i}/{total_nodes}: {node_id}")
                
                # Print progress every 10 nodes or for first/last nodes
                if i == 1 or i == total_nodes or i % 10 == 0:
                    print(f"   Entrenando nodo {i}/{total_nodes}...", flush=True)
                
                # Fase 4: Check quarantine status before training
                quarantine_status = self.check_quarantine_status(node_id)
                if quarantine_status == 'FORCED_ROOT':
                    logger.info(
                        f"Nodo marcado como ROOT permanente | node_id={node_id} | "
                        f"training_mode=root_only | inheritance=disabled"
                    )
                    # Continue training but will skip inheritance
                elif quarantine_status == 'IN_QUARANTINE':
                    logger.info(
                        f"Nodo en cuarentena, saltando entrenamiento | node_id={node_id} | "
                        f"action=skipped"
                    )
                    failed += 1
                    continue
                elif quarantine_status == 'QUARANTINE_EXPIRED':
                    # Mark as exited quarantine
                    try:
                        self.repository.exit_quarantine(node_id)
                        logger.info(
                            f"Nodo sale de cuarentena | node_id={node_id} | "
                            f"action=can_train"
                        )
                    except Exception as exit_error:
                        logger.warning(
                            f" Error marcando salida de cuarentena | node_id={node_id} | "
                            f"error={str(exit_error)[:100]}",
                            exc_info=True
                        )
                
                # Get estimated epochs for this node
                estimated_epochs = self.repository.get_estimated_epochs(node_id)
                epochs = estimated_epochs if estimated_epochs is not None else 3
                
                # CRITICAL: Verify source_ids BEFORE starting training
                # This prevents misleading "Training node iniciado" messages
                source_ids = self.repository.get_training_pointers(node_id)
                
                if not source_ids:
                    logger.warning(
                        f" No source_ids found for node '{node_id}', skipping training"
                    )
                    print(f"   Nodo {i}/{len(ordered_pending)}: Sin source_ids, saltando entrenamiento", flush=True)
                    # Mark as skipped (not failed, just no data)
                    self.repository.clear_training_flag(node_id)
                    # Update cache
                    with self._pending_cache_lock:
                        self._pending_training_cache.discard(node_id)
                    continue  # Skip to next node
                
                # Verify texts are available before starting training
                texts_preview = self.data_manager.get_texts_from_pointers(source_ids[:1])  # Check first one
                if not texts_preview:
                    logger.warning(
                        f" No texts found for node '{node_id}', skipping training"
                    )
                    print(f"   Nodo {i}/{len(ordered_pending)}: Sin textos disponibles, saltando entrenamiento", flush=True)
                    # Mark as skipped
                    self.repository.clear_training_flag(node_id)
                    # Update cache
                    with self._pending_cache_lock:
                        self._pending_training_cache.discard(node_id)
                    continue  # Skip to next node
                
                # Calculate total batches based on actual source_ids
                num_texts = len(source_ids)
                # Limit to MAX_TRAINING_TEXTS (usually 100)
                num_texts = min(num_texts, MAX_TRAINING_TEXTS)
                # Calculate total batches using configurable batch size
                batch_size = TRAINING_BATCH_SIZE
                total_batches = (num_texts + batch_size - 1) // batch_size if num_texts > 0 else 1
                
                # CRITICAL: Ensure main repository connection commits any pending transactions
                # before creating thread_repo to avoid database locks
                try:
                    self.repository.conn.commit()
                except:
                    pass  # Ignore if no transaction is active
                
                try:
                    # Use existing _async_train_kn but run synchronously
                    # (we're already in a separate phase, no need for thread pool)
                    # Log initial training start (only if we have data)
                    logger.info(
                        f"Training node iniciado | node_id={node_id} | "
                        f"progress={i}/{total_nodes} | epochs={epochs} | "
                        f"total_batches={total_batches} | texts={num_texts}"
                    )
                    print(f"   Entrenando nodo {i}/{total_nodes}: {num_texts} textos, {epochs} épocas, {total_batches} batches", flush=True)
                    
                    # Pass callback to update epoch progress (logged instead of printed)
                    def _log_epoch_progress(epoch, total_epochs):
                        logger.debug(f"Training node {i}/{total_nodes} | Época {epoch}/{total_epochs} | Total batches: {total_batches}")

                    self._async_train_kn(node_id, epoch_callback=_log_epoch_progress)
                    
                    # Verify training actually happened by checking if node is marked as trained
                    if self.repository.is_trained(node_id):
                        trained += 1
                        logger.info(
                            f"Training node completado | node_id={node_id} | "
                            f"progress={i}/{total_nodes} | status=success"
                        )
                    else:
                        # Training was skipped inside _async_train_kn (no source_ids/texts)
                        failed += 1
                        logger.warning(
                            f" Training node saltado | node_id={node_id} | "
                            f"progress={i}/{total_nodes} | reason=no_data"
                        )
                        print(f"   Nodo {i}/{total_nodes}: Entrenamiento saltado (sin datos)", flush=True)
                    
                    # ========================================================================
                    # Fase 8: Procesar Pool de Re-evaluación
                    # ========================================================================
                    # Después de entrenar cada nodo, procesar la cola de re-evaluación
                    # Esto permite que nodos huérfanos encuentren nuevos padres
                    # ========================================================================
                    try:
                        re_evaluated_count = self.process_re_evaluation_queue()
                        if re_evaluated_count > 0:
                            logger.info(
                                f"Re-evaluación completada | trigger_node={node_id} | "
                                f"nodes_reevaluated={re_evaluated_count} | phase=post_training"
                            )
                    except Exception as e:
                        logger.warning(
                            f" Error procesando cola de re-evaluación | trigger_node={node_id} | "
                            f"error={str(e)[:100]}",
                            exc_info=True
                        )
                        # Don't fail training if re-evaluation fails
                    
                    # ========================================================================
                    # Phase 4.1: Progressive Recalculation (NUEVO)
                    # ========================================================================
                    # Si el nodo recién entrenado es un Titán (no tiene padre),
                    # lanzar un micro-scan para ver si algún huérfano restante puede adoptarlo.
                    # ========================================================================
                    current_sig = self.repository.get_signature(node_id)
                    if current_sig and not current_sig.get('parent_node_id'):
                         # Es un nuevo Titán disponible!
                         new_titan_id = node_id
                         
                         # Fase 3: Verify that new Titan is trained and not FAILED before using it
                         if not self.repository.is_trained(new_titan_id):
                             logger.debug(f"🔍 Progressive Adoption: Titan {new_titan_id} no está entrenado, saltando")
                             continue
                         
                         try:
                             training_status = self.repository.get_peft_training_status(new_titan_id)
                             if training_status == 'FAILED':
                                 logger.debug(f"🔍 Progressive Adoption: Titan {new_titan_id} está FAILED, saltando")
                                 continue
                         except Exception as e:
                             logger.debug(f" Error verificando estado de Titan {new_titan_id}: {e}")
                             continue
                         
                         logger.debug(f"Progressive Scan: New Titan '{new_titan_id}' available. Scanning orphans...")
                         
                         orphans_rescued = 0
                         # Escanear SOLO los nodos pendientes que aún son huérfanos
                         for pending_id in ordered_pending[i:]: # Slicing: solo los que faltan
                             p_sig = self.repository.get_signature(pending_id)
                             if p_sig and not p_sig.get('parent_node_id'):
                                 # Verificar compatibilidad con este NUEVO Titán
                                 p_centroid = p_sig.get('centroid')
                                 if p_centroid is not None:
                                     # Usar afinidad especifica con este Titan
                                     # _calculate_affinity ya verifica que el Titan esté entrenado y no FAILED
                                     # Fase 7: Use progressive threshold (0.60) for Progressive Adoption
                                     affinity = self._calculate_affinity(new_titan_id, p_centroid, use_progressive_threshold=True)
                                     if affinity:
                                         pid, sim = affinity
                                         # Si pasa el umbral, adoptar inmediatamente
                                         # CRITICAL: Validar que no sea self-adoption
                                         # Fase 7: Already checked against PROGRESSIVE_ADOPTION_THRESHOLD in _calculate_affinity
                                         if pending_id != new_titan_id:
                                             try:
                                                 self.repository.update_parent_info(pending_id, new_titan_id, sim)
                                                 logger.info(
                                                     f"Progressive Adoption | child={pending_id} | "
                                                     f"parent={new_titan_id} | similarity={sim:.3f}"
                                                 )
                                                 orphans_rescued += 1
                                             except Exception as e:
                                                 logger.warning(
                                                     f"Failed progressive adoption | child={pending_id} | "
                                                     f"parent={new_titan_id} | error={str(e)[:100]}",
                                                     exc_info=True
                                                 )
                                                 # Continue with other nodes even if this one fails
                        
                         if orphans_rescued > 0:
                             # Log progressive adoption
                             logger.info(
                                 f"🧬 Progressive Adoption completado | titan_id={new_titan_id} | "
                                 f"orphans_rescued={orphans_rescued} | "
                                 f"phase=post_training"
                             )

                    if i % progress_interval == 0:
                        logger.info(f"Progress: {i}/{len(ordered_pending)} nodes trained ({trained} successful, {failed} failed)")
                        
                except Exception as e:
                    logger.error(
                        f"Training node falló | node_id={node_id} | "
                        f"progress={i}/{len(ordered_pending)} | error={str(e)[:100]}",
                        exc_info=True
                    )
                    failed += 1
                    # Ensure flags are cleared even on error
                    try:
                        kn = self.active_nodes.get(node_id)
                        if kn:
                            kn.is_training = False
                        self.repository.set_peft_training_status(node_id, 'FAILED')
                        self.repository.clear_training_flag(node_id)
                        # Update cache
                        with self._pending_cache_lock:
                            self._pending_training_cache.discard(node_id)
                        
                        # Fase 4: Evaluate quarantine increment
                        # Determine failure type (simplified: assume TRAINING_ERROR for now)
                        # TODO: Improve failure type detection based on exception
                        failure_type = "TRAINING_ERROR"
                        training_progress = None  # TODO: Track training progress
                        
                        if self.should_increment_quarantine(node_id, failure_type, training_progress):
                            try:
                                failure_reason = f"Training failure: {str(e)[:100]}"
                                new_count = self.repository.increment_quarantine_count(
                                    node_id, 
                                    reason=failure_reason,
                                    failure_type=failure_type
                                )
                                logger.warning(
                                    f"Nodo entra en cuarentena | node_id={node_id} | "
                                    f"quarantine_count={new_count} | reason={failure_reason} | "
                                    f"failure_type={failure_type} | phase=post_failure"
                                )
                            except Exception as quarantine_error:
                                logger.error(
                                    f"Error incrementando cuarentena | node_id={node_id} | "
                                    f"error={str(quarantine_error)[:100]}",
                                    exc_info=True
                                )
                        
                        # Sanitize orphans immediately after failure (Fase 2)
                        # This ensures nodes pointing to this failed node are cleaned up
                        sanitized = self.repository.sanitize_orphans()
                        if sanitized > 0:
                            logger.info(
                                f"🛡️  Sanitización inmediata | trigger_node={node_id} | "
                                f"affected={sanitized} | phase=post_failure"
                            )
                        
                        # Fase 8: Add children of failed node to re-evaluation queue
                        # Find all nodes that have this failed node as parent
                        try:
                            cursor = self.repository.conn.cursor()
                            cursor.execute(
                                "SELECT node_id FROM nodes WHERE status = 'KN' AND parent_node_id = ?",
                                (node_id,)
                            )
                            children = [row[0] for row in cursor.fetchall()]
                            cursor.close()
                            
                            for child_id in children:
                                self.add_to_re_evaluation_queue(child_id)
                                logger.debug(
                                    f"Agregando hijo {child_id} a cola de re-evaluación "
                                    f"(padre {node_id} falló)"
                                )
                        except Exception as queue_error:
                            logger.warning(
                                f" Error agregando hijos a cola de re-evaluación | "
                                f"parent_id={node_id} | error={str(queue_error)[:100]}",
                                exc_info=True
                            )
                    except Exception:
                        pass  # Don't fail on cleanup
                    continue
        
        except KeyboardInterrupt:
            logger.warning(
                f" Training interrumpido por usuario | "
                f"current_node={current_node_id if 'current_node_id' in locals() else 'unknown'} | "
                f"progress={current_node_index}/{len(ordered_pending)} | "
                f"trained={trained} | failed={failed}"
            )
            logger.warning(" Training interrupted by user. Cleaning up...")
            
            # Clean up state of current node (if any)
            current_node_id = ordered_pending[current_node_index-1] if current_node_index > 0 and current_node_index <= len(ordered_pending) else None
            if current_node_id:
                try:
                    self.repository.set_peft_training_status(current_node_id, None)
                    kn = self.active_nodes.get(current_node_id)
                    if kn:
                        kn.is_training = False
                    logger.info(f"Cleaned up state for interrupted node '{current_node_id}'")
                except Exception as e:
                    logger.warning(f"Failed to clean up interrupted node '{current_node_id}': {e}")
            
            # Return partial statistics
            return {
                "total_pending": len(ordered_pending),
                "trained": trained,
                "failed": failed,
                "interrupted": True
            }
        
        # CRITICAL: Recalculate actual pending nodes after training completes
        # Some nodes may have been marked as trained but flags not yet cleared
        # This ensures accurate reporting
        actual_pending = self.repository.get_pending_training_nodes()
        actual_pending_count = len(actual_pending)
        
        logger.info(
            f"Deferred training completado | total_pending={actual_pending_count} | "
            f"trained={trained} | failed={failed} | "
            f"success_rate={trained/len(ordered_pending)*100:.1f}% if all flags cleared"
        )
        logger.info(f"Deferred training completed:")
        logger.info(f"   - Trained: {trained} nodes")
        logger.info(f"   - Failed: {failed} nodes")
        logger.info(f"   - Total processed: {len(ordered_pending)} nodes")
        logger.info(f"   - Still pending: {actual_pending_count} nodes")
        
        # Print summary to console
        print(f"\n   Entrenamiento completado:")
        print(f"      - Nodos entrenados: {trained}")
        print(f"      - Nodos fallidos: {failed}")
        print(f"      - Total procesados: {len(ordered_pending)}")
        print(f"      - Pendientes restantes: {actual_pending_count}")
        sys.stdout.flush()
        
        # ========================================================================
        # Phase 3: Evaluation (2026-01-XX)
        # ========================================================================
        if trained > 0:
            logger.info(f"Starting evaluation for {trained} newly trained nodes...")
            print(f"\n   Iniciando evaluacion de calidad...", flush=True)
            
            # CRITICAL: Create a FRESH repository connection for evaluation
            # This bypasses any transaction isolation issues in the main connection
            # caused by long-running transactions during deferred training.
            from .repository import KNRepository
            from .evaluation import Evaluator
            
            try:
                # Create ephemeral repository connection
                eval_repo = KNRepository(self.repository.db_path)
                logger.debug("Created ephemeral DB connection for evaluation")
                
                # Create ephemeral evaluator reusing the transformer/tokenizer/device
                # to avoid reloading the model (heavy operation)
                ephemeral_evaluator = Evaluator(
                    repository=eval_repo,
                    transformer_base=self.evaluator.transformer,
                    data_manager=self.evaluator.data_manager
                )
                
                # Evaluate using the fresh connection
                ephemeral_evaluator.evaluate_all_adapters(
                    node_ids=ordered_pending,
                    progress_interval=progress_interval
                )
                
                # Close the ephemeral connection explicitly
                try:
                    eval_repo.conn.close()
                except:
                    pass
                logger.debug("Ephemeral evaluation connection closed")
                
            except Exception as e:
                logger.error(f"Error during evaluation phase: {e}", exc_info=True)
        
        return {
            "total_pending": actual_pending_count,
            "trained": trained,
            "failed": failed
        }
    
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

    def _handle_new_buffer(self, embedding: torch.Tensor, source_id: Optional[str] = None) -> str:
        """
        Helper to create buffer and check promotion.
        
        Phase 2: Also accepts source_id for training data provenance.
        
        Args:
            embedding: Embedding tensor
            source_id: Optional source ID (pointer to original dataset)
        
        Returns:
            Buffer ID or promoted KnowledgeNode ID
        """
        buffer_id = self.create_buffer(embedding, source_id=source_id)
        
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
                # Try to load from Repository first (preferred - has full state)
                kn = KnowledgeNode.from_repository(self.repository, node_id)
                self.active_nodes[node_id] = kn
            except ValueError:
                # Node doesn't exist in Repository - this can happen if:
                # 1. Node was just promoted but not yet committed (race condition)
                # 2. Node exists only in FilterBayesian memory but not in BD (inconsistency)
                # Try to verify if node exists in BD before creating new one
                signature_check = self.repository.get_signature(node_id)
                if signature_check:
                    # Node exists in BD but from_repository() failed - try loading again
                    try:
                        kn = KnowledgeNode.from_repository(self.repository, node_id)
                        self.active_nodes[node_id] = kn
                    except Exception as e:
                        logger.warning(f"Failed to load node '{node_id}' from Repository after signature check: {e}. Creating new node.")
                        kn = KnowledgeNode(node_id=node_id)
                        self.active_nodes[node_id] = kn
                else:
                    # Node truly doesn't exist - create new one
                    # This should only happen for newly promoted buffers
                    logger.debug(f"Node '{node_id}' not found in Repository, creating new node instance")
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
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # ========================================================================
        # Phase 2: Save Source ID Pointer (2025-01-XX)
        # ========================================================================
        # Save pointer to original dataset for training data provenance
        # Only save if source_id is provided and embedding was accepted
        # ========================================================================
        if source_id is not None:
            try:
                # Verify node exists in database before saving pointer
                # This handles cases where node was just promoted and may not be committed yet
                signature_check = self.repository.get_signature(node_id)
                if signature_check:
                    self.repository.save_pointer(node_id, source_id)
                    logger.debug(f"Saved pointer for node '{node_id}': source_id='{source_id}'")
                else:
                    # Node doesn't exist in DB yet - this can happen if node was just promoted
                    # Try to create signature first using current node state
                    try:
                        current_sig = kn.get_signature()
                        if current_sig and all(k in current_sig for k in ["node_id", "centroid", "mass", "variance"]):
                            # Check again if node was created by another thread/process (race condition)
                            signature_check_retry = self.repository.get_signature(node_id)
                            if signature_check_retry:
                                # Node now exists (was created between checks), save pointer
                                self.repository.save_pointer(node_id, source_id)
                                logger.debug(f"Node now exists, saved pointer for node '{node_id}': source_id='{source_id}'")
                            else:
                                # Node still doesn't exist, create it
                                # This should only happen for newly promoted buffers
                                try:
                                    self.repository.save_new_kn(
                                        node_id=current_sig["node_id"],
                                        centroid=current_sig["centroid"],
                                        mass=current_sig["mass"],
                                        variance=current_sig["variance"]
                                    )
                                    # Force commit to ensure node is available
                                    self.repository.conn.commit()
                                    # Now save pointer
                                    self.repository.save_pointer(node_id, source_id)
                                    logger.debug(f"Created signature and saved pointer for node '{node_id}': source_id='{source_id}'")
                                except Exception as create_error:
                                    # Check if it's an IntegrityError (node already exists - UNIQUE constraint violation)
                                    error_str = str(create_error).lower()
                                    if "unique constraint" in error_str or "integrity" in error_str or "already exists" in error_str:
                                        # Node was created by another thread/process between checks
                                        # This is expected in concurrent scenarios - just save the pointer
                                        try:
                                            self.repository.save_pointer(node_id, source_id)
                                            logger.debug(f"Node created concurrently, saved pointer for node '{node_id}': source_id='{source_id}'")
                                        except Exception as pointer_error:
                                            logger.debug(f"Failed to save pointer after concurrent node creation: {node_id}, error: {pointer_error}")
                                    else:
                                        logger.debug(f"Failed to create signature for node '{node_id}': {create_error}")
                        else:
                            logger.debug(f"Node '{node_id}' signature invalid, skipping pointer save")
                    except Exception as e2:
                        logger.debug(f"Failed to create signature for node '{node_id}': {e2}")
            except Exception as e:
                # Don't interrupt flow if pointer save fails
                logger.debug(f"Failed to save pointer for node '{node_id}': {e}")
        
        # Update Repository
        try:
            signature = kn.get_signature()
        except ValueError as e:
            # Node has no centroid (hasn't processed embeddings yet)
            # This shouldn't happen if process() was successful, but handle gracefully
            logger.warning(f"Node '{node_id}' has no signature: {e}, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Validate signature is not None
        if signature is None:
            logger.warning(f"Node '{node_id}' returned None signature, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Validate signature has required keys
        if not all(key in signature for key in ["node_id", "centroid", "mass", "variance"]):
            logger.warning(f"Node '{node_id}' signature missing required keys, redirecting to buffer")
            return self._handle_new_buffer(embedding, source_id=source_id)
        
        # Update node stats (will handle non-existent nodes internally)
        self.repository.update_node_stats(
            node_id=signature["node_id"],
            centroid=signature["centroid"],
            mass=signature["mass"],
            variance=signature["variance"]
        )
        
        # ========================================================================
        # CRITICAL: Update FilterBayesian signature immediately (2025-01-14)
        # ========================================================================
        # When a node accepts an embedding, its centroid moves immediately.
        # FilterBayesian must be updated immediately so the new centroid is used
        # for the next routing decisions. This allows the node to "attract" more
        # similar embeddings as its centroid evolves.
        # ========================================================================
        # Optimization: Use partial_update (O(1)) instead of refresh_signatures (O(N))
        self.filter.partial_update(
            node_id=signature["node_id"],
            new_centroid=signature["centroid"],
            new_mass=signature["mass"]
        )
        logger.debug(
            f"FilterBayesian signature updated incrementally for node '{node_id}': "
            f"mass={signature['mass']}, centroid updated"
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
        Check if node should be marked for training (deferred training).
        
        CHANGED: 2025-01-14 - Now only marks needs_training flag instead of starting training.
        This allows processing to continue without blocking.
        
        Phase 2: Deferred Training Trigger Logic
        
        Conditions for first training:
        1. Node mass >= TRAINING_THRESHOLD
        2. Node is not already trained
        3. Node is not already marked for training
        
        Conditions for re-training (Training Delta):
        1. Node is already trained
        2. Node mass >= last_training_mass * TRAINING_DELTA_MULTIPLIER (default: 2x)
        
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
            
            # Check if already marked for training (use cache to avoid DB query)
            # OPTIMIZATION: Use in-memory cache instead of expensive DB query
            with self._pending_cache_lock:
                if node_id in self._pending_training_cache:
                    return  # Already marked
            
            # ========================================================================
            # Case 1: First Training (threshold reached)
            # ========================================================================
            if mass >= TRAINING_THRESHOLD and not self.repository.is_trained(node_id):
                # Verify node exists before marking (optimization: avoid DB call if node doesn't exist)
                # Note: mark_for_training() will handle non-existent nodes, but we can avoid
                # the DB call by checking cache first. If node was deleted, it won't be in cache.
                try:
                    # Mark for training (will handle non-existent nodes internally)
                    self.repository.mark_for_training(node_id, is_retraining=False)
                    # Update cache if marking was successful (no exception means success)
                    with self._pending_cache_lock:
                        self._pending_training_cache.add(node_id)
                    logger.debug(
                        f"Node '{node_id}' marked for training (mass={mass} >= {TRAINING_THRESHOLD})"
                    )
                except Exception:
                    # Node doesn't exist or other error - don't update cache
                    pass
                return
            
            # ========================================================================
            # Case 2: Re-training (Training Delta - mass doubling)
            # ========================================================================
            if TRAINING_DELTA_MULTIPLIER > 0 and self.repository.is_trained(node_id):
                last_training_mass = self.repository.get_last_training_mass(node_id)
                if last_training_mass > 0 and mass >= last_training_mass * TRAINING_DELTA_MULTIPLIER:
                    # Mark for re-training (will handle non-existent nodes internally)
                    try:
                        self.repository.mark_for_training(node_id, is_retraining=True)
                        # Update cache if marking was successful
                        with self._pending_cache_lock:
                            self._pending_training_cache.add(node_id)
                        logger.debug(
                            f"Node '{node_id}' marked for re-training "
                            f"(mass={mass} >= {last_training_mass * TRAINING_DELTA_MULTIPLIER})"
                        )
                    except Exception:
                        # Node doesn't exist or other error - don't update cache
                        pass
                    return
            
            # ========================================================================
            # Case 3: Timeout-based re-training (optional)
            # ========================================================================
            # TODO: Implement timeout-based re-training if TRAINING_DELTA_TIMEOUT_DAYS > 0
            # This would require storing last_training_timestamp in DB
            
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
    
    def _async_train_kn(self, node_id: str, epoch_callback: Optional[Callable[[int, int], None]] = None) -> None:
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
        # ========================================================================
        # CRITICAL: Thread Safety for SQLite (2025-01-14)
        # ========================================================================
        # We MUST create a fresh Repository instance for this thread.
        # Sharing self.repository (and its self.conn) across threads causes
        # "database is locked" errors and potential hangs/crashes because
        # sqlite3 connection objects are not thread-safe for concurrent cursor usage.
        # ========================================================================
        thread_repo = KNRepository(self.repository.db_path)
        
        try:
            logger.debug(f"Starting training for node '{node_id}'")
            
            # 1. Activate training buffer (embeddings that arrive during training will be buffered)
            thread_repo.set_peft_training_status(node_id, 'TRAINING')
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = True
                logger.debug(f"Training buffer activated for node '{node_id}'")
            
            # 2. RE-FETCH: Get ALL current source_ids (not just those at threshold time)
            # This ensures we train with the most up-to-date data
            source_ids = thread_repo.get_training_pointers(node_id)
            
            if not source_ids:
                logger.warning(f" No source_ids found for node '{node_id}', cannot train")
                # Clean up
                if kn:
                    kn.is_training = False
                thread_repo.set_peft_training_status(node_id, None)
                thread_repo.clear_training_flag(node_id)
                # Update cache
                with self._pending_cache_lock:
                    self._pending_training_cache.discard(node_id)
                # Return without marking as trained
                return
            
            logger.debug(f"Re-fetched {len(source_ids)} source_ids for node '{node_id}' (current state)")
            
            # 3. Translate source_ids to texts using DataManager
            # DataManager generally reads from creating files or separate DB, verify safety
            # Assuming DataManager is thread-safe (read-only usually) or handles its own locking
            texts = self.data_manager.get_texts_from_pointers(source_ids)
            
            if not texts:
                logger.warning(f" No texts found for node '{node_id}', cannot train")
                # Clean up
                if kn:
                    kn.is_training = False
                thread_repo.set_peft_training_status(node_id, None)
                thread_repo.clear_training_flag(node_id)
                # Update cache
                with self._pending_cache_lock:
                    self._pending_training_cache.discard(node_id)
                # Return without marking as trained
                return
            
            logger.debug(f"Retrieved {len(texts)} texts for node '{node_id}' (re-fetched at training time)")
            
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
                thread_repo.set_peft_training_status(node_id, None)
                return
            
            # ========================================================================
            # Get estimated_epochs from repository (if available) (2026-01-14)
            # ========================================================================
            # Use calculated estimated_epochs if available, otherwise use default (3)
            # This enables adaptive training based on node complexity
            # ========================================================================
            estimated_epochs = thread_repo.get_estimated_epochs(node_id)
            epochs = estimated_epochs if estimated_epochs is not None else 3
            
            if estimated_epochs is not None:
                logger.debug(
                    f"Node '{node_id}' will train with {epochs} epochs "
                    f"(estimated from complexity analysis)"
                )
            else:
                logger.debug(
                    f"Node '{node_id}' will train with {epochs} epochs "
                    f"(default, estimated_epochs not set)"
                )
            
            # ========================================================================
            # Phase 3: Inheritance Setup (2025-01-XX)
            # ========================================================================
            parent_state_dict = None
            parent_similarity = None
            
            try:
                # Get current signature to check for parent
                current_sig = thread_repo.get_signature(node_id)
                if current_sig and current_sig.get('parent_node_id'):
                    parent_node_id = current_sig['parent_node_id']
                    parent_similarity = current_sig.get('parent_similarity')
                    
                    # ========================================================================
                    # VALIDACIÓN CRÍTICA: Detectar bucles infinitos
                    # ========================================================================
                    if parent_node_id == node_id:
                        logger.error(
                            f" BUCLE DETECTADO: Node '{node_id}' es su propio padre. "
                            f"Ignorando herencia. Esto indica corrupción de datos."
                        )
                        parent_state_dict = None
                        parent_similarity = None
                    else:
                        # ========================================================================
                        # VALIDACIÓN CRÍTICA: Verificar que el padre esté entrenado
                        # ========================================================================
                        if not thread_repo.is_trained(parent_node_id):
                            logger.warning(
                                f" Parent '{parent_node_id}' is not trained yet for node '{node_id}'. "
                                f"Skipping inheritance. Node will train without inheritance."
                            )
                            parent_state_dict = None
                            parent_similarity = None
                        else:
                            # Padre está entrenado, cargar pesos
                            parent_weights = thread_repo.get_lora_weights(parent_node_id)
                            
                            if parent_weights:
                                # Extract state_dict from new format (dict with 'state_dict' key) or use directly
                                if isinstance(parent_weights, dict) and 'state_dict' in parent_weights:
                                    parent_state_dict = parent_weights['state_dict']
                                else:
                                    # Backward compatibility: treat as direct state_dict
                                    parent_state_dict = parent_weights
                                
                                logger.debug(
                                    f"Loaded parent weights from '{parent_node_id}' for inheritance "
                                    f"(similarity: {parent_similarity:.3f})"
                                )
                            else:
                                logger.warning(
                                    f" Parent '{parent_node_id}' marked as trained but has no weights. "
                                    f"Skipping inheritance for '{node_id}'."
                                )
                                parent_state_dict = None
                                parent_similarity = None
                else:
                    # No parent, normal training
                    parent_state_dict = None
                    parent_similarity = None
                        
            except Exception as e:
                logger.warning(f"Error preparing inheritance for '{node_id}': {e}")
                # Continue without inheritance
                parent_state_dict = None
                parent_similarity = None

            # ========================================================================
            # CRITICAL: Thread Safety for TransformerBase Singleton (2025-01-14)
            # ========================================================================
            # TransformerBase singleton is NOT thread-safe for concurrent training.
            # We MUST acquire the lock before training to prevent race conditions.
            # This ensures only one training task modifies the shared base model at a time.
            # ========================================================================
            with self._training_lock:
                training_result = self.transformer.train_kn_adapter(
                    node_id, 
                    texts,
                    epochs=epochs,  # Use calculated epochs
                    batch_size=TRAINING_BATCH_SIZE,  # Use configurable batch size (optimized for GPU)
                    epoch_callback=epoch_callback,  # Pass callback for epoch updates
                    parent_state_dict=parent_state_dict,  # Soft Inheritance
                    parent_similarity=parent_similarity   # Dynamic L2 Lambda
                )
            
            # Handle new format: training_result can be dict with 'weights_bytes' and 'config' or just weights_bytes
            if training_result is None:
                if parent_state_dict is not None:
                    # FALLBACK TO ROOT TRAINING (2025-01-27)
                    # If inheritance failed (Life Insurance or divergent weights), retry as clean ROOT
                    logger.warning(f"Inheritance failed for node '{node_id}'. Retrying as ROOT training.")
                    with self._training_lock:
                        training_result = self.transformer.train_kn_adapter(
                            node_id, 
                            texts,
                            epochs=epochs,
                            batch_size=32, # TRAINING_BATCH_SIZE hardcoded for safety in script context
                            epoch_callback=epoch_callback,
                            parent_state_dict=None, # Clean retry
                            parent_similarity=None
                        )
                
                # Check again if still None after possible retry
                if training_result is None:
                    # Final training failed
                    logger.error(f"Training failed for node '{node_id}' (even after possible ROOT retry). Marking as FAILED.")
                    kn = self.active_nodes.get(node_id)
                    if kn:
                        kn.is_training = False
                    
                    try:
                        thread_repo.mark_as_failed(node_id)
                    except Exception as e:
                        logger.error(f"Failed to mark node {node_id} as failed in DB: {e}")
                    
                    return
            
            # Extract weights_bytes and config from result
            if isinstance(training_result, dict) and 'weights_bytes' in training_result:
                weights_bytes = training_result['weights_bytes']
                peft_config = training_result.get('config')
            else:
                # Backward compatibility: treat as direct weights_bytes
                weights_bytes = training_result
                peft_config = None
            
            if weights_bytes is None:
                logger.debug(f"Training returned None weights for node '{node_id}'")
                # Mark as failed
                kn = self.active_nodes.get(node_id)
                if kn:
                    kn.is_training = False
                thread_repo.set_peft_training_status(node_id, 'FAILED')
                
                # Check for FALLBACK condition (failed inheritance)
                if parent_state_dict:
                    # We attempted inheritance but failed (likely Life Insurance or training error)
                    # Update inheritance_type to FALLBACK
                    try:
                        thread_repo.update_inheritance_type(node_id, "FALLBACK")
                        logger.warning(
                            f"Node '{node_id}' training failed during inheritance. "
                            f"Downgraded type to 'FALLBACK'."
                        )
                    except Exception as e:
                        logger.error(f"Failed to update fallback type for {node_id}: {e}")

                return
            
            # 4. Save weights to Repository (with config if available)
            thread_repo.save_peft_weights(
                node_id,
                weights_bytes,
                format="safetensors",
                config=peft_config  # Pass saved config to ensure correct loading later
            )
            logger.debug(f"PEFT weights saved for node '{node_id}'")
            
            # 5. Mark as trained and update last_training_mass
            thread_repo.mark_as_trained(node_id)
            thread_repo.set_peft_training_status(node_id, 'COMPLETED')
            
            # Update last_training_mass for Training Delta (re-training when mass doubles)
            current_signature = thread_repo.get_signature(node_id)
            if current_signature:
                current_mass = current_signature.get('mass', 0)
                thread_repo.update_last_training_mass(node_id, current_mass)
                logger.debug(f"Node '{node_id}' marked as trained (mass={current_mass})")
            else:
                logger.debug(f"Node '{node_id}' marked as trained")
            
            # Clear needs_training flag
            thread_repo.clear_training_flag(node_id)
            # Update cache
            with self._pending_cache_lock:
                self._pending_training_cache.discard(node_id)
            
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
                                thread_repo.save_pointer(node_id, src_id)
                            except Exception:
                                pass  # Don't interrupt buffer processing
                    except Exception as e:
                        logger.debug(f"Error processing buffered embedding {i} for node '{node_id}': {e}")
                
                # Update statistics after processing buffer
                try:
                    signature = kn.get_signature()
                    if signature is None or not all(key in signature for key in ["node_id", "centroid", "mass", "variance"]):
                        logger.warning(f"Invalid signature for node '{node_id}' after buffer processing, skipping update")
                    else:
                        thread_repo.update_node_stats(
                            node_id=signature["node_id"],
                            centroid=signature["centroid"],
                            mass=signature["mass"],
                            variance=signature["variance"]
                        )
                except ValueError as e:
                    logger.warning(f"Error getting signature for node '{node_id}' after buffer processing: {e}")
                
                # Check if new threshold reached (recursive training trigger)
                # Note: recursive trigger uses the MAIN loop logic (check_and_trigger), 
                # but calls queue_training_task -> submits new future.
                # Since we are in a thread, calling self._check_and_trigger_training(node_id, signature)
                # is safe IF it only reads/submits. 
                # But _check_and_trigger uses self.repository which is the SHARED one.
                # However, it only does READS (is_trained) which *might* be ok, 
                # but ideally we should avoid self.repository in thread.
                # For safety, let's skip recursive trigger for now or replicate logic.
                # Or just let the next update trigger it.
                
                # Clear buffer
                kn.training_buffer.clear()
                kn.training_buffer_source_ids.clear()
                logger.debug(f"Training buffer cleared for node '{node_id}'")
            
            # 7. Clear training flag
            if kn:
                kn.is_training = False
                
        except KeyboardInterrupt:
            # Interruption by user (Ctrl+C)
            logger.warning(f"Training interrupted (KeyboardInterrupt) for node '{node_id}'")
            try:
                # Clear training status - node will be re-trained on next run
                thread_repo.set_peft_training_status(node_id, None)
                # DO NOT clear needs_training - we want it to re-train
            except Exception:
                pass  # Don't fail on cleanup
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = False
            raise  # Re-raise to propagate interruption
        
        except Exception as e:
            logger.error(f"Error in _async_train_kn for node '{node_id}': {e}", exc_info=True)
            # Ensure flag is cleared
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = False
                # Clear buffer on error (embeddings will be reprocessed on next routing)
                kn.training_buffer.clear()
                if hasattr(kn, 'training_buffer_source_ids'):
                    kn.training_buffer_source_ids.clear()
            
            # Mark as failed in thread-local repository
            try:
                thread_repo.set_peft_training_status(node_id, 'FAILED')
            except Exception:
                pass  # Don't fail on status update
        finally:
            # CRITICAL: Always ensure state is cleaned up
            kn = self.active_nodes.get(node_id)
            if kn:
                kn.is_training = False
            
            # If still in TRAINING status, clean it up (interruption or error)
            try:
                current_status = thread_repo.get_peft_training_status(node_id)
                if current_status == 'TRAINING':
                    # Check if node has weights
                    if not thread_repo.is_trained(node_id):
                        # No weights - clear status for re-training
                        thread_repo.set_peft_training_status(node_id, None)
                        logger.debug(f"Cleaned TRAINING status for node '{node_id}' (no weights, will re-train)")
                    else:
                        # Has weights but status incorrect - mark as COMPLETED
                        thread_repo.set_peft_training_status(node_id, 'COMPLETED')
                        thread_repo.clear_training_flag(node_id)
                        # Update cache
                        with self._pending_cache_lock:
                            self._pending_training_cache.discard(node_id)
                        logger.debug(f"Fixed TRAINING status for node '{node_id}' (has weights → COMPLETED)")
            except Exception:
                pass  # Don't fail on cleanup
            # CRITICAL: Close the thread-local repository connection
            thread_repo.conn.close()
            
    # ========================================================================
    # BLOCK PROCESSING OPTIMIZATION (2025-01-25)
    # ========================================================================
    # PROBLEM: process_batch() called individually causes DB overhead
    # SOLUTION: process_all_batches() handles all batches with caching
    # BENEFITS: Pre-load weights, cache stats, batch commits
    # ========================================================================
    
    def process_all_batches(self, all_batches, pbar):
        """Process all batches as optimized block.
        
        Optimizations:
        1. Pre-load all LoRA weights to avoid repeated DB calls
        2. Cache stats updates every 10 batches instead of every batch
        3. Batch database commits at the end
        4. Optimized progress updates
        """
        total_batches = len(all_batches)
        processed_embeddings = 0
        
        # Pre-load all unique node weights to minimize DB calls
        unique_nodes = set()
        for batch_idx, batch_embeddings, batch_source_ids in all_batches:
            # Quick routing to identify nodes (without full processing)
            routing_decisions = self.filter.route_batch(batch_embeddings)
            for node_id, _ in routing_decisions:
                if node_id != "NEW_BUFFER":
                    unique_nodes.add(node_id)
        
        # Pre-load all LoRA weights
        cached_weights = {}
        for node_id in unique_nodes:
            cached_weights[node_id] = self.repository.get_lora_weights(node_id)
        
        # Process batches with optimizations
        for batch_idx, batch_embeddings, batch_source_ids in all_batches:
            # Use cached weights instead of DB calls
            routing_decisions = self.filter.route_batch(batch_embeddings)
            node_groups: Dict[str, List[int]] = {}
            for i, (node_id, _) in enumerate(routing_decisions):
                if node_id not in node_groups:
                    node_groups[node_id] = []
                node_groups[node_id].append(i)
            
            final_results = [None] * len(batch_embeddings)
            
            # Process each node group with cached weights
            for node_id, indices in node_groups.items():
                if node_id == "NEW_BUFFER":
                    # Handle buffers (fast path)
                    for i, idx in enumerate(indices):
                        src_id = batch_source_ids[i] if i < len(batch_source_ids) else None
                        res = self._handle_new_buffer(batch_embeddings[i], source_id=src_id)
                        final_results[idx] = res
                        # Update progress per embedding
                        processed_embeddings += 1
                        pbar.update(1)
                    pbar.refresh()
                    continue
                
                # Use cached weights (no DB call) - with fallback
                lora_weights = cached_weights.get(node_id)
                if lora_weights is None:
                    # Fallback: load weights if not in cache
                    lora_weights = self.repository.get_lora_weights(node_id)
                    cached_weights[node_id] = lora_weights
                
                group_embeddings = batch_embeddings[indices]
                
                # Process with transformer (same logic as original)
                trans_results = self.transformer.forward_batch_with_node(
                    group_embeddings, 
                    lora_weights
                )
                
                # Apply feedback (same logic as original)
                for i, idx in enumerate(indices):
                    emb = group_embeddings[i]
                    src_id = batch_source_ids[i]
                    res = trans_results[i]
                    
                    feedback = self.post_processor.evaluate(res)
                    if feedback is None:
                        final_results[idx] = self._handle_new_buffer(emb, source_id=src_id)
                    elif feedback.status == "OK":
                        self.repository.apply_feedback(node_id, feedback)
                        # Layer 1 Update (feedback already applied above)
                        self._process_kn_update(node_id, emb, src_id)
                    else:
                        final_results[idx] = self._handle_new_buffer(emb, source_id=src_id)
                    
                    # Update progress per embedding (not per batch) for real-time feedback
                    processed_embeddings += 1
                    pbar.update(1)
                
                # Refresh after each node group for visual update
                pbar.refresh()
            
            # Update stats every 10 batches (not every batch)
            if (batch_idx + 1) % 10 == 0:
                counts = self.get_counts()
                # Fix: Ensure all values are strings, not node_ids
                postfix_dict = {
                    "KNs": str(counts["kn_count"]),
                    "Buf": str(counts["buffer_count"])
                }
                if self.has_active_training():
                    postfix_dict["T"] = "ON"
                pbar.set_postfix(postfix_dict, refresh=False)
        
        # Final stats update
        counts = self.get_counts()
        # Fix: Ensure all values are strings, not node_ids
        postfix_dict = {
            "KNs": str(counts["kn_count"]),
            "Buf": str(counts["buffer_count"])
        }
        if self.has_active_training():
            postfix_dict["T"] = "ON"
        pbar.set_postfix(postfix_dict, refresh=True)
        
        logger.info(f"Block processing completed: {processed_embeddings} embeddings processed")