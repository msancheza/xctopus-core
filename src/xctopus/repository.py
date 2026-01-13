"""
Repository for Clustering Layer.

Bridge to SQLite for storing statistical signatures and buffers.
Stores FP16 tensors as BLOBs for efficiency.
"""

import sqlite3
import io
import functools
from collections import OrderedDict
import numpy as np
import torch
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .settings import (
    DB_PATH,
    SAVE_BATCH_SIZE,
    DTYPE,
    DEVICE,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


class KNRepository:
    # LRU cache for LoRA weights (node_id -> state_dict)
    _lora_cache = functools.lru_cache(maxsize=128)(lambda self, node_id: self._load_lora_weights(node_id))
    """
    Repository for Knowledge Nodes.
    
    Manages:
    - Statistical signatures (centroids, mass, variance)
    - Physical memory of embeddings
    - Temporary buffers
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Repository.
        
        Args:
            db_path: Path to SQLite database (default: from settings)
        """
        self.db_path = db_path or DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # To access columns by name
        self._update_counter = 0  # For batch commits
        
        self._create_tables()
        logger.info(f"Repository initialized: {self.db_path}")

    def _create_tables(self) -> None:
        """Create tables necessary for Clustering Layer."""
        cursor = self.conn.cursor()
        
        # ========================================================================
        # SQLite Optimizations for High Performance
        # ========================================================================
        # WAL (Write-Ahead Logging): Allows concurrent reads while writing
        # synchronous=NORMAL: Balance between security and performance
        # cache_size=-64000: 64MB cache in RAM (improves query performance)
        # ========================================================================
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=-64000')  # 64MB cache in RAM
        logger.debug("WAL mode and SQLite optimizations activated")
        
        # 1. Signatures Table (The Shells for Clustering)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                centroid BLOB NOT NULL,
                mass INTEGER NOT NULL DEFAULT 0,
                variance REAL,
                status TEXT DEFAULT 'KN',
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. Physical Memory Table (FP16 embedding storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS node_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            )
        ''')
        
        # 3. Buffers Table (Temporary territories)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS buffers (
                buffer_id TEXT PRIMARY KEY,
                size INTEGER NOT NULL DEFAULT 0,
                centroid BLOB,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Add centroid column if it doesn't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE buffers ADD COLUMN centroid BLOB')
            logger.debug("Centroid column added to buffers table (migration)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass
        
        # 4. Buffer embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS buffer_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                buffer_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(buffer_id) REFERENCES buffers(buffer_id) ON DELETE CASCADE
            )
        ''')
        
        # Indexes to optimize queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_id ON nodes(node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_memory_node_id ON node_memory(node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_buffer_embeddings_buffer_id ON buffer_embeddings(buffer_id)')
        
        # ========================================================================
        # PEFT/LoRA Weights Support (2025-12-27)
        # ========================================================================
        # Migration: Add PEFT columns if they don't exist (for existing databases)
        # These columns are optional and can be NULL for Layer 1 nodes
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_weights BLOB')
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_format TEXT')
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_size INTEGER')
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_checksum TEXT')
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_config TEXT')
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_trained_timestamp TIMESTAMP')
            logger.debug("PEFT columns added to nodes table (migration 2025-12-27)")
        except sqlite3.OperationalError:
            # Columns already exist, not an error
            pass
        
        # ========================================================================
        # PEFT Training Status Support (2025-12-27)
        # ========================================================================
        # Migration: Add training status column for Layer 2
        # Values: 'TRAINING', 'COMPLETED', 'FAILED', or NULL (not training)
        # Purpose: Allow Layer 2 to know if a node is currently being trained
        # This enables intelligent fallback decisions (use Transformer base vs wait)
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN peft_training_status TEXT')
            logger.debug("peft_training_status column added (migration 2025-12-27)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass
        
        # ========================================================================
        # Phase 2: Training Support - Pointers and Training Status (2025-01-XX)
        # ========================================================================
        # Migration: Add is_trained column to track if node has trained adapter
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN is_trained INTEGER DEFAULT 0')
            logger.debug("is_trained column added (migration Phase 2)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass
        
        # 5. Node Data Mapping Table (Pointers to original dataset)
        # Purpose: Associate node_id with source_id (pointer to original dataset)
        # This enables training by retrieving original texts from datasets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS node_data_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                UNIQUE(node_id, source_id)
            )
        ''')
        
        # Indexes for node_data_mapping
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_data_mapping_node_id ON node_data_mapping(node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_data_mapping_source_id ON node_data_mapping(source_id)')
        
        self.conn.commit()
        logger.debug("Tables created/initialized correctly")

    # --- FP16 Tensor Management ---

    def _tensor_to_blob(self, tensor: torch.Tensor) -> bytes:
        """
        Convert a tensor to bytes in FP16 for storage.
        
        Uses numpy.tobytes() which is more efficient than torch.save() for 1D tensors.
        
        Args:
            tensor: PyTorch tensor (must be on CPU)
        
        Returns:
            bytes: Tensor serialized as BLOB
        
        Raises:
            ValueError: If tensor is empty or has incorrect shape
        """
        # Validation
        if tensor is None:
            raise ValueError("Tensor cannot be None")
        
        if tensor.numel() == 0:
            raise ValueError("Tensor cannot be empty")
        
        # Ensure it's on CPU and is float16
        tensor_np = tensor.detach().cpu().to(dtype=torch.float16).numpy()
        return tensor_np.tobytes()

    def _blob_to_tensor(self, blob: bytes, shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Convert bytes from DB to a PyTorch tensor.
        
        Args:
            blob: BLOB bytes
            shape: Tensor shape (if None, assumes 1D with EMBEDDING_DIM)
        
        Returns:
            Tensor on the DEVICE specified in settings
        """
        array = np.frombuffer(blob, dtype=np.float16)
        if shape:
            array = array.reshape(shape)
        else:
            # If shape not specified, assume 1D with EMBEDDING_DIM
            array = array.reshape((EMBEDDING_DIM,))
        
        # Make copy of array to make it writable (avoids PyTorch warning)
        # np.frombuffer() creates a read-only array
        array = array.copy()
        
        tensor = torch.from_numpy(array).to(dtype=DTYPE)
        return tensor.to(device=DEVICE)

    def _maybe_commit(self) -> None:
        """Periodic commit based on SAVE_BATCH_SIZE."""
        self._update_counter += 1
        if self._update_counter >= SAVE_BATCH_SIZE:
            self.conn.commit()
            self._update_counter = 0
            logger.debug(f"Batch commit performed (every {SAVE_BATCH_SIZE} updates)")

    # --- Clustering Operations (Signatures) ---

    def get_all_signatures(self) -> List[Dict[str, Any]]:
        """
        Retrieve all centroids and statistics for FilterBayesian.
        
        Returns:
            List of signatures with centroids as tensors on DEVICE
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT node_id, centroid, mass, variance FROM nodes WHERE status = 'KN'")
        rows = cursor.fetchall()
        
        signatures = []
        for row in rows:
            centroid = self._blob_to_tensor(row["centroid"], shape=(EMBEDDING_DIM,))
            
            signatures.append({
                "node_id": row["node_id"],
                "centroid": centroid,
                "mass": row["mass"],
                "variance": row["variance"]
            })
        
        logger.debug(f"Retrieved {len(signatures)} signatures from Repository")
        return signatures

    def get_signature(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the signature of a specific node.
        
        Args:
            node_id: Node ID
        
        Returns:
            Node signature or None if it doesn't exist
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT node_id, centroid, mass, variance FROM nodes WHERE node_id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        
        if row:
            centroid = self._blob_to_tensor(row["centroid"], shape=(EMBEDDING_DIM,))
            return {
                "node_id": row["node_id"],
                "centroid": centroid,
                "mass": row["mass"],
                "variance": row["variance"]
            }
        return None

    def has_peft_weights(self, node_id: str) -> bool:
        """
        Check if a Knowledge Node has PEFT/LoRA weights trained.
        
        Added: 2025-12-27
        Purpose: Allow Layer 2 to check if a KN needs training or already has weights.
        
        Args:
            node_id: Node ID to check
        
        Returns:
            True if PEFT weights exist, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT peft_weights FROM nodes WHERE node_id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        
        if row and row["peft_weights"] is not None:
            return True
        return False

    def save_new_kn(self, node_id: str, centroid: torch.Tensor, mass: int, variance: float) -> None:
        """
        Register a new KnowledgeNode after buffer promotion.
        
        Args:
            node_id: Unique node ID
            centroid: FP16 tensor of the centroid
            mass: Number of embeddings
            variance: Scalar variance
        
        Raises:
            ValueError: If parameters are invalid
            sqlite3.Error: If there's a database error
        """
        # Validation
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, received: {node_id}")
        
        if mass < 0:
            raise ValueError(f"mass must be >= 0, received: {mass}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO nodes (node_id, centroid, mass, variance, status) VALUES (?, ?, ?, ?, 'KN')",
                (node_id, self._tensor_to_blob(centroid), mass, variance)
            )
            self._maybe_commit()
            logger.info(f"New KN created: {node_id} (mass={mass}, variance={variance:.4f})")
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error creating KN {node_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating KN {node_id}: {e}")
            raise

    def update_node_stats(self, node_id: str, centroid: torch.Tensor, mass: int, variance: float) -> None:
        """
        Update the statistical signature of an existing node.
        
        Args:
            node_id: Node ID
            centroid: FP16 tensor of the updated centroid
            mass: New mass
            variance: New variance
        
        Raises:
            ValueError: If parameters are invalid
            sqlite3.Error: If there's a database error
        """
        # Validation
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, received: {node_id}")
        
        if mass < 0:
            raise ValueError(f"mass must be >= 0, received: {mass}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE nodes SET centroid = ?, mass = ?, variance = ?, last_updated = CURRENT_TIMESTAMP WHERE node_id = ?",
                (self._tensor_to_blob(centroid), mass, variance, node_id)
            )
            
            if cursor.rowcount == 0:
                logger.warning(f"Attempt to update non-existent node: {node_id}")
                return
            
            self._maybe_commit()
            logger.debug(f"Signature updated: {node_id} (mass={mass}, variance={variance:.4f})")
        except sqlite3.Error as e:
            logger.error(f"SQLite error updating KN {node_id}: {e}")
            raise

    def save_peft_weights(
        self,
        node_id: str,
        weights_bytes: bytes,
        format: str = "safetensors",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save PEFT/LoRA weights for an existing Knowledge Node.
        
        Added: 2025-12-27
        Purpose: Store trained adapter weights linked to a KN signature.
        
        Args:
            node_id: Node ID (must exist with signature)
            weights_bytes: Complete serialized file (.safetensors or .bin) as bytes
            format: 'safetensors' or 'bin' (default: 'safetensors')
            config: Optional dict with PEFT configuration (r, alpha, target_modules, etc.)
        
        Raises:
            ValueError: If node doesn't exist, invalid format, or empty weights
            sqlite3.Error: If there's a database error
        """
        # Validation: Node must exist
        signature = self.get_signature(node_id)
        if not signature:
            raise ValueError(f"KN {node_id} does not exist. Create signature first.")

        # Validation: weights_bytes must not be None
        if weights_bytes is None:
            raise ValueError("weights_bytes cannot be None")

        # Validation: Format
        if format not in ['safetensors', 'bin']:
            raise ValueError(f"Invalid format: {format}. Must be 'safetensors' or 'bin'")

        # Validation: Non-empty weights
        if not isinstance(weights_bytes, bytes):
            raise ValueError(f"weights_bytes must be bytes, got {type(weights_bytes)}")
        if len(weights_bytes) == 0:
            raise ValueError("weights_bytes cannot be empty")

        # Validation: Size limit (optional safety check)
        MAX_PEFT_SIZE = 50 * 1024 * 1024  # 50 MB
        weights_size = len(weights_bytes)
        if weights_size > MAX_PEFT_SIZE:
            raise ValueError(f"PEFT weights too large: {weights_size} bytes (max: {MAX_PEFT_SIZE})")

        # Calculate checksum
        checksum = hashlib.sha256(weights_bytes).hexdigest()

        # Serialize config to JSON string
        config_json = json.dumps(config) if config else None

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """UPDATE nodes SET 
                    peft_weights = ?,
                    peft_format = ?,
                    peft_size = ?,
                    peft_checksum = ?,
                    peft_config = ?,
                    peft_trained_timestamp = CURRENT_TIMESTAMP
                WHERE node_id = ?""",
                (weights_bytes, format, weights_size, checksum, config_json, node_id)
            )
            if cursor.rowcount == 0:
                logger.warning(f"Attempt to save PEFT weights for non-existent node: {node_id}")
                return
            self._maybe_commit()
            logger.debug(f"PEFT weights saved for {node_id} (format={format}, size={weights_size} bytes)")
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving PEFT weights for {node_id}: {e}")
            raise

    def _load_lora_weights(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Internal helper to load LoRA weights from DB and return a state_dict.
        Returns None if no weights are stored for the node.
        """
        peft = self.get_peft_weights(node_id)
        if not peft:
            return None
        buffer = io.BytesIO(peft["weights_bytes"])
        state_dict = torch.load(buffer, map_location=DEVICE)
        return state_dict

    def get_lora_weights(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Public method that returns the LoRA state_dict for a node, cached via LRU.
        Uses the internal _load_lora_weights function.
        """
        return self._lora_cache(node_id)

    def get_peft_weights(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve PEFT/LoRA weights for a Knowledge Node.
        Returns a dict with 'weights_bytes' and metadata, or None if not present.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT peft_weights, peft_format, peft_size, peft_checksum, 
                      peft_config, peft_trained_timestamp
               FROM nodes WHERE node_id = ?""",
            (node_id,)
        )
        row = cursor.fetchone()
        if not row or row["peft_weights"] is None:
            return None
            
        # Extract values
        weights_bytes = row["peft_weights"]
        format_type = row["peft_format"]
        size = row["peft_size"]
        stored_checksum = row["peft_checksum"]
        config_json = row["peft_config"]
        trained_timestamp = row["peft_trained_timestamp"]

        if stored_checksum is None:
            logger.error(f"peft_checksum is NULL for {node_id}, data may be corrupted")
            raise ValueError(f"PEFT weights data corrupted for {node_id}: checksum is NULL")
        
        if format_type is None:
            logger.error(f"peft_format is NULL for {node_id}, data may be corrupted")
            raise ValueError(f"PEFT weights data corrupted for {node_id}: format is NULL")
        
        # Validation: Ensure size is a valid integer
        if not isinstance(size, int) or size < 0:
            logger.error(f"Invalid size value for {node_id}: {size} (type: {type(size)})")
            raise ValueError(f"PEFT weights data corrupted for {node_id}: invalid size value")
        
        # Verify checksum (integrity check)
        calculated_checksum = hashlib.sha256(weights_bytes).hexdigest()
        if calculated_checksum != stored_checksum:
            logger.error(f"Checksum mismatch for {node_id}: stored={stored_checksum}, calculated={calculated_checksum}")
            raise ValueError(f"PEFT weights corrupted for {node_id}: checksum mismatch")
        
        # Parse config JSON
        config = None
        if config_json:
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON config for {node_id}, returning None")
                config = None
        
        return {
            'weights_bytes': weights_bytes,
            'format': format_type,
            'size': size,
            'checksum': stored_checksum,
            'config': config,
            'trained_timestamp': trained_timestamp
        }

    def delete_peft_weights(self, node_id: str) -> None:
        """
        Delete PEFT/LoRA weights for a Knowledge Node, keeping the signature intact.
        
        Added: 2025-12-27
        Purpose: Allow re-training from scratch or removing weights without deleting the KN.
        
        Args:
            node_id: Node ID
        
        Note:
            This only removes PEFT weights. The signature (centroid, mass, variance) remains.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE nodes SET 
                peft_weights = NULL,
                peft_format = NULL,
                peft_size = NULL,
                peft_checksum = NULL,
                peft_config = NULL,
                peft_trained_timestamp = NULL
            WHERE node_id = ?""",
            (node_id,)
        )
        
        if cursor.rowcount == 0:
            logger.warning(f"Attempt to delete PEFT weights for non-existent node: {node_id}")
            return
        
        self._maybe_commit()
        logger.debug(f"PEFT weights deleted for {node_id} (signature preserved)")

    def set_peft_training_status(self, node_id: str, status: Optional[str]) -> None:
        """
        Set PEFT training status for a Knowledge Node.
        
        Added: 2025-12-27
        Purpose: Track training state to enable intelligent fallback decisions in Layer 2.
        
        Args:
            node_id: Node ID
            status: 'TRAINING', 'COMPLETED', 'FAILED', or None (not training)
        
        Raises:
            ValueError: If status is invalid or node doesn't exist
            sqlite3.Error: If there's a database error
        """
        # Validation: Node must exist
        signature = self.get_signature(node_id)
        if not signature:
            raise ValueError(f"KN {node_id} does not exist. Create signature first.")
        
        # Validation: Status must be valid or None
        valid_statuses = ['TRAINING', 'COMPLETED', 'FAILED']
        if status is not None and status not in valid_statuses:
            raise ValueError(
                f"Invalid training status: {status}. "
                f"Must be one of {valid_statuses} or None"
            )
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE nodes SET peft_training_status = ? WHERE node_id = ?",
                (status, node_id)
            )
            
            if cursor.rowcount == 0:
                logger.warning(f"Attempt to set training status for non-existent node: {node_id}")
                return
            
            self._maybe_commit()
            logger.debug(f"Training status set for {node_id}: {status}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error setting training status for {node_id}: {e}")
            raise

    def get_peft_training_status(self, node_id: str) -> Optional[str]:
        """
        Get PEFT training status for a Knowledge Node.
        
        Added: 2025-12-27
        Purpose: Check if node is currently being trained for Layer 2 decision making.
        
        Args:
            node_id: Node ID
        
        Returns:
            Status string ('TRAINING', 'COMPLETED', 'FAILED') or None if not set
            Returns None if node doesn't exist (doesn't raise exception)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT peft_training_status FROM nodes WHERE node_id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            # Node doesn't exist - return None (don't raise exception)
            return None
        
        # Return status (can be None if not set)
        status = row["peft_training_status"]
        return status if status else None

    # --- Phase 2: Training Support - Pointers and Training Status ---
    
    def save_pointer(self, node_id: str, source_id: str) -> None:
        """
        Save a pointer (source_id) to the original dataset for a Knowledge Node.
        
        Added: Phase 2 (2025-01-XX)
        Purpose: Track which original data points (texts) contributed to each KN.
        This enables training by retrieving original texts from datasets.
        
        Args:
            node_id: Node ID (must exist)
            source_id: ID from the original dataset (e.g., arXiv paper ID, 20newsgroups post ID)
        
        Raises:
            ValueError: If node_id or source_id is invalid
            sqlite3.Error: If there's a database error
        
        Note:
            UNIQUE constraint prevents duplicate pointers (pointer integrity).
            If pointer already exists, operation is silently ignored (idempotent).
        """
        # Validation
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, received: {node_id}")
        
        if not source_id or not isinstance(source_id, str):
            raise ValueError(f"source_id must be a non-empty string, received: {source_id}")
        
        # Verify node exists
        signature = self.get_signature(node_id)
        if not signature:
            raise ValueError(f"KN {node_id} does not exist. Create signature first.")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO node_data_mapping (node_id, source_id) VALUES (?, ?)",
                (node_id, source_id)
            )
            self._maybe_commit()
            logger.debug(f"Pointer saved: {node_id} -> {source_id}")
        except sqlite3.IntegrityError:
            # UNIQUE constraint violation: pointer already exists (idempotent)
            # This is expected and not an error - silently ignore
            logger.debug(f"Pointer already exists: {node_id} -> {source_id} (ignoring duplicate)")
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving pointer for {node_id}: {e}")
            raise
    
    def get_training_pointers(self, node_id: str) -> List[str]:
        """
        Retrieve all source_ids (pointers) associated with a Knowledge Node.
        
        Added: Phase 2 (2025-01-XX)
        Purpose: Get all original data IDs when node reaches training threshold.
        These source_ids are used to retrieve original texts for training.
        
        Args:
            node_id: Node ID
        
        Returns:
            List of source_ids (pointers to original dataset)
            Returns empty list if node has no pointers or doesn't exist
        
        Example:
            source_ids = repository.get_training_pointers("KN_45")
            # Returns: ["arxiv:1234.5678", "arxiv:2345.6789", ...]
            texts = data_manager.get_texts_from_pointers(source_ids)
        """
        # Validation
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, received: {node_id}")
        
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id FROM node_data_mapping WHERE node_id = ? ORDER BY created_timestamp",
            (node_id,)
        )
        rows = cursor.fetchall()
        
        source_ids = [row["source_id"] for row in rows]
        
        logger.debug(f"Retrieved {len(source_ids)} pointers for {node_id}")
        return source_ids
    
    def is_trained(self, node_id: str) -> bool:
        """
        Check if a Knowledge Node has a trained adapter.
        
        Added: Phase 2 (2025-01-XX)
        Purpose: Verify if node has trained LoRA adapter before triggering training
        or using it for inference.
        
        Args:
            node_id: Node ID
        
        Returns:
            True if node has trained adapter (is_trained=1), False otherwise
            Returns False if node doesn't exist (doesn't raise exception)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT is_trained FROM nodes WHERE node_id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            # Node doesn't exist - return False (don't raise exception)
            return False
        
        # is_trained is INTEGER: 1 = True, 0 = False, NULL = False
        is_trained_value = row["is_trained"]
        return bool(is_trained_value) if is_trained_value is not None else False
    
    def mark_as_trained(self, node_id: str) -> None:
        """
        Mark a Knowledge Node as having a trained adapter.
        
        Added: Phase 2 (2025-01-XX)
        Purpose: Update flag after successful training to prevent re-training
        and enable inference with trained adapter.
        
        Args:
            node_id: Node ID (must exist)
        
        Raises:
            ValueError: If node doesn't exist
            sqlite3.Error: If there's a database error
        """
        # Validation: Node must exist
        signature = self.get_signature(node_id)
        if not signature:
            raise ValueError(f"KN {node_id} does not exist. Create signature first.")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE nodes SET is_trained = 1 WHERE node_id = ?",
                (node_id,)
            )
            
            if cursor.rowcount == 0:
                logger.warning(f"Attempt to mark non-existent node as trained: {node_id}")
                return
            
            self._maybe_commit()
            logger.debug(f"Node marked as trained: {node_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error marking node as trained {node_id}: {e}")
            raise

    # --- Physical Memory Management (Embeddings) ---
    
    def apply_feedback(self, node_id: str, feedback: Any) -> None:
        """Apply Feedback deltas to node statistics.
        Expected feedback object has attributes `delta_mass` and `delta_variance`.
        """
        # Retrieve current stats
        signature = self.get_signature(node_id)
        if not signature:
            logger.warning(f"apply_feedback called for unknown node {node_id}")
            return
        new_mass = signature["mass"] + getattr(feedback, "delta_mass", 0.0)
        new_variance = signature["variance"] + getattr(feedback, "delta_variance", 0.0)
        # Ensure non‑negative values
        new_mass = max(int(new_mass), 0)
        new_variance = max(new_variance, 0.0)
        # Update centroid unchanged (could be recomputed elsewhere)
        self.update_node_stats(node_id, signature["centroid"], new_mass, new_variance)

    def add_embedding_to_memory(self, node_id: str, embedding: torch.Tensor) -> None:
        """
        Physical anchor: saves the actual embedding linked to the node.
        
        Args:
            node_id: Node ID
            embedding: FP16 tensor of the embedding
        
        Raises:
            ValueError: If parameters are invalid
            sqlite3.Error: If there's a database error
        """
        # Validation
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"node_id must be a non-empty string, received: {node_id}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO node_memory (node_id, embedding) VALUES (?, ?)",
                (node_id, self._tensor_to_blob(embedding))
            )
            self._maybe_commit()
            logger.debug(f"Embedding added to memory: {node_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error adding embedding to memory of {node_id}: {e}")
            raise

    def get_node_embeddings(self, node_id: str) -> List[torch.Tensor]:
        """
        Retrieve all embeddings of a node.
        
        Args:
            node_id: Node ID
        
        Returns:
            List of FP16 tensors on DEVICE
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT embedding FROM node_memory WHERE node_id = ?",
            (node_id,)
        )
        rows = cursor.fetchall()
        
        embeddings = []
        for row in rows:
            embedding = self._blob_to_tensor(row["embedding"], shape=(EMBEDDING_DIM,))
            embeddings.append(embedding)
        
        return embeddings
    
    def delete_kn(self, node_id: str) -> None:
        """
        Delete a Knowledge Node and all its embeddings (after fusion).
        
        Embeddings are automatically deleted by ON DELETE CASCADE.
        
        Args:
            node_id: ID of the node to delete
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
        # Embeddings are automatically deleted by ON DELETE CASCADE
        self._maybe_commit()
        logger.debug(f"KN deleted: {node_id}")

    # --- Buffer Management ---

    def create_buffer(self, buffer_id: str) -> None:
        """
        Create a new temporary buffer.
        
        Args:
            buffer_id: Unique buffer ID
        
        Raises:
            ValueError: If buffer_id is invalid
            sqlite3.Error: If there's a database error
        """
        # Validation
        if not buffer_id or not isinstance(buffer_id, str):
            raise ValueError(f"buffer_id must be a non-empty string, received: {buffer_id}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO buffers (buffer_id, size) VALUES (?, 0)",
                (buffer_id,)
            )
            self._maybe_commit()
            logger.debug(f"Buffer created: {buffer_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Buffer {buffer_id} already exists")
            raise
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating buffer {buffer_id}: {e}")
            raise

    def add_to_buffer(self, buffer_id: str, embedding: torch.Tensor) -> None:
        """
        Add an embedding to a buffer and update the centroid incrementally.
        
        Incremental centroid update:
        - If it's the first embedding (n=0): C_new = E_new
        - If n > 0: C_new = (C_old * n + E_new) / (n + 1)
        
        Args:
            buffer_id: Buffer ID
            embedding: FP16 tensor of the embedding
        
        Raises:
            ValueError: If parameters are invalid
            sqlite3.Error: If there's a database error
        """
        # Validation
        if not buffer_id or not isinstance(buffer_id, str):
            raise ValueError(f"buffer_id must be a non-empty string, received: {buffer_id}")
        
        # Asegurar que embedding está en el formato correcto
        embedding = embedding.to(device=DEVICE, dtype=DTYPE)
        
        try:
            cursor = self.conn.cursor()
            
            # Obtener centroide actual y tamaño
            cursor.execute(
                "SELECT centroid, size FROM buffers WHERE buffer_id = ?",
                (buffer_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Buffer {buffer_id} no existe")
            
            current_centroid_blob = row["centroid"]
            current_size = row["size"]
            
            # Calcular nuevo centroide incrementalmente
            if current_centroid_blob is None or current_size == 0:
                # Primer embedding: centroide = embedding
                new_centroid = embedding
            else:
                # Actualización incremental: C_new = (C_old * n + E_new) / (n + 1)
                current_centroid = self._blob_to_tensor(current_centroid_blob, shape=(EMBEDDING_DIM,))
                new_centroid = (current_centroid * current_size + embedding) / (current_size + 1)
            
            # Agregar embedding
            cursor.execute(
                "INSERT INTO buffer_embeddings (buffer_id, embedding) VALUES (?, ?)",
                (buffer_id, self._tensor_to_blob(embedding))
            )
            
            # Actualizar tamaño y centroide
            cursor.execute(
                "UPDATE buffers SET size = size + 1, centroid = ? WHERE buffer_id = ?",
                (self._tensor_to_blob(new_centroid), buffer_id)
            )
            
            self._maybe_commit()
            logger.debug(f"Embedding added to buffer: {buffer_id} (size={current_size + 1})")
        except sqlite3.Error as e:
            logger.error(f"SQLite error adding embedding to buffer {buffer_id}: {e}")
            raise

    def get_buffer_size(self, buffer_id: str) -> int:
        """
        Get the size of a buffer.
        
        Args:
            buffer_id: Buffer ID
        
        Returns:
            Buffer size (0 if it doesn't exist)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT size FROM buffers WHERE buffer_id = ?", (buffer_id,))
        row = cursor.fetchone()
        return row["size"] if row else 0

    def get_buffer_embeddings(self, buffer_id: str) -> List[torch.Tensor]:
        """
        Get all embeddings of a buffer.
        
        Args:
            buffer_id: Buffer ID
        
        Returns:
            List of FP16 tensors on DEVICE
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT embedding FROM buffer_embeddings WHERE buffer_id = ?",
            (buffer_id,)
        )
        rows = cursor.fetchall()
        
        embeddings = []
        for row in rows:
            embedding = self._blob_to_tensor(row["embedding"], shape=(EMBEDDING_DIM,))
            embeddings.append(embedding)
        
        return embeddings

    def delete_buffer(self, buffer_id: str) -> None:
        """
        Delete a buffer and all its embeddings (after promotion).
        
        Args:
            buffer_id: Buffer ID
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM buffers WHERE buffer_id = ?", (buffer_id,))
        # Embeddings are automatically deleted by ON DELETE CASCADE
        self._maybe_commit()
        logger.debug(f"Buffer deleted: {buffer_id}")
    
    def get_all_active_buffers(self) -> List[str]:
        """
        Get all IDs of active buffers.
        
        Returns:
            List of buffer_ids
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT buffer_id FROM buffers")
        rows = cursor.fetchall()
        return [row["buffer_id"] for row in rows]
    
    def get_buffer_centroid(self, buffer_id: str) -> Optional[torch.Tensor]:
        """
        Get the centroid of a buffer from the database.
        
        If the centroid is not in the database (NULL), calculates it on-the-fly as fallback.
        This can occur with old buffers created before the migration.
        
        Args:
            buffer_id: Buffer ID
        
        Returns:
            Centroid tensor [EMBEDDING_DIM] or None if the buffer is empty
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT centroid FROM buffers WHERE buffer_id = ?",
            (buffer_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        centroid_blob = row["centroid"]
        
        if centroid_blob is not None:
            # Centroid stored in database (normal case)
            return self._blob_to_tensor(centroid_blob, shape=(EMBEDDING_DIM,))
        else:
            # Fallback: calculate on-the-fly (for old buffers without centroid)
            embeddings = self.get_buffer_embeddings(buffer_id)
            if not embeddings:
                return None
            
            embeddings_tensor = torch.stack(embeddings).to(device=DEVICE, dtype=DTYPE)
            centroid = embeddings_tensor.mean(dim=0)
            
            # Optional: update centroid in database for future queries
            cursor.execute(
                "UPDATE buffers SET centroid = ? WHERE buffer_id = ?",
                (self._tensor_to_blob(centroid), buffer_id)
            )
            self._maybe_commit()
            
            return centroid
    
    def get_all_buffer_centroids(self) -> Tuple[List[str], torch.Tensor]:
        """
        Get all centroids of active buffers in a single operation.
        
        OPTIMIZED: Reads centroids directly from database (no calculations).
        Centroids are kept updated incrementally in add_to_buffer().
        
        Returns:
            Tuple with:
            - List of valid buffer_ids (with centroids)
            - Centroids tensor [N, EMBEDDING_DIM] on DEVICE and DTYPE
        """
        cursor = self.conn.cursor()
        
        # ========================================================================
        # Simple SQL query: Direct SELECT of stored centroids
        # ========================================================================
        # No longer need JOIN or calculations: centroids are in the database
        cursor.execute('''
            SELECT buffer_id, centroid
            FROM buffers
            WHERE centroid IS NOT NULL AND size > 0
            ORDER BY buffer_id
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            return [], torch.empty((0, EMBEDDING_DIM), device=DEVICE, dtype=DTYPE)
        
        # ========================================================================
        # Convert BLOBs to tensors directly
        # ========================================================================
        buffer_ids = []
        centroids = []
        
        for row in rows:
            buffer_id = row["buffer_id"]
            centroid_blob = row["centroid"]
            
            if centroid_blob is not None:
                centroid = self._blob_to_tensor(centroid_blob, shape=(EMBEDDING_DIM,))
                buffer_ids.append(buffer_id)
                centroids.append(centroid)
        
        if not centroids:
            return [], torch.empty((0, EMBEDDING_DIM), device=DEVICE, dtype=DTYPE)
        
        # Stack into a single matrix [N, EMBEDDING_DIM]
        centroids_tensor = torch.stack(centroids).to(device=DEVICE, dtype=DTYPE)
        
        logger.debug(
            f"Buffer centroids retrieved: {len(buffer_ids)} buffers "
            f"(direct SELECT from database, no calculations)"
        )
        
        return buffer_ids, centroids_tensor

    def close(self) -> None:
        """Close the database connection."""
        if self._update_counter > 0:
            self.conn.commit()
            logger.debug("Final commit before closing")
        self.conn.close()
        logger.info("Repository closed")
