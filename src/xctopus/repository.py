"""
Repository for Clustering Layer.

Bridge to SQLite for storing statistical signatures and buffers.
Stores FP16 tensors as BLOBs for efficiency.
"""

import sqlite3
import io
import functools
import time
import random
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
        # Enable thread-safe access: WAL mode allows concurrent reads/writes from different threads
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('PRAGMA foreign_keys = ON')  # CRITICAL: Enabled (2025-01-27)
        # CRITICAL: Set busy_timeout to handle concurrent writes gracefully (2025-01-14)
        # This makes SQLite wait up to 15 seconds before raising "database is locked"
        # This is more efficient than manual retries and handles transient locks automatically
        # Increased from 5s to 20s to handle high concurrency during training
        # Higher timeout reduces lock errors but may cause longer waits
        self.conn.execute('PRAGMA busy_timeout = 20000')  # 20 seconds timeout
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
                origin_type TEXT DEFAULT 'ORGANIC',
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
        
        # ========================================================================
        # Migrations (Auto-Update Schema)
        # ========================================================================
        # Add 'label' column for Gemini integration if not exists
        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN label TEXT")
            logger.info("Schema Update: Added 'label' column to nodes table")
        except sqlite3.OperationalError:
            pass # Column likely exists

        
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
        
        # 5. Buffer source_ids table (Phase 2: Training support)
        # Stores source_ids for each embedding in a buffer
        # When buffer is promoted to KnowledgeNode, these source_ids are transferred
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS buffer_source_ids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                buffer_id TEXT NOT NULL,
                embedding_id INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                FOREIGN KEY(buffer_id) REFERENCES buffers(buffer_id) ON DELETE CASCADE,
                FOREIGN KEY(embedding_id) REFERENCES buffer_embeddings(id) ON DELETE CASCADE,
                UNIQUE(buffer_id, embedding_id)
            )
        ''')
        
        # Indexes to optimize queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_id ON nodes(node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_memory_node_id ON node_memory(node_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_buffer_embeddings_buffer_id ON buffer_embeddings(buffer_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_buffer_source_ids_buffer_id ON buffer_source_ids(buffer_id)')
        
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
        
        # ========================================================================
        # Phase 3: Node Identity & Roles (2026-01-29)
        # ========================================================================
        # Migration: Add origin_type column to distinguish SEED vs ORGANIC nodes
        # SEED nodes are immune to being absorbed and have priority
        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN origin_type TEXT DEFAULT 'ORGANIC'")
            logger.debug("origin_type column added (migration Phase 3)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass
        
        # ========================================================================
        # Training Sessions Map (2026-01-26)
        # ========================================================================
        # Stores mapping between short session IDs and actual dataset files
        # This decouples source_id prefixes from brittle filenames
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                dataset_path TEXT,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON training_sessions(session_id)')
        
        # Migration: Add needs_training column for deferred training
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN needs_training INTEGER DEFAULT 0')
            logger.debug("needs_training column added (migration deferred training)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass
        
        # Migration: Add last_training_mass for Training Delta (re-training when mass doubles)
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN last_training_mass INTEGER DEFAULT 0')
            logger.debug("last_training_mass column added (migration Training Delta)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass

        # ========================================================================
        # Adaptive Training Support (2026-01-14)
        # ========================================================================
        # Migration: Add estimated_epochs for adaptive epoch calculation
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN estimated_epochs INTEGER')
            logger.debug("estimated_epochs column added (migration adaptive training)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass

        # Migration: Add training_priority for training order optimization
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN training_priority REAL')
            logger.debug("training_priority column added (migration adaptive training)")
        except sqlite3.OperationalError:
            # Column already exists, not an error
            pass

        # 5. Node Data Mapping Table (Pointers to original dataset)
        # Purpose: Associate node_id with source_id (pointer to original dataset)
        
        # ========================================================================
        # Inheritance Support (2025-01-XX)
        # ========================================================================
        # Migration: Add parent_node_id to track genealogy
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN parent_node_id TEXT')
            logger.debug("parent_node_id column added (migration inheritance)")
        except sqlite3.OperationalError:
            pass

        # Migration: Add parent_similarity to track how similar child is to parent
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN parent_similarity REAL')
            logger.debug("parent_similarity column added (migration inheritance)")
        except sqlite3.OperationalError:
            pass

        # Migration: Add inheritance_type ('ROOT', 'HERITAGE', 'FALLBACK') (2025-01-XX)
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN inheritance_type TEXT DEFAULT "ROOT"')
            logger.debug("inheritance_type column added (migration provenance)")
        except sqlite3.OperationalError:
            pass

        # ========================================================================
        # Migration: Add quarantine and inheritance tracking columns (2025-01-XX)
        # ========================================================================
        # Protocolo de cuarentena para nodos inestables
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN quarantine_count INTEGER DEFAULT 0')
            logger.debug("quarantine_count column added (migration quarantine protocol)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN last_quarantine_exit DATETIME')
            logger.debug("last_quarantine_exit column added (migration quarantine protocol)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN forced_root INTEGER DEFAULT 0')
            logger.debug("forced_root column added (migration quarantine protocol)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN forced_root_reason TEXT')
            logger.debug("forced_root_reason column added (migration quarantine protocol)")
        except sqlite3.OperationalError:
            pass

        # Campos para tracking de fallos
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN total_failures INTEGER DEFAULT 0')
            logger.debug("total_failures column added (migration tracking)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN last_failure_time DATETIME')
            logger.debug("last_failure_time column added (migration tracking)")
        except sqlite3.OperationalError:
            pass

        # Historial de herencia (JSON)
        try:
            cursor.execute('ALTER TABLE nodes ADD COLUMN inheritance_history TEXT')
            logger.debug("inheritance_history column added (migration tracking)")
        except sqlite3.OperationalError:
            pass

        # Crear índices para performance
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quarantine_count ON nodes(quarantine_count)')
            logger.debug("Index idx_quarantine_count created")
        except sqlite3.OperationalError as e:
            logger.debug(f"Index idx_quarantine_count already exists or error: {e}")

        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_forced_root ON nodes(forced_root)')
            logger.debug("Index idx_forced_root created")
        except sqlite3.OperationalError as e:
            logger.debug(f"Index idx_forced_root already exists or error: {e}")

        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_quarantine_exit ON nodes(last_quarantine_exit)')
            logger.debug("Index idx_last_quarantine_exit created")
        except sqlite3.OperationalError as e:
            logger.debug(f"Index idx_last_quarantine_exit already exists or error: {e}")

        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_peft_training_status ON nodes(peft_training_status)')
            logger.debug("Index idx_peft_training_status created")
        except sqlite3.OperationalError as e:
            logger.debug(f"Index idx_peft_training_status already exists or error: {e}")

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
        
        # 6. Evaluation Metrics Table (Phase 3: Model Evaluation)
        # Purpose: Store quality metrics for trained adapters
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            )
        ''')
        
        # Index for fast retrieval of metrics by node
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_node_id ON evaluation_metrics(node_id)')
        
        self.conn.commit()
        
        # Validate migration: Check that new columns exist
        self._validate_quarantine_migration()
        
        logger.info("Migración de BD: Campos de cuarentena agregados correctamente")
        logger.debug("Tables created/initialized correctly")

    def _validate_quarantine_migration(self) -> None:
        """
        Validate that quarantine migration columns exist in the database.
        
        Raises:
            ValueError if required columns are missing (logged as warning, not raised)
        """
        cursor = self.conn.cursor()
        try:
            # Get table info
            cursor.execute("PRAGMA table_info(nodes)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_columns = {
                'quarantine_count': 'INTEGER',
                'last_quarantine_exit': 'DATETIME',
                'forced_root': 'INTEGER',
                'forced_root_reason': 'TEXT',
                'total_failures': 'INTEGER',
                'last_failure_time': 'DATETIME',
                'inheritance_history': 'TEXT'
            }
            
            missing_columns = []
            for col_name, expected_type in required_columns.items():
                if col_name not in columns:
                    missing_columns.append(col_name)
                else:
                    logger.debug(f"Columna {col_name} existe (tipo: {columns[col_name]})")
            
            if missing_columns:
                logger.warning(f" Columnas faltantes en migración: {missing_columns}")
                # Don't raise error - columns might be added in next init
            else:
                logger.debug("Todas las columnas de cuarentena están presentes")
                
        except Exception as e:
            logger.error(f"Error validando migración de cuarentena: {e}", exc_info=True)
            # Don't raise - migration might be partial
        finally:
            cursor.close()

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
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT node_id, centroid, mass, variance, parent_node_id, parent_similarity, inheritance_type FROM nodes WHERE status = 'KN'")
            rows = cursor.fetchall()
            
            signatures = []
            for row in rows:
                centroid = self._blob_to_tensor(row["centroid"], shape=(EMBEDDING_DIM,))
                
                signatures.append({
                    "node_id": row["node_id"],
                    "centroid": centroid,
                    "mass": row["mass"],
                    "variance": row["variance"],
                    # Inheritance fields (optional)
                    "parent_node_id": row["parent_node_id"] if "parent_node_id" in row.keys() else None,
                    "parent_similarity": row["parent_similarity"] if "parent_similarity" in row.keys() else None,
                    "inheritance_type": row["inheritance_type"] if "inheritance_type" in row.keys() else None
                })
            
            logger.debug(f"Retrieved {len(signatures)} signatures from Repository")
            return signatures
        finally:
            if cursor:
                cursor.close()

    def get_signature(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the signature of a specific node.
        
        Args:
            node_id: Node ID
        
        Returns:
            Node signature or None if it doesn't exist
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT node_id, centroid, mass, variance, parent_node_id, parent_similarity, inheritance_type FROM nodes WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if row:
                centroid = self._blob_to_tensor(row["centroid"], shape=(EMBEDDING_DIM,))
                return {
                    "node_id": row["node_id"],
                    "centroid": centroid,
                    "mass": row["mass"],
                    "variance": row["variance"],
                    "parent_node_id": row["parent_node_id"] if "parent_node_id" in row.keys() else None,
                    "parent_similarity": row["parent_similarity"] if "parent_similarity" in row.keys() else None,
                    "inheritance_type": row["inheritance_type"] if "inheritance_type" in row.keys() else "ROOT"
                }
            return None
        finally:
            if cursor:
                cursor.close()

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

    def save_new_kn(
        self, 
        node_id: str, 
        centroid: torch.Tensor, 
        mass: int, 
        variance: float,
        parent_node_id: Optional[str] = None,
        parent_similarity: Optional[float] = None,
        inheritance_type: str = "ROOT",
        origin_type: str = "ORGANIC"
    ) -> None:
        """
        Register a new KnowledgeNode after buffer promotion.
        
        Args:
            node_id: Unique node ID
            centroid: FP16 tensor of the centroid
            mass: Number of embeddings
            variance: Scalar variance
            origin_type: 'ORGANIC' (default) or 'SEED'
        
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
                "INSERT INTO nodes (node_id, centroid, mass, variance, status, parent_node_id, parent_similarity, inheritance_type, origin_type) VALUES (?, ?, ?, ?, 'KN', ?, ?, ?, ?)",
                (node_id, self._tensor_to_blob(centroid), mass, variance, parent_node_id, parent_similarity, inheritance_type, origin_type)
            )
            self._maybe_commit()
            logger.info(f"New KN created: {node_id} (mass={mass}, variance={variance:.4f}, type={origin_type})")
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error creating KN {node_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating KN {node_id}: {e}")
            raise

    def add_inheritance_history_entry(
        self,
        node_id: str,
        from_type: str,
        to_type: str,
        reason: str,
        parent_id: Optional[str] = None,
        similarity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an entry to inheritance_history for a node.
        
        Added: Fase 6 (2025-01-XX)
        
        Args:
            node_id: Node ID
            from_type: Previous inheritance type ('ROOT', 'HERITAGE', 'FALLBACK')
            to_type: New inheritance type
            reason: Reason for change ('LATE_ADOPTION', 'PROGRESSIVE_ADOPTION', 'SANITIZATION', etc.)
            parent_id: Parent node ID (if applicable)
            similarity: Similarity with parent (if applicable)
            metadata: Additional metadata (JSON dict)
        """
        # Fase 9: Validaciones de entrada
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            logger.warning(f" add_inheritance_history_entry: node_id inválido: {node_id}")
            return
        
        # Fase 9: Validación de tipos de herencia
        valid_types = ('ROOT', 'HERITAGE', 'FALLBACK')
        if from_type not in valid_types:
            logger.warning(f" add_inheritance_history_entry: from_type inválido: {from_type}, usando ROOT")
            from_type = 'ROOT'
        if to_type not in valid_types:
            logger.warning(f" add_inheritance_history_entry: to_type inválido: {to_type}, usando ROOT")
            to_type = 'ROOT'
        
        # Fase 9: Validación de reason
        if not reason or not isinstance(reason, str):
            logger.warning(f" add_inheritance_history_entry: reason inválido: {reason}, usando UNKNOWN")
            reason = 'UNKNOWN'
        
        # Fase 9: Validación de similarity
        if similarity is not None:
            try:
                similarity = float(similarity)
                if not (0.0 <= similarity <= 1.0):
                    logger.warning(
                        f" add_inheritance_history_entry: similarity fuera de rango [0.0, 1.0]: "
                        f"{similarity}, usando None"
                    )
                    similarity = None
            except (ValueError, TypeError):
                logger.warning(f" add_inheritance_history_entry: similarity inválido: {similarity}, usando None")
                similarity = None
        
        # Fase 9: Validación de metadata
        if metadata is not None and not isinstance(metadata, dict):
            logger.warning(f" add_inheritance_history_entry: metadata no es dict: {metadata}, usando {{}}")
            metadata = {}
        
        try:
            from datetime import datetime
            now = datetime.utcnow().isoformat()
            
            # Create new entry
            new_entry = {
                "timestamp": now,
                "from": from_type,
                "to": to_type,
                "reason": reason,
                "parent_id": parent_id,
                "similarity": similarity,
                "metadata": metadata or {}
            }
            
            # Get current history
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT inheritance_history FROM nodes WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Node {node_id} does not exist")
            
            # Fase 9: Manejo robusto de NULLs y JSON inválido
            history_json = row['inheritance_history']
            if history_json:
                try:
                    if isinstance(history_json, str):
                        history = json.loads(history_json)
                    else:
                        # Already parsed
                        history = history_json
                    
                    # Fase 9: Validar estructura del JSON
                    if not isinstance(history, dict):
                        raise ValueError("History is not a dict")
                    if 'entries' not in history:
                        history['entries'] = []
                    if 'change_count' not in history:
                        history['change_count'] = len(history.get('entries', []))
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(
                        f" Invalid JSON en inheritance_history para {node_id}: {e}, reinicializando"
                    )
                    history = {"change_count": 0, "entries": []}
            else:
                # Fase 9: Inicializar si es NULL
                history = {"change_count": 0, "entries": []}
            
            # Add new entry
            history["entries"].append(new_entry)
            history["change_count"] = len(history["entries"])
            
            # Update database
            updated_json = json.dumps(history)
            cursor.execute(
                "UPDATE nodes SET inheritance_history = ? WHERE node_id = ?",
                (updated_json, node_id)
            )
            self._maybe_commit()
            
            logger.debug(
                f"📝 Agregando entrada a inheritance_history para {node_id}: "
                f"{from_type} -> {to_type} (razón: {reason})"
            )
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error agregando entrada a inheritance_history para {node_id}: {e}", exc_info=True)
            # Don't raise - history tracking is not critical

    def update_inheritance_type(self, node_id: str, inheritance_type: str, reason: str = "MANUAL") -> None:
        """
        Update the inheritance type of a node.
        
        Args:
            node_id: Node ID
            inheritance_type: New type ('ROOT', 'HERITAGE', 'FALLBACK')
            reason: Reason for change (default: 'MANUAL')
        """
        try:
            # Get current type for history
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT inheritance_type FROM nodes WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Node {node_id} does not exist")
            
            from_type = row['inheritance_type'] or 'ROOT'
            
            # Update type
            cursor.execute(
                "UPDATE nodes SET inheritance_type = ? WHERE node_id = ?",
                (inheritance_type, node_id)
            )
            self._maybe_commit()
            logger.debug(f"Updated inheritance_type for {node_id} to '{inheritance_type}'")
            
            # Add to history
            if from_type != inheritance_type:
                self.add_inheritance_history_entry(
                    node_id=node_id,
                    from_type=from_type,
                    to_type=inheritance_type,
                    reason=reason
                )
            
            cursor.close()
            
        except sqlite3.Error as e:
            logger.error(f"Error updating inheritance_type for {node_id}: {e}")
            raise

    def sanitize_orphans(self) -> int:
        """
        Clean nodes that point to parents that failed in training.
        
        Converts orphaned nodes (with FAILED parents) back to ROOT status.
        This ensures database consistency and puts nodes back in the "adoption pool".
        
        Added: 2025-01-XX (Fase 2: Sanitización Activa)
        
        Returns:
            Number of nodes sanitized (converted to ROOT)
        """
        max_retries = 10
        retry_delay = 0.2
        affected_count = 0
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # REMOVED: self.conn.rollback() (2025-01-27)
                
                # Add jitter to avoid simultaneous writes
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    time.sleep(random.uniform(0.01, 0.05))
                
                cursor = self.conn.cursor()
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except sqlite3.OperationalError:
                    pass
                
                # First, SELECT nodes that will be affected (before UPDATE)
                # This allows us to track them for history entries
                select_query = """
                    SELECT node_id, inheritance_type, parent_node_id
                    FROM nodes 
                    WHERE status = 'KN'
                    AND parent_node_id IS NOT NULL
                    AND parent_node_id IN (
                        SELECT node_id FROM nodes 
                        WHERE status = 'KN' 
                        AND peft_training_status = 'FAILED'
                    )
                """
                
                cursor.execute(select_query)
                affected_nodes = cursor.fetchall()
                affected_count = len(affected_nodes)
                
                if affected_count > 0:
                    # Now UPDATE the affected nodes
                    update_query = """
                        UPDATE nodes 
                        SET parent_node_id = NULL, 
                            inheritance_type = 'ROOT',
                            parent_similarity = NULL
                        WHERE status = 'KN'
                        AND parent_node_id IS NOT NULL
                        AND parent_node_id IN (
                            SELECT node_id FROM nodes 
                            WHERE status = 'KN' 
                            AND peft_training_status = 'FAILED'
                        )
                    """
                    
                    cursor.execute(update_query)
                    self.conn.commit()
                    logger.info(
                        f" Sanitización completada: {affected_count} huérfanos convertidos a ROOT"
                    )
                    
                    # Add history entries for affected nodes (after commit)
                    for node_row in affected_nodes:
                        node_id = node_row['node_id']
                        from_type = node_row['inheritance_type'] or 'HERITAGE'
                        parent_id = node_row['parent_node_id']
                        try:
                            self.add_inheritance_history_entry(
                                node_id=node_id,
                                from_type=from_type,
                                to_type='ROOT',
                                reason='SANITIZATION',
                                parent_id=parent_id
                            )
                        except Exception as hist_error:
                            logger.warning(f" Error agregando historial para {node_id} durante sanitización: {hist_error}")
                else:
                    # No orphans to clean, but commit anyway to clear any pending transactions
                    self.conn.commit()
                    logger.debug(" Sanitización: No se encontraron nodos huérfanos")
                
                # Restore original busy_timeout
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                
                if cursor:
                    cursor.close()
                return affected_count
                
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked during sanitization, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error durante sanitización: {e}")
                    raise
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"SQLite error durante sanitización: {e}")
                raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Error inesperado durante sanitización: {e}", exc_info=True)
                raise
        
        # Should not reach here, but return 0 if all retries failed
        logger.warning(f" Sanitización: No se pudo completar después de {max_retries} intentos")
        return 0

    def update_parent_info(self, node_id: str, parent_node_id: str, parent_similarity: float) -> None:
        """
        Link a node to a parent (Late Adoption).
        
        Args:
            node_id: Child node ID
            parent_node_id: Parent node ID
            parent_similarity: Similarity score
        
        Raises:
            ValueError: If node_id or parent_node_id is invalid, or similarity is out of range
        """
        # Fase 9: Validaciones de entrada
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"node_id debe ser un string no vacío, recibido: {node_id}")
        
        if not parent_node_id or not isinstance(parent_node_id, str) or not parent_node_id.strip():
            raise ValueError(f"parent_node_id debe ser un string no vacío, recibido: {parent_node_id}")
        
        # Fase 9: Validación de rango para similarity
        try:
            parent_similarity = float(parent_similarity)
            if not (0.0 <= parent_similarity <= 1.0):
                raise ValueError(
                    f"parent_similarity debe estar en rango [0.0, 1.0], recibido: {parent_similarity}"
                )
        except (ValueError, TypeError) as e:
            if isinstance(e, ValueError) and "rango" in str(e):
                raise
            raise ValueError(f"parent_similarity debe ser un float válido, recibido: {parent_similarity}")
        
        # Validation: Infinite Loop
        if node_id == parent_node_id:
            logger.error(f"Cannot link node {node_id} to itself (Infinite Loop detected).")
            raise ValueError("Node cannot be its own parent")

        # ========================================================================
        # CRITICAL: Retry logic for database locks (2025-01-14)
        # ========================================================================
        # This method is called during Progressive Adoption which can be concurrent
        # ========================================================================
        max_retries = 10
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # REMOVED: self.conn.rollback() (2025-01-27)
                
                # On retries, wait a bit longer with jitter
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    # Even on first attempt, add small random delay to avoid simultaneous writes
                    time.sleep(random.uniform(0.01, 0.05))  # 10-50ms jitter
                
                cursor = self.conn.cursor()
                
                # Get current inheritance info for history
                cursor.execute(
                    "SELECT inheritance_type, parent_node_id FROM nodes WHERE node_id = ?",
                    (node_id,)
                )
                current_row = cursor.fetchone()
                
                if not current_row:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    raise ValueError(f"Node {node_id} does not exist")
                
                from_type = current_row['inheritance_type'] or 'ROOT'
                old_parent_id = current_row['parent_node_id']
                
                # Execute the update
                # This may block if database is locked, respecting busy_timeout
                
                cursor.execute(
                    """UPDATE nodes 
                       SET parent_node_id = ?, 
                           parent_similarity = ?, 
                           inheritance_type = 'HERITAGE' 
                       WHERE node_id = ?""",
                    (parent_node_id, parent_similarity, node_id)
                )
                
                if cursor.rowcount == 0:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    raise ValueError(f"Node {node_id} does not exist")
                
                self.conn.commit()
                
                # Add to history
                try:
                    # Determine reason based on whether parent changed
                    reason = 'PROGRESSIVE_ADOPTION' if old_parent_id is None else 'PARENT_CHANGE'
                    self.add_inheritance_history_entry(
                        node_id=node_id,
                        from_type=from_type,
                        to_type='HERITAGE',
                        reason=reason,
                        parent_id=parent_node_id,
                        similarity=parent_similarity
                    )
                except Exception as hist_error:
                    logger.warning(f" Error agregando historial para {node_id}: {hist_error}")
                
                # Restore original busy_timeout
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                logger.debug(f"Linked node {node_id} to parent {parent_node_id} (sim={parent_similarity:.3f})")
                if cursor:
                    cursor.close()
                return  # Success
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when updating parent info for {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error updating parent info for {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"SQLite error updating parent info for {node_id}: {e}")
                raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Unexpected error updating parent info for {node_id}: {e}")
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
                # Node may have been fused or deleted during processing - this is expected
                logger.debug(f"Attempt to update non-existent node (likely fused/deleted): {node_id}")
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

        # ========================================================================
        # CRITICAL: Retry logic for database locks (2025-01-14)
        # ========================================================================
        # SQLite with WAL mode can have temporary locks when multiple threads
        # write simultaneously, especially with large BLOBs. Retry with backoff.
        # ========================================================================
        max_retries = 10  # Increased retries for large BLOBs
        retry_delay = 0.2  # Start with 200ms
        
        for attempt in range(max_retries):
            cursor = None
            try:
                # Ensure connection is still valid
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # CRITICAL: Clean up any pending transactions before attempting
                try:
                    self.conn.rollback()
                except:
                    pass  # Ignore errors if no transaction is active
                
                # On retries, wait a bit longer to let other transactions complete
                # Add random jitter to prevent thundering herd problem
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    # Even on first attempt, add small random delay to avoid simultaneous writes
                    time.sleep(random.uniform(0.01, 0.05))  # 10-50ms jitter
                
                # CRITICAL: Execute the UPDATE directly.
                # SQLite will acquire the necessary lock automatically, respecting busy_timeout.
                # We do NOT use BEGIN IMMEDIATE here to avoid conflicts with isolation_level.
                cursor = self.conn.cursor()
                
                # Execute the update (may block if database is locked, respecting busy_timeout)
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
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    return
                # CRITICAL: Immediate commit for PEFT weights (large BLOBs can cause locks)
                self.conn.commit()
                # Restore original busy_timeout
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                logger.debug(f"PEFT weights saved for {node_id} (format={format}, size={weights_size} bytes)")
                if cursor:
                    cursor.close()
                return  # Success
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                
                # Rollback any pending transaction
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Retry with exponential backoff + jitter (capped at 4 seconds)
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when saving PEFT weights for {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Either not a lock error or max retries reached
                    logger.error(f"SQLite error saving PEFT weights for {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                # Other SQLite errors - don't retry
                logger.error(f"SQLite error saving PEFT weights for {node_id}: {e}")
                raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Unexpected error saving PEFT weights for {node_id}: {e}")
                raise

    def _load_lora_weights(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Internal helper to load LoRA weights from DB and return a dict with state_dict and config.
        Returns None if no weights are stored for the node.
        Returns dict with keys: 'state_dict', 'config' (optional)
        """
        peft = self.get_peft_weights(node_id)
        if not peft:
            return None
        
        weights_bytes = peft["weights_bytes"]
        format_type = peft.get("format", "safetensors")
        config = peft.get("config")  # PEFT configuration dict
        
        # CRITICAL: Handle different formats correctly
        # safetensors format cannot be loaded with torch.load() (expects pickle)
        if format_type == "safetensors":
            try:
                # Try to import safetensors
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    logger.error(
                        f"safetensors library not available but weights for '{node_id}' "
                        f"are in safetensors format. Cannot load weights."
                    )
                    return None
                
                # Load from bytes buffer
                # load_file from safetensors.torch accepts file-like objects
                # but we need to ensure the buffer is at position 0
                buffer = io.BytesIO(weights_bytes)
                buffer.seek(0)  # Ensure we're at the start
                try:
                    state_dict = load_file(buffer)
                except (TypeError, AttributeError) as e:
                    # If load_file doesn't work with BytesIO, use temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                        tmp_file.write(weights_bytes)
                        tmp_file.flush()
                        state_dict = load_file(tmp_file.name)
                
                # Move tensors to correct device
                state_dict = {k: v.to(device=DEVICE) for k, v in state_dict.items()}
                # Return dict with state_dict and config
                return {"state_dict": state_dict, "config": config}
            except Exception as e:
                logger.error(
                    f"Error loading safetensors weights for '{node_id}': {e}",
                    exc_info=True
                )
                return None
        else:
            # Format is 'bin' (pickle format) - use torch.load
            buffer = io.BytesIO(weights_bytes)
            # CRITICAL: PyTorch 2.6 changed default weights_only=True, but our weights contain
            # metadata that requires weights_only=False. These weights are from our own
            # trusted training process, so it's safe.
            state_dict = torch.load(buffer, map_location=DEVICE, weights_only=False)
            # Return dict with state_dict and config
            return {"state_dict": state_dict, "config": config}

    def get_lora_weights(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Public method that returns the LoRA weights for a node, cached via LRU.
        Returns dict with keys: 'state_dict', 'config' (optional), or None if not found.
        For backward compatibility, if the result is a dict with 'state_dict' key, 
        it can be used directly as state_dict in older code.
        """
        result = self._lora_cache(node_id)
        # For backward compatibility: if result is a dict with 'state_dict', return it
        # Otherwise return as-is (could be None or old format)
        return result

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
        
        # ========================================================================
        # CRITICAL: Retry logic for database locks (2025-01-14)
        # ========================================================================
        # SQLite with WAL mode can have temporary locks when multiple threads
        # write simultaneously. Retry with exponential backoff + jitter.
        # Jitter prevents "thundering herd" problem when multiple operations retry simultaneously.
        # ========================================================================
        max_retries = 15  # Increased retries for high concurrency
        retry_delay = 0.2  # Start with 200ms
        
        for attempt in range(max_retries):
            cursor = None
            try:
                # Ensure connection is still valid
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # CRITICAL: Clean up any pending transactions before attempting
                # This ensures we start with a clean state
                try:
                    self.conn.rollback()
                except:
                    pass  # Ignore errors if no transaction is active
                
                # On retries, wait a bit longer to let other transactions complete
                # Add random jitter to prevent thundering herd problem
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    # Even on first attempt, add small random delay to avoid simultaneous writes
                    time.sleep(random.uniform(0.01, 0.05))  # 10-50ms jitter
                
                # CRITICAL: Just execute the UPDATE.
                # SQLite will acquire the necessary lock automatically.
                # We do NOT use BEGIN IMMEDIATE here because it conflicts with default isolation_level
                # and can cause unwarranted "database is locked" errors in some setups.
                
                cursor = self.conn.cursor()
                
                # Execute the update
                # This may block if database is locked, respecting busy_timeout
                cursor.execute(
                    "UPDATE nodes SET peft_training_status = ? WHERE node_id = ?",
                    (status, node_id)
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Attempt to set training status for non-existent node: {node_id}")
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    return
                
                # CRITICAL: Immediate commit for training status updates (critical state changes)
                self.conn.commit()
                # Restore original busy_timeout
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                logger.debug(f"Training status set for {node_id}: {status}")
                if cursor:
                    cursor.close()
                return  # Success
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                
                # Rollback any pending transaction
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Retry with exponential backoff + jitter (capped at 4 seconds)
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when setting training status for {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Either not a lock error or max retries reached
                    logger.error(f"SQLite error setting training status for {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                # Other SQLite errors - don't retry
                logger.error(f"SQLite error setting training status for {node_id}: {e}")
                raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Unexpected error setting training status for {node_id}: {e}")
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
    
    def reassign_all_pointers(self, from_node_ids: List[str], to_node_id: str) -> int:
        """
        Reassign all data pointers from old nodes to a new merged node.
        
        Added: 2025-01-27
        Purpose: Atomic and fast migration during fusion. Prevents "phantom nodes" (nodes with mass but no texts).
        
        Args:
            from_node_ids: List of source node IDs to migrate from
            to_node_id: Destination node ID
            
        Returns:
            int: Number of pointers reassigned
        """
        if not from_node_ids: return 0
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Atomic update for all pointers in one go using OR IGNORE to handle duplicates
            # If two fused nodes have the same source_id, the redundant one remains with 
            # the old node_id and is later deleted by CASCADE.
            placeholders = ', '.join(['?'] * len(from_node_ids))
            query = f"UPDATE OR IGNORE node_data_mapping SET node_id = ? WHERE node_id IN ({placeholders})"
            params = [to_node_id] + from_node_ids
            
            cursor.execute(query, params)
            count = cursor.rowcount
            
            # NO maybe_commit here, the caller (fusion) should handle final commit
            logger.debug(f"Atomic reassignment: {count} pointers moved to {to_node_id}")
            return count
        except sqlite3.Error as e:
            logger.error(f"Failed atomic pointer reassignment to {to_node_id}: {e}")
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
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT source_id FROM node_data_mapping WHERE node_id = ? ORDER BY created_timestamp",
                (node_id,)
            )
            rows = cursor.fetchall()
            
            source_ids = [row["source_id"] for row in rows]
            
            logger.debug(f"Retrieved {len(source_ids)} pointers for {node_id}")
            return source_ids
        finally:
            if cursor:
                cursor.close()
    
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
        cursor = None
        try:
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
        finally:
            if cursor:
                cursor.close()
    
    def get_trained_node_ids(self) -> List[str]:
        """
        Get all node IDs that are trained (is_trained = 1).
        
        Added: 2026-01-17
        Purpose: Batch query to get all trained nodes at once, avoiding multiple
        individual is_trained() calls that can cause database locks.
        
        Returns:
            List of node IDs that have is_trained = 1
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT node_id FROM nodes WHERE is_trained = 1"
            )
            rows = cursor.fetchall()
            return [row["node_id"] for row in rows]
        finally:
            if cursor:
                cursor.close()
    
    def mark_as_trained(self, node_id: str) -> None:
        """
        Mark a Knowledge Node as having a trained adapter.
        """
        max_retries = 10
        retry_delay = 0.2
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn: raise sqlite3.Error("No connection")
                # REMOVED: self.conn.rollback() (Destructive to atomic transactions)
                if attempt > 0:
                    time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 0.5))
                cursor = self.conn.cursor()
                cursor.execute("UPDATE nodes SET is_trained = 1, peft_training_status = 'COMPLETED' WHERE node_id = ?", (node_id,))
                self.conn.commit()
                if cursor: cursor.close()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1: continue
                if cursor: cursor.close()
                raise

    def mark_as_failed(self, node_id: str, error_msg: str = None) -> None:
        """
        Mark a Knowledge Node training as FAILED.
        """
        max_retries = 10
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn: return
                # REMOVED: self.conn.rollback() (Destructive to atomic transactions)
                cursor = self.conn.cursor()
                cursor.execute("""
                    UPDATE nodes 
                    SET peft_training_status = 'FAILED',
                        is_trained = 0,
                        needs_training = 0,
                        total_failures = total_failures + 1,
                        last_failure_time = CURRENT_TIMESTAMP
                    WHERE node_id = ?
                """, (node_id,))
                self.conn.commit()
                if cursor: cursor.close()
                return
            except sqlite3.OperationalError:
                if attempt < max_retries - 1: time.sleep(0.5)
                if cursor: cursor.close()

    def mark_for_training(self, node_id: str, is_retraining: bool = False) -> None:
        """
        Mark a node as needing training (deferred training).
        
        Added: 2025-01-14
        Purpose: Instead of starting training immediately, mark node for later training.
        This allows processing to continue without blocking.
        
        Args:
            node_id: Knowledge Node ID to mark
            is_retraining: If True, this is a re-training (Training Delta)
        """
        max_retries = 10  # Increased retries
        retry_delay = 0.2  # Increased delay
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # REMOVED: self.conn.rollback() (2025-01-27)
                
                # On retries, wait a bit longer with jitter
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    # Even on first attempt, add small random delay to avoid simultaneous writes
                    time.sleep(random.uniform(0.01, 0.05))  # 10-50ms jitter
                
                cursor = self.conn.cursor()
                
                # Execute the update
                cursor.execute(
                    "UPDATE nodes SET needs_training = 1 WHERE node_id = ?",
                    (node_id,)
                )
                
                if cursor.rowcount == 0:
                    # Node may have been fused or deleted during processing - this is expected
                    logger.debug(f"Attempt to mark non-existent node for training (likely fused/deleted): {node_id}")
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    return
                
                self.conn.commit()
                # Restore original busy_timeout
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                retraining_str = " (re-training)" if is_retraining else ""
                logger.debug(f"Node marked for training{retraining_str}: {node_id}")
                if cursor:
                    cursor.close()
                return
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when marking node for training {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error marking node for training {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"SQLite error marking node for training {node_id}: {e}")
                raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Unexpected error marking node for training {node_id}: {e}")
                raise
    
    def get_pending_training_nodes(self) -> List[str]:
        """
        Get all node IDs that need training.
        
        Added: 2025-01-14
        Purpose: Retrieve list of nodes marked for deferred training.
        
        Returns:
            List of node IDs that have needs_training = 1
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT node_id FROM nodes WHERE needs_training = 1 ORDER BY mass DESC"
            )
            rows = cursor.fetchall()
            return [row["node_id"] for row in rows]
        finally:
            if cursor:
                cursor.close()
    
    def cleanup_stale_training_states(self) -> int:
        """
        Clean up training states that were left in 'TRAINING' due to interruptions.
        
        Added: 2025-01-XX
        Purpose: Ensure consistent state after interruptions (Ctrl+C, crashes, etc.)
        
        Returns:
            Number of nodes cleaned up
        """
        max_retries = 5
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                # REMOVED: self.conn.rollback() (2025-01-27)
                
                # On retries, wait a bit longer with jitter
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    # Even on first attempt, add small random delay to avoid simultaneous writes
                    time.sleep(random.uniform(0.01, 0.05))  # 10-50ms jitter
                
                cursor = self.conn.cursor()
                try:
                    cursor.execute("PRAGMA busy_timeout = 30000")
                    cursor.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as begin_error:
                    try:
                        self.conn.rollback()
                    except:
                        pass
                    if cursor:
                        cursor.close()
                    if "database is locked" in str(begin_error).lower() and attempt < max_retries - 1:
                        base_wait = min(retry_delay * (2 ** attempt), 4.0)
                        jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                        wait_time = base_wait + jitter
                        logger.debug(
                            f"Database locked (BEGIN IMMEDIATE) when cleaning stale states, "
                            f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
                
                # Read stale nodes within transaction
                cursor.execute(
                    "SELECT node_id, is_trained FROM nodes WHERE peft_training_status = 'TRAINING'"
                )
                stale_nodes = cursor.fetchall()
                
                cleaned = 0
                for row in stale_nodes:
                    node_id = row['node_id']
                    is_trained = bool(row['is_trained'])
                    
                    if is_trained:
                        # Has weights but incorrect status - mark as COMPLETED
                        cursor.execute(
                            "UPDATE nodes SET peft_training_status = 'COMPLETED', needs_training = 0 WHERE node_id = ?",
                            (node_id,)
                        )
                        logger.info(
                            f"Fixed inconsistent state: Node '{node_id}' has weights but status was TRAINING → COMPLETED"
                        )
                    else:
                        # No weights - clear status for re-training
                        cursor.execute(
                            "UPDATE nodes SET peft_training_status = NULL WHERE node_id = ?",
                            (node_id,)
                        )
                        logger.info(
                            f"Cleaned stale TRAINING status for node '{node_id}' (will re-train)"
                        )
                        cleaned += 1
                
                if stale_nodes:
                    self.conn.commit()
                    # Restore original busy_timeout
                    try:
                        cursor.execute("PRAGMA busy_timeout = 20000")
                    except:
                        pass
                    logger.info(f"Cleaned {len(stale_nodes)} stale training states ({cleaned} will re-train)")
                
                if cursor:
                    cursor.close()
                return len(stale_nodes)
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)  # 0-30% jitter
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when cleaning stale states, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error cleaning stale training states: {e}")
                    return 0
            except sqlite3.Error as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"SQLite error cleaning stale training states: {e}")
                return 0
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Unexpected error cleaning stale training states: {e}")
                return 0
        
        return 0  # If all retries failed

    def clear_training_flag(self, node_id: str) -> None:
        """
        Clear needs_training flag after training completes.
        
        Added: 2025-01-14
        Purpose: Reset the flag after training is done.
        
        Args:
            node_id: Knowledge Node ID
        """
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "UPDATE nodes SET needs_training = 0 WHERE node_id = ?",
                    (node_id,)
                )
                # CRITICAL: Immediate commit for training completion (critical state changes)
                # Changed from _maybe_commit() to ensure flag is cleared immediately
                self.conn.commit()
                logger.debug(f"Training flag cleared for node: {node_id}")
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error clearing training flag {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                logger.error(f"SQLite error clearing training flag {node_id}: {e}")
                raise
    
    def get_last_training_mass(self, node_id: str) -> int:
        """
        Get the mass at which node was last trained (for Training Delta).
        
        Added: 2025-01-14
        Purpose: Track mass at last training to enable re-training when mass doubles.
        
        Args:
            node_id: Knowledge Node ID
        
        Returns:
            Mass at last training, or 0 if never trained
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT last_training_mass FROM nodes WHERE node_id = ?",
            (node_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return 0
        
        return row["last_training_mass"] or 0
    
    def update_last_training_mass(self, node_id: str, mass: int) -> None:
        """
        Update last_training_mass after training completes.
        
        Added: 2025-01-14
        Purpose: Store mass at which training occurred for Training Delta calculation.
        
        Args:
            node_id: Knowledge Node ID
            mass: Current mass of the node
        """
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "UPDATE nodes SET last_training_mass = ? WHERE node_id = ?",
                    (mass, node_id)
                )
                self._maybe_commit()
                logger.debug(f"Last training mass updated for node {node_id}: {mass}")
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error updating last training mass {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                logger.error(f"SQLite error updating last training mass {node_id}: {e}")
                raise
    
    def get_estimated_epochs(self, node_id: str) -> Optional[int]:
        """
        Get estimated epochs for a node.
        
        Added: 2026-01-14
        Purpose: Retrieve calculated estimated_epochs for training.
        
        Args:
            node_id: Knowledge Node ID
        
        Returns:
            Estimated epochs, or None if not set
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT estimated_epochs FROM nodes WHERE node_id = ?",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if row and row["estimated_epochs"] is not None:
                return int(row["estimated_epochs"])
            return None
        finally:
            if cursor:
                cursor.close()
    
    def update_training_metadata(
        self, 
        node_id: str, 
        estimated_epochs: Optional[int] = None, 
        training_priority: Optional[float] = None
    ) -> None:
        """
        Update training metadata (estimated_epochs and training_priority) for a node.
        
        Added: 2026-01-14
        Purpose: Store calculated training metadata after fusion.
        
        Args:
            node_id: Knowledge Node ID
            estimated_epochs: Estimated number of epochs for training (optional)
            training_priority: Training priority based on information density (optional)
        """
        max_retries = 5
        retry_delay = 0.1
        
        # Build UPDATE query dynamically based on provided values
        updates = []
        values = []
        
        if estimated_epochs is not None:
            updates.append("estimated_epochs = ?")
            values.append(estimated_epochs)
        
        if training_priority is not None:
            updates.append("training_priority = ?")
            values.append(training_priority)
        
        if not updates:
            # Nothing to update
            return
        
        values.append(node_id)  # For WHERE clause
        
        query = f"UPDATE nodes SET {', '.join(updates)} WHERE node_id = ?"
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                cursor.execute(query, values)
                self._maybe_commit()
                logger.debug(
                    f"Training metadata updated for node {node_id}: "
                    f"epochs={estimated_epochs}, priority={training_priority}"
                )
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error updating training metadata {node_id}: {e}")
                    raise
            except sqlite3.Error as e:
                logger.error(f"SQLite error updating training metadata {node_id}: {e}")
                raise

    # --- Physical Memory Management (Embeddings) ---
    
    def apply_feedback(self, node_id: str, feedback: Any) -> None:
        """Apply Feedback deltas to node statistics.
        Expected feedback object has attributes `delta_mass` and `delta_variance`.
        """
        # Retrieve current stats
        signature = self.get_signature(node_id)
        if not signature:
            # Node may have been fused or deleted during processing - this is expected
            logger.debug(f"apply_feedback called for unknown node (likely fused/deleted): {node_id}")
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

    def add_to_buffer(self, buffer_id: str, embedding: torch.Tensor, source_id: Optional[str] = None) -> None:
        """
        Add an embedding to a buffer and update the centroid incrementally.
        
        Incremental centroid update:
        - If it's the first embedding (n=0): C_new = E_new
        - If n > 0: C_new = (C_old * n + E_new) / (n + 1)
        
        Phase 2: Also saves source_id if provided (for training data provenance).
        
        Args:
            buffer_id: Buffer ID
            embedding: FP16 tensor of the embedding
            source_id: Optional source ID (pointer to original dataset)
                If provided, will be saved for later transfer to KnowledgeNode
        
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
            embedding_id = cursor.lastrowid
            
            # Guardar source_id si se proporcionó (Phase 2: Training support)
            if source_id is not None:
                cursor.execute(
                    "INSERT INTO buffer_source_ids (buffer_id, embedding_id, source_id) VALUES (?, ?, ?)",
                    (buffer_id, embedding_id, source_id)
                )
                logger.debug(f"Saved source_id '{source_id}' for embedding in buffer '{buffer_id}'")
            
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

    def get_buffer_source_ids(self, buffer_id: str) -> List[str]:
        """
        Get all source_ids for embeddings in a buffer.
        
        Phase 2: Training support - retrieves source_ids that will be transferred
        to KnowledgeNode when buffer is promoted.
        
        Args:
            buffer_id: Buffer ID
            
        Returns:
            List of source_ids (may be empty if buffer has no source_ids)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT source_id FROM buffer_source_ids WHERE buffer_id = ? ORDER BY id",
                (buffer_id,)
            )
            rows = cursor.fetchall()
            return [row["source_id"] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting source_ids for buffer {buffer_id}: {e}")
            return []
    
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

    # --- Quarantine Protocol Support (2025-01-XX) ---
    
    def get_quarantine_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get quarantine information for a node.
        
        Added: Fase 4 (2025-01-XX)
        
        Args:
            node_id: Node ID
        
        Returns:
            Dictionary with quarantine info or None if node doesn't exist:
            - quarantine_count: int
            - last_quarantine_exit: datetime or None
            - forced_root: bool
            - forced_root_reason: str or None
            - total_failures: int
            - last_failure_time: datetime or None
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """SELECT quarantine_count, last_quarantine_exit, forced_root, 
                          forced_root_reason, total_failures, last_failure_time
                   FROM nodes WHERE node_id = ?""",
                (node_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Fase 9: Manejo robusto de NULLs y validación de tipos
            quarantine_count = row['quarantine_count']
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
            
            total_failures = row['total_failures']
            if total_failures is None:
                total_failures = 0
            else:
                try:
                    total_failures = int(total_failures)
                    if total_failures < 0:
                        logger.warning(f" total_failures negativo para {node_id}: {total_failures}, corrigiendo a 0")
                        total_failures = 0
                except (ValueError, TypeError):
                    logger.warning(f" total_failures inválido para {node_id}: {total_failures}, usando 0")
                    total_failures = 0
            
            forced_root_val = row['forced_root']
            forced_root = bool(forced_root_val) if forced_root_val is not None else False
            
            return {
                'quarantine_count': quarantine_count,
                'last_quarantine_exit': row['last_quarantine_exit'],
                'forced_root': forced_root,
                'forced_root_reason': row['forced_root_reason'],
                'total_failures': total_failures,
                'last_failure_time': row['last_failure_time']
            }
        except Exception as e:
            logger.error(f"Error getting quarantine info for {node_id}: {e}")
            return None
        finally:
            cursor.close()
    
    def increment_quarantine_count(
        self, 
        node_id: str, 
        reason: str = "Training failure",
        failure_type: Optional[str] = None
    ) -> int:
        """
        Increment quarantine count for a node.
        
        Added: Fase 4 (2025-01-XX)
        
        Logic:
        1. Increment quarantine_count
        2. Set last_quarantine_exit = NULL (node enters quarantine)
        3. If quarantine_count >= 3, set forced_root = True
        4. Update total_failures
        5. Set last_failure_time = now()
        
        Args:
            node_id: Node ID
            reason: Reason for quarantine
            failure_type: Type of failure (optional)
        
        Returns:
            New quarantine_count value
        
        Raises:
            ValueError: If node_id is invalid or node doesn't exist
        """
        # Fase 9: Validaciones de entrada
        if not node_id or not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"node_id debe ser un string no vacío, recibido: {node_id}")
        
        if reason is None:
            reason = "Training failure"
        if not isinstance(reason, str):
            reason = str(reason)
        
        max_retries = 10
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    time.sleep(random.uniform(0.01, 0.05))
                
                cursor = self.conn.cursor()
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                
                # Get current quarantine_count
                cursor.execute(
                    "SELECT quarantine_count, total_failures FROM nodes WHERE node_id = ?",
                    (node_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    raise ValueError(f"Node {node_id} does not exist")
                
                # Fase 9: Manejo robusto de NULLs y validación de rangos
                current_count = row['quarantine_count']
                if current_count is None:
                    current_count = 0
                else:
                    try:
                        current_count = int(current_count)
                        if current_count < 0:
                            logger.warning(f" quarantine_count negativo para {node_id}: {current_count}, corrigiendo a 0")
                            current_count = 0
                    except (ValueError, TypeError):
                        logger.warning(f" quarantine_count inválido para {node_id}: {current_count}, usando 0")
                        current_count = 0
                
                current_failures = row['total_failures']
                if current_failures is None:
                    current_failures = 0
                else:
                    try:
                        current_failures = int(current_failures)
                        if current_failures < 0:
                            logger.warning(f" total_failures negativo para {node_id}: {current_failures}, corrigiendo a 0")
                            current_failures = 0
                    except (ValueError, TypeError):
                        logger.warning(f" total_failures inválido para {node_id}: {current_failures}, usando 0")
                        current_failures = 0
                
                new_count = current_count + 1
                
                # Fase 9: Validación de límite máximo (safety check)
                MAX_QUARANTINE_COUNT = 10  # Safety limit
                if new_count > MAX_QUARANTINE_COUNT:
                    logger.error(
                        f"quarantine_count excede límite máximo ({MAX_QUARANTINE_COUNT}) para {node_id}, "
                        f"usando {MAX_QUARANTINE_COUNT}"
                    )
                    new_count = MAX_QUARANTINE_COUNT
                
                # Determine if we should set forced_root
                forced_root = new_count >= 3
                forced_root_reason = None
                if forced_root:
                    forced_root_reason = f"Reached quarantine limit ({new_count})"
                
                # Update quarantine fields
                from datetime import datetime
                now = datetime.utcnow().isoformat()
                
                cursor.execute(
                    """UPDATE nodes 
                       SET quarantine_count = ?,
                           last_quarantine_exit = NULL,
                           forced_root = ?,
                           forced_root_reason = ?,
                           total_failures = ?,
                           last_failure_time = ?
                       WHERE node_id = ?""",
                    (new_count, 1 if forced_root else 0, forced_root_reason, 
                     current_failures + 1, now, node_id)
                )
                
                if cursor.rowcount == 0:
                    raise ValueError(f"Node {node_id} does not exist")
                
                self.conn.commit()
                
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                
                if cursor:
                    cursor.close()
                
                if forced_root:
                    logger.error(
                        f"🔒 Nodo {node_id} alcanzó límite de cuarentenas ({new_count}), "
                        f"marcado como ROOT permanente"
                    )
                    # Add history entry for forced root
                    try:
                        # Get current inheritance type
                        cursor.execute(
                            "SELECT inheritance_type FROM nodes WHERE node_id = ?",
                            (node_id,)
                        )
                        current_row = cursor.fetchone()
                        from_type = current_row['inheritance_type'] if current_row else 'ROOT'
                        
                        self.add_inheritance_history_entry(
                            node_id=node_id,
                            from_type=from_type or 'ROOT',
                            to_type='ROOT',
                            reason='FORCED_ROOT',
                            metadata={'quarantine_count': new_count, 'reason': reason}
                        )
                    except Exception as hist_error:
                        logger.warning(f" Error agregando historial para FORCED_ROOT en {node_id}: {hist_error}")
                else:
                    logger.warning(
                        f"🔒 Nodo {node_id} entra en cuarentena (count={new_count}, razón: {reason})"
                    )
                    # Add history entry for quarantine entry
                    try:
                        cursor.execute(
                            "SELECT inheritance_type FROM nodes WHERE node_id = ?",
                            (node_id,)
                        )
                        current_row = cursor.fetchone()
                        from_type = current_row['inheritance_type'] if current_row else 'ROOT'
                        
                        self.add_inheritance_history_entry(
                            node_id=node_id,
                            from_type=from_type or 'ROOT',
                            to_type=from_type or 'ROOT',  # Type doesn't change, just quarantine status
                            reason='QUARANTINE_ENTRY',
                            metadata={'quarantine_count': new_count, 'reason': reason, 'failure_type': failure_type}
                        )
                    except Exception as hist_error:
                        logger.warning(f" Error agregando historial para QUARANTINE_ENTRY en {node_id}: {hist_error}")
                
                return new_count
                
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when incrementing quarantine for {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error incrementing quarantine for {node_id}: {e}")
                    raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Error incrementing quarantine for {node_id}: {e}")
                raise
        
        # Should not reach here
        raise RuntimeError(f"Failed to increment quarantine for {node_id} after {max_retries} attempts")
    
    def exit_quarantine(self, node_id: str) -> None:
        """
        Mark node as exiting quarantine.
        
        Added: Fase 4 (2025-01-XX)
        
        Logic:
        1. Set last_quarantine_exit = datetime.utcnow()
        
        Args:
            node_id: Node ID
        """
        max_retries = 10
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            cursor = None
            try:
                if not self.conn:
                    raise sqlite3.Error("Database connection is closed")
                
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if attempt > 0:
                    base_wait = min(retry_delay * (2 ** (attempt - 1)), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    time.sleep(wait_time)
                else:
                    time.sleep(random.uniform(0.01, 0.05))
                
                cursor = self.conn.cursor()
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                
                from datetime import datetime
                now = datetime.utcnow().isoformat()
                
                cursor.execute(
                    """UPDATE nodes 
                       SET last_quarantine_exit = ?
                       WHERE node_id = ? AND quarantine_count > 0""",
                    (now, node_id)
                )
                
                if cursor.rowcount == 0:
                    # Node might not be in quarantine, that's okay
                    logger.debug(f"Node {node_id} is not in quarantine, skipping exit")
                else:
                    self.conn.commit()
                    logger.info(f"Nodo {node_id} sale de cuarentena")
                
                try:
                    cursor.execute("PRAGMA busy_timeout = 20000")
                except:
                    pass
                
                if cursor:
                    cursor.close()
                return
                
            except sqlite3.OperationalError as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                try:
                    self.conn.rollback()
                except:
                    pass
                
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    base_wait = min(retry_delay * (2 ** attempt), 4.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.debug(
                        f"Database locked when exiting quarantine for {node_id}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"SQLite error exiting quarantine for {node_id}: {e}")
                    raise
            except Exception as e:
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                logger.error(f"Error exiting quarantine for {node_id}: {e}")
                raise

    # --- Evaluation Metrics Operations ---
    
    def save_evaluation_metrics(
        self,
        node_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save evaluation metrics for a node.
        
        Args:
            node_id: Node ID
            metrics: Dictionary of metric_name -> value
            metadata: Optional metadata dictionary
        """
        if not node_id or not metrics:
            return
            
        try:
            cursor = self.conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            
            for name, value in metrics.items():
                cursor.execute(
                    "INSERT INTO evaluation_metrics (node_id, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
                    (node_id, name, value, metadata_json)
                )
            
            self._maybe_commit()
            logger.debug(f"Saved {len(metrics)} evaluation metrics for node {node_id}")
            
        except sqlite3.Error as e:
            logger.error(f"Error saving evaluation metrics for {node_id}: {e}")
            raise

    def get_evaluation_metrics(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all evaluation metrics for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of metric entries
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT metric_name, metric_value, evaluation_timestamp, metadata FROM evaluation_metrics WHERE node_id = ? ORDER BY evaluation_timestamp DESC",
                (node_id,)
            )
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append({
                    "metric_name": row["metric_name"],
                    "metric_value": row["metric_value"],
                    "timestamp": row["evaluation_timestamp"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                })
            return metrics
        finally:
            if cursor:
                cursor.close()

    # ========================================================================
    # Training Sessions Management (2026-01-26)
    # ========================================================================
    def register_session(self, session_id: str, dataset_name: str, dataset_path: str) -> None:
        """
        Registers a training session and its associated dataset.
        
        Args:
            session_id: Unique short ID for the session (e.g., 's7A9x')
            dataset_name: Human-readable name (e.g., 'arxiv')
            dataset_path: Full path to the dataset file
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO training_sessions (session_id, dataset_name, dataset_path) VALUES (?, ?, ?)",
                (session_id, dataset_name, dataset_path)
            )
            self.conn.commit()
            logger.debug(f"Session registered: {session_id} -> {dataset_name} ({dataset_path})")
        except Exception as e:
            logger.error(f"Error registering session {session_id}: {e}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieves dataset information for a given session ID.
        
        Args:
            session_id: The short session ID to lookup
            
        Returns:
            Dict with 'dataset_name' and 'dataset_path', or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT dataset_name, dataset_path FROM training_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "dataset_name": row["dataset_name"],
                    "dataset_path": row["dataset_path"]
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving session info for {session_id}: {e}")
            return None

    # ========================================================================
    # Labeling Methods (Gemini Integration)
    # ========================================================================
    
    def get_nodes_without_label(self, limit: int = 100) -> List[str]:
        """Get nodes that are trained but have no semantic label."""
        try:
            cursor = self.conn.cursor()
            # Prioritize trained nodes first, as they are "mature"
            # But also consider nodes with enough mass if training is deferred
            cursor.execute('''
                SELECT node_id FROM nodes 
                WHERE (label IS NULL OR label = '')
                AND (is_trained = 1 OR mass > 5)
                ORDER BY mass DESC
                LIMIT ?
            ''', (limit,))
            return [row['node_id'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting nodes without label: {e}")
            return []

    def set_node_label(self, node_id: str, label: str) -> None:
        """Set the semantic label for a node."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('UPDATE nodes SET label = ? WHERE node_id = ?', (label, node_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error setting label for node {node_id}: {e}")

    def close(self) -> None:
        """Close the database connection."""
        if self._update_counter > 0:
            self.conn.commit()
            logger.debug("Final commit before closing")
        self.conn.close()
        logger.info("Repository closed")
