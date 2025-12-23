"""
Repository for Clustering Layer.

Bridge to SQLite for storing statistical signatures and buffers.
Stores FP16 tensors as BLOBs for efficiency.
"""

import sqlite3
import numpy as np
import torch
import logging
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

    # --- Physical Memory Management (Embeddings) ---

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
            logger.debug(f"Buffer creado: {buffer_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Buffer {buffer_id} ya existe")
            raise
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al crear buffer {buffer_id}: {e}")
            raise

    def add_to_buffer(self, buffer_id: str, embedding: torch.Tensor) -> None:
        """
        Agrega un embedding a un buffer y actualiza el centroide incrementalmente.
        
        Actualización incremental del centroide:
        - Si es el primer embedding (n=0): C_new = E_new
        - Si n > 0: C_new = (C_old * n + E_new) / (n + 1)
        
        Args:
            buffer_id: ID del buffer
            embedding: Tensor FP16 del embedding
        
        Raises:
            ValueError: Si los parámetros son inválidos
            sqlite3.Error: Si hay error en la base de datos
        """
        # Validación
        if not buffer_id or not isinstance(buffer_id, str):
            raise ValueError(f"buffer_id debe ser un string no vacío, recibido: {buffer_id}")
        
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
            logger.debug(f"Embedding agregado a buffer: {buffer_id} (size={current_size + 1})")
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al agregar embedding a buffer {buffer_id}: {e}")
            raise

    def get_buffer_size(self, buffer_id: str) -> int:
        """
        Obtiene el tamaño de un buffer.
        
        Args:
            buffer_id: ID del buffer
        
        Returns:
            Tamaño del buffer (0 si no existe)
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
