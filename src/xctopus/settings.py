"""
Centralized configuration for Clustering Layer.

All configurations must be here. NO hardcoded values in the code.
"""

import torch
from pathlib import Path

# ============================================================================
# Technical Configuration
# ============================================================================

DTYPE = torch.float16  # Force medium precision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# ============================================================================
# Similarity and Routing Parameters (FilterBayesianNode)
# ============================================================================

S_MIN = 0.65  # Minimum Cosine Similarity to accept an embedding in a KN.
               # Note: In MiniLM, 0.60-0.65 already indicates strong thematic relationship.
               # Higher values (0.75+) are too restrictive for diverse datasets.
LAMBDA_FACTOR = 0.1  # "Critical Mass" strength. Higher values mean larger nodes have more attraction.

# ============================================================================
# Data Structure Parameters
# ============================================================================

EMBEDDING_DIM = 384  # Dimension of your vectors (e.g., 384 for All-MiniLM-L6-v2).
BUFFER_THRESHOLD = 3  # How many embeddings a Buffer needs before promoting to KnowledgeNode.
                      # Note: Reduced value for fast concept validation. Once the
                      # KnowledgeNode is born, its LocalFilter will refine semantic purity.

# ============================================================================
# Persistence Parameters (KN Repository)
# ============================================================================

DB_PATH = "knowledge_base.sqlite"  # Path to SQLite database
SAVE_BATCH_SIZE = 10  # How many updates before committing to SQLite to avoid blocking the flow.

# ============================================================================
# Orchestrator Parameters
# ============================================================================

REFRESH_INTERVAL = 10  # How many processed embeddings before refreshing FilterBayesian signatures

# ============================================================================
# Knowledge Nodes Fusion Parameters
# ============================================================================

FUSION_SIMILARITY_THRESHOLD = 0.85  # Minimum cosine similarity to fuse KNs (default: 0.85)
FUSION_MIN_MASS = 10  # Maximum mass to consider a KN as "Small Stable" (adjusted: 10, based on average mass 9.2)
FUSION_MAX_VARIANCE = 0.5  # Maximum variance to consider a KN as "Stable" (default: 0.5)
FUSION_VARIANCE_INCREASE_THRESHOLD = 0.1  # Maximum variance increase allowed after fusion (default: 0.1)

# ============================================================================
# Layer 1 Identity Parameters (Clustering)
# ============================================================================
# (These are used to initialize KnowledgeNodes even if Transformer is on standby)

MODEL_BASE_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Base model for embeddings
LORA_RANK_DEFAULT = 4  # Default rank for LoRA adapters

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_DIR = Path("logs")  # Directory for log files
LOG_FILE = LOG_DIR / "xctopus.log"  # Main log file
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
LOG_BACKUP_COUNT = 5  # Keep 5 backup files

# Logging levels
LOG_LEVEL_FILE = "DEBUG"  # Level for file (everything)
LOG_LEVEL_CONSOLE = "WARNING"  # Level for console (only warnings/errors)
