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

# Device detection: prioritize CUDA, then MPS, fallback to CPU
# Note: MPS may have issues with BFloat16, so we can force CPU if needed
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    # MPS available but may have dtype compatibility issues
    # For now, use MPS if available (can be forced to "cpu" if issues occur)
    DEVICE = "mps"
else:
    DEVICE = "cpu"

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
# Phase 2: Training Parameters
# ============================================================================

TRAINING_THRESHOLD = 20  # Minimum mass (number of embeddings) to trigger training
                         # When a KnowledgeNode reaches this mass, it will be queued for training
                         # Range: 20-50 for testing/development, 50-100 for production
                         # Lower values = faster training triggers, but less stable gradients
MAX_CONCURRENT_TRAINING = 2  # Maximum concurrent training tasks (limited by ThreadPoolExecutor)
MIN_TRAINING_TEXTS = 10  # Minimum number of texts required for stable training
                          # Too few texts (e.g., < 10) can lead to unstable gradients and overfitting
                          # This ensures training quality and prevents gradient instability

# ============================================================================
# Knowledge Nodes Fusion Parameters
# ============================================================================

FUSION_SIMILARITY_THRESHOLD = 0.85  # Minimum cosine similarity to fuse KNs (default: 0.85)
FUSION_MIN_MASS = 10  # Maximum mass to consider a KN as "Small Stable" (adjusted: 10, based on average mass 9.2)
FUSION_MAX_VARIANCE = 0.5  # Maximum variance to consider a KN as "Stable" (default: 0.5)
FUSION_VARIANCE_INCREASE_THRESHOLD = 0.1  # Maximum variance increase allowed after fusion (default: 0.1)

# ============================================================================
# Layer 2 & 3 Identity Parameters
# ============================================================================
# (These are used to initialize KnowledgeNodes even if Transformer is on standby)

MODEL_BASE_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Base model for embeddings
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Layer 2 Model (Open source, small for testing)
LOAD_IN_8BIT = False  # Set to True only if bitsandbytes is fully working (Linux/CUDA often required)
LORA_RANK_DEFAULT = 4  # Default rank for LoRA adapters

# ============================================================================
# Post-Processing Parameters (Layer 3 Judgment)
# ============================================================================

PP_HIGH_THRESHOLD = 0.85  # Minimum confidence to reinforce a node
PP_LOW_THRESHOLD = 0.50   # Maximum confidence to suggest NEW_BUFFER (rejection)
PP_ETA = 0.05             # Learning rate for variance adjustment

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
