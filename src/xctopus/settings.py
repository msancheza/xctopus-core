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

S_MIN = 0.60  # Minimum Cosine Similarity to accept an embedding in a KN.
               # Adjusted: 0.60 (was 0.65) to allow more embeddings to be assigned to existing nodes
               # Note: In MiniLM, 0.60-0.65 already indicates strong thematic relationship.
               # Higher values (0.75+) are too restrictive for diverse datasets.
LAMBDA_FACTOR = 0.2  # "Critical Mass" strength. Higher values mean larger nodes have more attraction.
                      # Adjusted: 0.2 (was 0.1) to increase gravitational pull of existing nodes
THRESH_DECAY = 0.15  # Dynamic threshold decay factor for FilterBayesian
                      # Formula: dynamic_threshold = S_MIN - (THRESH_DECAY / log1p(mass))
                      # Higher values = more permissive for small nodes
                      # Lower values = stricter even for small nodes
                      # Range: 0.10-0.20 recommended
THRESH_MIN_LOG = 0.1  # Minimum value for log1p(mass) in dynamic threshold calculation
                       # Prevents division by zero and ensures numerical stability
                       # Should be > 0 and < log1p(1) ≈ 0.69

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

# REFRESH_INTERVAL = 10  # DEPRECATED: No longer used
#                         # FilterBayesian signatures are now updated immediately after each
#                         # embedding is accepted (see orchestrator.py _process_kn_update)
#                         # This ensures centroids evolve in real-time for better routing decisions

# ============================================================================
# Phase 2: Training Parameters
# ============================================================================

TRAINING_THRESHOLD = 10  # Minimum mass (number of embeddings) to trigger training
                         # When a KnowledgeNode reaches this mass, it will be queued for training
                         # Adjusted: 10 (was 20) based on actual node mass distribution (avg ~6)
                         # Lower values = faster training triggers, but less stable gradients
                         # Range: 8-15 for testing/development, 15-30 for production
MAX_CONCURRENT_TRAINING = 1  # Maximum concurrent training tasks (limited by ThreadPoolExecutor)
                             # CRITICAL: Set to 1 because TransformerBase singleton is not thread-safe for 
                             # concurrent training (PeftModel modifies the shared base model).
                             # Future improvement: Use multiple processes or separate model instances.
MIN_TRAINING_TEXTS = 6  # Minimum number of texts required for stable training
                         # Adjusted: 6 (was 10) to match realistic node sizes
                         # Too few texts (e.g., < 5) can lead to unstable gradients and overfitting
                         # This ensures training quality and prevents gradient instability

# ============================================================================
# Deferred Training Parameters (Training Delta)
# ============================================================================

TRAINING_DELTA_MULTIPLIER = 2.0  # Re-train when mass doubles (10→20→40→80...)
                                  # Set to 0.0 to disable re-training based on mass doubling
                                  # This enables incremental learning without saturating GPU

TRAINING_DELTA_TIMEOUT_DAYS = 30  # Re-train if last training was > 30 days ago
                                   # Set to 0 to disable timeout-based re-training
                                   # Ensures nodes are updated even if they grow slowly

MAX_TRAINING_TEXTS = 100  # Maximum number of texts to use for training
                          # If node has more, use last N texts (most recent)
                          # Set to 0 for no limit (use all available)
                          # Prevents training from being too slow with very large nodes

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
