"""
Centralized configuration for Clustering Layer.

All configurations must be here. NO hardcoded values in the code.
"""

import torch
from pathlib import Path

# ============================================================================
# Parameters Technical Configuration
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
SAVE_BATCH_SIZE = 200  # How many updates before committing to SQLite to avoid blocking the flow.
PROCESS_BATCH_SIZE = 64 # Size of batches for main processing loop (Orchestrator.process_batch)
                        # Higher = Better GPU utilization (>50 emb/s)
                        # Lower = Lower VRAM usage


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

TRAINING_THRESHOLD = 12  # Minimum mass to trigger training (increased for quality)
                         # Increased to 12 (from 6) to ensure higher knowledge density.
                         # Micro-nodes will be forcibly fused until they reach this mass.
MAX_CONCURRENT_TRAINING = 1  # Maximum concurrent training tasks (limited by ThreadPoolExecutor)
                             # CRITICAL: Set to 1 because TransformerBase singleton is not thread-safe for 
                             # concurrent training (PeftModel modifies the shared base model).
                             # Future improvement: Use multiple processes or separate model instances.
MIN_TRAINING_TEXTS = 12  # Minimum texts required for stable training
                         # Elevated to 12 (from 6). Forced fusion will act on nodes <= 12.

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

FUSION_SIMILARITY_THRESHOLD = 0.60  # Ultra-aggressive fusion (was 0.75)
FUSION_MIN_MASS = 15  # Maximum mass to consider a KN as "Small Stable"
FUSION_MAX_VARIANCE = 0.5  # Maximum variance to consider a KN as "Stable" (default: 0.5)
FUSION_VARIANCE_INCREASE_THRESHOLD = 0.1  # Maximum variance increase allowed after fusion (default: 0.1)

# ============================================================================
# Inheritance & Titan Selection Parameters (2025-01-XX)
# ============================================================================

INHERITANCE_ENABLED = True  # Master flag to enable/disable inheritance
TITAN_MIN_MASS = 20  # Minimum mass to be a "Titan" (Parent candidate)
                     # Recommended: TRAINING_THRESHOLD * 2
TITAN_MAX_VARIANCE = 0.5  # Maximum variance to be reliable parent
                          # Should match FUSION_MAX_VARIANCE for consistnecy
TITAN_SIMILARITY_THRESHOLD = 0.55  # Lowered (was 0.65) to maximize adoption of micro-fragments.

PROGRESSIVE_ADOPTION_THRESHOLD = 0.60  # Minimum similarity for Progressive Adoption (Fase 7)
                                       # Lower threshold allows more adoptions during training

# Inheritance Training Parameters (2025-01-XX)
LIFE_INSURANCE_THRESHOLD = 2.0  # Abort training if loss > threshold * baseline (Relaxed for safety)
INHERITANCE_L2_BASE_LAMBDA = 0.01  # Base lambda for L2 regularization (multiplied by (1 + similarity))
INHERITANCE_LR_MULTIPLIER = 0.3  # Learning rate multiplier for children (0.3x = 30% of normal LR)

# ============================================================================
# Layer 2 & 3 Identity Parameters
# ============================================================================
# (These are used to initialize KnowledgeNodes even if Transformer is on standby)

MODEL_BASE_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Base model for embeddings

# ============================================================================
# Adaptive Model Selection (2025-01-25)
# ============================================================================
# Select LLM model based on available hardware:
# - CUDA (NVIDIA GPU): Use TinyLlama 1.1B for best quality
# - CPU: Use GPT-2 Small for speed (testing/development)
# ============================================================================
if DEVICE == "cuda":
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params - best quality
elif DEVICE == "mps":
    LLM_MODEL_ID = "gpt2-medium"  # 355M params - balance quality/speed
else:
    LLM_MODEL_ID = "gpt2"  # 124M params - fast for CPU testing
LOAD_IN_8BIT = False  # Set to True only if bitsandbytes is fully working (Linux/CUDA often required)
LORA_RANK_DEFAULT = 4  # Default rank for LoRA adapters

# ============================================================================
# IMPORTANT: LoRA Target Modules - Model-Specific (2025-01-25)
# ============================================================================
# Different model architectures use different names for attention layers.
# Using wrong target modules causes training to fail silently!
#
# - LLaMA/TinyLlama: Uses "q_proj", "k_proj", "v_proj", "o_proj"
# - GPT-2: Uses "c_attn" (combined QKV) and "c_proj" (output)
#
# DO NOT hardcode target modules - always use LORA_TARGET_MODULES!
# ============================================================================
def _get_lora_target_modules():
    """
    IMPORTANT: Returns correct LoRA target modules based on selected model.
    
    This function MUST be called to get target modules - DO NOT hardcode them!
    Different models have different layer names:
    - LLaMA/TinyLlama: q_proj, k_proj, v_proj, o_proj
    - GPT-2: c_attn, c_proj
    
    Hardcoding the wrong modules causes silent training failures.
    """
    if "gpt2" in LLM_MODEL_ID.lower():
        return ["c_attn", "c_proj"]  # GPT-2 architecture
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA architecture

LORA_TARGET_MODULES = _get_lora_target_modules()

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

# ============================================================================
# Evaluation Parameters (Phase 3: Model Evaluation)
# ============================================================================

EVAL_VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation (0.0-1.0)
                              # 0.0 = use all data for training, no validation split
                              # 0.2 = 80% training, 20% validation (recommended)
                              # Set to 0.0 to disable validation split (use all data for training)

EVAL_METRICS = ["perplexity", "validation_loss"]  # Metrics to calculate
                                                   # Options: "perplexity", "validation_loss", "thematic_coherence"
                                                   # "perplexity": Standard language model metric
                                                   # "validation_loss": Loss on validation set
                                                   # "thematic_coherence": Semantic similarity with node embeddings

TRAINING_BATCH_SIZE = 32  # Batch size for training (optimized for GPU efficiency)
                         # REDUCED from 64 to 8 due to MPS memory constraints (98.7% usage)
                         # Higher batch sizes (64, 128, 256) maximize GPU utilization
                         # Lower batch sizes (4, 8, 16) waste GPU potential with overhead
                         # Adjust based on available VRAM: 64 for 8GB+, 128 for 16GB+, 256 for 24GB+
                         # NOTE: For MPS with high memory usage, use 4-8 to prevent OOM errors

TRAINING_LEARNING_RATE_BASE = 1e-4  # Base learning rate for batch_size=4 (reference)
                                     # Learning rate scales automatically with batch size
                                     # Formula: LR = LR_base * sqrt(batch_size / 4)
                                     # This maintains effective learning rate per example
                                     # For batch_size=64: LR ≈ 4e-4 (4x increase)
                                     # For batch_size=128: LR ≈ 5.66e-4 (5.66x increase)

EVAL_BATCH_SIZE = 4  # Batch size for evaluation (can be smaller than training)

EVAL_MIN_TEXTS = 1  # Minimum texts required for evaluation
                    # Lowered to 1 (from 3) to prevent skipping small nodes in reports (2025-01-27)

EVAL_SAVE_RESULTS = True  # Save evaluation results to database
                          # If True, creates/updates evaluation_metrics table

EVAL_REPORT_FORMAT = "both"  # Report format: "console", "file", or "both"
                             # "console": Print to console only
                             # "file": Save to file only
                             # "both": Print and save to file

# ============================================================================
# Google Gemini Integration (Labeling Agent)
# ============================================================================
import os
ENABLE_GEMINI_LABELING = os.getenv("ENABLE_GEMINI_LABELING", "False").lower() == "true"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Standard Google Env Var
GEMINI_MODEL = "gemini-1.5-flash" # Cost-effective, high context model
LABELING_BATCH_SIZE = 5 # Number of representative texts to send to Gemini for context
