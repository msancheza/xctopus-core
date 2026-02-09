"""
Centralized configuration for Clustering Layer.

All configurations must be here. NO hardcoded values in the code.
"""

import os
import torch
from pathlib import Path

# Hugging Face Authentication (Required for gated models like Llama 3.1)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# ============================================================================
# Path Configuration (Standardized for Studio Root)
# ============================================================================
# settings.py is in xctopus/src/xctopus/
_SETTINGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = _SETTINGS_DIR.parent.parent.parent  # 3 levels to reach Studio Root

# DTYPE Selection: float16 is standard for CUDA/MPS, but CPU/TPU often need float32 for stability
def _get_optimal_dtype(device_name: str) -> torch.dtype:
    if device_name in ["cuda", "mps"]:
        return torch.float16
    return torch.float32  # Safer for CPU and some TPU-v2/v3 environments

# Device detection: prioritize TPU (XLA), then CUDA, then MPS, fallback to CPU
# On Lightning.ai, TPU environments might report CUDA available on the host, 
# so we check TPU first for "Fidelity to Hardware".
try:
    import torch_xla.core.xla_model as xm
    DEVICE = "tpu"
except ImportError:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

DTYPE = _get_optimal_dtype(DEVICE)

# Safeguard for quantization (requires bitsandbytes and accelerate)
def _check_quantization_availability(mode="4bit"):
    """
    Checks if bitsandbytes and accelerate are available for quantization.
    Defaulting to 4-bit as it is much safer for 16GB GPUs (T4).
    """
    if DEVICE != "cuda":
        return False
    try:
        import bitsandbytes
        import accelerate
        return True
    except ImportError:
        return False

# ============================================================================
# Similarity and Routing Parameters (FilterBayesianNode)
# ============================================================================

S_MIN = 0.55  # Minimum Cosine Similarity to accept an embedding in a KN.
               # Adjusted: 0.48 (was 0.68) to bridge the Semantic Gap between 
               # JSON SEEDs and Narrative Evidence flow.
LAMBDA_FACTOR = 0.0  # Mass contribution to routing score (Gravity)
                       # Setting to 0.0 for pure semantic routing (Zero-G Mode)
                       # Prevents large nodes from "sucking" irrelevant queries.
THRESH_DECAY = 0.20  # Dynamic threshold decay factor for FilterBayesian
                      # Formula: dynamic_threshold = S_MIN - (THRESH_DECAY / log1p(mass))
                      # Higher values = more permissive for small nodes
                      # Lower values = stricter even for small nodes
                      # Range: 0.10-0.20 recommended (Adjusted to 0.20 for Long Tail Retrieval)
THRESH_MIN_LOG = 0.1  # Minimum value for log1p(mass) in dynamic threshold calculation
                       # Prevents division by zero and ensures numerical stability
                       # Should be > 0 and < log1p(1) ≈ 0.69

# [GOLD]: Structural Maturity Score (SMS) Weights (Fase 7)
SMS_WEIGHT_COHERENCE = 5.0  # Dominio de la coherencia (1 - Variance)
SMS_WEIGHT_MASS = 1.0       # Contribución de la masa crítica (log1p mass)
SMS_DEFAULT_THRESHOLD = 4.0 # Minimum SMS to be considered "Mature"

# ============================================================================
# Data Structure Parameters
# ============================================================================

# [ADAPTATIVIDAD 384d/768d]: La dimensión ahora se detecta dinámicamente. 
# Se mantiene este valor solo como fallback o para inicialización por defecto.
DEFAULT_EMBEDDING_DIM = 768  # all-mpnet-base-v2: 768 | MiniLM: 384 | SapBERT: 768
EMBEDDING_DIM = DEFAULT_EMBEDDING_DIM 
BUFFER_THRESHOLD = 3  # How many embeddings a Buffer needs before promoting to KnowledgeNode.
                      # Note: Reduced value for fast concept validation. Once the
                      # KnowledgeNode is born, its LocalFilter will refine semantic purity.

# ============================================================================
# Persistence Parameters (KN Repository)
# ============================================================================

DB_PATH = PROJECT_ROOT_DIR / "knowledge_base.sqlite"  # Path to SQLite database
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

TRAINING_THRESHOLD = 2  # Minimum mass to trigger training (Lowered to 2 for Long Tail)
                         # [RISK MODE]: Adjusted to 2 (was 12) to capture rare diseases/genes.
                         # Allows training on minimal evidence.
MAX_CONCURRENT_TRAINING = 1  # Maximum concurrent training tasks (limited by ThreadPoolExecutor)
                             # CRITICAL: Set to 1 because TransformerBase singleton is not thread-safe for 
                             # concurrent training (PeftModel modifies the shared base model).
                             # Future improvement: Use multiple processes or separate model instances.
MIN_TRAINING_TEXTS = 2   # Minimum texts required for stable training
                         # [RISK MODE]: Adjusted to 2 (was 12). 

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
# Phase 4: Atomic Injection Parameters (2026-02-05)
# ============================================================================
ATOMIC_SYSTEM_PROMPT = (
    "Identity: XCTOPUS Genomic AI. "
    "Task: Extract clinical data from [RAW_ATOMIC_EVIDENCE] and return a validation JSON. "
    "Format: Return ONLY a JSON object. No intro, no markdown text outside JSON. "
    "Example Output: "
    "{ \"gene\": \"BRCA1\", \"variant\": \"c.4327C>T\", \"significance\": \"Pathogenic\", \"cadd_score\": \"25.4\", \"phenotype\": \"Hereditary Breast Cancer\", \"analysis\": \"The variant is classified as pathogenic...\" } "
    "JSON Structure: "
    "{ "
    "  \"gene\": \"[Gene Symbol]\", "
    "  \"variant\": \"[HGVS or Genomic Coord]\", "
    "  \"significance\": \"[Clinical Significance]\", "
    "  \"cadd_score\": \"[Score or null]\", "
    "  \"phenotype\": \"[Associated Disease]\", "
    "  \"analysis\": \"[Concise bio-clinical interpretation based on evidence]\" "
    "} "
    "Rule: If a field is not in evidence, use null. DO NOT Hallucinate. "
)

# ============================================================================
# Knowledge Nodes Fusion Parameters
# ============================================================================

FUSION_SIMILARITY_THRESHOLD = 0.90  # Ultra-aggressive fusion (was 0.75)
FUSION_MIN_MASS = 15  # Maximum mass to consider a KN as "Small Stable"
FUSION_MAX_VARIANCE = 0.5  # Maximum variance to consider a KN as "Stable" (default: 0.5)
FUSION_VARIANCE_INCREASE_THRESHOLD = 0.05  # Maximum variance increase allowed after fusion (default: 0.1)

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
INHERITANCE_L2_BASE_LAMBDA = 0.10  # [GOLD]: Aumento de regularización para anclaje de pesos.
INHERITANCE_LR_MULTIPLIER = 0.1    # [GOLD]: Fine-tuning conservador para herencia.

# ============================================================================
# Layer 2 & 3 Identity Parameters
# ============================================================================
# (These are used to initialize KnowledgeNodes even if Transformer is on standby)

# [UPGRADE 2026-02-06]: MPNet for higher precision similarity.
# Note: all-mpnet-base-v2 has 768 dimensions. La arquitectura lo detectará.
MODEL_BASE_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================================================
# Adaptive Model Selection (Hardware Optimized)
# ============================================================================
# STANDARD: Llama 3.1 8B for CUDA (Professional grade, low hallucination)
# LIGHTWEIGHT: TinyLlama for MPS/CPU (Development/Compatibility)
# ============================================================================
if DEVICE in ["cuda", "tpu"]:
    # For TinyLlama, we don't need quantization on a 16GB GPU.
    # Disabling it eliminates bitsandbytes/accelerate conflicts.
    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
else:
    # LIGHTWEIGHT Fallback
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LOAD_IN_8BIT = False
    LOAD_IN_4BIT = False

# ============================================================================
# LoRA Weights Management (Isolation by Model)
# ============================================================================
# IMPORTANT: Weights trained for TinyLlama ARE NOT COMPATIBLE with Llama 3.
# We isolate paths to prevent loading errors during hybrid deployment.
# ============================================================================
def _get_lora_path():
    """Returns specialized path for LoRA weights based on active model."""
    model_slug = LLM_MODEL_ID.split("/")[-1].replace(".", "_")
    path = PROJECT_ROOT_DIR / "lora_weights" / model_slug
    path.mkdir(parents=True, exist_ok=True)
    return path

LORA_PATH = _get_lora_path()

LORA_RANK_DEFAULT = 16  # Default rank for LoRA adapters (Optimized for Standard nodes)
LORA_RANK_MICRO = 8    # Rank for small nodes (mass < 10) to prevent memorization
LORA_RANK_STANDARD = 16 # Rank for medium nodes (10-100)
LORA_RANK_TITAN = 32    # Rank for large clusters (mass > 100) for high resolution

def get_adaptive_lora_rank(mass: int) -> int:
    """
    Returns the optimal LoRA rank based on node mass.
    """
    if mass < 10:
        return LORA_RANK_MICRO
    elif mass <= 100:
        return LORA_RANK_STANDARD
    else:
        return LORA_RANK_TITAN

# ============================================================================
# IMPORTANT: LoRA Target Modules - Model-Specific (2025-01-25)
# ============================================================================
# Different model architectures use different names for attention layers.
# Using wrong target modules causes training to fail silently!
#
# - LLaMA/Qwen-based: Uses "q_proj", "k_proj", "v_proj", "o_proj"
# - GPT-2: Uses "c_attn" (combined QKV) and "c_proj" (output)
#
# DO NOT hardcode target modules - always use LORA_TARGET_MODULES!
# ============================================================================
def _get_lora_target_modules():
    """
    IMPORTANT: Returns correct LoRA target modules based on selected model.
    """
    m_id = LLM_MODEL_ID.lower()
    if "gpt2" in m_id:
        return ["c_attn", "c_proj"]
    # Both TinyLlama and Llama-3 use standard Llama projections
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

LORA_TARGET_MODULES = _get_lora_target_modules()

# ============================================================================
# Post-Processing Parameters (Layer 3 Judgment)
# ============================================================================

PP_HIGH_THRESHOLD = 0.85  # Minimum confidence to reinforce a node
PP_LOW_THRESHOLD = 0.50   # Maximum confidence to suggest NEW_BUFFER (rejection)
PP_ETA = 0.05             # Learning rate for variance adjustment

# ============================================================================
# Logging Parameters
# ============================================================================

LOG_DIR = PROJECT_ROOT_DIR / "logs"  # Fixed: Targets the Studio-level /logs folder
LOG_FILE = LOG_DIR / "xctopus.log"  # Main log file
LOG_MAX_BYTES = 100 * 1024 * 1024  # 100 MB per file
LOG_BACKUP_COUNT = 10  # Keep 10 backup files
LOG_LEVEL_FILE = "DEBUG"   # Detailed logs to file
LOG_LEVEL_CONSOLE = "WARNING" # High-level summaries to terminal

# ============================================================================
# Evaluation Parameters (Phase 3: Model Evaluation)
# ============================================================================

EVAL_VALIDATION_SPLIT = 0.0  # Fraction of data to use for validation (0.0-1.0)
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
# [GOLD]: Structural Health & Mental Age (PPL Control)
# ============================================================================
PPL_MAX_THRESHOLD = 15.0   # Upper limit for "confused" nodes (pruning candidate)
PPL_SPLIT_THRESHOLD = 25.0 # Critical limit for immediate split of "obese" nodes
MIN_TEXTS_FOR_SPLIT = 15 # Minimum mass to allow splitting (Raised to 15 for stability)
TRAINING_EPOCHS = 5      # Default training epochs (Raised to 5 for better memorization)

# ============================================================================
# Google Gemini Integration (Labeling Agent)
# ============================================================================
import os
ENABLE_GEMINI_LABELING = os.getenv("ENABLE_GEMINI_LABELING", "False").lower() == "true"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Standard Google Env Var
GEMINI_MODEL = "gemini-1.5-flash" # Cost-effective, high context model
LABELING_BATCH_SIZE = 5 # Number of representative texts to send to Gemini for context