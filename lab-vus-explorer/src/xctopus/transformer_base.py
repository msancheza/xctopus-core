import torch
import numpy as np
import io
import os
import tempfile
import logging
import warnings
import math

# Suppress specific Hugging Face warnings about padding (2025-01-25)
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

# Also suppress via transformers logging (they use logging, not warnings)
import transformers
transformers.logging.set_verbosity_error()
from typing import List, Optional, Callable, Dict
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from .settings import (
    LLM_MODEL_ID, LOAD_IN_8BIT, LOAD_IN_4BIT, DEVICE, DTYPE, LORA_RANK_DEFAULT,
    LORA_TARGET_MODULES,  # IMPORTANT: Model-specific target modules (GPT-2 vs LLaMA)
    MIN_TRAINING_TEXTS, MAX_TRAINING_TEXTS,
    LIFE_INSURANCE_THRESHOLD,
    INHERITANCE_L2_BASE_LAMBDA,
    INHERITANCE_LR_MULTIPLIER,
    TRAINING_BATCH_SIZE,
    TRAINING_LEARNING_RATE_BASE,
    LORA_PATH,
    HF_TOKEN,
)

# Suppress transformers deprecation warning about torch.tensor()
# This is an internal transformers warning, not from our code
warnings.filterwarnings("ignore", message=".*torch_dtype.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", category=UserWarning)
# Suppress PEFT warnings (we clean adapters properly by resetting to base model)
warnings.filterwarnings("ignore", message=".*Already found a `peft_config` attribute.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*You are trying to modify a model with PEFT for a second time.*", category=UserWarning)

logger = logging.getLogger(__name__)

# Try to import safetensors (optional dependency)
try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.debug("safetensors not available, will use torch.save as fallback")


def get_dynamic_batch_size(base_batch_size: int, num_texts: int, safety_margin: float = 0.15) -> int:
    """
    Calculate dynamic batch size based on available GPU memory.
    
    This function:
    1. Detects available GPU memory (MPS or CUDA)
    2. Estimates memory needed for training
    3. Adjusts batch_size to fit available memory with safety margin
    4. Ensures batch_size is never too small (minimum 4) or too large
    
    Args:
        base_batch_size: Desired batch size from settings (e.g., 64)
        num_texts: Number of texts to train on
        safety_margin: Safety margin for memory (0.15 = 15% buffer)
    
    Returns:
        Adjusted batch_size that fits in available memory
    """
    try:
        # Get memory info based on device
        if torch.cuda.is_available():
            # CUDA: Get allocated and reserved memory
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            available = total - reserved
            memory_used_pct = (reserved / total) * 100 if total > 0 else 0
        elif DEVICE == "mps":
            # MPS: PyTorch doesn't provide direct memory API, but we can estimate
            # Based on error messages, MPS typically has ~20GB limit on Apple Silicon
            # Strategy: Use conservative approach - if we've had recent OOM errors,
            # assume high memory usage and reduce batch size
            try:
                # Try to get memory info if available (newer PyTorch versions)
                if hasattr(torch.backends.mps, "current_allocated_memory"):
                    allocated = torch.backends.mps.current_allocated_memory() / (1024**3)  # GB
                    total = 20.0  # Approximate MPS limit
                    reserved = allocated
                    available = total - reserved
                    memory_used_pct = (reserved / total) * 100 if total > 0 else 0
                else:
                    # Fallback: Use conservative estimate for MPS
                    # Since we can't detect MPS memory directly and we've seen OOM errors,
                    # assume high memory usage (90%) to be safe
                    # This will trigger aggressive batch_size reduction
                    memory_used_pct = 90.0
                    logger.warning(
                        " MPS: Cannot detect memory directly, using conservative estimate (90%) "
                        "to prevent OOM errors. Batch size will be reduced aggressively."
                    )
            except Exception as e:
                # Fallback: Assume high memory usage if we can't detect
                logger.warning(f" Cannot detect MPS memory ({e}), assuming high usage (80%)")
                memory_used_pct = 80.0
                available = 4.0  # Conservative estimate
        else:
            # CPU or unknown: Check system RAM
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            
            logger.debug(f"CPU Memory detected: Total={total_ram_gb:.1f}GB, Available={available_ram_gb:.1f}GB")
            
            # Si hay suficiente RAM (>12GB libres), permitimos batch 8 o incluso 16
            if available_ram_gb > 12:
                adjusted_batch = min(base_batch_size, 16)
                logger.debug(f"High RAM detected ({available_ram_gb:.1f}GB available), using batch_size={adjusted_batch}")
            elif available_ram_gb > 6:
                adjusted_batch = min(base_batch_size, 8)
                logger.debug(f"Moderate RAM detected ({available_ram_gb:.1f}GB available), using batch_size={adjusted_batch}")
            else:
                # Poca RAM, ser conservador
                adjusted_batch = min(base_batch_size, 4)
                logger.debug(f"Low RAM detected ({available_ram_gb:.1f}GB available), forcing conservative batch_size={adjusted_batch}")
            
            return adjusted_batch
        
        # Calculate safe batch size based on memory availability
        # If memory usage > 85%, reduce batch size aggressively
        # If memory usage > 70%, reduce batch size moderately
        # If memory usage < 70%, use base batch size
        
        if memory_used_pct > 90:
            # Critical: Very high memory usage, use minimal batch size
            # For base_batch_size=8, this gives: 8//8 = 1, but we want at least 4
            # So we use max(4, base_batch_size // 4) instead to get 2, then max with 4
            adjusted_batch = max(4, base_batch_size // 4)  # 8//4 = 2, but min is 4
            logger.warning(
                f" GPU memory usage critical ({memory_used_pct:.1f}%), "
                f"reducing batch_size from {base_batch_size} to {adjusted_batch}"
            )
        elif memory_used_pct > 85:
            # High: Reduce batch size significantly
            adjusted_batch = max(4, base_batch_size // 2)  # 8//2 = 4
            logger.warning(
                f" GPU memory usage high ({memory_used_pct:.1f}%), "
                f"reducing batch_size from {base_batch_size} to {adjusted_batch}"
            )
        elif memory_used_pct > 75:
            # Moderate: Reduce batch size moderately
            adjusted_batch = max(8, base_batch_size // 2)
            logger.info(
                f"ℹ** GPU memory usage moderate ({memory_used_pct:.1f}%), "
                f"reducing batch_size from {base_batch_size} to {adjusted_batch}"
            )
        elif memory_used_pct > 65:
            # Low-Moderate: Slight reduction
            adjusted_batch = max(16, int(base_batch_size * 0.75))
            logger.debug(
                f"GPU memory usage low-moderate ({memory_used_pct:.1f}%), "
                f"reducing batch_size from {base_batch_size} to {adjusted_batch}"
            )
        else:
            # Low: Use base batch size
            adjusted_batch = base_batch_size
            logger.debug(f"GPU memory usage low ({memory_used_pct:.1f}%), using base batch_size={base_batch_size}")
        
        # Ensure batch_size doesn't exceed number of texts
        final_batch_size = min(adjusted_batch, num_texts)
        
        # Ensure minimum batch size of 4 (for stable gradients)
        final_batch_size = max(4, final_batch_size)
        
        if final_batch_size != base_batch_size:
            logger.info(
                f"**Dynamic batch_size adjustment: {base_batch_size} → {final_batch_size} "
                f"(memory_used={memory_used_pct:.1f}%, num_texts={num_texts})"
            )
        
        return final_batch_size
        
    except Exception as e:
        logger.warning(f" Error calculating dynamic batch size: {e}, using base batch_size={base_batch_size}")
        return base_batch_size

class TransformerBase:
    """
    Layer 2 Inference Engine.
    Manages the base model (frozen) and dynamic injection of LoRA adapters.
    """
    _base_model = None
    _tokenizer = None

    def __init__(self, model_id: str = LLM_MODEL_ID, device: str = DEVICE):
        self.device = device
        self.model_id = model_id
        
        if TransformerBase._base_model is None:
            logger.info(f"Loading base model: {model_id} (Singleton)")
            # Use token=None to avoid interactive prompts
            # If you need a token for private models, set it via environment variable HF_TOKEN
            # Set local_files_only=False to allow downloads, but avoid interactive prompts
            import os
            # Disable interactive prompts and progress bars from HuggingFace
            # Disable interactive prompts and progress bars from HuggingFace
            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Suppress progress bars
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only errors, no warnings
            
            # [GOLD]: Inicialización Limpia de Tokenizador
            TransformerBase._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                padding_side='left'
            )
            
            # Compatibilidad con GPT-2 / Decoders
            if TransformerBase._tokenizer.pad_token is None:
                TransformerBase._tokenizer.pad_token = TransformerBase._tokenizer.eos_token
                TransformerBase._tokenizer.pad_token_id = TransformerBase._tokenizer.eos_token_id
            
            # Configuración de Carga Dinámica según Hardware
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": DTYPE
            }
            
            if DEVICE == "cuda":
                # CUDA: Carga directa (TinyLlama cabe en 16GB)
                load_kwargs["device_map"] = "cuda"
                logger.info(f"Cargando {model_id} en modo nativo CUDA ({DTYPE})...")
                
                # Liberar caché antes de carga pesada
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                TransformerBase._base_model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    local_files_only=False, 
                    **load_kwargs
                )
            else:
                # Otros (MPS, TPU, CPU): Carga y movimiento manual
                if DEVICE == "mps":
                    load_kwargs["device_map"] = None
                elif DEVICE == "cpu":
                    load_kwargs["device_map"] = "cpu"
                
                TransformerBase._base_model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    local_files_only=False, 
                    **load_kwargs
                )
                
                # Desplazamiento explícito para arquitecturas no-CUDA
                if DEVICE == "tpu":
                    import torch_xla.core.xla_model as xm
                    TransformerBase._base_model = TransformerBase._base_model.to(xm.xla_device())
                elif DEVICE == "mps":
                    TransformerBase._base_model = TransformerBase._base_model.to("mps")
                
                # Asegurar DTYPE en movimiento manual
                TransformerBase._base_model = TransformerBase._base_model.to(dtype=DTYPE)
            
            logger.info("Base model loaded in unique memory.")
        
        self.model = TransformerBase._base_model
        self.tokenizer = TransformerBase._tokenizer
        logger.debug("TransformerBase initialized with shared instance.")

    def apply_lora(self, state_dict_or_dict: dict):
        """
        Safely applies LoRA weights to the base model.
        
        Args:
            state_dict_or_dict: Either a state_dict directly, or a dict with keys:
                - 'state_dict': The actual state_dict to load
                - 'config': Optional PEFT configuration dict (if available)
        """
        # Handle backward compatibility: check if it's a dict with 'state_dict' key
        # or a direct state_dict
        if isinstance(state_dict_or_dict, dict) and 'state_dict' in state_dict_or_dict:
            state_dict = state_dict_or_dict['state_dict']
            saved_config = state_dict_or_dict.get('config')
        else:
            # Backward compatibility: treat as direct state_dict
            state_dict = state_dict_or_dict
            saved_config = None
        
        # 1. Always reset to base model from singleton to ensure clean state
        # This avoids PEFT warnings about modifying a model with PEFT for a second time
        if isinstance(self.model, PeftModel):
            logger.debug("Unloading previous adapter...")
            try:
                # Try unload() first (cleaner, doesn't merge weights)
                if hasattr(self.model, 'unload'):
                    self.model = self.model.unload()
                else:
                    # Fallback to merge_and_unload() if unload() not available
                    self.model = self.model.merge_and_unload()
            except Exception as e:
                logger.debug(f"Error unloading adapter: {e}, resetting to base model")
                self.model = TransformerBase._base_model
        
        # 2. Always use fresh base model from singleton to ensure no residual PEFT state
        # This is the safest approach to avoid PEFT warnings
        self.model = TransformerBase._base_model
            
        # 3. Inject new adapter using PeftModel.from_pretrained() for proper key matching
        # This is more robust than creating PeftModel and loading state_dict directly
        # because PEFT handles key prefixes correctly when loading from directory
        if state_dict:
            try:
                import tempfile
                import json
                import os
                
                # Create temporary directory with adapter files
                with tempfile.TemporaryDirectory() as tmpdir:
                    adapter_path = Path(tmpdir) / "adapter"
                    adapter_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save state_dict to safetensors or bin file
                    if SAFETENSORS_AVAILABLE:
                        safetensors_path = adapter_path / "adapter_model.safetensors"
                        save_file(state_dict, str(safetensors_path))
                    else:
                        bin_path = adapter_path / "adapter_model.bin"
                        torch.save(state_dict, bin_path)
                    
                    # Save adapter_config.json if available, otherwise create from defaults
                    if saved_config:
                        # Use saved config directly (already a dict)
                        config_dict = saved_config
                        logger.debug(f"Using saved LoRA config: r={config_dict.get('r', 'N/A')}")
                    else:
                        # Create default config matching training configuration
                        # IMPORTANT: Use LORA_TARGET_MODULES from settings (model-specific)
                        config_dict = {
                            "peft_type": "LORA",
                            "task_type": "CAUSAL_LM",
                            "inference_mode": True,
                            "r": LORA_RANK_DEFAULT,
                            "lora_alpha": LORA_RANK_DEFAULT * 2,
                            "target_modules": LORA_TARGET_MODULES,  # Model-specific (GPT-2 vs LLaMA)
                            "lora_dropout": 0.1,
                            "bias": "none"
                        }
                        logger.debug(f"Using default LoRA config: r={LORA_RANK_DEFAULT}, target_modules={LORA_TARGET_MODULES}")
                    
                    # Save adapter_config.json
                    config_path = adapter_path / "adapter_config.json"
                    with open(config_path, 'w') as f:
                        json.dump(config_dict, f, indent=2)
                    
                    # Load adapter using PeftModel.from_pretrained() - this handles key prefixes correctly
                    try:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            str(adapter_path),
                            adapter_name="default"
                        )
                        self.model.eval()  # Ensure inference mode
                        logger.debug("LoRA adapter injected successfully using from_pretrained()")
                    except Exception as load_error:
                        # Fallback: try manual loading if from_pretrained fails
                        logger.warning(f"PeftModel.from_pretrained() failed: {load_error}, trying manual load")
                        # Reconstruct LoraConfig and load manually
                        try:
                            # IMPORTANT: Use LORA_TARGET_MODULES as fallback (model-specific)
                            peft_config = LoraConfig(
                                inference_mode=True,
                                r=config_dict.get('r', LORA_RANK_DEFAULT),
                                lora_alpha=config_dict.get('lora_alpha', LORA_RANK_DEFAULT * 2),
                                target_modules=config_dict.get('target_modules', LORA_TARGET_MODULES),
                                lora_dropout=config_dict.get('lora_dropout', 0.1),
                                bias=config_dict.get('bias', 'none'),
                                task_type=TaskType.CAUSAL_LM
                            )
                            self.model = PeftModel(self.model, peft_config, adapter_name="default")
                            # Load weights with strict=False
                            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                            if missing:
                                logger.warning(f"Missing keys in LoRA (manual load): {len(missing)}")
                                if len(missing) <= 5:
                                    logger.debug(f"Missing keys: {list(missing)[:5]}")
                            if unexpected:
                                logger.debug(f"Unexpected keys in LoRA: {len(unexpected)}")
                            self.model.eval()
                            logger.debug("LoRA adapter injected successfully using manual load")
                        except Exception as manual_error:
                            # 2026-01-30: SELF-HEALING FOR INFERENCE
                            # If manual load fails due to target modules mismatch, try to auto-detect
                            if "target modules" in str(manual_error).lower() and "not found" in str(manual_error).lower():
                                logger.warning("LoRA target mismatch detected during inference. Attempting Self-Healing...")
                                try:
                                    # Auto-detect modules
                                    detected_modules = set()
                                    for name, module in self.model.named_modules():
                                        if isinstance(module, torch.nn.Linear) or "Linear" in str(type(module)):
                                            layer_name = name.split('.')[-1]
                                            if layer_name in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query_key_value"]:
                                                detected_modules.add(layer_name)
                                    
                                    if detected_modules:
                                        new_targets = list(detected_modules)
                                        logger.warning(f"Self-healing detected valid modules: {new_targets}")
                                        
                                        # Reconstruct config with correct targets
                                        peft_config_rescue = LoraConfig(
                                            inference_mode=True,
                                            r=config_dict.get('r', LORA_RANK_DEFAULT),
                                            lora_alpha=config_dict.get('lora_alpha', LORA_RANK_DEFAULT * 2),
                                            target_modules=new_targets, # Use detected targets
                                            lora_dropout=config_dict.get('lora_dropout', 0.1),
                                            bias=config_dict.get('bias', 'none'),
                                            task_type=TaskType.CAUSAL_LM
                                        )
                                        
                                        self.model = PeftModel(self.model, peft_config_rescue, adapter_name="default")
                                        
                                        # Load weights loosely
                                        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                                        logger.warning(f"Self-healing loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                                        
                                        self.model.eval()
                                        logger.warning("Self-Healing SUCCESS! Adapter injected with recovered config.")
                                        return # Success
                                    else:
                                        logger.error("Self-healing found no valid linear layers.")
                                except Exception as rescue_err:
                                    logger.error(f"Self-healing failed: {rescue_err}")

                            # If self-healing failed or wasn't applicable, re-raise original error
                            logger.error(f"Manual load failed: {manual_error}")
                            raise
                
            except Exception as e:
                logger.error(f"Error applying LoRA: {e}", exc_info=True)
                # Fallback: continue with base model and logs error
                self.model = TransformerBase._base_model

    def generate(
        self, 
        prompt: str, 
        lora_weights: Optional[dict] = None, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> dict:
        """
        Generates text from a prompt using specific node weights and calculates pure perplexity.
        
        Perplexity is calculated ONLY on the generated tokens to measure 
        how 'confident' the adapter is with the produced content.
        
        Args:
            prompt: User input text
            lora_weights: Optional adapter weights for the node
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dict with 'text', 'perplexity', 'confidence', and 'node_id'
        """
        # 1. Clean state and apply weights
        if lora_weights:
            self.apply_lora(lora_weights)
        
        # 2. Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        
        try:
            with torch.no_grad():
                # 3. Generate with scores to calculate perplexity
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # 4. Extract generated sequence (excluding prompt)
            generated_sequence = outputs.sequences[0][prompt_len:]
            
            # 5. Calculate Pure Perplexity (Only on generated tokens)
            # outputs.scores is a tuple of logits [batch, vocab] for each generated token
            log_probs = []
            
            for i, logits in enumerate(outputs.scores):
                # Get the actual token generated at this step
                token_id = generated_sequence[i]
                
                # Calculate softmax log probability for this token
                probs = torch.nn.functional.softmax(logits[0], dim=-1)
                token_prob = probs[token_id].item()
                
                # Avoid log(0)
                log_probs.append(math.log(max(token_prob, 1e-10)))
            
            # PPL = exp(-1/N * sum(log(p_i)))
            if log_probs:
                avg_log_prob = sum(log_probs) / len(log_probs)
                perplexity = math.exp(-avg_log_prob)
            else:
                perplexity = 0.0
                
            response_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            confidence = self._calculate_confidence(outputs.scores)
            
            return {
                "text": response_text,
                "perplexity": perplexity,
                "confidence": confidence,
                "tokens_generated": len(generated_sequence)
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return {
                "text": f"Error: {str(e)}",
                "perplexity": float('inf'),
                "confidence": 0.0
            }

            logger.error(f"Error in inference: {e}", exc_info=True)
            return {"text": "", "confidence": 0.0, "logits": None}

    def forward_batch_with_node(self, embeddings: torch.Tensor, lora_weights: dict) -> List[dict]:
        """
        Complete pipeline for BATCH: Inject LoRA (Once) -> Batch Inference -> Return results.
        
        Optimized for high throughput:
        1. Loads adapter once.
        2. Generates for B items in parallel.
        
        Args:
            embeddings: Tensor of shape [BATCH_SIZE, 384]
            lora_weights: Weights for the node (shared for all items in batch)
            
        Returns:
            List of dicts [{"text":..., "confidence":...}, ...]
        """
        if embeddings is None or embeddings.numel() == 0:
            return []
            
        batch_size = embeddings.shape[0] if embeddings.dim() > 1 else 1
        
        # 1. Apply node weights (ONCE for the whole batch)
        self.apply_lora(lora_weights)
        
        try:
            with torch.no_grad():
                # Prepare batch of start tokens [BATCH_SIZE, 1]
                # Use bos_token_id if available, otherwise fallback to eos_token_id (GPT-2 compatibility)
                bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
                start_tokens = torch.full(
                    (batch_size, 1), 
                    bos_id, 
                    dtype=torch.long, 
                    device=self.device
                )
                
                # Batch Generation
                # This runs the transformer on all B items simultaneously
                outputs = self.model.generate(
                    input_ids=start_tokens,
                    attention_mask=torch.ones_like(start_tokens),
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id  # GPT-2 compatibility
                )
            
            # Process results
            results = []
            for i in range(batch_size):
                # Decode text
                seq = outputs.sequences[i]
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                
                # Calculate confidence (extract scores for this batch item)
                # scores is a tuple of logits tensors, one per step.
                # Each tensor is [BATCH_SIZE, VOCAB_SIZE]
                # We need to reconstruct the sequence scores for item i
                item_scores = [step_logits[i] for step_logits in outputs.scores]
                confidence = self._calculate_confidence(item_scores)
                
                results.append({
                    "text": text,
                    "confidence": confidence,
                    # We skip 'logits' in batch mode to save memory/bandwidth unless needed
                    "logits": None 
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in batch inference: {e}", exc_info=True)
            # Return empty/error results
            return [{"text": "", "confidence": 0.0, "logits": None}] * batch_size

    def forward_with_node(self, embedding: torch.Tensor, lora_weights: dict, gene_identity: Optional[str] = None,
            knowledge_summary: Optional[str] = None
    ) -> dict:
        """
        Complete pipeline for SINGLE ITEM: Inject LoRA -> Inference -> Return result.
        
        Args:
            embedding: embedding tensor
            lora_weights: Weights for the node
            gene_identity: Optional gene identity string (e.g. "BRCA1") for system prompt injection.
            knowledge_summary: Optional node summary (e.g. "BRCA1 is a tumor suppressor...")
            
        Returns:
            Dict {"text":..., "confidence":..., "logits":...}
        """
        if embedding is None:
            return {"text": "", "confidence": 0.0, "logits": None}
            
        # Helper to treat single item as batch of 1
        # This keeps logic centralized in forward_batch_with_node
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        results = self.forward_batch_with_node(embedding, lora_weights)
        return results[0] if results else {"text": "", "confidence": 0.0, "logits": None}

    def _calculate_confidence(self, scores):
        """Calculates average probability of generated tokens."""
        probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
        return sum(probs) / len(probs) if probs else 0.0
    
    def train_kn_adapter(
        self,
        node_id: str,
        texts: List[str],
        epochs: int = 3,
        learning_rate: Optional[float] = None,  # None = auto-scale based on batch_size
        batch_size: int = TRAINING_BATCH_SIZE,  # Use configurable batch size from settings
        max_length: int = 512,
        epoch_callback: Optional[Callable[[int, int], None]] = None,
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        parent_similarity: Optional[float] = None,
        mental_age_score: Optional[float] = None,
        lora_rank: Optional[int] = None
    ) -> Optional[bytes]:
        """
        Train LoRA adapter for a Knowledge Node.
        
        Phase 2: Training Implementation
        
        This method:
        1. Validates inputs (null safety)
        2. Configures LoRA adapter
        3. Prepares training data (tokenization)
        4. Trains adapter on text corpus
        5. Serializes weights to bytes (safetensors format)
        6. Cleans up memory
        
        Args:
            node_id: Knowledge Node ID (for adapter naming)
            texts: List of training texts (must not be empty)
            epochs: Number of training epochs (default: 3)
            learning_rate: Learning rate for optimizer (default: None = auto-scale with batch_size)
                          If None, automatically scales: LR = LR_base * sqrt(batch_size / 4)
                          For batch_size=64: LR ≈ 4e-4 (4x increase from base 1e-4)
            batch_size: Training batch size (default: TRAINING_BATCH_SIZE from settings)
            max_length: Maximum sequence length (default: 512)
        
        Returns:
            bytes: Serialized adapter weights in safetensors format, or None if training fails
        
        Raises:
            ValueError: If texts is empty or None
        """
        # TEST EDIT (2025-01-27)
        # ========================================================================
        # Phase 2: Input Validation (Null Safety)
        # ========================================================================
        if texts is None:
            logger.debug(f"train_kn_adapter: texts is None for node '{node_id}'")
            return None
        
        if not isinstance(texts, list):
            logger.debug(f"train_kn_adapter: texts must be a list, got {type(texts)} for node '{node_id}'")
            return None
        
        if len(texts) == 0:
            logger.debug(f"train_kn_adapter: texts list is empty for node '{node_id}'")
            return None
        
        # ========================================================================
        # Apply MAX_TRAINING_TEXTS limit if configured (2025-01-14)
        # ========================================================================
        if MAX_TRAINING_TEXTS > 0 and len(texts) > MAX_TRAINING_TEXTS:
            logger.debug(
                f"Node '{node_id}' has {len(texts)} texts, "
                f"limiting to last {MAX_TRAINING_TEXTS} (most recent)"
            )
            texts = texts[-MAX_TRAINING_TEXTS:]  # Use last N texts (most recent)
        
        # Filter out None/empty texts
        valid_texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
        
        if len(valid_texts) == 0:
            logger.debug(f"train_kn_adapter: no valid texts after filtering for node '{node_id}'")
            return None
        
        # ========================================================================
        # Phase 2: Minimum Dataset Size Validation (2025-01-XX)
        # ========================================================================
        # Too few texts can lead to unstable gradients and overfitting
        # This validation ensures training quality and prevents gradient instability
        # ========================================================================
        if len(valid_texts) < MIN_TRAINING_TEXTS:
            logger.debug(
                f"train_kn_adapter: insufficient texts for stable training in node '{node_id}': "
                f"{len(valid_texts)} < {MIN_TRAINING_TEXTS} (minimum required). "
                f"Skipping training to prevent gradient instability."
            )
            return None
        
        logger.debug(
            f"Starting training for node '{node_id}': "
            f"{len(valid_texts)} valid texts (>= {MIN_TRAINING_TEXTS} minimum), {epochs} epochs"
        )
        
        # ========================================================================
        # Phase 2: Aggressive Memory Optimization - Clear Inference Cache
        # ========================================================================
        # [GOLD]: Resiliencia Molecular. Clean memory aggressively before training to prevent OOM errors
        # This is especially important for MPS which doesn't release memory automatically
        # ========================================================================
        import gc
        try:
            # 1. Unload any existing adapters to free memory
            if isinstance(self.model, PeftModel):
                logger.debug("Unloading existing adapter before training")
                self.model = self.model.merge_and_unload()
            
            # 2. Force Python garbage collection
            gc.collect()
            
            # 3. Clear GPU cache (CRITICAL: Do this before calculating batch_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(f"CUDA memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            elif DEVICE == "mps":
                # MPS: Try multiple cleanup methods
                if hasattr(torch.backends.mps, "empty_cache"):
                    torch.backends.mps.empty_cache()
                    logger.debug("MPS cache cleared")
                # Force another garbage collection after MPS cleanup
                gc.collect()
                logger.debug("MPS memory cleanup completed (forced GC)")
            
            # 4. Final garbage collection pass
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during aggressive memory cleanup: {e}")
            # Continue anyway - try to train with whatever memory is available
        
        # ========================================================================
        # Phase 2: Dynamic Batch Size Adjustment (2025-01-19)
        # ========================================================================
        # Adjust batch_size dynamically based on available GPU memory
        # This prevents OOM errors while maintaining training speed when memory is available
        # ========================================================================
        batch_size = get_dynamic_batch_size(batch_size, len(valid_texts))
        
        # ========================================================================
        # Phase 2: Configure LoRA Adapter
        # ========================================================================
        try:
            # ========================================================================
            # IMPORTANT: Use LORA_TARGET_MODULES from settings (2025-01-25)
            # ========================================================================
            # Different models have different layer names:
            # - LLaMA/TinyLlama: q_proj, k_proj, v_proj, o_proj
            # - GPT-2: c_attn, c_proj
            # DO NOT hardcode target modules - training will fail silently!
            # ========================================================================
            # Determine rank (use dynamic rank if provided, else from settings)
            rank = lora_rank if lora_rank is not None else LORA_RANK_DEFAULT
            alpha = rank * 2
            
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=LORA_TARGET_MODULES,  # IMPORTANT: Model-specific (GPT-2 vs LLaMA)
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False  # Training mode
            )
            
            # Create PEFT model
            peft_model = get_peft_model(self.model, lora_config)
            peft_model.train()  # Set to training mode
            
            logger.debug(f"LoRA adapter configured for node '{node_id}': r={rank}, target_modules={LORA_TARGET_MODULES}")
            
        except Exception as e:
            logger.error(f"Error configuring LoRA adapter for node '{node_id}': {e}")
            # Diagnostic: Print available modules if target modules not found
            if "target modules" in str(e).lower() and "not found" in str(e).lower():
                logger.error("DIAGNOSTIC - Available modules in base model:")
                for name, module in self.model.named_modules():
                    logger.error(f" - {name}")
            
            # MULTI-ATTEMPT RECOVERY (Self-Healing)
            logger.warning("Attempting self-healing for LoRA configuration...")
            try:
                # Scan model for valid linear layers
                detected_modules = set()
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear) or "Linear" in str(type(module)):
                        # Extract the last part of the name (e.g. model.layers.0.self_attn.q_proj -> q_proj)
                        layer_name = name.split('.')[-1]
                        if layer_name in ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query_key_value"]:
                            detected_modules.add(layer_name)
                
                if detected_modules:
                    new_targets = list(detected_modules)
                    logger.warning(f"Self-healing detected valid modules: {new_targets}")
                    
                    lora_config_rescue = LoraConfig(
                        r=rank,
                        lora_alpha=alpha,
                        target_modules=new_targets,
                        lora_dropout=0.1,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False
                    )
                    peft_model = get_peft_model(self.model, lora_config_rescue)
                    peft_model.train()
                    logger.warning(f"Self-healing SUCCESS! LoRA configured with auto-detected modules.")
                else:
                    logger.error("Self-healing failed: No standard attention layers found.")
                    return None
            except Exception as rescue_error:
                logger.error(f"Self-healing failed: {rescue_error}")
                return None

        
        # ========================================================================
        # Phase 3: Soft Inheritance (Initialization)
        # ========================================================================
        if parent_state_dict:
            try:
                # Load parent weights into new adapter
                # strict=False because adapter names might vary slightly (usually consistent)
                missing, unexpected = peft_model.load_state_dict(parent_state_dict, strict=False)
                if missing:
                    logger.debug(f"Inheritance: missing keys from parent: {len(missing)}")
                logger.info(f"Node '{node_id}' inherited weights from parent Titan.")
            except Exception as e:
                logger.error(f"Error inheriting weights for '{node_id}': {e}")
                # Fallback: continue with random initialization (orphan)
                parent_state_dict = None  # Disable reg since init failed

        # ========================================================================
        # Phase 3: Safety Configuration
        # ========================================================================
        # L2 Regularization Lambda
        # Fase 7: Increase L2 regularization for low similarity (< 0.70)
        lambda_val = 0.0
        if parent_state_dict and parent_similarity:
            # Dynamic Lambda: Base * (1 + similarity)
            # Sim=0.9 -> Lambda=0.019 (High protection)
            # Sim=0.6 -> Lambda=0.016 (Lower protection)
            lambda_val = INHERITANCE_L2_BASE_LAMBDA * (1.0 + parent_similarity)
            
            # Fase 7: Increase L2 by 50% for low similarity (< 0.70)
            if parent_similarity < 0.70:
                lambda_val = lambda_val * 1.5
                logger.debug(
                    f" Similitud baja ({parent_similarity:.3f} < 0.70), "
                    f"aumentando L2 regularización en 50%: lambda={lambda_val:.4f}"
                )
            else:
                logger.debug(f"Dynamic L2 Regularization enabled: lambda={lambda_val:.4f}")
        
        # ========================================================================
        # Phase 8: Madurez Sináptica Adaptativa (Sintonización Dinámica)
        # ========================================================================
        # Rationale: Higher SMS (maturity) allows for more 'flexibility' (less L2),
        # while lower SMS requires more 'anchoring' (higher L2).
        # ========================================================================
        if mental_age_score is not None:
            # Shift lambda_val based on absolute maturity
            # Reference matures at ~5.0. 
            # Maturity factor: (5.0 / (actual_sms + eps))
            # e.g., SMS=4.0 -> factor=1.25 (More reg)
            # e.g., SMS=6.0 -> factor=0.83 (Less reg)
            maturity_factor = 5.0 / (mental_age_score + 1e-6)
            maturity_factor = np.clip(maturity_factor, 0.5, 2.0)
            
            old_lambda = lambda_val
            lambda_val *= maturity_factor
            logger.info(
                f"Maturity-Linked L2: SMS={mental_age_score:.2f} | "
                f"Factor={maturity_factor:.2f} | Lambda: {old_lambda:.4f} -> {lambda_val:.4f}"
            )
        
        # ========================================================================
        # Learning Rate Auto-Scaling with Batch Size (2025-01-19)
        # ========================================================================
        # Rule: When batch size increases, learning rate should scale proportionally
        # Formula: LR = LR_base * sqrt(batch_size / reference_batch_size)
        # Reference batch_size = 4 (original default)
        # This maintains effective learning rate per example
        # ========================================================================
        # ========================================================================
        # Condensed Learning Rate Calculation (Cleanup 2026-02-02)
        # ========================================================================
        
        lr_base = TRAINING_LEARNING_RATE_BASE if isinstance(TRAINING_LEARNING_RATE_BASE, (int, float)) else 1e-4
        if batch_size is None or not isinstance(batch_size, (int, float)) or batch_size <= 0:
             batch_size = 64
             
        lr_scale_factor = 1.0
        lr_inheritance_multiplier = 1.0
        
        # 1. Base / Auto-scale
        if learning_rate is None:
            # Formula: LR = LR_base * sqrt(batch_size / 4)
            lr_scale_factor = math.sqrt(batch_size / 4.0)
            learning_rate = lr_base * lr_scale_factor
        else:
            lr_scale_factor = learning_rate / lr_base if lr_base > 0 else 1.0

        # 2. Inheritance Adjustment
        if parent_state_dict:
            lr_inheritance_multiplier = INHERITANCE_LR_MULTIPLIER if isinstance(INHERITANCE_LR_MULTIPLIER, (int, float)) else 1.0
            learning_rate *= lr_inheritance_multiplier
            
        # 3. Maturity Adjustment (kept from original)
        if mental_age_score is not None:
             # Higher maturity (experts) can afford slightly higher LR for fine-tuning
             lr_maturity_multiplier = np.clip(mental_age_score / 5.0, 0.5, 1.5)
             old_lr = learning_rate
             learning_rate *= lr_maturity_multiplier
             logger.info(
                 f"Maturity-Linked LR: SMS={mental_age_score:.2f} | "
                 f"Multiplier={lr_maturity_multiplier:.2f} | LR: {old_lr:.6f} -> {learning_rate:.6f}"
             )

        # 4. Rank Adjustment (Fase 2 X-DETOX)
        # Rule: Higher rank nodes need lower LR for stability, lower rank nodes need higher LR to learn
        # Formula: Multiplier = sqrt(16 / rank)
        lr_rank_multiplier = 1.0
        if lora_rank is not None and lora_rank > 0:
            lr_rank_multiplier = math.sqrt(16.0 / lora_rank)
            learning_rate *= lr_rank_multiplier
            logger.debug(f"Rank-Linked LR Adjustment: Rank={lora_rank} | Multiplier={lr_rank_multiplier:.2f}")

        # 5. Final Safety & Log
        # Ensure it is a valid float > 0
        try:
            lr_final = float(learning_rate)
            if lr_final <= 0: lr_final = 1e-4
        except (ValueError, TypeError):
            lr_final = 1e-4
            
        parent_similarity_str = f"{parent_similarity:.3f}" if parent_similarity is not None else "N/A"
        
        logger.info(
            f"** LR Final para nodo '{node_id}': {lr_final:.6f} | "
            f"Desglose: Base={lr_base:.6f} × Escala={lr_scale_factor:.2f}x × "
            f"Herencia={lr_inheritance_multiplier:.2f}x × Rango={lr_rank_multiplier:.2f}x | "
            f"batch_size={batch_size} | "
            f"parent_similarity={parent_similarity_str}"
        )
        
        # Propagate final value
        learning_rate = lr_final
        
        # Life Insurance (Early Stopping)
        baseline_loss = None
        
        # ========================================================================
        # Phase 2: Prepare Training Data
        # ========================================================================
        try:
            # Tokenize texts (for causal LM, we need to add padding token if not exists)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure left padding for decoder-only models (GPT-2 compatibility)
            self.tokenizer.padding_side = 'left'
            
            # Tokenize all texts
            tokenized = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = tokenized["input_ids"].to(DEVICE)
            attention_mask = tokenized["attention_mask"].to(DEVICE)
            
            logger.debug(f"Tokenized {len(valid_texts)} texts for node '{node_id}'")
            
        except Exception as e:
            logger.debug(f"Error tokenizing texts for node '{node_id}': {e}")
            # Clean up
            try:
                if isinstance(peft_model, PeftModel):
                    peft_model = peft_model.merge_and_unload()
            except Exception:
                pass
            return None
        
        # ========================================================================
        # Phase 3: Calculate Baseline Loss (if inheritance)
        # ========================================================================
        # Fase 7: Life Insurance estricto para similitud < 0.70
        use_strict_life_insurance = False
        if parent_state_dict and parent_similarity is not None and parent_similarity < 0.70:
            use_strict_life_insurance = True
            logger.warning(
                f" Similitud baja ({parent_similarity:.3f} < 0.70), "
                f"usando Life Insurance estricto"
            )
        
        if parent_state_dict:
            # Calculate baseline on small validation subset before training
            validation_subset = valid_texts[:min(10, len(valid_texts))]
            baseline_loss = self._calculate_validation_loss(peft_model, validation_subset, max_length)
            logger.debug(f"Baseline loss (with parent): {baseline_loss:.4f}")
        
        # ========================================================================
        # Phase 2: Training Loop
        # ========================================================================
        try:
            # Use lr_final (validated) instead of learning_rate for optimizer
            # This ensures we always use a valid learning rate
            logger.debug(f"🔍 [L904] VALIDATION: Validating lr_final for optimizer | value={lr_final}, type={type(lr_final)}")
            optimizer_lr = lr_final if isinstance(lr_final, (int, float)) and lr_final > 0 else 1e-4
            logger.debug(f"🔍 [L904] MODIFY: optimizer_lr set | line=904 | value={optimizer_lr}, type={type(optimizer_lr)}")
            if optimizer_lr != lr_final:
                logger.warning(f" [L904] optimizer_lr differs from lr_final, using fallback | lr_final={lr_final}, optimizer_lr={optimizer_lr}")
            optimizer = torch.optim.AdamW(
                peft_model.parameters(),
                lr=optimizer_lr  # Use validated lr_final
            )
            logger.debug(f"🔍 [L907] MODIFY: Optimizer created with lr | line=907 | optimizer_lr={optimizer_lr}")
            
            peft_model.train()
            total_loss = 0.0
            num_samples = len(valid_texts)
            
            # Move parent weights to device once for efficiency
            parent_params = {}
            if parent_state_dict and lambda_val > 0:
                for n, p in peft_model.named_parameters():
                    if p.requires_grad and n in parent_state_dict:
                        parent_params[n] = parent_state_dict[n].to(DEVICE)

            # Gradient Accumulation Logic (2025-01-27)
            # Maintain stability of batch=32 using 4 steps of physical batch size
            accumulation_steps = 4
            
            for epoch in range(epochs):
                current_epoch = epoch + 1
                if epoch_callback:
                    epoch_callback(current_epoch, epochs)
                
                epoch_loss = 0.0
                num_batches = 0
                i = 0
                
                while i < num_samples:
                    # Current accumulation window
                    retry_count = 0
                    max_retries = 3
                    
                    # Target window size (requested batch_size)
                    current_window_size = min(batch_size, num_samples - i)
                    # Physical batch size (micro-batch)
                    micro_batch_size = max(1, current_window_size // accumulation_steps)
                    
                    batch_total_loss = 0.0
                    
                    while retry_count < max_retries:
                        try:
                            optimizer.zero_grad()
                            temp_accumulated_loss = 0.0
                            steps_done = 0
                            
                            # Process the window in micro-batches
                            for step_start in range(i, i + current_window_size, micro_batch_size):
                                step_end = min(step_start + micro_batch_size, i + current_window_size)
                                micro_input_ids = input_ids[step_start:step_end].to(DEVICE)
                                micro_attention_mask = attention_mask[step_start:step_end].to(DEVICE)
                                
                                actual_steps = math.ceil(current_window_size / micro_batch_size)
                                
                                # Automatic Mixed Precision (2025-01-27)
                                with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", enabled=("cuda" in DEVICE)):
                                    # Mask padding labels with -100 to ignore them in loss calculation
                                    micro_labels = micro_input_ids.clone()
                                    if self.tokenizer.pad_token_id is not None:
                                        micro_labels[micro_labels == self.tokenizer.pad_token_id] = -100

                                    outputs = peft_model(
                                        input_ids=micro_input_ids,
                                        attention_mask=micro_attention_mask,
                                        labels=micro_labels
                                    )
                                    loss = outputs.loss / actual_steps
                                    
                                    # Life Insurance
                                    if baseline_loss is not None:
                                        current_full_loss = loss.item() * actual_steps
                                        if use_strict_life_insurance and i == 0 and steps_done == 0:
                                            if current_full_loss > baseline_loss:
                                                logger.error(f"Life Insurance estricto: Loss inicial ({current_full_loss:.4f}) > baseline ({baseline_loss:.4f}).")
                                                return None
                                        elif current_full_loss > baseline_loss * LIFE_INSURANCE_THRESHOLD:
                                            logger.warning(f" Life Insurance Triggered! Loss {current_full_loss:.4f} > {LIFE_INSURANCE_THRESHOLD}x Baseline.")
                                            return None

                                    # Dynamic L2 Regularization
                                    if parent_params and lambda_val > 0:
                                        l2_reg = torch.tensor(0.0, device=DEVICE)
                                        for n, p in peft_model.named_parameters():
                                            if p.requires_grad and n in parent_params:
                                                l2_reg += torch.norm(p - parent_params[n])**2
                                        loss += (lambda_val * l2_reg) / actual_steps
                                
                                # Backward pass - INSIDE THE TRY BLOCK
                                loss.backward()
                                temp_accumulated_loss += loss.item() * actual_steps
                                steps_done += 1
                            
                            # Success: Optimization step
                            optimizer.step()
                            optimizer.zero_grad()
                            batch_total_loss = temp_accumulated_loss / steps_done if steps_done > 0 else 0
                            break 
                            
                        except RuntimeError as e:
                            error_msg = str(e)
                            if "out of memory" in error_msg.lower():
                                retry_count += 1
                                if micro_batch_size <= 1:
                                    logger.error(f"OOM with micro_batch=1. Node {node_id} failed.")
                                    raise
                                micro_batch_size = max(1, micro_batch_size // 2)
                                import gc; gc.collect()
                                if torch.cuda.is_available(): torch.cuda.empty_cache()
                                optimizer.zero_grad()
                            else: raise
                    
                    epoch_loss += batch_total_loss
                    num_batches += 1
                    i += current_window_size
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                total_loss += avg_loss
                logger.debug(f"Node '{node_id}' epoch {epoch + 1}/{epochs}: avg_loss={avg_loss:.4f}")
            avg_total_loss = total_loss / epochs if epochs > 0 else 0.0
            logger.debug(f"Training completed for node '{node_id}': avg_loss={avg_total_loss:.4f}")
            
            # ========================================================================
            # Phase 3: Post-training Validation (for inheritance quality check)
            # ========================================================================
            if parent_state_dict and baseline_loss is not None:
                validation_subset = valid_texts[:min(10, len(valid_texts))]
                final_loss = self._calculate_validation_loss(peft_model, validation_subset, max_length)
                logger.debug(
                    f"Post-training validation loss: {final_loss:.4f} "
                    f"(baseline: {baseline_loss:.4f}, ratio: {final_loss/baseline_loss:.2f}x)"
                )
            
        except Exception as e:
            logger.debug(f"Error during training for node '{node_id}': {e}", exc_info=True)
            # Clean up
            try:
                if isinstance(peft_model, PeftModel):
                    peft_model = peft_model.merge_and_unload()
            except Exception:
                pass
            return None
        
        # ========================================================================
        # Phase 2: Serialize Weights to Bytes
        # ========================================================================
        try:
            # [HARDWARE FIDELITY]: Save adapter to persistent, model-specific path
            # This ensures isolation between Llama 3 and TinyLlama adapters.
            adapter_path = LORA_PATH / node_id
            adapter_path.mkdir(parents=True, exist_ok=True)
            
            peft_model.save_pretrained(str(adapter_path))
                
            # CRITICAL: Extract and save PEFT configuration
            # PEFT saves adapter_config.json when save_pretrained() is called
            adapter_config_path = adapter_path / "adapter_config.json"
            peft_config_dict = None
            if adapter_config_path.exists():
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        peft_config_dict = json.load(f)
                    logger.debug(f"Loaded PEFT config for node '{node_id}': r={peft_config_dict.get('r', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Error reading adapter_config.json for '{node_id}': {e}")
                    peft_config_dict = None
            else:
                # Fallback: Extract config from peft_model if available
                try:
                    if hasattr(peft_model, 'peft_config') and peft_model.peft_config:
                        # Get the first (and usually only) adapter config
                        adapter_name = list(peft_model.peft_config.keys())[0] if peft_model.peft_config else None
                        if adapter_name:
                            config_obj = peft_model.peft_config[adapter_name]
                            # Convert LoraConfig to dict
                            # IMPORTANT: Use LORA_TARGET_MODULES as fallback (model-specific)
                            peft_config_dict = {
                                'r': getattr(config_obj, 'r', LORA_RANK_DEFAULT),
                                'lora_alpha': getattr(config_obj, 'lora_alpha', LORA_RANK_DEFAULT * 2),
                                'target_modules': getattr(config_obj, 'target_modules', LORA_TARGET_MODULES),
                                'lora_dropout': getattr(config_obj, 'lora_dropout', 0.1),
                                'bias': getattr(config_obj, 'bias', 'none'),
                                'task_type': str(getattr(config_obj, 'task_type', 'CAUSAL_LM'))
                            }
                            logger.debug(f"Extracted PEFT config from model for node '{node_id}': r={peft_config_dict['r']}")
                except Exception as e:
                    logger.warning(f"Error extracting config from peft_model for '{node_id}': {e}")
                    peft_config_dict = None
            
            # Load safetensors file
            safetensors_path = adapter_path / "adapter_model.safetensors"
            
            if not safetensors_path.exists():
                # Fallback: try adapter_model.bin
                bin_path = adapter_path / "adapter_model.bin"
                if bin_path.exists():
                    if SAFETENSORS_AVAILABLE:
                        # Convert to safetensors
                        # CRITICAL: PyTorch 2.6 changed default weights_only=True, but our weights contain
                        # metadata that requires weights_only=False. These weights are from our own
                        # trusted training process, so it's safe.
                        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
                        safetensors_path = adapter_path / "adapter_model.safetensors"
                        save_file(state_dict, str(safetensors_path))
                    else:
                        # Use bin file directly
                        safetensors_path = bin_path
                else:
                    logger.debug(f"No adapter weights found for node '{node_id}'")
                    return None
            
            # Read as bytes
            with open(safetensors_path, "rb") as f:
                weights_bytes = f.read()
            
            logger.debug(
                f"Adapter weights serialized for node '{node_id}': "
                f"{len(weights_bytes)} bytes"
            )
                
        except Exception as e:
            logger.debug(f"Error serializing weights for node '{node_id}': {e}", exc_info=True)
            return None
        
        # ========================================================================
        # Phase 2: Aggressive Clean Up Memory (2025-01-19)
        # ========================================================================
        # CRITICAL: Clean memory aggressively after training to free GPU memory
        # This prevents memory accumulation across multiple training runs
        # ========================================================================
        import gc
        try:
            # 1. Unload adapter to free memory
            if isinstance(peft_model, PeftModel):
                peft_model = peft_model.merge_and_unload()
                logger.debug(f"Adapter unloaded for node '{node_id}'")
            
            # 2. Delete peft_model reference
            del peft_model
            peft_model = None
            
            # 3. Force Python garbage collection
            gc.collect()
            
            # 4. Clear GPU cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(f"CUDA memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            elif DEVICE == "mps":
                # MPS: Multiple cleanup passes
                if hasattr(torch.backends.mps, "empty_cache"):
                    torch.backends.mps.empty_cache()
                # Force another garbage collection after MPS cleanup
                gc.collect()
                logger.debug(f"MPS memory cleaned after training for node '{node_id}'")
            
            # 5. Final garbage collection pass
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during aggressive memory cleanup after training for node '{node_id}': {e}")
            # Continue anyway - weights are already serialized
        
        # Return dict with weights_bytes and config for proper configuration saving
        if peft_config_dict:
            return {"weights_bytes": weights_bytes, "config": peft_config_dict}
        else:
            # Backward compatibility: return just weights_bytes if no config
            return weights_bytes
    
    def _calculate_validation_loss(
        self, 
        model: torch.nn.Module, 
        texts: List[str], 
        max_length: int = 512
    ) -> float:
        """
        Calculate validation loss on a set of texts.
        
        Used for:
        - Baseline loss calculation (before training with inheritance)
        - Post-training validation (to check if inheritance improved quality)
        
        Args:
            model: Model to evaluate (with adapter applied)
            texts: List of texts to evaluate on
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Average validation loss (float), or inf if error
        """
        if not texts:
            return float('inf')
        
        model.eval()
        # [GOLD] Use singleton tokenizer to ensure consistency
        tokenizer = TransformerBase._tokenizer
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Ensure left padding for decoder-only models (GPT-2 compatibility)
        tokenizer.padding_side = 'left'
        
        total_loss = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                batch_size = 4  # Small batch for validation
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    tokenized = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    # [GOLD] Use model.device if available (safest), else fall back to DEVICE
                    target_device = model.device if hasattr(model, "device") else DEVICE
                    input_ids = tokenized["input_ids"].to(target_device)
                    attention_mask = tokenized["attention_mask"].to(target_device)
                    
                    # Mask padding labels with -100 to ignore them in loss calculation
                    labels = input_ids.clone()
                    if tokenizer.pad_token_id is not None:
                        labels[labels == tokenizer.pad_token_id] = -100

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    total_loss += outputs.loss.item()
                    num_batches += 1
        except Exception as e:
            logger.debug(f"Error calculating validation loss: {e}")
            return float('inf')
        finally:
            model.train()  # Return to training mode
        
        return total_loss / num_batches if num_batches > 0 else float('inf')