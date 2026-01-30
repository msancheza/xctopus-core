"""
Evaluation Module for Xctopus Layer 2 (LoRA Adapters).

This module provides evaluation capabilities for trained LoRA adapters without
modifying existing classes. It depends on TransformerBase, KNRepository, and
optionally DataManager, but these classes remain unchanged.

Added: 2026-01-15
Purpose: Evaluate quality of trained LoRA adapters after training phase.
"""

import torch
import torch.nn.functional as F
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from .repository import KNRepository
from .transformer_base import TransformerBase
from .data_manager import DataManager
from peft import PeftModel
from .settings import (
    DEVICE,
    DTYPE,
    EVAL_VALIDATION_SPLIT,
    EVAL_METRICS,
    EVAL_BATCH_SIZE,
    EVAL_MIN_TEXTS,
    EVAL_SAVE_RESULTS,
    EVAL_REPORT_FORMAT,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for trained LoRA adapters.
    
    This class evaluates the quality of trained LoRA adapters by:
    1. Loading each adapter and applying it to the base model
    2. Evaluating on validation/test data
    3. Calculating metrics (perplexity, loss, thematic coherence)
    4. Generating reports
    
    Dependencies:
    - TransformerBase: For loading base model and applying adapters
    - KNRepository: For accessing node data and trained adapters
    - DataManager (optional): For retrieving original texts
    
    Note: This class does NOT modify any existing classes.
    """
    
    def __init__(
        self,
        repository: KNRepository,
        transformer_base: TransformerBase,
        data_manager: Optional[DataManager] = None
    ):
        """
        Initialize the Evaluator.
        
        Args:
            repository: KNRepository instance for accessing node data
            transformer_base: TransformerBase instance (singleton, shared)
            data_manager: Optional DataManager for retrieving original texts
                         If None, will try to get texts from repository pointers
        """
        self.repository = repository
        self.transformer = transformer_base
        self.data_manager = data_manager
        
        logger.info("Evaluator initialized")
    
    def evaluate_adapter(
        self,
        node_id: str,
        validation_texts: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single LoRA adapter.
        
        Args:
            node_id: Knowledge Node ID to evaluate
            validation_texts: Optional list of validation texts
                            If None, will retrieve from repository/data_manager
            metrics: Optional list of metrics to calculate
                    If None, uses EVAL_METRICS from settings
        
        Returns:
            Dictionary with evaluation results:
            {
                "node_id": str,
                "perplexity": float (if calculated),
                "validation_loss": float (if calculated),
                "thematic_coherence": float (if calculated),
                "num_texts": int,
                "status": "success" | "failed" | "skipped",
                "error": str (if failed)
            }
        """
        metrics = metrics or EVAL_METRICS
        result = {
            "node_id": node_id,
            "status": "pending",
            "num_texts": 0
        }
        
        try:
            # 1. Check if node exists and is trained
            if not self.repository.is_trained(node_id):
                result["status"] = "skipped"
                result["error"] = "Node not trained"
                logger.debug(f"Node '{node_id}' not trained, skipping evaluation")
                return result
            
            # 2. Get validation texts
            if validation_texts is None:
                validation_texts = self._get_validation_texts(node_id)
            
            if not validation_texts or len(validation_texts) < EVAL_MIN_TEXTS:
                result["status"] = "skipped"
                result["error"] = f"Insufficient texts for evaluation: {len(validation_texts) if validation_texts else 0} < {EVAL_MIN_TEXTS}"
                logger.debug(f"Node '{node_id}' has insufficient texts for evaluation")
                return result
            
            result["num_texts"] = len(validation_texts)
            
            # 3. Load and apply adapter
            logger.debug(f"Evaluating adapter for node '{node_id}' with {len(validation_texts)} texts")
            adapter_weights = self.repository.get_lora_weights(node_id)
            
            if adapter_weights is None:
                result["status"] = "failed"
                result["error"] = "Adapter weights not found"
                logger.warning(f"Adapter weights not found for node '{node_id}'")
                return result
            
            # Apply adapter to base model
            # Note: apply_lora() already resets to base model before applying new adapter
            try:
                self.transformer.apply_lora(adapter_weights)
                model = self.transformer.model
                model.eval()  # Set to evaluation mode
                
                # 4. Calculate metrics
                with torch.no_grad():
                    if "perplexity" in metrics:
                        result["perplexity"] = self._calculate_perplexity(model, validation_texts)
                    
                    if "validation_loss" in metrics:
                        result["validation_loss"] = self._calculate_validation_loss(model, validation_texts)
                    
                    if "thematic_coherence" in metrics:
                        result["thematic_coherence"] = self._calculate_thematic_coherence(node_id, model, validation_texts)
                
                result["status"] = "success"
                logger.debug(
                    f"Evaluation completed for node '{node_id}': "
                    f"perplexity={result.get('perplexity', 'N/A')}, "
                    f"loss={result.get('validation_loss', 'N/A')}"
                )
                
            finally:
                # Reset to base model after evaluation (important: don't leave adapter applied)
                # We'll reset by calling apply_lora with an empty dict or None, which will reset to base
                # Alternatively, we can directly access the base model through the transformer instance
                if isinstance(self.transformer.model, PeftModel):
                    try:
                        if hasattr(self.transformer.model, 'unload'):
                            self.transformer.model = self.transformer.model.unload()
                        else:
                            self.transformer.model = self.transformer.model.merge_and_unload()
                    except Exception as e:
                        logger.debug(f"Error unloading adapter after evaluation: {e}, resetting to base model")
                        # Reset by calling apply_lora with empty dict (will reset to base)
                        self.transformer.apply_lora({})
                else:
                    # Already base model, no need to reset
                    pass
        
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Error evaluating adapter for node '{node_id}': {e}", exc_info=True)
        
        return result
    
    def evaluate_all_adapters(
        self,
        node_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        progress_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate all trained adapters.
        
        Args:
            node_ids: Optional list of node IDs to evaluate
                    If None, evaluates all trained nodes
            metrics: Optional list of metrics to calculate
            progress_interval: How often to log progress (default: every node)
        
        Returns:
            Dictionary with evaluation results:
            {
                "total_nodes": int,
                "evaluated": int,
                "failed": int,
                "skipped": int,
                "results": List[Dict]  # Results for each node
            }
        """
        metrics = metrics or EVAL_METRICS
        
        # Get list of nodes to evaluate
        if node_ids is None:
            # Get all trained nodes
            all_signatures = self.repository.get_all_signatures()
            node_ids = [
                sig["node_id"] for sig in all_signatures
                if self.repository.is_trained(sig["node_id"])
            ]
        
        logger.info(f"Starting evaluation of {len(node_ids)} adapters")
        
        results = []
        evaluated = 0
        failed = 0
        skipped = 0
        
        for i, node_id in enumerate(node_ids, 1):
            logger.info(f"Evaluating adapter {i}/{len(node_ids)}: {node_id}")
            
            result = self.evaluate_adapter(node_id, metrics=metrics)
            results.append(result)
            
            if result["status"] == "success":
                evaluated += 1
                logger.info(f"  Adapter {i}/{len(node_ids)} evaluated successfully")
            elif result["status"] == "failed":
                failed += 1
                logger.warning(f"  Adapter {i}/{len(node_ids)} failed: {result.get('error', 'Unknown error')}")
            else:
                skipped += 1
                logger.info(f"   Adapter {i}/{len(node_ids)} skipped: {result.get('error', 'Unknown reason')}")
            
            if i % progress_interval == 0:
                logger.info(f"Progress: {i}/{len(node_ids)} adapters evaluated ({evaluated} success, {failed} failed, {skipped} skipped)")
        
        summary = {
            "total_nodes": len(node_ids),
            "evaluated": evaluated,
            "failed": failed,
            "skipped": skipped,
            "results": results
        }
        
        # Save results if configured
        if EVAL_SAVE_RESULTS:
            self._save_evaluation_results(summary)
        
        # Generate report
        self._generate_report(summary)
        
        logger.info(
            f"Evaluation completed: {evaluated} evaluated, {failed} failed, "
            f"{skipped} skipped (out of {len(node_ids)} total)"
        )
        logger.info(f"Evaluation completed:")
        logger.info(f"   - Evaluated: {evaluated} adapters")
        logger.info(f"   - Failed: {failed} adapters")
        logger.info(f"   - Skipped: {skipped} adapters")
        logger.info(f"   - Total: {len(node_ids)} adapters")
        
        return summary
    
    def _get_validation_texts(self, node_id: str) -> List[str]:
        """
        Get validation texts for a node.
        
        If EVAL_VALIDATION_SPLIT > 0, splits training texts.
        Otherwise, uses all available texts.
        
        Args:
            node_id: Knowledge Node ID
        
        Returns:
            List of validation texts
        """
        # Get all source_ids for this node
        source_ids = self.repository.get_training_pointers(node_id)
        
        if not source_ids:
            logger.debug(f"No source_ids found for node '{node_id}'")
            return []
        
        # Get texts from DataManager or repository
        if self.data_manager:
            # Try to get texts from DataManager
            try:
                texts = self.data_manager.get_texts_from_pointers(source_ids)
            except Exception as e:
                logger.debug(f"Error getting texts from DataManager for node '{node_id}': {e}")
                texts = []
        else:
            # DataManager not available, cannot retrieve texts
            logger.warning(f"DataManager not available, cannot retrieve texts for node '{node_id}'")
            return []
        
        if not texts:
            return []
        
        # Apply validation split if configured
        if EVAL_VALIDATION_SPLIT > 0 and len(texts) > EVAL_MIN_TEXTS:
            split_idx = int(len(texts) * (1 - EVAL_VALIDATION_SPLIT))
            validation_texts = texts[split_idx:]  # Last N% for validation
            logger.debug(
                f"Node '{node_id}': {len(texts)} total texts, "
                f"{len(validation_texts)} validation texts "
                f"({EVAL_VALIDATION_SPLIT * 100:.0f}% split)"
            )
            return validation_texts
        else:
            # Use all texts (no split or too few texts)
            logger.debug(f"Node '{node_id}': using all {len(texts)} texts for evaluation (no split)")
            return texts
    
    def _calculate_perplexity(
        self,
        model: torch.nn.Module,
        texts: List[str]
    ) -> float:
        """
        Calculate perplexity on a set of texts.
        
        Perplexity = exp(mean_loss)
        Lower is better.
        
        Args:
            model: Model to evaluate (with adapter applied)
            texts: List of texts to evaluate on
        
        Returns:
            Perplexity value (float)
        """
        total_loss = 0.0
        num_batches = 0
        
        tokenizer = self.transformer.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), EVAL_BATCH_SIZE):
                batch_texts = texts[i:i + EVAL_BATCH_SIZE]
                
                # Tokenize
                tokenized = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = tokenized["input_ids"].to(DEVICE)
                attention_mask = tokenized["attention_mask"].to(DEVICE)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For causal LM
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches == 0:
            return float('inf')
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def _calculate_validation_loss(
        self,
        model: torch.nn.Module,
        texts: List[str]
    ) -> float:
        """
        Calculate validation loss on a set of texts.
        
        Args:
            model: Model to evaluate (with adapter applied)
            texts: List of texts to evaluate on
        
        Returns:
            Average validation loss (float)
        """
        total_loss = 0.0
        num_batches = 0
        
        tokenizer = self.transformer.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), EVAL_BATCH_SIZE):
                batch_texts = texts[i:i + EVAL_BATCH_SIZE]
                
                # Tokenize
                tokenized = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = tokenized["input_ids"].to(DEVICE)
                attention_mask = tokenized["attention_mask"].to(DEVICE)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches == 0:
            return float('inf')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _calculate_thematic_coherence(
        self,
        node_id: str,
        model: torch.nn.Module,
        texts: List[str]
    ) -> float:
        """
        Calculate thematic coherence: how well generated texts match node's theme.
        
        This is a Xctopus-specific metric that measures if the adapter maintains
        the semantic theme of the knowledge node.
        
        Args:
            node_id: Knowledge Node ID
            model: Model to evaluate (with adapter applied)
            texts: List of texts to evaluate on
        
        Returns:
            Thematic coherence score (0.0-1.0, higher is better)
        """
        # Get node signature for comparison
        signature = self.repository.get_signature(node_id)
        if not signature:
            return 0.0
        
        node_centroid = signature["centroid"]  # This is a tensor
        
        # Generate a sample text from the model and compare with node centroid
        # For now, use a simple approach: generate embeddings of validation texts
        # and compare with node centroid
        
        # This is a simplified version - in production, you might want to:
        # 1. Generate texts from the model
        # 2. Embed those generated texts
        # 3. Compare with node centroid
        
        # For now, we'll use the validation texts themselves as a proxy
        # (assuming they represent the node's theme)
        
        try:
            # Use the embedding model to get embeddings of validation texts
            # Note: This requires access to the embedding model, which might not be available
            # For now, return a placeholder or skip this metric
            
            # Placeholder: return average similarity (would need embedding model)
            # For now, return 0.0 to indicate not calculated
            logger.debug(f"Thematic coherence calculation not fully implemented for node '{node_id}'")
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating thematic coherence for node '{node_id}': {e}")
            return 0.0
    
    def _save_evaluation_results(self, summary: Dict[str, Any]) -> None:
        """
        Save evaluation results to database.
        
        Args:
            summary: Evaluation summary dictionary
        """
        results = summary["results"]
        for result in results:
            if result["status"] == "success":
                node_id = result["node_id"]
                metrics = {}
                
                if "perplexity" in result:
                    metrics["perplexity"] = result["perplexity"]
                if "validation_loss" in result:
                    metrics["validation_loss"] = result["validation_loss"]
                if "thematic_coherence" in result:
                    metrics["thematic_coherence"] = result["thematic_coherence"]
                
                # Metadata (num_texts, etc.)
                metadata = {
                    "num_texts": result.get("num_texts", 0)
                }
                
                if metrics:
                    self.repository.save_evaluation_metrics(node_id, metrics, metadata)
                    logger.debug(f"Saved evaluation metrics for node {node_id}")
    
    def _generate_report(self, summary: Dict[str, Any]) -> None:
        """
        Generate evaluation report.
        
        Args:
            summary: Evaluation summary dictionary
        """
        results = summary["results"]
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            logger.warning("No successful evaluations to report")
            return
        
        # Calculate aggregate statistics
        perplexities = [r.get("perplexity") for r in successful_results if r.get("perplexity") is not None]
        losses = [r.get("validation_loss") for r in successful_results if r.get("validation_loss") is not None]
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("== EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTotal nodes evaluated: {summary['total_nodes']}")
        report_lines.append(f"  - Successfully evaluated: {summary['evaluated']}")
        report_lines.append(f"  - Failed: {summary['failed']}")
        report_lines.append(f"  - Skipped: {summary['skipped']}")
        
        if perplexities:
            report_lines.append(f"\n== PERPLEXITY STATISTICS:")
            report_lines.append(f"  - Min: {min(perplexities):.2f}")
            report_lines.append(f"  - Max: {max(perplexities):.2f}")
            report_lines.append(f"  - Average: {sum(perplexities) / len(perplexities):.2f}")
            report_lines.append(f"  - Median: {sorted(perplexities)[len(perplexities) // 2]:.2f}")
        
        if losses:
            report_lines.append(f"\n== VALIDATION LOSS STATISTICS:")
            report_lines.append(f"  - Min: {min(losses):.4f}")
            report_lines.append(f"  - Max: {max(losses):.4f}")
            report_lines.append(f"  - Average: {sum(losses) / len(losses):.4f}")
            report_lines.append(f"  - Median: {sorted(losses)[len(losses) // 2]:.4f}")
        
        # Top and bottom performers
        if perplexities:
            sorted_by_ppl = sorted(successful_results, key=lambda x: x.get("perplexity", float('inf')))
            report_lines.append(f"\n🏆 TOP 5 ADAPTERS (Lowest Perplexity):")
            for i, result in enumerate(sorted_by_ppl[:5], 1):
                node_id_short = result["node_id"][:20] + "..." if len(result["node_id"]) > 20 else result["node_id"]
                ppl = result.get("perplexity", "N/A")
                report_lines.append(f"  {i}. {node_id_short}: {ppl:.2f}")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Output based on EVAL_REPORT_FORMAT
        if EVAL_REPORT_FORMAT in ["console", "both"]:
            logger.info("\n" + report_text)
        
        if EVAL_REPORT_FORMAT in ["file", "both"]:
            report_file = Path("notebooks_build/evaluation_report.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to: {report_file}")
        
        logger.info("Evaluation report generated")
