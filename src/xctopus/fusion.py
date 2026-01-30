"""
Knowledge Nodes Fusion Module.

Implements post-clustering fusion protocol to consolidate similar KNs
and reduce fragmentation, improving the final semantic architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from uuid import uuid4
import sys

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .repository import KNRepository
from .orchestrator import Orchestrator
from .knowledgenode import KnowledgeNode
from .settings import (
    TRAINING_THRESHOLD,
    DEVICE,
    DTYPE,
    EMBEDDING_DIM,
    S_MIN,
    FUSION_SIMILARITY_THRESHOLD,
    FUSION_MIN_MASS,
    FUSION_MAX_VARIANCE,
    FUSION_VARIANCE_INCREASE_THRESHOLD,
    MIN_TRAINING_TEXTS,
)

logger = logging.getLogger(__name__)
console = Console() if RICH_AVAILABLE else None


def diagnose_fusion_potential(
    repository: KNRepository,
    similarity_threshold: Optional[float] = None,
    min_mass: Optional[int] = None,
    max_variance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Diagnose fusion potential of current KNs.
    
    Shows statistics and suggests more appropriate parameters.
    
    Args:
        repository: KNRepository instance
        similarity_threshold: Similarity threshold to test (default: FUSION_SIMILARITY_THRESHOLD)
        min_mass: Maximum mass to test (default: FUSION_MIN_MASS)
        max_variance: Maximum variance to test (default: FUSION_MAX_VARIANCE)
    
    Returns:
        Dict with statistics and recommendations
    """
    signatures = repository.get_all_signatures()
    
    if len(signatures) < 2:
        return {
            'total_kns': len(signatures),
            'message': 'At least 2 KNs are needed to fuse'
        }
    
    similarity_threshold = similarity_threshold or FUSION_SIMILARITY_THRESHOLD
    min_mass = min_mass or FUSION_MIN_MASS
    max_variance = max_variance or FUSION_MAX_VARIANCE
    
    # Calculate basic statistics
    masses = [sig["mass"] for sig in signatures]
    variances = [sig["variance"] for sig in signatures]
    
    stats = {
        'total_kns': len(signatures),
        'avg_mass': sum(masses) / len(masses),
        'min_mass': min(masses),
        'max_mass': max(masses),
        'avg_variance': sum(variances) / len(variances),
        'min_variance': min(variances),
        'max_variance': max(variances),
        'small_kns': sum(1 for m in masses if m <= min_mass),
        'stable_kns': sum(1 for v in variances if v <= max_variance),
        'small_stable_kns': sum(1 for sig in signatures 
                               if sig["mass"] <= min_mass and sig["variance"] <= max_variance),
    }
    
    # Calculate adjacency matrix for different thresholds
    # OPTIMIZATION: Calculate entire similarity matrix at once (vectorized)
    centroids = torch.stack([sig["centroid"].to(device=DEVICE, dtype=DTYPE) for sig in signatures])
    n = len(signatures)
    
    # Normalize centroids for cosine similarity (should already be normalized, but for safety)
    centroids_normalized = F.normalize(centroids, p=2, dim=1)
    
    # Calculate complete cosine similarity matrix (vectorized)
    # similarity_matrix[i, j] = cosine_similarity(centroids[i], centroids[j])
    similarity_matrix = torch.mm(centroids_normalized, centroids_normalized.t())
    
    # Extract only upper triangular part (without diagonal) to avoid duplicates
    # Create upper triangular mask
    upper_triangular_mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    similarities_upper = similarity_matrix[upper_triangular_mask].cpu().numpy()
    
    # Count pairs with different similarity thresholds (vectorized)
    similarity_counts = {}
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    for threshold in thresholds:
        # Count how many pairs exceed the threshold (vectorized)
        count = int((similarities_upper >= threshold).sum())
        similarity_counts[threshold] = count
    
    stats['similarity_pairs'] = similarity_counts
    
    # Recommendations
    recommendations = []
    
    if stats['small_stable_kns'] < 2:
        recommendations.append(
            f"⚠️  Only {stats['small_stable_kns']} KNs meet criteria (mass<={min_mass}, variance<={max_variance}). "
            f"Consider increasing FUSION_MIN_MASS or FUSION_MAX_VARIANCE."
        )
    
    if similarity_counts[similarity_threshold] == 0:
        # Try lower thresholds
        for threshold in sorted(similarity_counts.keys()):
            if similarity_counts[threshold] > 0:
                recommendations.append(
                    f"💡 With similarity>={threshold} you would find {similarity_counts[threshold]} pairs. "
                    f"Consider reducing FUSION_SIMILARITY_THRESHOLD from {similarity_threshold} to {threshold}."
                )
                break
    
    if stats['avg_mass'] > min_mass * 2:
        recommendations.append(
            f"💡 Average mass ({stats['avg_mass']:.1f}) is much greater than FUSION_MIN_MASS ({min_mass}). "
            f"Consider increasing FUSION_MIN_MASS to include more KNs."
        )
    
    if stats['avg_variance'] > max_variance * 2:
        recommendations.append(
            f"💡 Average variance ({stats['avg_variance']:.4f}) is greater than FUSION_MAX_VARIANCE ({max_variance}). "
            f"Consider increasing FUSION_MAX_VARIANCE to include more KNs."
        )
    
    stats['recommendations'] = recommendations
    
    return stats


def fuse_knowledge_nodes(
    repository: KNRepository,
    orchestrator: Orchestrator,
    progress_interval: int = 10,
    calculate_metadata: bool = True,
) -> Dict[str, int]:
    """
    Execute Knowledge Nodes fusion protocol.
    
    Flow:
    1. Calculate semantic adjacency matrix
    2. Identify fusion candidates (Small Stable Nodes)
    3. Simulate variance after fusion
    4. Execute valid fusions
    5. Re-assign orphan buffers
    6. Calculate training metadata (optional)
    
    Args:
        repository: KNRepository instance
        orchestrator: Orchestrator instance
        progress_interval: How many fusions before showing progress (default: 10)
        calculate_metadata: If True, calculate training metadata after fusion (default: True)
    
    Returns:
        Dict with statistics: {
            'initial_kns': int,
            'final_kns': int,
            'fusions_performed': int,
            'buffers_reassigned': int,
            'metadata_updated': int (if calculate_metadata=True)
        }
    """
    logger.info("=" * 60)
    logger.info("Starting Knowledge Nodes fusion protocol")
    logger.info("=" * 60)
    
    # Get initial signatures
    initial_signatures = repository.get_all_signatures()
    initial_kns_count = len(initial_signatures)
    
    if initial_kns_count < 2:
        logger.info("Less than 2 KNs, cannot perform fusion")
        return {
            'initial_kns': initial_kns_count,
            'final_kns': initial_kns_count,
            'fusions_performed': 0,
            'buffers_reassigned': 0
        }
    
    logger.info(f"Initial KNs: {initial_kns_count}")
    
    # Detect if we're in a notebook
    in_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    
    # Phase 1: Calculate semantic adjacency matrix
    logger.info("Phase 1: Calculating semantic adjacency matrix...")
    adjacency_matrix = _calculate_semantic_adjacency_matrix(
        initial_signatures,
        in_notebook=in_notebook
    )
    
    # Phase 2: Identify fusion candidates
    logger.info("Phase 2: Identifying fusion candidates...")
    fusion_candidates = _identify_fusion_candidates(
        initial_signatures,
        adjacency_matrix
    )
    
    if not fusion_candidates:
        logger.info("No fusion candidates found")
        final_signatures = repository.get_all_signatures()
        return {
            'initial_kns': initial_kns_count,
            'final_kns': len(final_signatures),
            'fusions_performed': 0,
            'buffers_reassigned': 0
        }
    
    logger.info(f"Fusion candidates identified: {len(fusion_candidates)} pairs")
    
    # Phase 3: Execute fusions
    logger.info("Phase 3: Executing fusions...")
    fusions_performed = _execute_fusions(
        fusion_candidates,
        repository,
        orchestrator,
        in_notebook=in_notebook,
        progress_interval=progress_interval
    )
    
    # ========================================================================
    # Post-Fusion: Marcar nodos que alcancen threshold (2025-01-14)
    # ========================================================================
    # Después de fusionar, algunos nodos pueden haber alcanzado masa >= 10
    # Necesitamos marcarlos para training si no están ya marcados
    # Esto evita entrenar nodos que se fusionarán después
    # ========================================================================
    logger.info("Post-fusion: Checking for nodes that reached training threshold...")
    final_signatures_after_fusion = repository.get_all_signatures()
    
    nodes_marked_post_fusion = 0
    for sig in final_signatures_after_fusion:
        node_id = sig['node_id']
        mass = sig['mass']
        
        # Si masa >= threshold y no está entrenado y no está ya marcado
        if mass >= TRAINING_THRESHOLD:
            if not repository.is_trained(node_id):
                pending = repository.get_pending_training_nodes()
                if node_id not in pending:
                    repository.mark_for_training(node_id, is_retraining=False)
                    nodes_marked_post_fusion += 1
                    logger.debug(
                        f"Node '{node_id}' marked for training post-fusion "
                        f"(mass={mass} >= {TRAINING_THRESHOLD})"
                    )
    
    if nodes_marked_post_fusion > 0:
        logger.info(
            f"Post-fusion: {nodes_marked_post_fusion} nodes marked for training "
            f"(reached threshold after fusion)"
        )
    else:
        logger.debug("Post-fusion: No additional nodes needed marking for training")
    
    # Phase 4: Consolidate micro-nodes (2025-01-27)
    # Forced merge for nodes with < 6 texts to avoid information loss
    logger.info("Phase 4: Consolidating micro-nodes (< 6 texts)...")
    micro_consolidated = consolidate_micro_nodes(
        repository,
        orchestrator,
        min_training_texts=MIN_TRAINING_TEXTS
    )
    
    # Phase 5: Re-assign orphan buffers
    logger.info("Phase 5: Re-assigning orphan buffers...")
    buffers_reassigned = _reassign_orphan_buffers(
        repository,
        orchestrator,
        in_notebook=in_notebook
    )
    
    # Phase 6: Calculate training metadata (optional)
    metadata_updated = 0
    if calculate_metadata:
        logger.info("Phase 6: Calculating training metadata...")
        try:
            metadata_stats = _calculate_training_metadata(repository, orchestrator)
            metadata_updated = metadata_stats.get('updated', 0)
            logger.info(
                f"Training metadata calculated: {metadata_updated} nodes updated "
                f"(avg_epochs={metadata_stats.get('avg_epochs', 0):.2f}, "
                f"range={metadata_stats.get('min_epochs', 0)}-{metadata_stats.get('max_epochs', 0)})"
            )
        except Exception as e:
            logger.warning(
                f"Error calculating training metadata: {e}. "
                f"Fusion completed successfully, but metadata calculation failed.",
                exc_info=True
            )
            # Don't fail fusion if metadata calculation fails
            metadata_updated = 0
    
    # Final statistics
    final_signatures = repository.get_all_signatures()
    final_kns_count = len(final_signatures)
    
    logger.info("=" * 60)
    logger.info("Fusion completed")
    logger.info(f"Initial KNs: {initial_kns_count}")
    logger.info(f"Final KNs: {final_kns_count}")
    logger.info(f"Fusions performed: {fusions_performed}")
    logger.info(f"Micro-nodes consolidated: {micro_consolidated}")
    logger.info(f"Buffers reassigned: {buffers_reassigned}")
    if calculate_metadata:
        logger.info(f"Metadata updated: {metadata_updated}")
    logger.info(f"Reduction: {initial_kns_count - final_kns_count} KNs ({((initial_kns_count - final_kns_count) / initial_kns_count * 100):.1f}%)")
    # Final commit to ensure all atomic migrations and deletions are persisted
    if repository.conn:
        repository.conn.commit()
    
    result = {
        'initial_kns': initial_kns_count,
        'final_kns': final_kns_count,
        'fusions_performed': fusions_performed,
        'micro_consolidated': micro_consolidated,
        'buffers_reassigned': buffers_reassigned
    }
    
    if calculate_metadata:
        result['metadata_updated'] = metadata_updated
    
    return result


def consolidate_micro_nodes(
    repository: KNRepository,
    orchestrator: Orchestrator,
    min_training_texts: int = 6,
    similarity_floor: float = 0.55,
) -> int:
    """
    Forcefully merge nodes with fewer than min_training_texts with their 
    most similar semantic neighbor to avoid information loss.
    
    Args:
        repository: KNRepository instance
        orchestrator: Orchestrator instance
        min_training_texts: Threshold to consider a node "micro"
        similarity_floor: Minimum similarity to allow forced merge (safety block)
        
    Returns:
        int: Number of micro-nodes consolidated
    """
    signatures = repository.get_all_signatures()
    # CRITICAL: Filter out SEED nodes from mandatory consolidation
    # SEED nodes, even if small, must preserve their identity
    micro_nodes = [
        sig for sig in signatures 
        if sig["mass"] <= min_training_texts and sig.get("origin_type", "ORGANIC") != "SEED"
    ]
    
    if not micro_nodes or len(signatures) < 2:
        return 0
        
    logger.info(f"Checking {len(micro_nodes)} micro-nodes (mass <= {min_training_texts}) for mandatory consolidation...")
    consolidated_count = 0
    processed_this_run: Set[str] = set()
    
    # Sort micro-nodes by mass ascending (smallest first)
    micro_nodes.sort(key=lambda x: x["mass"])
    
    for micro_sig in micro_nodes:
        micro_id = micro_sig["node_id"]
        if micro_id in processed_this_run:
            continue
            
        # Refetch all signatures to account for previous fusions in this loop
        current_sigs = repository.get_all_signatures()
        candidates = [s for s in current_sigs if s["node_id"] != micro_id]
        
        if not candidates:
            continue
            
        micro_centroid = micro_sig["centroid"].to(device=DEVICE, dtype=DTYPE)
        
        best_target_id = None
        best_sim = -1.0
        
        for cand in candidates:
            sim = F.cosine_similarity(
                micro_centroid.unsqueeze(0),
                cand["centroid"].to(device=DEVICE, dtype=DTYPE).unsqueeze(0),
                dim=1
            ).item()
            
            if sim > best_sim:
                best_sim = sim
                best_target_id = cand["node_id"]
        
        # Mandatory merge if above safety floor
        if best_target_id and best_sim >= similarity_floor:
            try:
                merged_id = _merge_two_kns(
                    micro_id,
                    best_target_id,
                    repository,
                    orchestrator
                )
                if merged_id:
                    consolidated_count += 1
                    processed_this_run.add(micro_id)
                    processed_this_run.add(best_target_id)
                    logger.debug(f"Micro-node {micro_id[:8]} consolidated with {best_target_id[:8]} (sim={best_sim:.3f})")
            except Exception as e:
                logger.error(f"Failed to consolidate micro-node {micro_id}: {e}")
                
    return consolidated_count


def _calculate_semantic_adjacency_matrix(
    signatures: List[Dict[str, Any]],
    in_notebook: bool = False,
) -> Dict[Tuple[str, str], float]:
    """
    Calculate semantic adjacency matrix between all pairs of KNs.
    
    Args:
        signatures: List of KN signatures
        in_notebook: If we're in a notebook (for progress bar)
    
    Returns:
        Dict with keys (node_id_1, node_id_2) and cosine similarity values
    """
    n = len(signatures)
    if n < 2:
        return {}
    
    adjacency = {}
    total_pairs = n * (n - 1) // 2
    
    # Progress bar - initialize variables first
    progress = None
    task = None
    
    if in_notebook and TQDM_AVAILABLE:
        pairs_iter = tqdm(
            range(n),
            desc="Calculating semantic adjacency",
            unit="KN",
            ncols=100
        )
    elif RICH_AVAILABLE and console and not in_notebook:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>6.2f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        task = progress.add_task(
            "[cyan]Calculating semantic adjacency...",
            total=total_pairs
        )
        pairs_iter = range(n)
        progress.start()
    else:
        pairs_iter = range(n)
    
    try:
        processed = 0
        for i in pairs_iter:
            sig_i = signatures[i]
            centroid_i = sig_i["centroid"].to(device=DEVICE, dtype=DTYPE)
            
            for j in range(i + 1, n):
                sig_j = signatures[j]
                centroid_j = sig_j["centroid"].to(device=DEVICE, dtype=DTYPE)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(
                    centroid_i.unsqueeze(0),
                    centroid_j.unsqueeze(0),
                    dim=1
                ).item()
                
                # Save in both directions (symmetric)
                adjacency[(sig_i["node_id"], sig_j["node_id"])] = similarity
                adjacency[(sig_j["node_id"], sig_i["node_id"])] = similarity
                
                processed += 1
                if progress and task is not None:
                    progress.update(task, advance=1)
    finally:
        if progress:
            progress.stop()
    
    logger.debug(f"Adjacency matrix calculated: {len(adjacency)} pairs")
    return adjacency


def _identify_fusion_candidates(
    signatures: List[Dict[str, Any]],
    adjacency_matrix: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str, float]]:
    """
    Identify fusion candidates based on:
    - Small Stable Nodes (low mass, low variance)
    - High semantic similarity
    
    Args:
        signatures: List of KN signatures
        adjacency_matrix: Semantic adjacency matrix
    
    Returns:
        List of tuples (node_id_1, node_id_2, similarity) sorted by similarity descending
    """
    candidates = []
    
    # Create dict for fast signature access
    signatures_dict = {sig["node_id"]: sig for sig in signatures}
    
    # Identify "Small Stable Nodes"
    small_stable_nodes = []
    stats = {
        'total_kns': len(signatures),
        'small_kns': 0,  # mass <= FUSION_MIN_MASS
        'stable_kns': 0,  # variance <= FUSION_MAX_VARIANCE
        'small_stable_kns': 0,  # both criteria
        'high_similarity_pairs': 0,  # pairs with similarity >= threshold
        'variance_rejected': 0  # rejected due to variance increase
    }
    
    for sig in signatures:
        # CRITICAL: SEED nodes are immune to being absorbed as "small" nodes
        is_seed = sig.get("origin_type", "ORGANIC") == "SEED"
        if is_seed:
            continue

        is_small = sig["mass"] <= FUSION_MIN_MASS
        is_stable = sig["variance"] <= FUSION_MAX_VARIANCE
        
        if is_small:
            stats['small_kns'] += 1
        if is_stable:
            stats['stable_kns'] += 1
        if is_small and is_stable:
            stats['small_stable_kns'] += 1
            small_stable_nodes.append(sig["node_id"])
    
    logger.info(
        f"Fusion candidates diagnosis:\n"
        f"  - Total KNs: {stats['total_kns']}\n"
        f"  - Small KNs (mass <= {FUSION_MIN_MASS}): {stats['small_kns']}\n"
        f"  - Stable KNs (variance <= {FUSION_MAX_VARIANCE}): {stats['stable_kns']}\n"
        f"  - Small and stable KNs: {stats['small_stable_kns']}"
    )
    
    # Search for pairs of Small Stable Nodes with high similarity
    for i, node_id_1 in enumerate(small_stable_nodes):
        for node_id_2 in small_stable_nodes[i + 1:]:
            similarity = adjacency_matrix.get((node_id_1, node_id_2), 0.0)
            
            if similarity >= FUSION_SIMILARITY_THRESHOLD:
                stats['high_similarity_pairs'] += 1
                # Verify that fusion doesn't increase variance too much
                sig_1 = signatures_dict[node_id_1]
                sig_2 = signatures_dict[node_id_2]
                
                # Simulate variance after fusion
                simulated_variance = _simulate_merged_variance(sig_1, sig_2)
                
                # Verify that variance increase is acceptable
                max_variance = max(sig_1["variance"], sig_2["variance"])
                variance_increase = simulated_variance - max_variance
                
                if variance_increase <= FUSION_VARIANCE_INCREASE_THRESHOLD:
                    candidates.append((node_id_1, node_id_2, similarity))
                else:
                    stats['variance_rejected'] += 1
    
    # Diagnostic logging
    if len(candidates) == 0:
        logger.warning(
            f"No fusion candidates found. Diagnosis:\n"
            f"  - Small and stable KNs: {stats['small_stable_kns']}\n"
            f"  - Pairs with similarity >= {FUSION_SIMILARITY_THRESHOLD}: {stats['high_similarity_pairs']}\n"
            f"  - Rejected due to variance increase: {stats['variance_rejected']}\n"
            f"  - Current parameters: similarity>={FUSION_SIMILARITY_THRESHOLD}, "
            f"mass<={FUSION_MIN_MASS}, variance<={FUSION_MAX_VARIANCE}"
        )
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    logger.debug(f"Valid fusion candidates: {len(candidates)}")
    return candidates


def _simulate_merged_variance(
    sig_1: Dict[str, Any],
    sig_2: Dict[str, Any],
) -> float:
    """
    Simulate the variance a merged KN would have.
    
    Uses the variance combination formula for two samples:
    - Combined variance = (n1*var1 + n2*var2 + n1*n2*(mean1-mean2)²/(n1+n2)) / (n1+n2-1)
    
    Args:
        sig_1: First KN signature
        sig_2: Second KN signature
    
    Returns:
        Simulated variance of the merged KN
    """
    n1 = sig_1["mass"]
    n2 = sig_2["mass"]
    var1 = sig_1["variance"]
    var2 = sig_2["variance"]
    centroid1 = sig_1["centroid"].to(device=DEVICE, dtype=DTYPE)
    centroid2 = sig_2["centroid"].to(device=DEVICE, dtype=DTYPE)
    
    # Calculate difference between centroids (Euclidean distance squared)
    centroid_diff = centroid1 - centroid2
    centroid_diff_sq = torch.dot(centroid_diff, centroid_diff).item()
    
    # Variance combination formula
    if n1 + n2 <= 1:
        return 0.0
    
    # Combined variance (approximation)
    # Note: This is an approximation. Real variance would require recalculating from embeddings.
    combined_variance = (
        (n1 - 1) * var1 + 
        (n2 - 1) * var2 + 
        (n1 * n2 * centroid_diff_sq) / (n1 + n2)
    ) / (n1 + n2 - 1)
    
    return max(0.0, combined_variance)


def _execute_fusions(
    candidates: List[Tuple[str, str, float]],
    repository: KNRepository,
    orchestrator: Orchestrator,
    in_notebook: bool = False,
    progress_interval: int = 10,
) -> int:
    """
    Execute fusions of identified candidates.
    
    Args:
        candidates: List of fusion candidates (node_id_1, node_id_2, similarity)
        repository: KNRepository instance
        orchestrator: Orchestrator instance
        in_notebook: If we're in a notebook (for progress bar)
        progress_interval: How many fusions before showing progress
    
    Returns:
        Number of successfully performed fusions
    """
    fusions_performed = 0
    processed_nodes: Set[str] = set()  # To avoid fusing the same node twice
    
    # Progress bar - initialize variables first
    progress = None
    task = None
    
    if in_notebook and TQDM_AVAILABLE:
        candidates_iter = tqdm(
            candidates,
            desc="Fusing KNs",
            unit="fusion",
            ncols=100
        )
    elif RICH_AVAILABLE and console and not in_notebook:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>6.2f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        task = progress.add_task(
            "[cyan]Fusing KNs...",
            total=len(candidates)
        )
        candidates_iter = candidates
        progress.start()
    else:
        candidates_iter = candidates
    
    try:
        for idx, (node_id_1, node_id_2, similarity) in enumerate(candidates_iter):
            # Verify that neither node has already been fused
            if node_id_1 in processed_nodes or node_id_2 in processed_nodes:
                if progress and task is not None:
                    progress.update(task, advance=1)
                continue
            
            # Verify that both nodes still exist
            sig_1 = repository.get_signature(node_id_1)
            sig_2 = repository.get_signature(node_id_2)
            
            if not sig_1 or not sig_2:
                if progress and task is not None:
                    progress.update(task, advance=1)
                continue
            
            # Execute fusion
            try:
                merged_node_id = _merge_two_kns(
                    node_id_1,
                    node_id_2,
                    repository,
                    orchestrator
                )
                
                if merged_node_id:
                    fusions_performed += 1
                    processed_nodes.add(node_id_1)
                    processed_nodes.add(node_id_2)
                    logger.debug(
                        f"Successful fusion: {node_id_1[:8]}... + {node_id_2[:8]}... -> "
                        f"{merged_node_id[:8]}... (similarity={similarity:.3f})"
                    )
                
                if progress and task is not None:
                    progress.update(task, advance=1)
                
                # Periodic logging
                if (idx + 1) % progress_interval == 0:
                    logger.info(
                        f"Fusions processed: {idx + 1}/{len(candidates)} | "
                        f"Successful fusions: {fusions_performed}"
                    )
            except Exception as e:
                logger.error(
                    f"Error fusing {node_id_1} and {node_id_2}: {e}",
                    exc_info=True
                )
                if progress and task is not None:
                    progress.update(task, advance=1)
    finally:
        if progress:
            progress.stop()
    
    return fusions_performed


def _merge_two_kns(
    node_id_1: str,
    node_id_2: str,
    repository: KNRepository,
    orchestrator: Orchestrator,
) -> Optional[str]:
    """
    Merge two Knowledge Nodes into a new one.
    
    Strategy:
    - If there are embeddings in node_memory, use them to calculate precise statistics
    - If no embeddings, use approximation based on centroids and masses
    
    Args:
        node_id_1: ID of first KN
        node_id_2: ID of second KN
        repository: KNRepository instance
        orchestrator: Orchestrator instance
    
    Returns:
        ID of the new merged KN, or None if it fails
    """
    # Get signatures
    sig_1 = repository.get_signature(node_id_1)
    sig_2 = repository.get_signature(node_id_2)
    
    if not sig_1 or not sig_2:
        logger.warning(f"Cannot get signatures for {node_id_1} or {node_id_2}")
        return None
    
    # Get embeddings from both nodes (may be empty)
    embeddings_1 = repository.get_node_embeddings(node_id_1)
    embeddings_2 = repository.get_node_embeddings(node_id_2)
    all_embeddings = embeddings_1 + embeddings_2
    
    # Calculate merged node statistics
    n1 = sig_1["mass"]
    n2 = sig_2["mass"]
    total_mass = n1 + n2
    
    centroid_1 = sig_1["centroid"].to(device=DEVICE, dtype=DTYPE)
    centroid_2 = sig_2["centroid"].to(device=DEVICE, dtype=DTYPE)
    
    # Combined centroid: weighted average
    merged_centroid = (centroid_1 * n1 + centroid_2 * n2) / total_mass
    
    # Calculate initial variance
    if all_embeddings and len(all_embeddings) > 1:
        # Case 1: We have individual embeddings - calculate precise M2
        embeddings_tensor = torch.stack(all_embeddings).to(device=DEVICE, dtype=DTYPE)
        initial_m2 = 0.0
        
        running_mean = embeddings_tensor[0].clone()
        for i in range(1, len(all_embeddings)):
            delta = embeddings_tensor[i] - running_mean
            running_mean += delta / (i + 1)
            delta2 = embeddings_tensor[i] - running_mean
            initial_m2 += torch.dot(delta, delta2).item()
    else:
        # Case 2: No individual embeddings - approximate M2 from variances
        # Use variance combination formula
        var1 = sig_1["variance"]
        var2 = sig_2["variance"]
        
        # M2 = variance * (mass - 1)
        m2_1 = var1 * (n1 - 1) if n1 > 1 else 0.0
        m2_2 = var2 * (n2 - 1) if n2 > 1 else 0.0
        
        # Difference between centroids (Euclidean distance squared)
        centroid_diff = centroid_1 - centroid_2
        centroid_diff_sq = torch.dot(centroid_diff, centroid_diff).item()
        
        # Variance combination formula for M2
        if total_mass > 1:
            initial_m2 = (
                m2_1 + m2_2 + 
                (n1 * n2 * centroid_diff_sq) / total_mass
            )
        else:
            initial_m2 = 0.0
        
        logger.debug(
            f"Fusion without individual embeddings: using approximation "
            f"(M2_1={m2_1:.4f}, M2_2={m2_2:.4f}, diff_sq={centroid_diff_sq:.4f})"
        )
    
    # Create new KnowledgeNode
    merged_node_id = f"kn_{uuid4()}"
    
    # ========================================================================
    # Create KnowledgeNode temporarily for processing embeddings (2025-12-27)
    # ========================================================================
    # We need to create the node directly first to process embeddings,
    # then save to Repository, then reload using from_repository() to get
    # PEFT metadata synchronized
    # ========================================================================
    kn = KnowledgeNode(
        node_id=merged_node_id,
        initial_centroid=merged_centroid,
        initial_mass=total_mass,
        initial_variance=initial_m2,
    )
    
    # If we have embeddings, process them to update statistics
    if all_embeddings:
        for emb in all_embeddings:
            kn.process(emb)
    
    # Get final signature
    signature = kn.get_signature()
    
    # Save new KN in Repository
    repository.save_new_kn(
        node_id=signature["node_id"],
        centroid=signature["centroid"],
        mass=signature["mass"],
        variance=signature["variance"],
    )
    
    # ========================================================================
    # Load KnowledgeNode using from_repository() (2025-12-27)
    # ========================================================================
    try:
        kn_loaded = KnowledgeNode.from_repository(repository, merged_node_id)
        kn = kn_loaded
    except ValueError as e:
        logger.error(f"Failed to load merged node '{merged_node_id}': {e}. Using direct node.")
        pass

    # ========================================================================
    # Herencia de Estado para Deferred Training (2025-01-14)
    # ========================================================================
    # Aplicar reglas de herencia antes de eliminar nodos originales
    pending_training_nodes = repository.get_pending_training_nodes()
    needs_training_1 = node_id_1 in pending_training_nodes
    needs_training_2 = node_id_2 in pending_training_nodes
    is_trained_1 = repository.is_trained(node_id_1)
    is_trained_2 = repository.is_trained(node_id_2)
    
    if needs_training_1 or needs_training_2:
        repository.mark_for_training(merged_node_id, is_retraining=False)
    
    if is_trained_1 or is_trained_2:
        repository.mark_for_training(merged_node_id, is_retraining=True)
    
    if total_mass >= TRAINING_THRESHOLD:
        if not repository.is_trained(merged_node_id):
            repository.mark_for_training(merged_node_id, is_retraining=False)

    # ========================================================================
    # CRITICAL: Atomic Pointer Migration (2025-01-27)
    # ========================================================================
    try:
        count = repository.reassign_all_pointers([node_id_1, node_id_2], merged_node_id)
        if count > 0:
            logger.debug(f"Atomically transferred {count} pointers to merged node '{merged_node_id}'")
    except Exception as e:
        logger.error(f"Error during atomic pointer migration: {e}.")
    
    # Delete original nodes
    repository.delete_kn(node_id_1)
    repository.delete_kn(node_id_2)
    
    # Update orchestrator
    if node_id_1 in orchestrator.active_nodes:
        del orchestrator.active_nodes[node_id_1]
    if node_id_2 in orchestrator.active_nodes:
        del orchestrator.active_nodes[node_id_2]
    
    orchestrator.active_nodes[merged_node_id] = kn
    orchestrator.kn_count = len(orchestrator.active_nodes)
    
    logger.info(
        f"Merged KN created: {merged_node_id} "
        f"(mass={signature['mass']}, variance={signature['variance']:.4f})"
    )
    
    return merged_node_id


def _reassign_orphan_buffers(
    repository: KNRepository,
    orchestrator: Orchestrator,
    in_notebook: bool = False,
) -> int:
    """
    Re-assign orphan buffers to existing KNs if they have sufficient similarity.
    
    Args:
        repository: KNRepository instance
        orchestrator: Orchestrator instance
        in_notebook: If we're in a notebook (for progress bar)
    
    Returns:
        Number of buffers reassigned
    """
    buffer_ids = repository.get_all_active_buffers()
    if not buffer_ids:
        return 0
    
    signatures = repository.get_all_signatures()
    if not signatures:
        return 0
    
    reassigned = 0
    
    # Progress bar - initialize variables first
    progress = None
    task = None
    
    if in_notebook and TQDM_AVAILABLE:
        buffers_iter = tqdm(
            buffer_ids,
            desc="Re-assigning buffers",
            unit="buffer",
            ncols=100
        )
    elif RICH_AVAILABLE and console and not in_notebook:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>6.2f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        task = progress.add_task(
            "[cyan]Re-assigning buffers...",
            total=len(buffer_ids)
        )
        buffers_iter = buffer_ids
        progress.start()
    else:
        buffers_iter = buffer_ids
    
    try:
        for buffer_id in buffers_iter:
            buffer_centroid = repository.get_buffer_centroid(buffer_id)
            if buffer_centroid is None:
                if progress and task is not None:
                    progress.update(task, advance=1)
                continue
            
            # Find the most similar KN
            best_node_id = None
            best_similarity = -1.0
            
            for sig in signatures:
                centroid = sig["centroid"].to(device=DEVICE, dtype=DTYPE)
                similarity = F.cosine_similarity(
                    buffer_centroid.unsqueeze(0),
                    centroid.unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_node_id = sig["node_id"]
            
            # If similarity is sufficient, re-assign
            if best_similarity >= S_MIN and best_node_id:
                buffer_embeddings = repository.get_buffer_embeddings(buffer_id)
                
                # Add embeddings to KN
                for emb in buffer_embeddings:
                    repository.add_embedding_to_memory(best_node_id, emb)
                    
                    # Update KN if it's in active_nodes
                    if best_node_id in orchestrator.active_nodes:
                        kn = orchestrator.active_nodes[best_node_id]
                        kn.process(emb)
                        
                        # Update signature in Repository
                        signature = kn.get_signature()
                        repository.update_node_stats(
                            node_id=signature["node_id"],
                            centroid=signature["centroid"],
                            mass=signature["mass"],
                            variance=signature["variance"],
                        )
                
                # Delete buffer
                repository.delete_buffer(buffer_id)
                reassigned += 1
                logger.debug(
                    f"Buffer {buffer_id} reassigned to KN {best_node_id} "
                    f"(similarity={best_similarity:.3f})"
                )
            
            if progress and task is not None:
                progress.update(task, advance=1)
    finally:
        if progress:
            progress.stop()
    
    return reassigned


def _calculate_training_epochs(
    mass: int, 
    variance: float, 
    num_texts: Optional[int] = None,
    parent_similarity: Optional[float] = None
) -> int:
    """
    Calculate estimated number of epochs based on node complexity.
    
    Formula: complexity = log1p(mass) * (1 + variance)
    Epochs = clip(complexity * 0.5, 1, 7)
    
    Inheritance Adjustment:
    If parent_similarity is provided, apply discount:
    discount = 1.0 - (parent_similarity * 0.5)
    complexity *= discount
    
    Volume Discount (2025-01-XX):
    If mass > 200, apply 0.7x multiplier to final epochs (after mapping).
    Rationale: Massive nodes have enough data diversity, multiple epochs
    are redundant and risk overfitting. This follows best practices from
    high-performance ML centers.
    
    Safety Floor:
    Always guarantees at least 1 epoch, regardless of discounts applied.
    Multiple validation layers ensure this: max(1, int(epochs * 0.7)) and
    final safety check: max(1, epochs).
    
    Args:
        mass: Node mass (number of embeddings)
        variance: Node variance (semantic dispersion)
        num_texts: Number of texts for training (optional, for adjustment)
        parent_similarity: Cosine similarity to parent node (0.0-1.0), or None
    
    Returns:
        int: Estimated number of epochs (1-7, always >= 1)
    """
    # Validate inputs
    if mass <= 0:
        logger.warning(f"Invalid mass for epoch calculation: {mass}, using default 2")
        return 2
    if variance < 0:
        logger.warning(f"Invalid variance for epoch calculation: {variance}, using 0")
        variance = 0.0
    
    # Base complexity: logarithmic growth with mass, penalized by variance
    complexity = np.log1p(mass) * (1 + variance)
    
    # Special adjustments
    if variance == 0:
        # Perfectly cohesive node: fewer epochs needed
        complexity *= 0.8
    elif variance > 1.5:
        # Very high variance: more epochs needed
        complexity *= 1.2
    
    if mass < 5:
        # Very small nodes: more epochs to compensate for low volume
        complexity *= 1.3
    
    # Adjust by number of texts if available
    if num_texts is not None:
        if num_texts < 20:
            complexity *= 1.2  # Small datasets: more epochs
        elif num_texts > 100:
            complexity *= 0.8  # Large datasets: fewer epochs

    # ========================================================================
    # Inheritance Discount (2025-01-XX)
    # ========================================================================
    if parent_similarity is not None and parent_similarity > 0:
        # Discount factor: 1.0 - (similarity * 0.5)
        # e.g., sim=0.9 -> discount=0.55 (45% reduction)
        # e.g., sim=0.6 -> discount=0.70 (30% reduction)
        discount = 1.0 - (parent_similarity * 0.5)
        
        # Safety clip for discount (shouldn't be needed if sim is 0-1, but safe)
        discount = np.clip(discount, 0.5, 1.0)
        
        complexity *= discount
        # logger.debug(f"Applied inheritance discount: {discount:.2f} (sim={parent_similarity:.2f})")
    
    # ========================================================================
    # PASO 1: Mapear complejidad a épocas (como siempre)
    # ========================================================================
    # Map to epochs (factor 0.5, range 1-7)
    # CRITICAL: Safety Floor = 1 epoch (never 0)
    epochs = int(np.clip(complexity * 0.5, 1, 7))
    
    # ========================================================================
    # PASO 2: Aplicar descuento por volumen a épocas finales (2025-01-XX)
    # ========================================================================
    # Rationale: Nodes with mass > 200 have enough data diversity
    # Multiple epochs are redundant and risk overfitting
    # This follows best practices from high-performance ML centers
    # IMPORTANT: This is applied AFTER mapping to epochs, not to complexity
    # It's an additional optimization layer, not a replacement for inheritance discount
    # ========================================================================
    if mass > 200:
        # Apply volume discount: 0.7x multiplier to final epochs
        # CRITICAL: Always maintain minimum of 1 epoch (safety floor)
        epochs_before = epochs
        epochs = max(1, int(epochs * 0.7))  # max(1, ...) ensures minimum of 1
        
        logger.debug(
            f"Applied volume discount to final epochs: "
            f"mass={mass}, epochs_before={epochs_before}, "
            f"epochs_after={epochs}"
        )
    
    # ========================================================================
    # Safety Floor: Always ensure at least 1 epoch (2025-01-XX)
    # ========================================================================
    # CRITICAL: No matter what discounts are applied, every node must train
    # for at least 1 epoch. This ensures the model sees the data at least once.
    # ========================================================================
    epochs = max(1, epochs)  # Final safety check
    
    return epochs


def _calculate_training_priority(mass: int, variance: float) -> float:
    """
    Calculate training priority based on information density.
    
    Formula: priority = mass / (1 + variance)
    Higher mass + lower variance = higher priority (better expert)
    
    Args:
        mass: Node mass (number of embeddings)
        variance: Node variance (semantic dispersion)
    
    Returns:
        float: Training priority (higher = more important)
    """
    if variance < 0:
        logger.warning(f"Invalid variance for priority calculation: {variance}, using 0")
        variance = 0.0
    
    # Information density: mass / (1 + variance)
    # +1 prevents division by zero and smooths variance effect
    priority = mass / (1 + variance)
    
    return priority


def _calculate_training_metadata(
    repository: KNRepository,
    orchestrator: Optional[Orchestrator] = None
) -> Dict[str, Any]:
    """
    Calculate estimated_epochs and training_priority for nodes that need training.
    
    This function is called after fusion to ensure all nodes have updated metadata.
    It analyzes nodes with needs_training=1 and calculates:
    - estimated_epochs: Based on complexity (mass, variance, num_texts)
    - training_priority: Based on information density (mass / (1 + variance))
    
    Args:
        repository: KNRepository instance
        orchestrator: Optional Orchestrator (for accessing active_nodes if needed)
    
    Returns:
        dict: Statistics about metadata calculation {
            'total': int,
            'updated': int,
            'errors': int,
            'avg_epochs': float,
            'min_epochs': int,
            'max_epochs': int
        }
    """
    try:
        # Get nodes that need training
        cursor = repository.conn.cursor()
        cursor.execute("""
            SELECT 
                n.node_id, 
                n.mass, 
                n.variance,
                n.parent_similarity,     -- Added Phase 3
                COUNT(ndm.source_id) as num_texts
            FROM nodes n
            LEFT JOIN node_data_mapping ndm ON n.node_id = ndm.node_id
            WHERE n.needs_training = 1 
            AND n.mass > 0 
            AND n.variance IS NOT NULL
            GROUP BY n.node_id, n.mass, n.variance, n.parent_similarity
        """)
        rows = cursor.fetchall()
        
        if not rows:
            logger.debug("No nodes marked for training, skipping metadata calculation")
            return {
                'total': 0,
                'updated': 0,
                'errors': 0,
                'avg_epochs': 0.0,
                'min_epochs': 0,
                'max_epochs': 0
            }
        
        updated_count = 0
        error_count = 0
        epochs_list = []
        
        for row in rows:
            node_id = row["node_id"]
            mass = row["mass"]
            variance = row["variance"] or 0.0
            num_texts = row["num_texts"] or None
            
            try:
                # Calculate epochs and priority
                # Phase 3: Pass parent_similarity if available in row (need to fetch it first)
                parent_similarity = row["parent_similarity"] if "parent_similarity" in row.keys() else None
                
                epochs = _calculate_training_epochs(mass, variance, num_texts, parent_similarity)
                priority = _calculate_training_priority(mass, variance)
                
                # Update in repository
                repository.update_training_metadata(
                    node_id=node_id,
                    estimated_epochs=epochs,
                    training_priority=priority
                )
                
                updated_count += 1
                epochs_list.append(epochs)
                
                logger.debug(
                    f"Node {node_id}: mass={mass}, variance={variance:.3f}, "
                    f"texts={num_texts}, epochs={epochs}, priority={priority:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error processing node {node_id} for metadata: {e}")
                error_count += 1
                continue
        
        # Calculate statistics
        avg_epochs = np.mean(epochs_list) if epochs_list else 0.0
        min_epochs = int(min(epochs_list)) if epochs_list else 0
        max_epochs = int(max(epochs_list)) if epochs_list else 0
        
        result = {
            'total': len(rows),
            'updated': updated_count,
            'errors': error_count,
            'avg_epochs': float(avg_epochs),
            'min_epochs': min_epochs,
            'max_epochs': max_epochs
        }
        
        logger.debug(
            f"Training metadata calculation completed: "
            f"{updated_count}/{len(rows)} nodes updated, "
            f"avg_epochs={avg_epochs:.2f}, range={min_epochs}-{max_epochs}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in _calculate_training_metadata: {e}", exc_info=True)
        raise
