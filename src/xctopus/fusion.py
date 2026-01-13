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
    DEVICE,
    DTYPE,
    EMBEDDING_DIM,
    FUSION_SIMILARITY_THRESHOLD,
    FUSION_MIN_MASS,
    FUSION_MAX_VARIANCE,
    FUSION_VARIANCE_INCREASE_THRESHOLD,
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
) -> Dict[str, int]:
    """
    Execute Knowledge Nodes fusion protocol.
    
    Flow:
    1. Calculate semantic adjacency matrix
    2. Identify fusion candidates (Small Stable Nodes)
    3. Simulate variance after fusion
    4. Execute valid fusions
    5. Re-assign orphan buffers
    
    Args:
        repository: KNRepository instance
        orchestrator: Orchestrator instance
        progress_interval: How many fusions before showing progress (default: 10)
    
    Returns:
        Dict with statistics: {
            'initial_kns': int,
            'final_kns': int,
            'fusions_performed': int,
            'buffers_reassigned': int
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
    
    # Phase 4: Re-assign orphan buffers
    logger.info("Phase 4: Re-assigning orphan buffers...")
    buffers_reassigned = _reassign_orphan_buffers(
        repository,
        orchestrator,
        in_notebook=in_notebook
    )
    
    # Final statistics
    final_signatures = repository.get_all_signatures()
    final_kns_count = len(final_signatures)
    
    logger.info("=" * 60)
    logger.info("Fusion completed")
    logger.info(f"Initial KNs: {initial_kns_count}")
    logger.info(f"Final KNs: {final_kns_count}")
    logger.info(f"Fusions performed: {fusions_performed}")
    logger.info(f"Buffers reassigned: {buffers_reassigned}")
    logger.info(f"Reduction: {initial_kns_count - final_kns_count} KNs ({((initial_kns_count - final_kns_count) / initial_kns_count * 100):.1f}%)")
    logger.info("=" * 60)
    
    return {
        'initial_kns': initial_kns_count,
        'final_kns': final_kns_count,
        'fusions_performed': fusions_performed,
        'buffers_reassigned': buffers_reassigned
    }


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
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
    # Changed from keeping the directly created KnowledgeNode to loading
    # from Repository using from_repository(). This ensures consistency:
    # - PEFT metadata is loaded correctly (even if None)
    # - Training status is loaded correctly (even if None)
    # - Node state matches Repository state exactly
    # ========================================================================
    try:
        # Reload from Repository to ensure metadata is synchronized
        kn_loaded = KnowledgeNode.from_repository(repository, merged_node_id)
        kn = kn_loaded
    except ValueError as e:
        # This should never happen since we just saved the node
        logger.error(
            f"Failed to load merged node '{merged_node_id}' from Repository after saving: {e}. "
            f"Using directly created node (metadata may be incomplete)."
        )
        # Fallback: use the directly created node (metadata may be incomplete)
        pass
    
    # Move embeddings to new node (if they exist)
    if all_embeddings:
        for emb in all_embeddings:
            repository.add_embedding_to_memory(merged_node_id, emb)
    
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
    from .settings import S_MIN
    
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
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
