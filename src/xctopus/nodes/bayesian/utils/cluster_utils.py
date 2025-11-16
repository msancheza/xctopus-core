"""
Cluster utility functions for orphan detection and cluster analysis.

This module provides unified functions for cluster analysis that can be used
across different contexts (dynamic_clustering_pipeline, ClusterAnalyzer, etc.)
to avoid code duplication.
"""

from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional import for DynamicClusteringConfig (for defaults)
try:
    from .clustering_config import DynamicClusteringConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    DynamicClusteringConfig = None


def identify_orphan_clusters(
    labels: Union[List, Any],
    min_size: int = 3,
    include_outliers: bool = True,
    return_dict: bool = False
) -> Union[Tuple[Dict[int, int], List[int]], Dict[str, Any]]:
    """
    Unified function to identify orphan clusters (too small) and outliers.
    
    This function can be used by both dynamic_clustering_pipeline.py and
    ClusterAnalyzer to avoid code duplication.
    
    Args:
        labels: Array or list of cluster labels (can be numpy array, list, etc.)
        min_size: Minimum size threshold to not be considered orphan. Default: 3
        include_outliers: Whether to detect and include outliers (label == -1). Default: True
        return_dict: If True, return structured dict; if False, return tuple for backward compatibility
    
    Returns:
        If return_dict=False (backward compatibility):
            Tuple (orphan_clusters_dict, outliers_list) where:
            - orphan_clusters_dict: {cluster_id: size} for clusters smaller than min_size
            - outliers_list: [index1, index2, ...] for indices with label == -1
        
        If return_dict=True:
            Dictionary with:
            - 'orphans': List of tuples [(cluster_id, size), ...] sorted by size
            - 'outliers': List of indices [i, j, ...] with outlier labels
            - 'n_orphans': Number of orphan clusters
            - 'n_outliers': Number of outliers
    
    Example:
        >>> labels = [0, 0, 0, 1, 1, 2, -1, -1]
        >>> orphans_dict, outliers = identify_orphan_clusters(labels, min_size=3)
        >>> print(orphans_dict)  # {2: 1}  (cluster 2 has only 1 element)
        >>> print(outliers)  # [6, 7]  (indices with label -1)
        
        >>> result = identify_orphan_clusters(labels, min_size=3, return_dict=True)
        >>> print(result['orphans'])  # [(2, 1)]
        >>> print(result['n_orphans'])  # 1
    """
    # Convert to list if needed (handles numpy arrays, pandas Series, etc.)
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    elif hasattr(labels, '__iter__') and not isinstance(labels, str):
        labels_list = list(labels)
    else:
        labels_list = [labels]
    
    # Count cluster sizes efficiently
    cluster_sizes = Counter(labels_list)
    
    # Detect orphan clusters (small clusters)
    orphans = []
    for cluster_id, size in cluster_sizes.items():
        if cluster_id == -1:
            # Outliers are handled separately
            continue
        elif size < min_size:
            orphans.append((cluster_id, size))
    
    # Detect outliers (label == -1) - calculate once, efficiently
    outliers = []
    if include_outliers:
        outliers = [i for i, label in enumerate(labels_list) if label == -1]
    
    # Return format based on return_dict parameter
    if return_dict:
        # Return structured dictionary with metadata
        return {
            'orphans': sorted(orphans, key=lambda x: x[1]),  # Sort by size
            'outliers': outliers,
            'n_orphans': len(orphans),
            'n_outliers': len(outliers)
        }
    else:
        # Return tuple for backward compatibility with dynamic_clustering_pipeline.py
        orphan_dict = {cluster_id: size for cluster_id, size in orphans}
        return orphan_dict, outliers


def evaluate_cluster_quality(
    embeddings: Union[np.ndarray, Any],
    labels: Union[List, np.ndarray, Any],
    include_std: bool = False,
    metric: str = 'cosine',
    return_legacy_format: bool = False
) -> Dict[str, Any]:
    """
    Unified function to evaluate cluster quality using multiple metrics.
    
    This function can be used by both dynamic_clustering_pipeline.py and
    ClusterAnalyzer to avoid code duplication. It combines the best features
    of both implementations.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
            Can be numpy array, torch tensor, or list
        labels: Array of cluster labels (can include -1 for outliers)
            Can be numpy array, list, or any iterable
        include_std: Whether to include standard deviations of cohesion/separation
            Default: False (for backward compatibility)
        metric: Distance metric to use ('cosine' or 'euclidean')
            Default: 'cosine'
        return_legacy_format: If True, returns format compatible with 
            dynamic_clustering_pipeline.py (intra_cluster_density, inter_cluster_distance)
            If False, returns unified format (cohesion, separation)
            Default: False
    
    Returns:
        Dictionary with metrics:
        - silhouette_score: Silhouette coefficient (higher is better, range: -1 to 1)
        - davies_bouldin_score: Davies-Bouldin index (lower is better)
        - cohesion: Average intra-cluster distance (lower is better)
            OR 'intra_cluster_density' if return_legacy_format=True
        - separation: Average inter-cluster distance (higher is better)
            OR 'inter_cluster_distance' if return_legacy_format=True
        - cohesion_std: Standard deviation of cohesion (if include_std=True)
        - separation_std: Standard deviation of separation (if include_std=True)
        - n_clusters: Number of valid clusters (excluding outliers)
        - n_outliers: Number of outliers (label == -1)
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*30 + [1]*40 + [2]*25 + [-1]*5
        >>> metrics = evaluate_cluster_quality(embeddings, labels)
        >>> print(metrics['silhouette_score'])
        >>> print(metrics['cohesion'])
    """
    if not SKLEARN_AVAILABLE:
        return {
            'error': 'scikit-learn required for cluster quality evaluation',
            'silhouette_score': None,
            'davies_bouldin_score': None,
            'cohesion': None,
            'separation': None,
            'n_clusters': 0,
            'n_outliers': 0
        }
    
    # Convert embeddings to numpy if needed
    if hasattr(embeddings, 'cpu'):
        # Torch tensor
        embeddings_np = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        # TensorFlow tensor or similar
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    # Convert labels to numpy array
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    elif hasattr(labels, '__iter__') and not isinstance(labels, str):
        labels_list = list(labels)
    else:
        labels_list = [labels]
    
    labels_np = np.array(labels_list)
    
    # Filter outliers for metrics (label == -1)
    valid_mask = labels_np != -1
    n_outliers = (~valid_mask).sum()
    
    # Handle edge cases
    if valid_mask.sum() < 2:
        result = {
            'silhouette_score': None if not return_legacy_format else -1,
            'davies_bouldin_score': None if not return_legacy_format else float('inf'),
            'cohesion': None if not return_legacy_format else float('inf'),
            'separation': None if not return_legacy_format else 0,
            'n_clusters': 0,
            'n_outliers': int(n_outliers)
        }
        
        # Add legacy format keys if needed
        if return_legacy_format:
            result['intra_cluster_density'] = result['cohesion']
            result['inter_cluster_distance'] = result['separation']
            result['silhouette'] = result['silhouette_score']
            result['davies_bouldin'] = result['davies_bouldin_score']
        
        # Add std if requested
        if include_std:
            result['cohesion_std'] = None
            result['separation_std'] = None
        
        return result
    
    valid_embeddings = embeddings_np[valid_mask]
    valid_labels = labels_np[valid_mask]
    unique_clusters = set(valid_labels)
    n_clusters = len(unique_clusters)
    
    # Handle case with only one cluster
    if n_clusters < 2:
        result = {
            'silhouette_score': None if not return_legacy_format else -1,
            'davies_bouldin_score': None if not return_legacy_format else float('inf'),
            'cohesion': None if not return_legacy_format else float('inf'),
            'separation': None if not return_legacy_format else 0,
            'n_clusters': n_clusters,
            'n_outliers': int(n_outliers)
        }
        
        if return_legacy_format:
            result['intra_cluster_density'] = result['cohesion']
            result['inter_cluster_distance'] = result['separation']
            result['silhouette'] = result['silhouette_score']
            result['davies_bouldin'] = result['davies_bouldin_score']
        
        if include_std:
            result['cohesion_std'] = None
            result['separation_std'] = None
        
        return result
    
    # Compute silhouette score
    try:
        if metric == 'cosine':
            silhouette = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
        else:
            silhouette = silhouette_score(valid_embeddings, valid_labels, metric=metric)
    except Exception:
        silhouette = None if not return_legacy_format else -1
    
    # Compute Davies-Bouldin index
    try:
        db_index = davies_bouldin_score(valid_embeddings, valid_labels)
    except Exception:
        db_index = None if not return_legacy_format else float('inf')
    
    # Compute cohesion (intra-cluster distances) and separation (inter-cluster distances)
    cohesion_scores = []
    separation_scores = []
    
    for cluster_id in unique_clusters:
        cluster_mask = valid_labels == cluster_id
        cluster_embeddings = valid_embeddings[cluster_mask]
        
        if len(cluster_embeddings) > 1:
            # Cohesion: average distance within cluster
            if metric == 'cosine':
                distances = cosine_distances(cluster_embeddings)
            else:
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(cluster_embeddings)
            
            # Remove diagonal (self-distances)
            mask = ~np.eye(len(distances), dtype=bool)
            cohesion_scores.append(distances[mask].mean())
        
        # Separation: average distance to other clusters
        other_mask = valid_labels != cluster_id
        if other_mask.sum() > 0:
            other_embeddings = valid_embeddings[other_mask]
            if metric == 'cosine':
                cross_distances = cosine_distances(cluster_embeddings, other_embeddings)
            else:
                from sklearn.metrics.pairwise import euclidean_distances
                cross_distances = euclidean_distances(cluster_embeddings, other_embeddings)
            separation_scores.append(cross_distances.mean())
    
    # Calculate averages
    cohesion = np.mean(cohesion_scores) if cohesion_scores else None
    separation = np.mean(separation_scores) if separation_scores else None
    
    # Build result dictionary
    result = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_index,
        'cohesion': cohesion,
        'separation': separation,
        'n_clusters': n_clusters,
        'n_outliers': int(n_outliers)
    }
    
    # Add standard deviations if requested
    if include_std:
        result['cohesion_std'] = np.std(cohesion_scores) if cohesion_scores else None
        result['separation_std'] = np.std(separation_scores) if separation_scores else None
    
    # Add legacy format keys if needed (for backward compatibility with dynamic_clustering_pipeline)
    if return_legacy_format:
        result['intra_cluster_density'] = cohesion
        result['inter_cluster_distance'] = separation
        result['silhouette'] = silhouette
        result['davies_bouldin'] = db_index
    
    return result


def adaptive_merge_clusters(
    embeddings: Union[np.ndarray, Any],
    labels: Union[List, np.ndarray, Any],
    max_iterations: Optional[int] = None,
    semantic_threshold: Optional[float] = None,
    fusion_percentile: Optional[float] = None,
    min_clusters_target: Optional[int] = None,
    min_texts_per_cluster: Optional[int] = None,
    return_stats: bool = False,
    verbose: bool = False,
    config: Optional[Any] = None
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Unified function to adaptively merge clusters based on distance/similarity.
    
    This function can be used by both dynamic_clustering_pipeline.py and
    ClusterAnalyzer to avoid code duplication. It combines the best features
    of both implementations.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
            Can be numpy array, torch tensor, or list
        labels: Array of cluster labels (can include -1 for outliers)
            Can be numpy array, list, or any iterable
        max_iterations: Maximum merge iterations (default: from config or 10)
        semantic_threshold: Semantic similarity threshold for merging (default: from config or 0.85)
            Range: 0.7-0.95 (higher = more strict, only very similar clusters merge)
        fusion_percentile: Percentile for adaptive distance threshold (default: from config or 25)
            Range: 10-50 (lower = more aggressive merging, higher = more conservative)
        min_clusters_target: Minimum target clusters (default: from config or 5)
            Range: 3-20 (stops merging when this number is reached)
        min_texts_per_cluster: Minimum texts per cluster (default: from config or 5)
            Range: 3-15 (clusters smaller than this are candidates for merging)
        return_stats: If True, return dict with statistics; if False, return labels only
        verbose: Whether to print progress messages
        config: Optional DynamicClusteringConfig or ClusterAnalysisConfig (used for defaults)
    
    Returns:
        If return_stats=False:
            Merged labels array (same format as input)
        If return_stats=True:
            Dictionary with:
            - 'labels': Merged labels array
            - 'initial_clusters': Number of clusters before merging
            - 'final_clusters': Number of clusters after merging
            - 'merges_performed': Total number of merges
            - 'iterations': Number of iterations performed
            - 'clusters_reduced': Difference in cluster count
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*30 + [1]*40 + [2]*25 + [-1]*5
        >>> merged = adaptive_merge_clusters(embeddings, labels, semantic_threshold=0.90)
        >>> print(f"Merged to {len(set(merged))} clusters")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for adaptive cluster merging")
    
    # Convert embeddings to numpy if needed
    if hasattr(embeddings, 'cpu'):
        embeddings_np = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    # Convert labels to numpy array
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    elif hasattr(labels, '__iter__') and not isinstance(labels, str):
        labels_list = list(labels)
    else:
        labels_list = [labels]
    
    labels_np = np.array(labels_list)
    
    # Resolve defaults from config or hardcoded values
    if config is not None:
        # Try to get values from config (supports both DynamicClusteringConfig and ClusterAnalysisConfig)
        max_iter = max_iterations
        if max_iter is None:
            max_iter = getattr(config, 'MAX_FUSION_ITERATIONS', None) or getattr(config, 'max_merge_iterations', None) or 10
        
        sem_thresh = semantic_threshold
        if sem_thresh is None:
            sem_thresh = getattr(config, 'SEMANTIC_FUSION_THRESHOLD', None) or getattr(config, 'semantic_fusion_threshold', None) or 0.85
        
        fusion_pct = fusion_percentile
        if fusion_pct is None:
            fusion_pct = getattr(config, 'FUSION_PERCENTILE', None) or getattr(config, 'fusion_percentile', None) or 25
        
        min_target = min_clusters_target
        if min_target is None:
            min_target = getattr(config, 'MIN_CLUSTERS_TARGET', None) or getattr(config, 'min_clusters_target', None) or 5
        
        min_texts = min_texts_per_cluster
        if min_texts is None:
            min_texts = getattr(config, 'MIN_TEXTS_PER_CLUSTER', None) or getattr(config, 'min_texts_per_cluster', None) or 5
    else:
        # Use provided values or hardcoded defaults
        max_iter = max_iterations if max_iterations is not None else 10
        sem_thresh = semantic_threshold if semantic_threshold is not None else 0.85
        fusion_pct = fusion_percentile if fusion_percentile is not None else 25
        min_target = min_clusters_target if min_clusters_target is not None else 5
        min_texts = min_texts_per_cluster if min_texts_per_cluster is not None else 5
    
    # Initialize tracking variables
    current_labels = labels_np.copy()
    initial_clusters = len(set(current_labels) - {-1})  # Exclude outliers
    iteration = 0
    total_merges = 0
    
    if verbose:
        print(f"\n[*] Starting adaptive cluster merging...")
        print(f"   Initial clusters: {initial_clusters}")
    
    # Main merging loop
    while iteration < max_iter:
        # Get unique cluster labels (excluding outliers)
        unique_labels = [l for l in set(current_labels) if l != -1]
        if len(unique_labels) < 2:
            break
        
        # Stop if we've reached minimum target
        if len(unique_labels) <= min_target:
            if verbose:
                print(f"   [OK] Reached minimum target clusters ({min_target})")
            break
        
        # Calculate centroids and sizes
        centroids = {}
        cluster_sizes = {}
        for cluster_id in unique_labels:
            cluster_mask = current_labels == cluster_id
            cluster_embeddings = embeddings_np[cluster_mask]
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
            cluster_sizes[cluster_id] = len(cluster_embeddings)
        
        # Calculate distances between centroids
        centroid_list = [centroids[cid] for cid in unique_labels]
        distance_matrix = cosine_distances(centroid_list)
        
        # Create pairs with distances
        pairs = []
        for i, cid1 in enumerate(unique_labels):
            for j, cid2 in enumerate(unique_labels[i+1:], start=i+1):
                distance = distance_matrix[i, j]
                pairs.append((cid1, cid2, distance, cluster_sizes[cid1], cluster_sizes[cid2]))
        
        if not pairs:
            break
        
        # Calculate adaptive threshold (percentile of distances)
        distances = [p[2] for p in pairs]
        threshold = np.percentile(distances, fusion_pct)
        
        # More restrictive threshold if we're close to minimum target
        if len(unique_labels) <= min_target + 2:
            threshold = min(threshold, np.percentile(distances, 5))
        
        # Identify pairs to merge
        pairs_to_merge = []
        for p in pairs:
            cid1, cid2, distance, size1, size2 = p
            similarity = 1 - distance
            
            # Merge criteria:
            # 1. High semantic similarity, OR
            # 2. Close distance AND (small cluster OR both small)
            should_merge = (
                similarity > sem_thresh or  # High semantic similarity
                (distance < threshold and (size1 < min_texts or size2 < min_texts))  # Small clusters
            )
            
            if should_merge:
                pairs_to_merge.append((cid1, cid2, distance, similarity, size1, size2))
        
        # Sort: prioritize merging small clusters first, then by similarity
        pairs_to_merge.sort(key=lambda x: (min(x[4], x[5]), -x[3]))
        
        # Perform merges
        merged_this_iteration = 0
        merged_clusters = set()
        
        for pair in pairs_to_merge:
            cid1, cid2, dist, similarity, size1, size2 = pair
            if cid1 in merged_clusters or cid2 in merged_clusters:
                continue
            
            # Merge: assign all items from cid2 to cid1
            current_labels[current_labels == cid2] = cid1
            merged_clusters.add(cid2)
            merged_this_iteration += 1
            total_merges += 1
            
            if verbose:
                print(f"   Merging: Cluster {cid2} ({size2} items, sim={similarity:.3f}) → Cluster {cid1} ({size1} items)")
        
        if merged_this_iteration == 0:
            break
        
        # Renumber clusters to be consecutive
        unique_labels_new = sorted(set(current_labels))
        label_mapping = {old: new for new, old in enumerate(unique_labels_new) if old != -1}
        label_mapping[-1] = -1  # Preserve outliers
        
        for i in range(len(current_labels)):
            if current_labels[i] != -1:
                current_labels[i] = label_mapping[current_labels[i]]
        
        iteration += 1
        n_clusters = len([l for l in set(current_labels) if l != -1])
        
        if verbose:
            print(f"   Iteration {iteration}: {merged_this_iteration} merges → {n_clusters} clusters remaining")
    
    final_clusters = len(set(current_labels) - {-1})
    
    if verbose:
        print(f"[OK] Adaptive merging completed: {initial_clusters} → {final_clusters} clusters ({total_merges} merges)")
    
    # Return format based on return_stats
    if return_stats:
        return {
            'labels': current_labels,
            'initial_clusters': initial_clusters,
            'final_clusters': final_clusters,
            'merges_performed': total_merges,
            'iterations': iteration,
            'clusters_reduced': initial_clusters - final_clusters
        }
    else:
        return current_labels


def calculate_cluster_centroids(
    embeddings: np.ndarray,
    labels: Union[List, np.ndarray],
    exclude_outliers: bool = True
) -> Dict[int, np.ndarray]:
    """
    Calculate centroids (mean embeddings) for each cluster.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        labels: Array of cluster labels (can include -1 for outliers)
        exclude_outliers: Whether to exclude outliers (label == -1) from calculation
    
    Returns:
        Dictionary mapping cluster_id to centroid array
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*30 + [1]*40 + [2]*25 + [-1]*5
        >>> centroids = calculate_cluster_centroids(embeddings, labels)
        >>> print(f"Cluster 0 centroid shape: {centroids[0].shape}")
    """
    labels = np.asarray(labels)
    valid_clusters = [l for l in set(labels) if l != -1] if exclude_outliers else [l for l in set(labels)]
    
    cluster_centroids = {}
    for cluster_id in valid_clusters:
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        if len(cluster_embeddings) > 0:
            cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    return cluster_centroids


def renumber_clusters(
    labels: Union[List, np.ndarray],
    preserve_outliers: bool = True
) -> np.ndarray:
    """
    Renumber cluster labels to be consecutive (0, 1, 2, ...).
    
    Args:
        labels: Array of cluster labels (can include -1 for outliers)
        preserve_outliers: Whether to preserve outlier label (-1)
    
    Returns:
        Renumbered labels array
    
    Example:
        >>> labels = np.array([5, 5, 3, 3, -1, 8, 8])
        >>> renumbered = renumber_clusters(labels)
        >>> print(renumbered)  # [0, 0, 1, 1, -1, 2, 2]
    """
    labels = np.asarray(labels)
    unique_labels_new = sorted(set(labels))
    
    if preserve_outliers and -1 in unique_labels_new:
        unique_labels_new.remove(-1)
        label_mapping = {old: new for new, old in enumerate(unique_labels_new)}
        label_mapping[-1] = -1
    else:
        label_mapping = {old: new for new, old in enumerate(unique_labels_new)}
    
    renumbered = labels.copy()
    for i in range(len(renumbered)):
        if renumbered[i] != -1 or not preserve_outliers:
            renumbered[i] = label_mapping[renumbered[i]]
    
    return renumbered


def calculate_cluster_coherence(
    cluster_embeddings: np.ndarray,
    metric: str = 'cosine'
) -> Dict[str, float]:
    """
    Calculate coherence metrics for a single cluster.
    
    Args:
        cluster_embeddings: Embeddings for cluster [n_samples, embedding_dim]
        metric: Distance metric to use ('cosine' or 'euclidean')
    
    Returns:
        Dictionary with:
        - 'coherence': Coherence score (1.0 / (1.0 + CV))
        - 'cv': Coefficient of variation
        - 'mean_distance': Mean intra-cluster distance
        - 'std_distance': Standard deviation of intra-cluster distances
    
    Example:
        >>> embeddings = np.random.rand(20, 128)
        >>> metrics = calculate_cluster_coherence(embeddings)
        >>> print(f"Coherence: {metrics['coherence']:.3f}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for calculate_cluster_coherence")
    
    if len(cluster_embeddings) <= 1:
        return {
            'coherence': 0.0,
            'cv': 1.0,
            'mean_distance': 0.0,
            'std_distance': 0.0
        }
    
    # Calculate intra-cluster distances
    distances = cosine_distances(cluster_embeddings)
    triu_indices = np.triu_indices(len(distances), k=1)
    intra_distances = distances[triu_indices]
    
    mean_dist = np.mean(intra_distances)
    std_dist = np.std(intra_distances)
    cv = std_dist / mean_dist if mean_dist > 0 else 0
    coherence = 1.0 / (1.0 + cv)  # Coherence inverse to CV
    
    return {
        'coherence': coherence,
        'cv': cv,
        'mean_distance': mean_dist,
        'std_distance': std_dist
    }


def assign_outliers_to_clusters(
    embeddings: np.ndarray,
    labels: Union[List, np.ndarray],
    similarity_threshold: float = 0.1,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assign outliers (label == -1) to nearest clusters based on similarity.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        labels: Array of cluster labels (can include -1 for outliers)
        similarity_threshold: Minimum similarity to assign outlier (default: 0.1)
        verbose: Whether to print progress
    
    Returns:
        Tuple of:
        - Updated labels array
        - Statistics dict with:
          - 'assigned_count': Number of outliers assigned
          - 'n_outliers_before': Number of outliers before assignment
          - 'n_outliers_after': Number of outliers after assignment
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*30 + [1]*40 + [-1]*30
        >>> updated_labels, stats = assign_outliers_to_clusters(embeddings, labels)
        >>> print(f"Assigned: {stats['assigned_count']} outliers")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for assign_outliers_to_clusters")
    
    labels = np.asarray(labels)
    outlier_indices = np.where(labels == -1)[0]
    n_outliers_before = len(outlier_indices)
    
    if n_outliers_before == 0:
        if verbose:
            print("  [OK] No hay outliers para asignar")
        return labels, {
            'assigned_count': 0,
            'n_outliers_before': 0,
            'n_outliers_after': 0
        }
    
    if verbose:
        print(f"  Outliers detectados: {n_outliers_before} ({100*n_outliers_before/len(labels):.1f}%)")
        print(f"  Asignando outliers a clusters más cercanos...")
    
    # Calculate centroids of existing clusters
    cluster_centroids = calculate_cluster_centroids(embeddings, labels, exclude_outliers=True)
    
    if not cluster_centroids:
        if verbose:
            print("  [WARNING]  No hay clusters válidos para asignar outliers")
        return labels, {
            'assigned_count': 0,
            'n_outliers_before': n_outliers_before,
            'n_outliers_after': n_outliers_before
        }
    
    # Assign each outlier to nearest cluster
    assigned_count = 0
    
    for outlier_idx in outlier_indices:
        outlier_emb = embeddings[outlier_idx].reshape(1, -1)
        best_cluster_id = None
        best_similarity = -1
        
        for cluster_id, centroid in cluster_centroids.items():
            similarity = cosine_similarity(
                outlier_emb,
                centroid.reshape(1, -1)
            )[0, 0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        # Assign if similarity is reasonable
        if best_cluster_id and best_similarity > similarity_threshold:
            labels[outlier_idx] = best_cluster_id
            assigned_count += 1
    
    n_outliers_after = np.sum(labels == -1)
    
    if verbose:
        print(f"  [OK] Outliers asignados: {assigned_count}/{n_outliers_before}")
        print(f"  Outliers restantes: {n_outliers_after} ({100*n_outliers_after/len(labels):.1f}%)")
    
    return labels, {
        'assigned_count': assigned_count,
        'n_outliers_before': n_outliers_before,
        'n_outliers_after': int(n_outliers_after)
    }


def merge_clusters_by_ratio(
    embeddings: np.ndarray,
    labels: Union[List, np.ndarray],
    min_texts_per_node: int,
    max_iterations: int = 5,
    similarity_threshold_small: float = 0.15,
    similarity_threshold_normal: float = 0.3,
    min_fusion_similarity: float = 0.3,
    log_fusion_decisions: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Merge clusters that don't meet minimum texts per node ratio.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        labels: Array of cluster labels
        min_texts_per_node: Minimum texts required per cluster/node
        max_iterations: Maximum merge iterations
        similarity_threshold_small: Threshold for very small clusters (<3 texts)
        similarity_threshold_normal: Threshold for normal clusters
        min_fusion_similarity: Minimum similarity for final orphan merge
        log_fusion_decisions: Whether to log fusion decisions
        verbose: Whether to print progress
    
    Returns:
        Tuple of:
        - Merged labels array
        - Fusion log (list of merge operations)
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*10 + [1]*5 + [2]*8 + [3]*77
        >>> merged, log = merge_clusters_by_ratio(embeddings, labels, min_texts_per_node=10)
        >>> print(f"Merged to {len(set(merged))} clusters")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for merge_clusters_by_ratio")
    
    labels = np.asarray(labels)
    merged_labels = labels.copy()
    fusion_log = []
    iteration = 0
    
    while iteration < max_iterations:
        # Calculate cluster sizes
        unique_labels = [l for l in set(merged_labels) if l != -1]
        cluster_sizes = {cid: np.sum(merged_labels == cid) for cid in unique_labels}
        
        # Identify clusters that don't meet minimum
        clusters_below_min = [cid for cid, size in cluster_sizes.items() 
                             if size < min_texts_per_node]
        
        if not clusters_below_min:
            if verbose:
                print(f"  [OK] Todos los clusters cumplen el mínimo de {min_texts_per_node} textos/nodo")
            break
        
        if verbose:
            print(f"  Iteración {iteration + 1}: {len(clusters_below_min)} clusters con <{min_texts_per_node} textos")
        
        merged_this_iteration = 0
        merged_clusters = set()
        
        # Calculate centroids for all clusters
        cluster_centroids = calculate_cluster_centroids(embeddings, merged_labels, exclude_outliers=True)
        
        for small_cid in clusters_below_min:
            if small_cid in merged_clusters or small_cid not in cluster_centroids:
                continue
            
            small_centroid = cluster_centroids[small_cid]
            best_match_id = None
            best_similarity = -1
            
            # Find most similar cluster
            for other_cid in unique_labels:
                if other_cid == small_cid or other_cid in merged_clusters or other_cid not in cluster_centroids:
                    continue
                
                other_centroid = cluster_centroids[other_cid]
                similarity = cosine_similarity(
                    small_centroid.reshape(1, -1),
                    other_centroid.reshape(1, -1)
                )[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = other_cid
            
            # Merge if similarity is reasonable
            threshold = similarity_threshold_small if cluster_sizes[small_cid] < 3 else similarity_threshold_normal
            
            if best_match_id and best_similarity > threshold:
                merged_labels[merged_labels == small_cid] = best_match_id
                merged_clusters.add(small_cid)
                merged_this_iteration += 1
                
                if log_fusion_decisions:
                    fusion_log.append({
                        'from': int(small_cid),
                        'to': int(best_match_id),
                        'size': int(cluster_sizes[small_cid]),
                        'similarity': float(best_similarity),
                        'iteration': iteration + 1
                    })
                
                if verbose:
                    print(f"    ✓ Cluster {small_cid} ({cluster_sizes[small_cid]} textos) → Cluster {best_match_id} (sim={best_similarity:.3f})")
        
        if merged_this_iteration == 0:
            break
        
        # Renumber clusters
        merged_labels = renumber_clusters(merged_labels, preserve_outliers=True)
        iteration += 1
    
    # Final pass: merge any remaining orphan clusters (<3 texts)
    merged_labels = renumber_clusters(merged_labels, preserve_outliers=True)
    
    if verbose:
        print(f"\n  [*] Pasada final: fusionando clusters huérfanos restantes...")
    
    unique_labels = [l for l in set(merged_labels) if l != -1]
    cluster_sizes = {cid: np.sum(merged_labels == cid) for cid in unique_labels}
    orphan_clusters = [cid for cid, size in cluster_sizes.items() if size < 3]
    
    if orphan_clusters:
        if verbose:
            print(f"    Detectados {len(orphan_clusters)} clusters huérfanos (<3 textos)")
        
        merged_orphans = set()
        cluster_centroids = calculate_cluster_centroids(embeddings, merged_labels, exclude_outliers=True)
        
        for orphan_cid in orphan_clusters:
            if orphan_cid in merged_orphans or orphan_cid not in cluster_centroids:
                continue
            
            orphan_centroid = cluster_centroids[orphan_cid]
            best_match_id = None
            best_similarity = -1
            
            for other_cid in unique_labels:
                if other_cid == orphan_cid or other_cid in merged_orphans or other_cid not in cluster_centroids:
                    continue
                
                other_centroid = cluster_centroids[other_cid]
                similarity = cosine_similarity(
                    orphan_centroid.reshape(1, -1),
                    other_centroid.reshape(1, -1)
                )[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = other_cid
            
            # Merge only if similarity is reasonable
            threshold = min_fusion_similarity if cluster_sizes[orphan_cid] >= 2 else 0.3
            
            if best_match_id and best_similarity > threshold:
                merged_labels[merged_labels == orphan_cid] = best_match_id
                merged_orphans.add(orphan_cid)
                
                if log_fusion_decisions:
                    fusion_log.append({
                        'from': int(orphan_cid),
                        'to': int(best_match_id),
                        'size': int(cluster_sizes[orphan_cid]),
                        'similarity': float(best_similarity),
                        'aggressive': best_similarity < min_fusion_similarity,
                        'iteration': 'final'
                    })
                
                if verbose:
                    print(f"    ✓ Cluster {orphan_cid} ({cluster_sizes[orphan_cid]} textos) → Cluster {best_match_id} (sim={best_similarity:.3f})")
            elif best_match_id and log_fusion_decisions:
                # Log rejected fusion
                fusion_log.append({
                    'from': int(orphan_cid),
                    'to': int(best_match_id),
                    'size': int(cluster_sizes[orphan_cid]),
                    'similarity': float(best_similarity),
                    'rejected': True,
                    'reason': f'Similitud {best_similarity:.3f} < threshold {threshold:.3f}',
                    'iteration': 'final'
                })
                
                if verbose:
                    print(f"    [WARNING]  Cluster {orphan_cid} ({cluster_sizes[orphan_cid]} textos) no fusionado: sim={best_similarity:.3f} < {threshold:.3f}")
    else:
        if verbose:
            print(f"    [OK] No hay clusters huérfanos restantes")
    
    # Final renumbering
    merged_labels = renumber_clusters(merged_labels, preserve_outliers=True)
    
    return merged_labels, fusion_log


def calculate_dynamic_threshold(
    embeddings: np.ndarray,
    sample_size: int = 50,
    percentile: float = 25,
    verbose: bool = True
) -> float:
    """
    Calculate dynamic threshold based on distance distribution in embeddings.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
        sample_size: Number of samples to use for calculation (default: 50)
        percentile: Percentile to use for threshold (default: 25)
        verbose: Whether to print statistics
    
    Returns:
        Dynamic threshold value (as similarity, 1 - distance_percentile)
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> threshold = calculate_dynamic_threshold(embeddings, sample_size=50)
        >>> print(f"Threshold: {threshold:.4f}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for calculate_dynamic_threshold")
    
    # Sample embeddings for efficiency
    actual_sample_size = min(sample_size, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), actual_sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # Calculate distances
    distances = cosine_distances(sample_embeddings)
    # Only upper triangular (without diagonal)
    triu_indices = np.triu_indices(len(distances), k=1)
    all_distances = distances[triu_indices]
    
    avg_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)
    percentile_value = np.percentile(all_distances, percentile)
    
    # Dynamic threshold: convert distance to similarity
    dynamic_threshold = 1 - percentile_value
    
    if verbose:
        print(f"   Distancia promedio: {avg_distance:.4f}")
        print(f"   Distancia mediana: {median_distance:.4f}")
        print(f"   Threshold dinámico (similitud): {dynamic_threshold:.4f}")
        print(f"   [TIP] Solo se creará cluster nuevo si similitud < {dynamic_threshold:.4f}")
    
    return dynamic_threshold


def calculate_dynamic_node_threshold(
    n_total_texts: int,
    pct_dataset: float = 0.08,
    min_cap: int = 5,
    max_cap: int = 50,
    verbose: bool = True
) -> int:
    """
    Calculate dynamic threshold for creating KnowledgeNodes using hybrid formula.
    
    Formula: max(min(ceil(total_texts * pct), max_cap), min_cap)
    
    Args:
        n_total_texts: Total number of texts in dataset
        pct_dataset: Percentage of dataset to use (default: 0.08 = 8%)
        min_cap: Minimum absolute threshold (default: 5)
        max_cap: Maximum threshold to avoid too large nodes (default: 50)
        verbose: Whether to print calculation details
    
    Returns:
        Dynamic threshold value
    
    Example:
        >>> threshold = calculate_dynamic_node_threshold(1000, pct_dataset=0.08)
        >>> print(f"Threshold: {threshold}")
    """
    # Calculate threshold using hybrid formula
    calculated = int(np.ceil(n_total_texts * pct_dataset))
    capped = min(calculated, max_cap)
    dynamic_threshold = max(capped, min_cap)
    
    if verbose:
        print(f"\n  [*] Cálculo de Umbral Dinámico (Fórmula Híbrida):")
        print(f"    - Total textos: {n_total_texts}")
        print(f"    - Porcentaje del dataset: {pct_dataset*100:.1f}%")
        print(f"    - Cálculo: ceil({n_total_texts} × {pct_dataset}) = {calculated}")
        print(f"    - Aplicando límites: min({calculated}, {max_cap}) = {capped}")
        print(f"    - Aplicando mínimo: max({capped}, {min_cap}) = {dynamic_threshold}")
        print(f"\n  [*] Umbral dinámico final: {dynamic_threshold} embeddings")
    
    return dynamic_threshold


def detect_domain_from_path(
    dataset_path: Optional[str],
    domain_config: Dict[str, int],
    verbose: bool = True
) -> Tuple[str, int]:
    """
    Detect domain from dataset file path and return appropriate minimum texts per node.
    
    Args:
        dataset_path: Path to dataset file
        domain_config: Dictionary with domain-specific minimums (e.g., {'mathematics': 20, 'aerospace': 10, 'default': 5})
        verbose: Whether to print detected domain
    
    Returns:
        Tuple of (domain_name, min_texts_per_node)
    
    Example:
        >>> config = {'mathematics': 20, 'aerospace': 10, 'default': 5}
        >>> domain, min_texts = detect_domain_from_path('math_dataset.csv', config)
        >>> print(f"Domain: {domain}, Min texts: {min_texts}")
    """
    if not dataset_path:
        domain_name = 'default'
        min_texts_per_node = domain_config.get('default', 5)
    else:
        import os
        dataset_name = os.path.basename(dataset_path).lower()
        
        if 'math' in dataset_name or 'mathematics' in dataset_name:
            domain_name = 'mathematics'
            min_texts_per_node = domain_config.get('mathematics', domain_config.get('default', 5))
        elif 'aerospace' in dataset_name or 'aero' in dataset_name:
            domain_name = 'aerospace'
            min_texts_per_node = domain_config.get('aerospace', domain_config.get('default', 5))
        else:
            domain_name = 'default'
            min_texts_per_node = domain_config.get('default', 5)
    
    if verbose:
        domain_display = {
            'mathematics': 'Matemáticas',
            'aerospace': 'Aeroespacial',
            'default': 'General'
        }.get(domain_name, domain_name)
        print(f"   Dominio detectado: {domain_display} → mínimo {min_texts_per_node} textos/nodo")
    
    return domain_name, min_texts_per_node


def load_domain_mapping(
    dataset_path: Optional[str],
    category_column: str = 'category',
    max_index: Optional[int] = None
) -> Dict[int, str]:
    """
    Load domain/category mapping from dataset.
    
    Args:
        dataset_path: Path to dataset CSV file
        category_column: Name of category column (default: 'category')
        max_index: Maximum index to map (default: None, use all)
    
    Returns:
        Dictionary mapping text index to category/domain
    
    Example:
        >>> mapping = load_domain_mapping('dataset.csv', category_column='category')
        >>> print(f"Loaded {len(mapping)} domain mappings")
    """
    domain_mapping = {}
    
    if not dataset_path:
        return domain_mapping
    
    try:
        import os
        import pandas as pd
        
        if not os.path.exists(dataset_path):
            return domain_mapping
        
        # Check if column exists
        df_check = pd.read_csv(dataset_path, sep=",", quotechar='"', dtype=str, nrows=1)
        if category_column not in df_check.columns:
            return domain_mapping
        
        # Load full dataset
        df_full = pd.read_csv(dataset_path, sep=",", quotechar='"', dtype=str)
        if category_column not in df_full.columns:
            return domain_mapping
        
        # Create mapping
        for i, cat in enumerate(df_full[category_column].dropna()):
            if max_index is None or i < max_index:
                domain_mapping[i] = cat
    except Exception:
        # Silently fail if there's any error
        pass
    
    return domain_mapping


def calculate_domain_purity(
    cluster_indices: np.ndarray,
    domain_mapping: Dict[int, str]
) -> float:
    """
    Calculate domain purity for a cluster (proportion of most common domain).
    
    Args:
        cluster_indices: Array of text indices in cluster
        domain_mapping: Dictionary mapping text index to domain
    
    Returns:
        Domain purity score (0.0 to 1.0, where 1.0 = all texts from same domain)
    
    Example:
        >>> indices = np.array([0, 1, 2, 3, 4])
        >>> mapping = {0: 'math', 1: 'math', 2: 'math', 3: 'physics', 4: 'math'}
        >>> purity = calculate_domain_purity(indices, mapping)
        >>> print(f"Purity: {purity:.3f}")  # 0.8 (4/5 are 'math')
    """
    if not domain_mapping:
        return 1.0  # Default: assume complete purity
    
    cluster_domains = {}
    for idx in cluster_indices:
        if idx in domain_mapping:
            domain = domain_mapping[idx]
            cluster_domains[domain] = cluster_domains.get(domain, 0) + 1
    
    if not cluster_domains:
        return 1.0  # No domain info available, assume purity
    
    total_domains = sum(cluster_domains.values())
    max_domain_count = max(cluster_domains.values())
    domain_purity = max_domain_count / total_domains if total_domains > 0 else 0.0
    
    return domain_purity


def determine_node_configuration(
    cluster_size: int,
    coherence: float = 1.0,
    action: Optional[str] = None
) -> Tuple[int, int]:
    """
    Determine optimal number of layers and LoRA rank for KnowledgeNode based on cluster characteristics.
    
    Args:
        cluster_size: Number of texts in cluster
        coherence: Cluster coherence score (default: 1.0)
        action: Bayesian decision action (e.g., "CREATE_NODE_MINIMAL", "PROMOTE_AS_CANDIDATE")
    
    Returns:
        Tuple of (num_layers, lora_rank)
    
    Example:
        >>> layers, rank = determine_node_configuration(cluster_size=50, coherence=0.8)
        >>> print(f"Layers: {layers}, Rank: {rank}")
    """
    is_minimal = (action == "CREATE_NODE_MINIMAL")
    
    if is_minimal or (cluster_size < 20 and coherence < 0.5):
        # Clusters pequeños heterogéneos o acción CREATE_NODE_MINIMAL: configuración mínima
        num_layers = 1
        lora_rank = 4
    elif cluster_size < 20:
        # Clusters pequeños coherentes: configuración mínima para evitar overfitting
        num_layers = 1
        lora_rank = 4
    elif cluster_size < 60:
        num_layers = 1
        lora_rank = 5
    elif cluster_size < 100:
        num_layers = 2
        lora_rank = 6
    elif cluster_size < 200:
        num_layers = 3
        lora_rank = 6
    else:  # Clusters muy grandes (>200 embeddings)
        num_layers = 3
        lora_rank = 6
    
    return num_layers, lora_rank


def get_current_layers(node: Any) -> int:
    """
    Get current number of layers from KnowledgeNode.
    
    Args:
        node: KnowledgeNode instance
    
    Returns:
        Current number of layers (default: 1 if cannot determine)
    
    Example:
        >>> layers = get_current_layers(knowledge_node)
        >>> print(f"Current layers: {layers}")
    """
    if hasattr(node, 'transformer') and hasattr(node.transformer, 'num_layers'):
        return node.transformer.num_layers
    elif hasattr(node, 'transformer') and hasattr(node.transformer, 'encoder_layers'):
        return len(node.transformer.encoder_layers)
    return 1


def get_current_lora_rank(node: Any) -> int:
    """
    Get current LoRA rank from KnowledgeNode.
    
    Args:
        node: KnowledgeNode instance
    
    Returns:
        Current LoRA rank (default: 4 if LoRA not enabled)
    
    Example:
        >>> rank = get_current_lora_rank(knowledge_node)
        >>> print(f"Current LoRA rank: {rank}")
    """
    try:
        lora_config = node.get_lora_config()
        if lora_config.get('enabled', False):
            return lora_config.get('rank', 4)
    except (AttributeError, TypeError):
        pass
    return 4


def update_node_configuration(
    node: Any,
    cluster_size: int,
    config: Any,
    coherence: float = 1.0,
    action: Optional[str] = None,
    preserve_embeddings: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Update KnowledgeNode configuration (layers and LoRA rank) based on cluster size.
    
    This function creates a new KnowledgeNode with optimal configuration while
    preserving embeddings and filter state.
    
    Args:
        node: Current KnowledgeNode to update
        cluster_size: Number of texts in cluster
        config: Configuration object with node parameters (SRC_VOCAB_SIZE, D_MODEL, etc.)
        coherence: Cluster coherence score (default: 1.0)
        action: Bayesian decision action (e.g., "CREATE_NODE_MINIMAL")
        preserve_embeddings: Whether to preserve embeddings and filter (default: True)
    
    Returns:
        Tuple of (updated_node, update_info_dict) where update_info contains:
        - 'needs_update': bool - Whether update was needed
        - 'current_layers': int - Previous number of layers
        - 'optimal_layers': int - Optimal number of layers
        - 'current_rank': int - Previous LoRA rank
        - 'optimal_rank': int - Optimal LoRA rank
    
    Example:
        >>> new_node, info = update_node_configuration(
        ...     node=old_node,
        ...     cluster_size=100,
        ...     config=config,
        ...     coherence=0.8
        ... )
        >>> if info['needs_update']:
        ...     print(f"Updated: {info['current_layers']}→{info['optimal_layers']} layers")
    """
    # Use determine_node_configuration to get optimal configuration
    optimal_layers, optimal_rank = determine_node_configuration(
        cluster_size=cluster_size,
        coherence=coherence,
        action=action
    )
    
    # Get current configuration
    current_layers = get_current_layers(node)
    current_rank = get_current_lora_rank(node)
    
    # Check if update is needed
    if current_layers >= optimal_layers and current_rank >= optimal_rank:
        return node, {
            'needs_update': False,
            'current_layers': current_layers,
            'optimal_layers': optimal_layers,
            'current_rank': current_rank,
            'optimal_rank': optimal_rank
        }
    
    # Import here to avoid circular imports
    try:
        import torch
        import torch.nn as nn
        from xctopus.nodes.bayesian.bayesian_node import KnowledgeNode
    except ImportError:
        raise ImportError("torch and KnowledgeNode are required for update_node_configuration")
    
    # Preserve embeddings and filter if requested
    old_filter = None
    old_input_proj = None
    cluster_id = None
    embeddings = []
    
    if preserve_embeddings:
        old_filter = getattr(node, 'filter', None)
        old_input_proj = getattr(node, 'input_proj', None)
        
        # Try to extract cluster_id and embeddings from filter.memory
        if old_filter and hasattr(old_filter, 'memory') and isinstance(old_filter.memory, dict):
            # Find cluster_id by searching memory
            for cid, mem_data in old_filter.memory.items():
                if isinstance(mem_data, list) and len(mem_data) > 0:
                    cluster_id = cid
                    embeddings = mem_data
                    break
    
    # Create new node with optimal configuration
    new_node = KnowledgeNode(
        src_vocab_size=config.SRC_VOCAB_SIZE,
        tgt_vocab_size=config.TGT_VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=optimal_layers,
        d_ff=config.D_FF,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout=config.DROPOUT,
        embedding_dim=None,
        use_lora=True,
        lora_r=optimal_rank,
        lora_alpha=config.LORA_ALPHA
    )
    
    # Restore filter and embeddings if preserved
    if preserve_embeddings and old_filter:
        new_node.filter = old_filter
        if old_input_proj:
            new_node.input_proj = old_input_proj.to(config.DEVICE)
    
    # Freeze base parameters
    new_node.transformer.freeze_base_parameters()
    new_node = new_node.to(config.DEVICE)
    
    return new_node, {
        'needs_update': True,
        'current_layers': current_layers,
        'optimal_layers': optimal_layers,
        'current_rank': current_rank,
        'optimal_rank': optimal_rank,
        'cluster_id': cluster_id
    }


def batch_update_node_configurations(
    knowledge_nodes: Dict[int, Any],
    cluster_sizes: Dict[int, int],
    config: Any,
    coherence_scores: Optional[Dict[int, float]] = None,
    specific_cluster_ids: Optional[List[int]] = None,
    min_cluster_size: int = 30,
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Batch update multiple KnowledgeNode configurations.
    
    Args:
        knowledge_nodes: Dictionary mapping cluster_id to KnowledgeNode
        cluster_sizes: Dictionary mapping cluster_id to cluster size
        config: Configuration object with node parameters
        coherence_scores: Optional dictionary mapping cluster_id to coherence
        specific_cluster_ids: Optional list of specific cluster IDs to update
        min_cluster_size: Minimum cluster size to consider for update (default: 30)
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary mapping cluster_id to update_info for updated nodes
    
    Example:
        >>> updates = batch_update_node_configurations(
        ...     knowledge_nodes=nodes_dict,
        ...     cluster_sizes={0: 50, 1: 100, 2: 20},
        ...     config=config,
        ...     coherence_scores={0: 0.8, 1: 0.9, 2: 0.6}
        ... )
        >>> print(f"Updated {len(updates)} nodes")
    """
    updates = {}
    nodes_to_update = []
    
    # Identify nodes that need updating
    for cluster_id, node in knowledge_nodes.items():
        # Filter by specific IDs if provided
        if specific_cluster_ids is not None and cluster_id not in specific_cluster_ids:
            continue
        
        cluster_size = cluster_sizes.get(cluster_id, 0)
        
        # Only update clusters above minimum size
        if cluster_size < min_cluster_size:
            continue
        
        coherence = coherence_scores.get(cluster_id, 1.0) if coherence_scores else 1.0
        
        # Check if update is needed
        updated_node, update_info = update_node_configuration(
            node=node,
            cluster_size=cluster_size,
            config=config,
            coherence=coherence
        )
        
        if update_info['needs_update']:
            nodes_to_update.append({
                'cluster_id': cluster_id,
                'node': updated_node,
                'update_info': update_info,
                'size': cluster_size
            })
    
    if not nodes_to_update:
        if verbose:
            print(f"\n[OK] Todos los clusters tienen configuración óptima")
        return updates
    
    if verbose:
        print(f"\n" + "=" * 70)
        print("[*] ACTUALIZACIÓN DE CONFIGURACIÓN DE CLUSTERS")
        print("=" * 70)
        print(f"\n[*] Clusters que necesitan actualización: {len(nodes_to_update)}")
        for n in sorted(nodes_to_update, key=lambda x: x['size'], reverse=True):
            info = n['update_info']
            print(f"  - Cluster {n['cluster_id']}: {n['size']} embeddings → "
                  f"{info['current_layers']} layers/{info['current_rank']} rank → "
                  f"{info['optimal_layers']} layers/{info['optimal_rank']} rank")
        
        print(f"\n[WARNING]  NOTA: Actualizar configuración recrea el nodo y se pierde el entrenamiento LoRA previo.")
        print(f"   Los embeddings se preservan, y se puede hacer fine-tuning después.")
    
    # Perform updates
    for update_data in nodes_to_update:
        cluster_id = update_data['cluster_id']
        updated_node = update_data['node']
        update_info = update_data['update_info']
        
        # Replace node in dictionary
        knowledge_nodes[cluster_id] = updated_node
        updates[cluster_id] = update_info
        
        if verbose:
            print(f"  ✓ Cluster {cluster_id} actualizado: "
                  f"{update_info['current_layers']}→{update_info['optimal_layers']} layers, "
                  f"{update_info['current_rank']}→{update_info['optimal_rank']} rank")
    
    if verbose:
        print(f"\n[OK] {len(updates)} clusters actualizados con configuración óptima")
        print(f"[TIP] Recomendación: Ejecutar fine-tuning para entrenar los nuevos parámetros LoRA")
    
    return updates


def analyze_cluster_distribution(
    cluster_sizes: Dict[int, int],
    orphan_threshold: int = 3,
    small_threshold: int = 5,
    medium_threshold: int = 20
) -> Dict[str, Any]:
    """
    Analyze cluster size distribution and categorize clusters.
    
    Args:
        cluster_sizes: Dictionary mapping cluster_id to size
        orphan_threshold: Size threshold for orphan clusters (default: 3)
        small_threshold: Size threshold for small clusters (default: 5)
        medium_threshold: Size threshold for medium clusters (default: 20)
    
    Returns:
        Dictionary with:
        - 'orphan_clusters': List of (cluster_id, size) for orphans
        - 'small_clusters': List of (cluster_id, size) for small
        - 'medium_clusters': List of (cluster_id, size) for medium
        - 'large_clusters': List of (cluster_id, size) for large
        - 'total_clusters': Total number of clusters
        - 'total_embeddings': Total number of embeddings
        - 'avg_size': Average cluster size
    
    Example:
        >>> sizes = {0: 2, 1: 4, 2: 15, 3: 50}
        >>> dist = analyze_cluster_distribution(sizes)
        >>> print(f"Orphans: {len(dist['orphan_clusters'])}")
    """
    orphan_clusters = []
    small_clusters = []
    medium_clusters = []
    large_clusters = []
    
    for cluster_id, size in cluster_sizes.items():
        if size < orphan_threshold:
            orphan_clusters.append((cluster_id, size))
        elif size < small_threshold:
            small_clusters.append((cluster_id, size))
        elif size < medium_threshold:
            medium_clusters.append((cluster_id, size))
        else:
            large_clusters.append((cluster_id, size))
    
    total_clusters = len(cluster_sizes)
    total_embeddings = sum(cluster_sizes.values())
    avg_size = total_embeddings / total_clusters if total_clusters > 0 else 0
    
    return {
        'orphan_clusters': orphan_clusters,
        'small_clusters': small_clusters,
        'medium_clusters': medium_clusters,
        'large_clusters': large_clusters,
        'total_clusters': total_clusters,
        'total_embeddings': total_embeddings,
        'avg_size': avg_size
    }


def find_similar_cluster_pairs(
    centroids: Dict[int, np.ndarray],
    similarity_threshold: float = 0.7,
    max_pairs: Optional[int] = None
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of clusters with high similarity for potential merging.
    
    Args:
        centroids: Dictionary mapping cluster_id to centroid array
        similarity_threshold: Minimum similarity to consider (default: 0.7)
        max_pairs: Maximum number of pairs to return (default: None, return all)
    
    Returns:
        List of tuples (cluster_id1, cluster_id2, similarity) sorted by similarity (descending)
    
    Example:
        >>> centroids = {
        ...     0: np.random.rand(128),
        ...     1: np.random.rand(128),
        ...     2: np.random.rand(128)
        ... }
        >>> pairs = find_similar_cluster_pairs(centroids, similarity_threshold=0.8)
        >>> print(f"Found {len(pairs)} similar pairs")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for find_similar_cluster_pairs")
    
    similar_pairs = []
    cluster_ids = list(centroids.keys())
    
    for i, cid1 in enumerate(cluster_ids):
        for cid2 in cluster_ids[i+1:]:
            if cid1 in centroids and cid2 in centroids:
                # Use cosine_similarity from sklearn for consistency
                similarity = cosine_similarity(
                    centroids[cid1].reshape(1, -1),
                    centroids[cid2].reshape(1, -1)
                )[0, 0]
                
                if similarity > similarity_threshold:
                    similar_pairs.append((cid1, cid2, float(similarity)))
    
    # Sort by similarity (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Limit number of pairs if requested
    if max_pairs is not None and max_pairs > 0:
        similar_pairs = similar_pairs[:max_pairs]
    
    return similar_pairs


def calculate_cluster_internal_similarity(
    embeddings: Union[np.ndarray, List, Any],
    metric: str = 'cosine'
) -> float:
    """
    Calculate average internal similarity within a cluster.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
            Can be numpy array, torch tensor, or list
        metric: Similarity metric ('cosine' or 'euclidean')
            Default: 'cosine'
    
    Returns:
        Average pairwise similarity within the cluster (0.0 to 1.0 for cosine)
    
    Example:
        >>> embeddings = np.random.rand(10, 128)
        >>> sim = calculate_cluster_internal_similarity(embeddings)
        >>> print(f"Internal similarity: {sim:.4f}")
    """
    # Convert to numpy if needed
    if hasattr(embeddings, 'cpu'):
        # Torch tensor
        embeddings = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        # Torch tensor (alternative)
        embeddings = embeddings.numpy()
    elif isinstance(embeddings, list):
        # List of tensors or arrays
        if len(embeddings) == 0:
            return 0.0
        if hasattr(embeddings[0], 'cpu'):
            embeddings = np.array([e.cpu().numpy() for e in embeddings])
        elif hasattr(embeddings[0], 'numpy'):
            embeddings = np.array([e.numpy() for e in embeddings])
        else:
            embeddings = np.array(embeddings)
    else:
        embeddings = np.asarray(embeddings)
    
    if len(embeddings) == 0:
        return 0.0
    if len(embeddings) == 1:
        return 1.0  # Single point has perfect similarity with itself
    
    if metric == 'cosine':
        if not SKLEARN_AVAILABLE:
            # Fallback to manual calculation
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    # Manual cosine similarity
                    dot_product = np.dot(embeddings[i], embeddings[j])
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    if norm_i > 0 and norm_j > 0:
                        sim = dot_product / (norm_i * norm_j)
                        similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0
        else:
            # Use sklearn for efficiency
            similarity_matrix = cosine_similarity(embeddings)
            # Get upper triangle (excluding diagonal)
            n = len(embeddings)
            upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
            return float(np.mean(upper_triangle))
    elif metric == 'euclidean':
        # For euclidean, we calculate average distance and convert to similarity
        # (inverse relationship: lower distance = higher similarity)
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        # Normalize to 0-1 range (assuming max distance is reasonable)
        # This is a simple normalization, may need adjustment based on data
        max_expected_distance = np.max(distances) if distances else 1.0
        similarity = 1.0 / (1.0 + avg_distance / max_expected_distance)
        return float(similarity)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'")


def analyze_cluster_quality_per_cluster(
    embeddings: Union[np.ndarray, List, Any],
    labels: Union[List, np.ndarray],
    return_centroids: bool = True,
    return_variance: bool = True,
    return_internal_similarity: bool = True,
    metric: str = 'cosine'
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze cluster quality by calculating metrics PER CLUSTER.
    
    This function calculates individual metrics for each cluster, unlike
    evaluate_cluster_quality() which calculates global metrics. It reuses
    existing functions from cluster_utils to avoid code duplication.
    
    Args:
        embeddings: Array of embeddings [n_samples, embedding_dim]
            Can be numpy array, torch tensor, or list
        labels: Array of cluster labels [n_samples]
            Can be numpy array, list, or any iterable
        return_centroids: Whether to calculate centroids (default: True)
        return_variance: Whether to calculate variance (default: True)
        return_internal_similarity: Whether to calculate internal similarity (default: True)
        metric: Similarity metric for internal similarity ('cosine' or 'euclidean')
            Default: 'cosine'
    
    Returns:
        Dictionary with stats per cluster:
        {
            cluster_id: {
                'size': int,
                'centroid': np.ndarray (if return_centroids=True),
                'variance': float (if return_variance=True),
                'avg_internal_similarity': float (if return_internal_similarity=True)
            }
        }
    
    Example:
        >>> embeddings = np.random.rand(100, 128)
        >>> labels = [0]*30 + [1]*40 + [2]*25 + [-1]*5
        >>> stats = analyze_cluster_quality_per_cluster(embeddings, labels)
        >>> print(stats[0]['size'])  # Size of cluster 0
        >>> print(stats[0]['avg_internal_similarity'])  # Internal similarity
    """
    # Convert embeddings to numpy if needed
    if hasattr(embeddings, 'cpu'):
        # Torch tensor
        embeddings_np = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        # TensorFlow tensor or similar
        embeddings_np = embeddings.numpy()
    elif isinstance(embeddings, list):
        # List of tensors or arrays
        if len(embeddings) == 0:
            return {}
        if hasattr(embeddings[0], 'cpu'):
            embeddings_np = np.array([e.cpu().numpy() for e in embeddings])
        elif hasattr(embeddings[0], 'numpy'):
            embeddings_np = np.array([e.numpy() for e in embeddings])
        else:
            embeddings_np = np.array(embeddings)
    else:
        embeddings_np = np.asarray(embeddings)
    
    # Convert labels to numpy array
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    elif hasattr(labels, '__iter__') and not isinstance(labels, str):
        labels_list = list(labels)
    else:
        labels_list = [labels]
    
    labels_np = np.array(labels_list)
    
    if len(embeddings_np) != len(labels_np):
        raise ValueError(f"Mismatch: {len(embeddings_np)} embeddings but {len(labels_np)} labels")
    
    # Get unique cluster IDs (excluding outliers if present)
    unique_clusters = [c for c in np.unique(labels_np) if c != -1]
    
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        # Get embeddings for this cluster
        cluster_mask = labels_np == cluster_id
        cluster_embeddings = embeddings_np[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            continue
        
        stats = {
            'size': len(cluster_embeddings)
        }
        
        # Calculate centroid using unified function
        if return_centroids:
            centroids = calculate_cluster_centroids(
                embeddings_np,
                labels_np,
                exclude_outliers=False
            )
            centroid = centroids.get(cluster_id)
            if centroid is not None:
                stats['centroid'] = centroid
        
        # Calculate variance
        if return_variance:
            variance = np.var(cluster_embeddings, axis=0).mean()
            stats['variance'] = float(variance)
        
        # Calculate internal similarity using unified function
        if return_internal_similarity:
            # Pass cluster embeddings directly
            internal_sim = calculate_cluster_internal_similarity(
                cluster_embeddings,
                metric=metric
            )
            stats['avg_internal_similarity'] = internal_sim
        
        cluster_stats[cluster_id] = stats
    
    return cluster_stats


def extract_embeddings_from_nodes(
    nodes_dict: Dict[int, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Any]]]:
    """
    Extract embeddings and labels from a dictionary of KnowledgeNodes.
    
    This function extracts embeddings from node.filter.memory and creates
    corresponding labels. It handles both dict and list memory formats.
    
    Args:
        nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
            Each node should have node.filter.memory containing embeddings
    
    Returns:
        Tuple of (embeddings, labels, raw_embeddings_dict):
        - embeddings: numpy array [n_samples, embedding_dim]
        - labels: numpy array [n_samples] (cluster_ids)
        - raw_embeddings_dict: dict mapping cluster_id to list of embeddings
            (useful if you need torch tensors)
    
    Example:
        >>> nodes_dict = {0: node0, 1: node1}
        >>> embeddings, labels, raw = extract_embeddings_from_nodes(nodes_dict)
        >>> stats = analyze_cluster_quality_per_cluster(embeddings, labels)
    """
    all_embeddings = []
    all_labels = []
    raw_embeddings_dict = {}
    
    for cluster_id, node in nodes_dict.items():
        if not hasattr(node, 'filter') or not hasattr(node.filter, 'memory'):
            continue
        
        memory = node.filter.memory
        
        # Handle dict format: {cluster_id: [embeddings]}
        if isinstance(memory, dict):
            if cluster_id in memory and len(memory[cluster_id]) > 0:
                cluster_memory = memory[cluster_id]
                raw_embeddings_dict[cluster_id] = cluster_memory
                
                # Convert to numpy
                for emb in cluster_memory:
                    if hasattr(emb, 'cpu'):
                        all_embeddings.append(emb.cpu().numpy())
                    elif hasattr(emb, 'numpy'):
                        all_embeddings.append(emb.numpy())
                    else:
                        all_embeddings.append(np.array(emb))
                    all_labels.append(cluster_id)
        
        # Handle list format: [embeddings]
        elif isinstance(memory, list) and len(memory) > 0:
            raw_embeddings_dict[cluster_id] = memory
            
            # Convert to numpy
            for emb in memory:
                if hasattr(emb, 'cpu'):
                    all_embeddings.append(emb.cpu().numpy())
                elif hasattr(emb, 'numpy'):
                    all_embeddings.append(emb.numpy())
                else:
                    all_embeddings.append(np.array(emb))
                all_labels.append(cluster_id)
    
    if len(all_embeddings) == 0:
        return np.array([]), np.array([]), {}
    
    embeddings_np = np.array(all_embeddings)
    labels_np = np.array(all_labels)
    
    return embeddings_np, labels_np, raw_embeddings_dict


def identify_large_clusters(
    nodes_dict: Dict[int, Any],
    min_size: int = 50
) -> List[Tuple[int, int, Any]]:
    """
    Identify large clusters from KnowledgeNodes.
    
    This function uses analyze_cluster_distribution() internally to identify
    clusters that meet the minimum size threshold.
    
    Args:
        nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
        min_size: Minimum cluster size to consider "large" (default: 50)
    
    Returns:
        List of tuples (cluster_id, cluster_size, node) for large clusters,
        sorted by size (descending)
    
    Example:
        >>> nodes_dict = {0: node0, 1: node1, 2: node2}
        >>> large = identify_large_clusters(nodes_dict, min_size=50)
        >>> print(f"Found {len(large)} large clusters")
    """
    # Calculate cluster sizes
    cluster_sizes = {}
    for cluster_id, node in nodes_dict.items():
        if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
            memory = node.filter.memory
            if isinstance(memory, dict):
                cluster_size = len(memory.get(cluster_id, []))
            elif isinstance(memory, list):
                cluster_size = len(memory)
            else:
                cluster_size = 0
            cluster_sizes[cluster_id] = cluster_size
    
    if not cluster_sizes:
        return []
    
    # Use unified function to analyze distribution
    distribution = analyze_cluster_distribution(
        cluster_sizes=cluster_sizes,
        orphan_threshold=3,
        small_threshold=5,
        medium_threshold=min_size  # Use min_size as threshold for large
    )
    
    # Build list of large clusters with nodes
    large_clusters = []
    for cluster_id, size in distribution['large_clusters']:
        if cluster_id in nodes_dict:
            large_clusters.append((cluster_id, size, nodes_dict[cluster_id]))
    
    # Sort by size (descending)
    large_clusters.sort(key=lambda x: x[1], reverse=True)
    
    return large_clusters


def fine_tune_cluster_with_lora(
    node: Any,
    embeddings: List[Any],
    optimizer: Optional[Any] = None,
    criterion: Optional[Any] = None,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    device: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Fine-tune a KnowledgeNode using LoRA on cluster embeddings.
    
    This function performs LoRA fine-tuning on a KnowledgeNode using embeddings
    from its cluster. It handles optimizer creation, training loop, and loss tracking.
    
    Args:
        node: KnowledgeNode to fine-tune
        embeddings: List of embeddings from the cluster (can be torch tensors or numpy arrays)
        optimizer: Optimizer instance (if None, creates Adam for LoRA params)
        criterion: Loss function (if None, uses MSE Loss)
        num_epochs: Number of training epochs (default: 3)
        learning_rate: Learning rate for optimizer (default: 1e-4)
        device: PyTorch device (if None, uses embeddings device)
    
    Returns:
        Dictionary with training results:
        - 'initial_loss': float - Loss at first epoch
        - 'final_loss': float - Loss at last epoch
        - 'improvement': float - Difference (initial - final)
        - 'epoch_losses': List[float] - Loss per epoch
        - 'num_epochs': int - Number of epochs trained
    
    Example:
        >>> node = KnowledgeNode(...)
        >>> embeddings = [torch.rand(128) for _ in range(50)]
        >>> results = fine_tune_cluster_with_lora(node, embeddings, num_epochs=5)
        >>> print(f"Improvement: {results['improvement']:.6f}")
    """
    import torch
    import torch.nn as nn
    
    if not embeddings:
        return {
            'initial_loss': 0.0,
            'final_loss': 0.0,
            'improvement': 0.0,
            'epoch_losses': [],
            'num_epochs': 0
        }
    
    # Get device
    if device is None:
        if hasattr(embeddings[0], 'device'):
            device = embeddings[0].device
        else:
            device = torch.device("cpu")
    
    # Convert embeddings to torch tensors if needed
    torch_embeddings = []
    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            torch_emb = torch.from_numpy(emb).to(device)
        elif hasattr(emb, 'cpu'):
            torch_emb = emb.to(device)
        else:
            torch_emb = torch.tensor(emb, device=device)
        torch_embeddings.append(torch_emb)
    
    # Create optimizer if not provided
    if optimizer is None:
        # Get LoRA parameters
        if hasattr(node, 'get_lora_params'):
            lora_params = node.get_lora_params()
        elif hasattr(node, 'transformer') and hasattr(node.transformer, 'lora_parameters'):
            lora_params = node.transformer.lora_parameters()
        else:
            # Fallback: get all trainable parameters
            lora_params = [p for p in node.parameters() if p.requires_grad]
        
        if len(lora_params) == 0:
            return {
                'initial_loss': 0.0,
                'final_loss': 0.0,
                'improvement': 0.0,
                'epoch_losses': [],
                'num_epochs': 0,
                'error': 'No LoRA parameters found'
            }
        
        optimizer = torch.optim.Adam(lora_params, lr=learning_rate)
    
    # Create criterion if not provided
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Set node to training mode
    node.train()
    
    # Training loop
    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = []
        
        for emb in torch_embeddings:
            optimizer.zero_grad()
            
            # Prepare input
            emb_input = emb.unsqueeze(0).unsqueeze(0)
            
            # Forward pass
            refined, _ = node(emb_input)
            
            # Prepare target
            if hasattr(node, 'input_proj') and node.input_proj is not None:
                target = node.input_proj(emb_input).mean(dim=1)
            else:
                target = emb_input.mean(dim=1)
            
            # Calculate loss
            loss = criterion(refined, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        epoch_losses.append(avg_loss)
    
    # Calculate results
    initial_loss = epoch_losses[0] if epoch_losses else 0.0
    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    improvement = initial_loss - final_loss if len(epoch_losses) > 1 else 0.0
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement': improvement,
        'epoch_losses': epoch_losses,
        'num_epochs': num_epochs
    }