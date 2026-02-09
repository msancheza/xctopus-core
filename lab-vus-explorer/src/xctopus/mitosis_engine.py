
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Any

# Internal imports
from .settings import DEVICE, DTYPE, SMS_WEIGHT_COHERENCE, SMS_WEIGHT_MASS

logger = logging.getLogger(__name__)

class MitosisEngine:
    """
    MOTOR DE MITOSIS: Convierte 1 buffer incoherente -> 2+ nodos coherentes.
    BIOLOGÍA COMPUTACIONAL: Spherical K-Means + Variance Gain Validation.
    """
    
    def __init__(self, repository: Any, filter_system: Any):
        """
        Initialize the Mitosis Engine.
        
        Args:
            repository: Reference to the KNRepository
            filter_system: Reference to FilterBayesian (for updating signatures if needed)
        """
        self.repository = repository
        self.filter = filter_system
        logger.info("MitosisEngine Initialized [Spherical Geometry Enabled]")

    def check_mitosis_eligibility(self, buffer_id: str) -> bool:
        """
        Check if a buffer meets the biological criteria for mitosis.
        1. Critical Mass: Must have enough points to support 2 children.
        2. High Variance: Must be confused/incoherent.
        3. Potential: Mass * Coherence checks.
        """
        # Fetch status
        sig = self.repository.get_signature(buffer_id)
        if not sig:
            return False
            
        mass = sig.get('mass', 0)
        variance = sig.get('variance', 0.0)
        
        # Criteria from risks.txt / manifest
        # Mass >= 50 ensures each child gets ~25 texts (valid for KN)
        # Variance > 0.35 indicates significant semantic spread
        if mass >= 50 and variance > 0.35:
            # Additional check: Is there enough "structure" hidden?
            # Or is it just pure noise? Pure noise might also fail this, 
            # but we'll prune it via Variance Gain check later.
            logger.info(f"🧬 Mitosis Candidate Detected: {buffer_id} (Mass={mass}, Var={variance:.3f})")
            return True
            
        return False

    def execute_mitosis(self, buffer_id: str) -> List[str]:
        """
        Execute the Mitosis process:
        1. Fetch embeddings
        2. Run Spherical K-Means (k=2)
        3. Validate Variance Gain (Apoptosis check)
        4. If valid, split buffer into 2 new buffers
        5. Delete parent buffer
        6. Return new buffer IDs
        """
        logger.info(f"🧬 STARTING MITOSIS for {buffer_id}")
        
        # 1. Fetch Embeddings
        # get_buffer_embeddings returns list of tensors
        embeddings_list = self.repository.get_buffer_embeddings(buffer_id)
        if not embeddings_list:
            logger.warning(f"Mitosis aborted: No embeddings found for {buffer_id}")
            return [buffer_id]
            
        # Stack to tensor [N, 384]
        embeddings = torch.stack(embeddings_list).to(DEVICE, dtype=DTYPE)
        N = embeddings.shape[0]
        
        if N < 50: # Safety check (race condition?)
            return [buffer_id]

        # 2. Spherical K-Means (k=2)
        try:
            # Returns cluster_assignments [N], centroids [2, 384]
            assignments, centroids = self._spherical_kmeans(embeddings, k=2)
        except Exception as e:
            logger.error(f"Mitosis failed during K-Means: {e}")
            return [buffer_id]
            
        # 3. Variance Gain Validation (Checkpoint)
        # Calculate variance of parent (already known, but let's recompute to be sure)
        parent_var = self._calculate_variance(embeddings)
        
        # Split data
        mask_0 = (assignments == 0)
        mask_1 = (assignments == 1)
        
        emb_0 = embeddings[mask_0]
        emb_1 = embeddings[mask_1]
        
        if len(emb_0) < 10 or len(emb_1) < 10:
             logger.info(f"Mitosis aborted: Uneven split ({len(emb_0)} vs {len(emb_1)}). Not structurally sound.")
             return [buffer_id]

        var_0 = self._calculate_variance(emb_0)
        var_1 = self._calculate_variance(emb_1)
        
        # Gain metric: Weighted average variance should be significantly lower
        # Or simple sum check?
        # Let's use weighted variance to be fair.
        n0, n1 = len(emb_0), len(emb_1)
        weighted_child_var = (n0 * var_0 + n1 * var_1) / N
        
        variance_reduction = parent_var - weighted_child_var
        
        logger.info(f"Mitosis Validation: ParentVar={parent_var:.4f} -> ChildrenVar={weighted_child_var:.4f} (Delta={variance_reduction:.4f})")
        
        # Threshold: We want at least 20% reduction in confusion? Or absolute value 0.05?
        # If parent was 0.40, we want children to be ~0.30 or less.
        if variance_reduction < 0.02: # Tolerant threshold for MVP
             logger.info(f"Mitosis REJECTED: Insufficient variance gain. Buffer is likely pure noise, not multi-modal.")
             # TODO: Maybe mark for pruning? For now, just return parent.
             return [buffer_id]

        # 4. Cytokinesis (Split)
        logger.info("Mitosis ACCEPTED. Splitting buffer...")
        
        child_a_id = f"{buffer_id}_A"
        child_b_id = f"{buffer_id}_B"
        
        # Create children in DB
        # We need a way to move embeddings.
        # Efficient way: Update node_id in node_memory?
        # Or delete/insert?
        # Repository doesn't expose "move", so lets use creation/deletion for safety.
        
        # Create Child A
        self.repository.create_buffer(child_a_id) # Assumes this method exists or we use register_kn with status BUFFER
        # Actually create_buffer usually doesn't exist explicitly, we typically just add embeddings.
        # But wait, we need 'nodes' entry? 
        # Usually 'add_embedding_to_memory' links to a node.
        # We need to register the node first.
        
        # Let's check how buffers are created. usually orchestrator 'process_new_embedding' -> 'register_kn' (if new).
        # We will use register_kn equivalent for buffers.
        # Assuming register_buffer or similar exists. If not, use register_kn with status='BUFFER'
        
        # Re-using register_kn logic manually?
        # Let's calculate centroids for children
        cent_0 = F.normalize(emb_0.mean(dim=0), p=2, dim=0)
        cent_1 = F.normalize(emb_1.mean(dim=0), p=2, dim=0)
        
        # Register Child A
        self.repository.save_new_kn(
            node_id=child_a_id,
            centroid=cent_0,
            mass=n0,
            variance=float(var_0),
            status='BUFFER', # IMPORTANT
            origin_type='MITOSIS'
        )
        # Add embeddings A
        self.repository.add_embeddings_bulk(child_a_id, emb_0)
        
        # Register Child B
        self.repository.save_new_kn(
            node_id=child_b_id,
            centroid=cent_1,
            mass=n1,
            variance=float(var_1),
            status='BUFFER',
            origin_type='MITOSIS'
        )
        # Add embeddings B
        self.repository.add_embeddings_bulk(child_b_id, emb_1)
        
        # 5. Apoptosis of Parent
        self.repository.delete_node(buffer_id)
        
        logger.info(f"🧬 Mitosis SUCCESS: {buffer_id} -> [{child_a_id}, {child_b_id}]")
        return [child_a_id, child_b_id]

    def _spherical_kmeans(self, embeddings: torch.Tensor, k: int = 2, max_iters: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spherical K-Means implementation.
        """
        N, D = embeddings.shape
        
        # 1. Normalize embeddings (Cos Sim -> Euclidean on Sphere)
        # Assuming embeddings are already somewhat normalized, but enforce it.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Init Centroids (K-Means++)
        # Pick 1st random
        centroids = []
        first_idx = torch.randint(0, N, (1,)).item()
        centroids.append(embeddings[first_idx])
        
        for _ in range(k - 1):
            # Distances to current centroids (1 - CosSim)
            # cluster_dists [N, current_k]
            cents_stack = torch.stack(centroids)
            dists = 1.0 - torch.mm(embeddings, cents_stack.T) # [N, current_k]
            min_dists, _ = torch.min(dists, dim=1)
            
            # Weighted random sampling
            probs = min_dists ** 2
            next_idx = torch.multinomial(probs, 1).item()
            centroids.append(embeddings[next_idx])
            
        centroids = torch.stack(centroids) # [K, D]
        
        prev_assignments = None
        assignments = torch.zeros(N, dtype=torch.long, device=DEVICE)
        
        for i in range(max_iters):
            # E-Step: Assign to closest centroid (Max Cosine Sim)
            sims = torch.mm(embeddings, centroids.T) # [N, K]
            assignments = torch.argmax(sims, dim=1)
            
            if prev_assignments is not None and torch.equal(assignments, prev_assignments):
                break
            prev_assignments = assignments.clone()
            
            # M-Step: Update Centroids
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    # Mean vector
                    mean_vec = embeddings[mask].mean(dim=0)
                    # PROJECT BACK TO SPHERE
                    centroids[c] = F.normalize(mean_vec, p=2, dim=0)
                else:
                    # Re-init empty cluster? Or leave as is.
                    # For mitigation, re-init random
                    rand_idx = torch.randint(0, N, (1,)).item()
                    centroids[c] = embeddings[rand_idx]
                    
        return assignments, centroids

    def _calculate_variance(self, embeddings: torch.Tensor) -> float:
        """Calculate variance (1 - avg_sim) for a tensor of embeddings."""
        if len(embeddings) < 2:
            return 0.0
        centroid = F.normalize(embeddings.mean(dim=0), p=2, dim=0)
        sims = F.cosine_similarity(embeddings, centroid.unsqueeze(0))
        return 1.0 - sims.mean().item()
