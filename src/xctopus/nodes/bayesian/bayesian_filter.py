import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class FilterBayesianNode(nn.Module):
      '''
          This class is structure 1 of the xctopus architecture.
          This filter receives the inputs to process them in training mode or in model evaluation mode, 
          depending on the stage we are working on.
          ** Discover cluster
          ** Save memory and uncertainty
          y=Wx+b
      '''    
      def __init__(self, mode='train', initial_threshold=0.4, min_threshold=0.1, 
                   max_threshold=0.9, threshold_decay=0.95, adaptive_threshold=True,
                   enable_bayesian_decision=True, cost_recreate=1.0, utility_threshold=0.0):
          assert mode in ["eval","train"]
          super().__init__()
          self.mode = mode
          self.initial_threshold = initial_threshold
          self.min_threshold = min_threshold
          self.max_threshold = max_threshold  # Upper limit to avoid clusters that are too large (now configurable)
          self.threshold_decay = threshold_decay
          self.adaptive_threshold = adaptive_threshold
          self.similarity_threshold = initial_threshold
          self.memory = {}
          self.uncertainty = {}
          
          # Bayesian Decision Making
          self.enable_bayesian_decision = enable_bayesian_decision
          self.cost_recreate = cost_recreate  # Cost to recreate node
          self.utility_threshold = utility_threshold  # Minimum expected utility threshold
          
          # Decision history for online learning
          self.decision_history = []  # List of {features, action, outcome}
          self.cluster_ages = {}  # Time since cluster creation
          self.cluster_growth_rates = {}  # Growth per epoch
          
          # Bayesian priors (less conservative: allows initial node creation)
          self.prior_node_good = 0.5  # More permissive prior: 50% initial probability (increased from 0.3)
          
          # Simple Bayesian model (Naive Bayes with Gaussian distributions)
          # Adjusted for REAL cluster values: coherence 0.3-0.5 is acceptable, doesn't need 0.7+
          # Feature 0: size_norm - clusters of 30-80 embeddings are good (normalized to 0.4-0.7)
          # Feature 1: coherence - values 0.3-0.5 are acceptable (don't need 0.7+)
          # Feature 2: cv_inverse - low CV (0.2-0.4) is good, normalized to 0.6-0.8
          # Feature 3: silhouette_norm - values 0.1-0.3 are acceptable (normalized to 0.55-0.65)
          # Feature 4: growth_rate_norm - moderate growth (0.1-0.3) is good
          # Feature 5: age_norm - young clusters (0-2) are normal, normalized to 0.0-0.2
          # Feature 6: loss_improving - stable loss (0.0-0.1) is acceptable
          self.feature_stats_good = defaultdict(lambda: {'mean': 0.5, 'std': 0.25})  # Realistic stats for good nodes (lower mean, wider std)
          self.feature_stats_bad = defaultdict(lambda: {'mean': 0.2, 'std': 0.15})   # Stats for bad nodes (more separated from good)
        
      def print_cluster_stats(self):
          print("\nCluster Overview:")
          for cid, examples in self.memory.items():
            if len(examples) == 0:
                print(f"  Cluster {cid}: 0 embeddings")
            else:
                stacked = torch.stack(examples)
                mean_vec = torch.mean(stacked, dim=0)
                if mean_vec.dim() > 0:
                    print(f"  Cluster {cid}: {len(examples)} embeddings, ejemplo media={mean_vec[:5]}...")
                else:
                    print(f"  Cluster {cid}: {len(examples)} embeddings, ejemplo media={mean_vec}...")
            
      def _update_memory(self, cluster_id, embedding):
         # Step 1: Updates the memory of beliefs in training mode
        embedding = embedding.squeeze()
        
        # Ensure embedding is on CPU for consistency
        embedding = embedding.cpu()

        if cluster_id not in self.memory:
            self.memory[cluster_id] = []

        self.memory[cluster_id].append(embedding)          
        
        
      def evaluate(self, embedding):
         # Evaluates the inputs and executes the related methods.
         # Step-1: To create new clusters or detect similarities with existing clusters
         
         # Ensure embedding is on CPU (to avoid device issues)
         embedding = embedding.cpu()
         
         if len(self.memory) == 0:
            new_cluster = 0
            self._update_memory(new_cluster, embedding)
            return new_cluster, 1.0

            # Compare embedding with existing cluster centroids
         for cluster_id, examples in self.memory.items():
            # Ensure all examples are on CPU
            examples_cpu = [ex.cpu() if ex.device.type != 'cpu' else ex for ex in examples]
            centroid = torch.mean(torch.stack(examples_cpu), dim=0)

            embedding = embedding.view(-1)
            centroid = centroid.view(-1)

            similarity = F.cosine_similarity(embedding, centroid, dim=0)
            if similarity >= self.similarity_threshold:
                self._update_memory(cluster_id, embedding)
                
                # Update adaptive threshold periodically during evaluation
                # (every 10 processed embeddings to avoid slowing down too much)
                if self.adaptive_threshold and len(self.memory[cluster_id]) % 10 == 0:
                    self._process_feedback(self.memory)
                
                return cluster_id, similarity.item()

         new_cluster = len(self.memory)
         self._update_memory(new_cluster, embedding)
         
         # Activate adaptive threshold immediately after creating a new cluster
         # This prevents excessive creation of small clusters
         if self.adaptive_threshold and len(self.memory) > 0:
             # Update adaptive threshold in real time
             self._process_feedback(self.memory)
         
         return new_cluster, 1.0

        
      def switch_mode(self, mode='train'):
         assert mode in ['train','eval']
         self.mode = mode   
        
      def _process_feedback(self, feedback):
            """
            Adjusts similarity_threshold adaptively.
            
            Strategy:
            - If there are many clusters (>20), increase threshold to merge
            - If there are few clusters (<5), reduce threshold to split
            - Maintains a range between min_threshold and max_threshold
            """
            if not self.adaptive_threshold:
                return
            
            total_embeddings = sum(len(v) for v in feedback.values())
            num_clusters = len(feedback)
            
            # Adaptive strategy based on number of clusters (less aggressive)
            if num_clusters > 8:
                # Too many clusters: increase threshold moderately to merge
                self.similarity_threshold = min(
                    self.similarity_threshold * 1.10,  # Reduced from 1.25 to 1.10 (less aggressive)
                    self.max_threshold
                )
            elif num_clusters > 5:
                # Moderate-high number: increase threshold to avoid more fragmentation
                if total_embeddings > 20:
                    self.similarity_threshold = min(
                        self.similarity_threshold * 1.05,  # Reduced from 1.20 to 1.05 (less aggressive)
                        self.max_threshold
                    )
            elif num_clusters > 5:
                # Moderate number: conservative adjustment
                if total_embeddings > 100:
                    self.similarity_threshold = max(
                        self.similarity_threshold * self.threshold_decay,
                        self.min_threshold
                    )
            else:
                # Few clusters: allow more splitting
                if total_embeddings > 50:
                    self.similarity_threshold = max(
                        self.similarity_threshold * self.threshold_decay,
                        self.min_threshold
                    )
            
            # Ensure it's in the valid range
            self.similarity_threshold = max(
                self.min_threshold,
                min(self.similarity_threshold, self.max_threshold)
            )
            
            if num_clusters % 10 == 0 or self.similarity_threshold <= self.min_threshold + 0.01:
                print(f"[GlobalFilter] Clusters: {num_clusters}, Threshold: {self.similarity_threshold:.3f}")
    
      def decide_create_node(self, cluster_id: int, cluster_stats: Dict, 
                            global_config: Dict, history: Optional[List] = None) -> Tuple[str, Dict]:
            """
            Bayesian decision to create KnowledgeNode based on cluster features.
            
            Args:
                cluster_id: Cluster ID
                cluster_stats: Dict with {size, coherence, CV, silhouette, loss_trend, growth_rate, age}
                global_config: Global configuration {total_texts, MIN_CLUSTER_SIZE_dynamic, COST_RECREATE, ...}
                history: Optional history of previous decisions for learning
                
            Returns:
                Tuple[action, decision_info]:
                    - action: "CREATE_NODE", "PROMOTE_AS_CANDIDATE", "DEFER"
                    - decision_info: Dict with {P_good, ExpectedUtility, features, reasoning}
            """
            if not self.enable_bayesian_decision:
                # Fallback to simple size-based decision
                min_size = global_config.get('MIN_CLUSTER_SIZE_dynamic', 30)
                if cluster_stats.get('size', 0) >= min_size:
                    return "CREATE_NODE", {'method': 'simple_threshold', 'threshold': min_size}
                return "DEFER", {'method': 'simple_threshold', 'reason': 'size_below_threshold'}
            
            # 1) Calculate minimum dynamic threshold (more flexible hybrid formula)
            # [WARNING] IMPORTANT: This threshold is only for normalization, NOT for filtering
            # FilterBayesian must evaluate ALL clusters, even small ones
            total_texts = global_config.get('total_texts', 100)
            PCT_DATASET = 0.05  # Reduced from 0.08 to 0.05 to allow smaller clusters
            MIN_CAP = 3  # Reduced from 5 to 3 to allow small but useful clusters
            MAX_CAP = 30  # Reduced from 50 to 30 to be less restrictive
            MIN_SIZE = max(
                min(int(np.ceil(total_texts * PCT_DATASET)), MAX_CAP),
                MIN_CAP
            )
            
            # DO NOT use MIN_SIZE to filter - only for feature normalization
            
            # 2) Extract normalized features
            size = cluster_stats.get('size', 0)
            coherence = cluster_stats.get('coherence', 0.0)
            cv = cluster_stats.get('CV', 1.0)
            silhouette = cluster_stats.get('silhouette', 0.0)
            domain_purity = cluster_stats.get('domain_purity', 1.0)  # New feature: domain purity
            growth_rate = cluster_stats.get('growth_rate', 0.0)
            age = cluster_stats.get('age', 0)
            loss_trend = cluster_stats.get('loss_trend', 1.0)  # 1.0 = stable, <1.0 = improving
            needs_subclustering = cluster_stats.get('needs_subclustering', False)  # Flag for sub-clustering
            
            # Normalize features to [0, 1] - Adjusted for REAL cluster values
            features = {
                'size_norm': min(size / max(MIN_SIZE * 1.5, 1), 1.0),  # Normalize relative to 1.5x MIN_SIZE (more permissive)
                'coherence': max(0, min(coherence, 1.0)),  # Already in [0, 1] - values 0.3-0.5 are acceptable
                'cv_inverse': max(0, min(1.0 / (1.0 + cv), 1.0)),  # Invert CV (low CV = good)
                'silhouette_norm': max(0, min((silhouette + 1) / 2, 1.0)),  # Silhouette in [-1, 1] -> [0, 1]
                'domain_purity': max(0, min(domain_purity, 1.0)),  # Domain purity [0, 1] - 1.0 = pure domain
                # Adjust growth_rate_norm: small values (1-5) are normal, don't penalize
                'growth_rate_norm': max(0, min(growth_rate / 5.0, 1.0)),  # Reduced from 10.0 to 5.0 to not penalize young clusters
                'age_norm': max(0, min(age / 20.0, 1.0)),  # Normalize age more smoothly (increased from 10.0 to 20.0)
                # Adjust loss_improving: stable loss (0.95-1.0) is acceptable, not just improving
                'loss_improving': max(0, min(2.0 - loss_trend, 1.0))  # Changed: loss_trend 1.0 -> 1.0, 0.95 -> 1.05 (cap 1.0)
            }
            
            # Feature vector for Bayesian model (now includes domain_purity)
            x = np.array([
                features['size_norm'],
                features['coherence'],
                features['cv_inverse'],
                features['silhouette_norm'],
                features['domain_purity'],  # New feature
                features['growth_rate_norm'],
                features['age_norm'],
                features['loss_improving']
            ])
            
            # If cluster needs sub-clustering, slightly penalize the decision
            if needs_subclustering:
                # Reduce P_good if sub-clustering is needed (heterogeneous cluster)
                # This will be done after calculating initial P_good
                subclustering_penalty = 0.15  # 15% penalty
            else:
                subclustering_penalty = 0.0
            
            # 3) Calculate posterior probability P(node_is_good | features)
            P_good = self._bayesian_predict(x)
            
            # Apply penalty if sub-clustering is needed
            if needs_subclustering:
                P_good = max(0.0, P_good - subclustering_penalty)
            
            # 4) Estimate Benefit and Cost
            Benefit = self._estimate_benefit(cluster_stats, global_config)
            Cost = self.cost_recreate * global_config.get('COST_RECREATE_MULTIPLIER', 1.0)
            
            # 5) Expected Utility
            ExpectedUtility = P_good * Benefit - (1 - P_good) * Cost
            
            # If needs sub-clustering and is large, consider DEFER or PROMOTE_AS_CANDIDATE
            if needs_subclustering and size >= 50:
                # Large heterogeneous clusters: better to split them before creating node
                if P_good < 0.6:
                    # If P_good is already low after penalty, defer
                    action = "DEFER"
                    reasoning = f"Large cluster ({size} texts) but heterogeneous - needs sub-clustering before creating node (P={P_good:.3f}, domain_purity={domain_purity:.3f})"
                    decision_info = {
                        'action': action,
                        'P_good': float(P_good),
                        'ExpectedUtility': float(ExpectedUtility),
                        'features': {k: float(v) for k, v in features.items()},
                        'cluster_stats': {k: float(v) if isinstance(v, (int, float)) else v 
                                        for k, v in cluster_stats.items()},
                        'reasoning': reasoning,
                        'MIN_SIZE': MIN_SIZE,
                        'Benefit': float(Benefit),
                        'Cost': float(Cost)
                    }
                    return action, decision_info
            
            # 6) Decision rules
            min_coherence_candidate = global_config.get('min_coherence_candidate', 0.5)
            
            # Adjusted thresholds: more permissive to allow node creation
            # With prior_node_good=0.5 and more realistic feature_stats, P_good should be in range 0.4-0.7
            # IMPORTANT: Evaluate small but coherent clusters as well
            
            # Decision based on size and quality
            is_small_cluster = size < MIN_SIZE
            is_medium_cluster = MIN_SIZE <= size < MIN_SIZE * 2
            is_large_cluster = size >= MIN_SIZE * 2
            
            # For small clusters: require high coherence and very high P_good
            # If small but heterogeneous (high dispersion), use minimum configuration
            if is_small_cluster:
                # Small but very dispersed clusters: create with minimum configuration
                is_small_heterogeneous = (size < 20 and coherence < 0.5 and cv > 0.5)
                
                if is_small_heterogeneous:
                    # Small but heterogeneous cluster: create minimal node (1 layer, rank 4)
                    if ExpectedUtility > self.utility_threshold and P_good >= 0.60:
                        action = "CREATE_NODE_MINIMAL"  # New action for minimal nodes
                        reasoning = f"Small cluster ({size} texts) but heterogeneous - create minimal node (coherence={coherence:.3f}, CV={cv:.3f}, P={P_good:.3f})"
                    else:
                        action = "DEFER"
                        reasoning = f"Small cluster ({size} texts) heterogeneous with insufficient probability/utility (P={P_good:.3f}, EU={ExpectedUtility:.3f})"
                elif ExpectedUtility > self.utility_threshold and P_good >= 0.70 and coherence >= 0.6:
                    action = "CREATE_NODE"
                    reasoning = f"Small cluster ({size} texts) but very coherent (coherence={coherence:.3f}, P={P_good:.3f}, EU={ExpectedUtility:.3f})"
                elif P_good >= 0.50 and coherence >= 0.5:
                    action = "PROMOTE_AS_CANDIDATE"
                    reasoning = f"Small cluster ({size} texts) with moderate quality (P={P_good:.3f}, coherence={coherence:.3f}) - experimental candidate"
                else:
                    action = "DEFER"
                    if P_good < 0.50:
                        reasoning = f"Small cluster ({size} texts) with low probability (P={P_good:.3f} < 0.50)"
                    elif coherence < 0.5:
                        reasoning = f"Small cluster ({size} texts) with insufficient coherence ({coherence:.3f} < 0.5)"
                    else:
                        reasoning = f"Small cluster ({size} texts) with negative utility (EU={ExpectedUtility:.3f})"
            
            # For medium and large clusters: more standard thresholds
            elif ExpectedUtility > self.utility_threshold and P_good >= 0.45:
                action = "CREATE_NODE"
                reasoning = f"High probability (P={P_good:.3f}) and positive utility (EU={ExpectedUtility:.3f})"
            elif P_good >= 0.30 and coherence >= min_coherence_candidate:
                action = "PROMOTE_AS_CANDIDATE"
                reasoning = f"Moderate probability (P={P_good:.3f}) with acceptable coherence ({coherence:.3f})"
            else:
                action = "DEFER"
                if P_good < 0.30:
                    reasoning = f"Low probability (P={P_good:.3f} < 0.30)"
                elif coherence < min_coherence_candidate:
                    reasoning = f"Insufficient coherence ({coherence:.3f} < {min_coherence_candidate})"
                else:
                    reasoning = f"Negative expected utility (EU={ExpectedUtility:.3f})"
            
            # 7) Logging and traceability
            decision_info = {
                'action': action,
                'P_good': float(P_good),
                'ExpectedUtility': float(ExpectedUtility),
                'features': {k: float(v) for k, v in features.items()},
                'cluster_stats': {k: float(v) if isinstance(v, (int, float)) else v 
                                for k, v in cluster_stats.items()},
                'reasoning': reasoning,
                'MIN_SIZE': MIN_SIZE,
                'Benefit': float(Benefit),
                'Cost': float(Cost)
            }
            
            # Save to history for online learning
            self.decision_history.append({
                'cluster_id': cluster_id,
                'features': features,
                'action': action,
                'P_good': P_good,
                'timestamp': len(self.decision_history)  # Simulate timestamp
            })
            
            return action, decision_info
    
      def _bayesian_predict(self, x: np.ndarray) -> float:
            """
            Bayesian prediction using Naive Bayes with Gaussian distributions.
            
            Calculates P(good | x) using Bayes' rule:
            P(good | x) = P(x | good) * P(good) / P(x)
            """
            # Prior
            prior_good = self.prior_node_good
            prior_bad = 1.0 - prior_good
            
            # Likelihoods (assuming independence - Naive Bayes)
            log_likelihood_good = 0.0
            log_likelihood_bad = 0.0
            
            for i, feature_value in enumerate(x):
                # Get stats for this feature
                stats_good = self.feature_stats_good[i]
                stats_bad = self.feature_stats_bad[i]
                
                # Calculate log-likelihood using normal distribution
                # P(x_i | good) ~ N(mean_good, std_good)
                mean_good = stats_good['mean']
                std_good = max(stats_good['std'], 0.01)  # Avoid std=0
                
                mean_bad = stats_bad['mean']
                std_bad = max(stats_bad['std'], 0.01)
                
                # Log-likelihood (log of normal density)
                log_likelihood_good += -0.5 * np.log(2 * np.pi * std_good**2) - \
                                      0.5 * ((feature_value - mean_good) / std_good)**2
                
                log_likelihood_bad += -0.5 * np.log(2 * np.pi * std_bad**2) - \
                                     0.5 * ((feature_value - mean_bad) / std_bad)**2
            
            # Posterior (using log for numerical stability)
            log_posterior_good = np.log(prior_good) + log_likelihood_good
            log_posterior_bad = np.log(prior_bad) + log_likelihood_bad
            
            # Normalize (log-sum-exp trick)
            log_sum = np.logaddexp(log_posterior_good, log_posterior_bad)
            log_posterior_good -= log_sum
            
            P_good = np.exp(log_posterior_good)
            
            # Ensure valid range
            P_good = max(0.0, min(1.0, P_good))
            
            return float(P_good)
    
      def _estimate_benefit(self, cluster_stats: Dict, global_config: Dict) -> float:
            """
            Estimates the expected benefit of creating a KnowledgeNode for this cluster.
            
            Benefit = expected_new_texts_per_epoch * improvement_value
            Adjusted so ExpectedUtility is positive more frequently.
            """
            size = cluster_stats.get('size', 0)
            growth_rate = cluster_stats.get('growth_rate', 0.0)
            coherence = cluster_stats.get('coherence', 0.0)
            silhouette = cluster_stats.get('silhouette', 0.0)
            
            # Base benefit: more texts correctly assigned
            # Increased scale factor so Benefit is higher
            # coherence 0.3-0.5 is acceptable, doesn't need to be 0.7+
            coherence_factor = max(coherence, 0.3)  # Minimum 0.3 to not penalize too much
            base_benefit = size * coherence_factor * 0.2  # Increased from 0.1 to 0.2
            
            # Benefit from cluster quality (silhouette)
            quality_benefit = max(0, (silhouette + 1) / 2) * size * 0.05  # Quality bonus
            
            # Benefit from expected growth (increased)
            growth_benefit = max(growth_rate, 0) * coherence_factor * 0.1  # Increased from 0.05 to 0.1
            
            # Minimum benefit for clusters of reasonable size
            if size >= 20:
                min_benefit = 2.0  # Guaranteed minimum benefit
            elif size >= 10:
                min_benefit = 1.0
            else:
                min_benefit = 0.5
            
            # Total benefit
            total_benefit = base_benefit + quality_benefit + growth_benefit
            total_benefit = max(total_benefit, min_benefit)  # Ensure minimum
            
            return float(total_benefit)
    
      def update_priors_from_outcome(self, cluster_id: int, outcome: Dict):
            """
            Updates priors and Bayesian model statistics based on outcomes.
            
            Args:
                cluster_id: Cluster ID
                outcome: Dict with {success: bool, improved_loss: float, coherence_change: float, ...}
            """
            # Find previous decision for this cluster
            decision = None
            for d in reversed(self.decision_history):
                if d['cluster_id'] == cluster_id:
                    decision = d
                    break
            
            if not decision:
                return
            
            # Determine if it was a "good" or "bad" node
            success = outcome.get('success', False)
            improved_loss = outcome.get('improved_loss', 0.0)
            coherence_maintained = outcome.get('coherence_maintained', True)
            
            is_good = success and (improved_loss > 0 or coherence_maintained)
            
            # Update feature statistics
            features = decision['features']
            for i, feature_value in enumerate(features.values()):
                if is_good:
                    # Update stats for good nodes (exponential moving average)
                    old_mean = self.feature_stats_good[i]['mean']
                    old_std = self.feature_stats_good[i]['std']
                    alpha = 0.1  # Learning rate
                    
                    self.feature_stats_good[i]['mean'] = (1 - alpha) * old_mean + alpha * feature_value
                    # Update std in simplified way
                    self.feature_stats_good[i]['std'] = max(0.05, old_std * 0.95 + abs(feature_value - old_mean) * 0.05)
                else:
                    # Update stats for bad nodes
                    old_mean = self.feature_stats_bad[i]['mean']
                    old_std = self.feature_stats_bad[i]['std']
                    alpha = 0.1
                    
                    self.feature_stats_bad[i]['mean'] = (1 - alpha) * old_mean + alpha * feature_value
                    self.feature_stats_bad[i]['std'] = max(0.05, old_std * 0.95 + abs(feature_value - old_mean) * 0.05)
            
            # Update prior (more permissive adjustment to allow it to increase)
            if is_good:
                # Increase prior if node was good (more aggressive)
                self.prior_node_good = min(0.7, self.prior_node_good * 1.10)  # Increased from 1.05 to 1.10, max from 0.5 to 0.7
            else:
                # Reduce prior if node was bad (less aggressive)
                self.prior_node_good = max(0.3, self.prior_node_good * 0.98)  # Reduced from 0.95 to 0.98, min from 0.05 to 0.3

# Alias for API compatibility
BayesianFilter = FilterBayesianNode 