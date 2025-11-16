import torch
import torch.nn as nn

class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for linear layers.
    Allows training only low-rank parameters while freezing original weights.
    """
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear  # Original Linear from Transformer
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(original_linear.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, original_linear.out_features) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        # Forward of original linear (frozen) + LoRA adaptation (trainable)
        base_output = self.original_linear(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output
    
    def get_lora_params(self):
        """Returns only trainable LoRA parameters"""
        return [self.lora_A, self.lora_B]
    
    def get_rank(self):
        """Returns the current rank of this LoRA instance"""
        return self.rank
    
    def get_alpha(self):
        """Returns the current alpha"""
        return self.alpha
    
    def get_config(self):
        """Returns complete configuration as dict"""
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'scaling': self.scaling,
            'in_features': self.original_linear.in_features,
            'out_features': self.original_linear.out_features
        }
    
    @staticmethod
    def calculate_optimal_rank(cluster_size, 
                                small_threshold=30, 
                                medium_threshold=60,
                                small_rank=4, 
                                medium_rank=5, 
                                large_rank=6):
        """
        Calculates optimal rank based on cluster size.
        
        Args:
            cluster_size: Number of embeddings in the cluster
            small_threshold: Threshold for small clusters (default: 30)
            medium_threshold: Threshold for medium clusters (default: 60)
            small_rank: Rank for small clusters (default: 4)
            medium_rank: Rank for medium clusters (default: 5)
            large_rank: Rank for large clusters (default: 6)
        
        Returns:
            int: Recommended optimal rank
        
        Strategy:
            - <30 embeddings: rank 4 (basic)
            - 30-60 embeddings: rank 5 (medium clusters, prevent limitations)
            - 60+ embeddings: rank 6 (large clusters, better capacity)
        """
        if cluster_size < small_threshold:
            return small_rank
        elif cluster_size < medium_threshold:
            return medium_rank
        else:
            return large_rank
