import torch
import torch.nn as nn
from xctopus.nodes.transformer.transformer import Transformer
from xctopus.modules.lora import LoRA

class KnowledgeNode(nn.Module):
    """
    Specialized knowledge node.
    Allows:
    - Incremental training with external embeddings
    - Use of LoRA adapters
    - Local memory for feedback
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_length, dropout, embedding_dim=None, use_lora=False, lora_r=4, lora_alpha=1.0):
        super().__init__()

        self.d_model = d_model
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            embedding_dim=embedding_dim,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # Input projection for external embeddings
        if embedding_dim and embedding_dim != d_model:
            self.input_proj = nn.Linear(embedding_dim, d_model)
        else:
            self.input_proj = None

        # Local memory for feedback
        self.memory = []

        # Placeholder for FilterBayesianNode
        self.filter = None

    def forward(self, embedding_seq):
        """
        embedding_seq: [batch, seq_len, embedding_dim]
        """
        # Projection if needed
        if self.input_proj:
            embedding_seq = self.input_proj(embedding_seq)

        # Forward to transformer
        refined_output = self.transformer.encode(external_embeddings=embedding_seq)

        # Save feedback embedding in local memory
        self.memory.append(refined_output.detach())

        return refined_output, refined_output  # second return as placeholder for feedback_embedding

    def get_lora_params(self):
        """Returns only LoRA parameters for incremental optimization"""
        return self.transformer.lora_parameters()
    
    def get_lora_config(self):
        """
        Retorna configuración LoRA actual del nodo.
        
        Returns:
            dict: Diccionario con configuración LoRA o None si LoRA no está habilitado.
                Keys: 'rank', 'alpha', 'enabled'
        """
        if not self.transformer.use_lora:
            return {'enabled': False}
        
        # Obtener rank de la primera capa LoRA encontrada
        first_lora = None
        for layer in self.transformer.encoder_layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'W_q'):
                if isinstance(layer.self_attn.W_q, LoRA):
                    first_lora = layer.self_attn.W_q
                    break
        
        if first_lora:
            return {
                'rank': first_lora.get_rank(),
                'alpha': first_lora.get_alpha(),
                'enabled': True
            }
        return {'enabled': False}
    
    def get_optimal_lora_rank(self, cluster_size, 
                              small_threshold=30, 
                              medium_threshold=60,
                              small_rank=4, 
                              medium_rank=5, 
                              large_rank=6):
        """
        Calcula rank óptimo para este nodo según tamaño del cluster.
        
        Args:
            cluster_size: Número de embeddings en el cluster
            small_threshold: Umbral para clusters pequeños (default: 30)
            medium_threshold: Umbral para clusters medianos (default: 60)
            small_rank: Rank para clusters pequeños (default: 4)
            medium_rank: Rank para clusters medianos (default: 5)
            large_rank: Rank para clusters grandes (default: 6)
        
        Returns:
            int: Rank óptimo recomendado
        """
        return LoRA.calculate_optimal_rank(
            cluster_size, 
            small_threshold=small_threshold,
            medium_threshold=medium_threshold,
            small_rank=small_rank,
            medium_rank=medium_rank,
            large_rank=large_rank
        )
    
    def needs_lora_upgrade(self, cluster_size, 
                           small_threshold=30, 
                           medium_threshold=60,
                           small_rank=4, 
                           medium_rank=5, 
                           large_rank=6):
        """
        Verifica si el nodo necesita upgrade de LoRA rank.
        
        Args:
            cluster_size: Número de embeddings en el cluster
            small_threshold: Umbral para clusters pequeños (default: 30)
            medium_threshold: Umbral para clusters medianos (default: 60)
            small_rank: Rank para clusters pequeños (default: 4)
            medium_rank: Rank para clusters medianos (default: 5)
            large_rank: Rank para clusters grandes (default: 6)
        
        Returns:
            bool: True si el rank actual es menor que el óptimo
        """
        current = self.get_lora_config()
        if not current or not current['enabled']:
            return False
        
        optimal = self.get_optimal_lora_rank(
            cluster_size,
            small_threshold=small_threshold,
            medium_threshold=medium_threshold,
            small_rank=small_rank,
            medium_rank=medium_rank,
            large_rank=large_rank
        )
        return current['rank'] < optimal
    
    def create_lora_optimizer(self, learning_rate):
        """
        Crea optimizer solo para parámetros LoRA.
        
        Args:
            learning_rate: Learning rate para el optimizer
        
        Returns:
            torch.optim.Optimizer o None si no hay parámetros LoRA
        """
        lora_params = self.get_lora_params()
        if len(lora_params) > 0:
            return torch.optim.Adam(lora_params, lr=learning_rate)
        return None

    def clear_memory(self):
        """Clear local memory (if desired)"""
        self.memory = []


# Alias for API compatibility
BayesianNode = KnowledgeNode
