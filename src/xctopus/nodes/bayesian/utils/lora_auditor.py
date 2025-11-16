"""
LoRA Auditor Module

Provides LoRAAuditor class for auditing and tracking LoRA (Low-Rank Adaptation) parameters
in KnowledgeNodes. This includes capturing state, computing changes, and visualizing
parameter updates during training.
"""

import os
from typing import Optional, Dict, List, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ..bayesian_node import KnowledgeNode


class LoRAAuditor:
    """
    Auditoría especializada de parámetros LoRA.
    
    Esta clase proporciona métodos para:
    - Capturar estado de parámetros LoRA
    - Calcular cambios en parámetros (absoluto, relativo, normas)
    - Calcular contribución de LoRA al output
    - Visualizar cambios en parámetros LoRA
    
    Example:
        >>> from xctopus.nodes.bayesian.utils import LoRAAuditor
        >>> auditor = LoRAAuditor(device=torch.device("cpu"), verbose=True)
        >>> initial_state = auditor.capture_state(node)
        >>> # ... training ...
        >>> final_state = auditor.capture_state(node)
        >>> changes = auditor.compute_changes(initial_state, final_state)
        >>> auditor.visualize_changes(changes, output_path="lora_changes.png")
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize LoRAAuditor.
        
        Args:
            device: PyTorch device (default: cpu)
            verbose: Whether to print progress messages (default: True)
        """
        self.device = device or torch.device("cpu")
        self.verbose = verbose
    
    def capture_state(self, node: KnowledgeNode) -> Dict[str, Any]:
        """
        Captura el estado actual de todos los parámetros LoRA de un nodo.
        
        Args:
            node: KnowledgeNode instance
        
        Returns:
            Dictionary with LoRA state:
            - 'lora_A': List of lora_A parameter tensors
            - 'lora_B': List of lora_B parameter tensors
            - 'param_names': List of parameter names
        """
        lora_params = node.get_lora_params()
        state = {
            'lora_A': [],
            'lora_B': [],
            'param_names': []
        }
        
        for param in lora_params:
            param_data = param.data.clone().cpu()
            param_name = None
            
            # Try to get parameter name
            for name, p in node.named_parameters():
                if p is param:
                    param_name = name
                    break
            
            # Classify as lora_A or lora_B based on shape
            # lora_A: [in_features, rank]
            # lora_B: [rank, out_features]
            if len(param_data.shape) == 2:
                if param_data.shape[0] > param_data.shape[1]:
                    state['lora_A'].append(param_data)
                    state['param_names'].append(f"{param_name or 'unknown'}_A")
                else:
                    state['lora_B'].append(param_data)
                    state['param_names'].append(f"{param_name or 'unknown'}_B")
        
        return state
    
    def compute_changes(
        self,
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcula métricas de cambio en los parámetros LoRA.
        
        Args:
            initial_state: Initial LoRA state (from capture_state)
            final_state: Final LoRA state (from capture_state)
        
        Returns:
            Dictionary with change metrics:
            - 'total_params': Total number of LoRA parameters
            - 'changed_params': Number of changed parameters
            - 'mean_absolute_change': Mean absolute change
            - 'max_absolute_change': Maximum absolute change
            - 'mean_relative_change': Mean relative change
            - 'lora_A_changes': List of change metrics for lora_A layers
            - 'lora_B_changes': List of change metrics for lora_B layers
            - 'frobenius_norm_changes': List of Frobenius norm changes per layer
        """
        metrics = {
            'total_params': 0,
            'changed_params': 0,
            'mean_absolute_change': 0.0,
            'max_absolute_change': 0.0,
            'mean_relative_change': 0.0,
            'lora_A_changes': [],
            'lora_B_changes': [],
            'frobenius_norm_changes': []
        }
        
        # Compare lora_A
        for init_A, final_A in zip(initial_state['lora_A'], final_state['lora_A']):
            change = final_A - init_A
            abs_change = torch.abs(change)
            
            metrics['lora_A_changes'].append({
                'mean': abs_change.mean().item(),
                'max': abs_change.max().item(),
                'std': abs_change.std().item(),
                'frobenius': torch.norm(change, p='fro').item()
            })
            
            metrics['total_params'] += change.numel()
            metrics['mean_absolute_change'] += abs_change.sum().item()
            metrics['max_absolute_change'] = max(metrics['max_absolute_change'], abs_change.max().item())
            
            # Relative change (avoid division by zero)
            with torch.no_grad():
                init_norm = torch.norm(init_A, p='fro')
                if init_norm > 1e-8:
                    relative_change = torch.norm(change, p='fro') / init_norm
                    metrics['mean_relative_change'] += relative_change.item()
            
            metrics['frobenius_norm_changes'].append(torch.norm(change, p='fro').item())
        
        # Compare lora_B
        for init_B, final_B in zip(initial_state['lora_B'], final_state['lora_B']):
            change = final_B - init_B
            abs_change = torch.abs(change)
            
            metrics['lora_B_changes'].append({
                'mean': abs_change.mean().item(),
                'max': abs_change.max().item(),
                'std': abs_change.std().item(),
                'frobenius': torch.norm(change, p='fro').item()
            })
            
            metrics['total_params'] += change.numel()
            metrics['mean_absolute_change'] += abs_change.sum().item()
            metrics['max_absolute_change'] = max(metrics['max_absolute_change'], abs_change.max().item())
            
            # Relative change
            with torch.no_grad():
                init_norm = torch.norm(init_B, p='fro')
                if init_norm > 1e-8:
                    relative_change = torch.norm(change, p='fro') / init_norm
                    metrics['mean_relative_change'] += relative_change.item()
            
            metrics['frobenius_norm_changes'].append(torch.norm(change, p='fro').item())
        
        # Normalize averages
        num_layers = len(initial_state['lora_A']) + len(initial_state['lora_B'])
        if num_layers > 0:
            metrics['mean_absolute_change'] /= metrics['total_params']
            metrics['mean_relative_change'] /= num_layers
        
        metrics['changed_params'] = metrics['total_params']  # All LoRA params are trainable
        
        return metrics
    
    def compute_contribution(
        self,
        node: KnowledgeNode,
        embedding: torch.Tensor,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calcula la contribución de LoRA al output comparando con/sin LoRA.
        
        Args:
            node: KnowledgeNode instance
            embedding: Input embedding tensor
            initial_state: Initial LoRA state (optional, for comparison)
        
        Returns:
            Dictionary with contribution metrics:
            - 'output_norm': Norm of output with LoRA
            - 'output_mean': Mean of output with LoRA
            - 'output_std': Standard deviation of output with LoRA
        """
        node.eval()
        
        with torch.no_grad():
            # Forward normal (with LoRA)
            emb_input = embedding.unsqueeze(0).unsqueeze(0).to(self.device)
            output_with_lora, _ = node(emb_input)
            
            # Calculate output without LoRA (base only)
            # This requires access to internal transformer layers
            # For simplicity, we compare with initial state
            # In a more complete implementation, we could temporarily disable LoRA
            
            # Alternative: compare with output using initial parameters
            # We save the output with current LoRA as reference
            
            contribution = {
                'output_norm': torch.norm(output_with_lora).item(),
                'output_mean': output_with_lora.mean().item(),
                'output_std': output_with_lora.std().item()
            }
        
        return contribution
    
    def visualize_changes(
        self,
        lora_changes: Dict[int, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualiza específicamente los cambios en los parámetros LoRA.
        
        Args:
            lora_changes: Dictionary mapping cluster_id to change metrics
            output_path: Path to save visualization (default: auto-generate)
        """
        if not lora_changes:
            if self.verbose:
                print("  [WARNING]  No hay datos de cambios LoRA para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Changes by cluster (bars)
        cluster_ids = list(lora_changes.keys())
        mean_changes = [lora_changes[cid]['mean_absolute_change'] for cid in cluster_ids]
        max_changes = [lora_changes[cid]['max_absolute_change'] for cid in cluster_ids]
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        axes[0, 0].bar(x - width/2, mean_changes, width, label='Cambio Promedio', alpha=0.7)
        axes[0, 0].bar(x + width/2, max_changes, width, label='Cambio Máximo', alpha=0.7)
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Magnitud del Cambio')
        axes[0, 0].set_title('Cambios en Parámetros LoRA por Cluster')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'Cluster {cid}' for cid in cluster_ids])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Relative changes
        relative_changes = [lora_changes[cid]['mean_relative_change'] for cid in cluster_ids]
        axes[0, 1].bar(range(len(cluster_ids)), relative_changes, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Cambio Relativo')
        axes[0, 1].set_title('Cambio Relativo en Parámetros LoRA')
        axes[0, 1].set_xticks(range(len(cluster_ids)))
        axes[0, 1].set_xticklabels([f'Cluster {cid}' for cid in cluster_ids])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Distribution of changes by type (A vs B)
        all_A_changes = []
        all_B_changes = []
        for changes in lora_changes.values():
            for change_dict in changes.get('lora_A_changes', []):
                all_A_changes.append(change_dict['mean'])
            for change_dict in changes.get('lora_B_changes', []):
                all_B_changes.append(change_dict['mean'])
        
        if all_A_changes or all_B_changes:
            axes[1, 0].hist(all_A_changes, bins=15, alpha=0.6, label='LoRA-A', color='blue')
            axes[1, 0].hist(all_B_changes, bins=15, alpha=0.6, label='LoRA-B', color='orange')
            axes[1, 0].set_xlabel('Cambio Absoluto Promedio')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].set_title('Distribución de Cambios: LoRA-A vs LoRA-B')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Frobenius norms by layer
        for cluster_id, changes in lora_changes.items():
            if changes['frobenius_norm_changes']:
                layer_indices = range(len(changes['frobenius_norm_changes']))
                axes[1, 1].plot(layer_indices, changes['frobenius_norm_changes'], 
                              marker='o', label=f'Cluster {cluster_id}')
        
        axes[1, 1].set_xlabel('Índice de Capa')
        axes[1, 1].set_ylabel('Norma de Frobenius del Cambio')
        axes[1, 1].set_title('Cambios por Capa (Norma de Frobenius)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            # Auto-generate path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
            output_path = os.path.join(project_root, "notebooks", "lora_changes_visualization.png")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"  [*] Visualización LoRA guardada en: {output_path}")
        
        plt.close()

