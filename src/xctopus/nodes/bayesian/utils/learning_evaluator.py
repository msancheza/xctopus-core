"""
Learning Evaluator Module

Provides LearningEvaluator class for advanced learning evaluation with detailed LoRA auditing,
training comparison, and comprehensive reporting. Extends LearningAuditor with additional
functionality for deep learning evaluation.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Import from local modules
from ..bayesian_node import KnowledgeNode
from .learning_auditor import LearningAuditor
from .lora_auditor import LoRAAuditor
from .cluster_utils import (
    calculate_cluster_coherence,
    calculate_domain_purity,
    fine_tune_cluster_with_lora
)


class LearningEvaluator(LearningAuditor):
    """
    Evaluador avanzado de aprendizaje con auditoría detallada de LoRA.
    
    Extiende LearningAuditor con:
    - Evaluación de rendimiento con múltiples embeddings
    - Comparación antes/después del entrenamiento completa
    - Visualización avanzada de progreso
    - Generación de reportes completos con auditoría LoRA
    
    Example:
        >>> from xctopus.nodes.bayesian.utils import LearningEvaluator
        >>> evaluator = LearningEvaluator(device=torch.device("cpu"), d_model=128)
        >>> report = evaluator.generate_learning_report(
        ...     pipeline_result, test_texts, train_epochs=5
        ... )
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        d_model: int = 128,
        verbose: bool = True
    ):
        """
        Initialize LearningEvaluator.
        
        Args:
            device: PyTorch device (default: cpu)
            d_model: Model dimension (default: 128)
            verbose: Whether to print progress messages (default: True)
        """
        super().__init__(device, d_model, verbose)
        self.lora_auditor = LoRAAuditor(device=device, verbose=verbose)
    
    def evaluate_node_performance(
        self,
        node: KnowledgeNode,
        embeddings_list: List[Any],
        node_id: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de un KnowledgeNode con múltiples embeddings.
        
        Args:
            node: KnowledgeNode to evaluate
            embeddings_list: List of embeddings (can be torch tensors or numpy arrays)
            node_id: Identifier for the node (default: "Unknown")
        
        Returns:
            Dictionary with performance metrics:
            - node_id: Node identifier
            - num_samples: Number of embeddings evaluated
            - similarities: List of cosine similarities per embedding
            - outputs: List of refined outputs per embedding
            - reconstruction_errors: List of reconstruction errors per embedding
            - avg_similarity: Average similarity
            - std_similarity: Standard deviation of similarities
            - avg_reconstruction_error: Average reconstruction error
            - std_reconstruction_error: Standard deviation of reconstruction errors
        """
        node.eval()
        metrics = {
            'node_id': node_id,
            'num_samples': len(embeddings_list),
            'similarities': [],
            'outputs': [],
            'reconstruction_errors': []
        }
        
        with torch.no_grad():
            for emb in embeddings_list:
                # Convert to tensor if needed
                if isinstance(emb, np.ndarray):
                    emb = torch.tensor(emb, dtype=torch.float32)
                emb_input = emb.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Forward pass
                refined_output, feedback = node(emb_input)
                
                # Calculate input/output similarity
                if node.input_proj:
                    input_projected = node.input_proj(emb_input).mean(dim=1)
                else:
                    input_projected = emb_input.mean(dim=1)
                
                # Cosine similarity
                input_norm = input_projected / (input_projected.norm(dim=1, keepdim=True) + 1e-8)
                output_norm = refined_output / (refined_output.norm(dim=1, keepdim=True) + 1e-8)
                similarity = (input_norm * output_norm).sum(dim=1).item()
                
                # Reconstruction error (MSE)
                reconstruction_error = nn.MSELoss()(refined_output, input_projected).item()
                
                metrics['similarities'].append(similarity)
                metrics['outputs'].append(refined_output.cpu())
                metrics['reconstruction_errors'].append(reconstruction_error)
        
        # Calculate statistics
        metrics['avg_similarity'] = np.mean(metrics['similarities'])
        metrics['std_similarity'] = np.std(metrics['similarities'])
        metrics['avg_reconstruction_error'] = np.mean(metrics['reconstruction_errors'])
        metrics['std_reconstruction_error'] = np.std(metrics['reconstruction_errors'])
        
        return metrics
    
    def compare_before_after_training(
        self,
        pipeline_result: Dict[str, Any],
        test_texts: Optional[List[str]] = None,
        train_epochs: int = 5,
        use_existing_optimizers: bool = True,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Compara el rendimiento antes y después del entrenamiento usando el resultado del pipeline dinámico.
        Incluye auditoría detallada de LoRA, coherencia y domain_purity.
        
        Args:
            pipeline_result: Resultado de dynamic_clustering_pipeline
            test_texts: Lista de textos de prueba (opcional, usa textos del pipeline si no se proporciona)
            train_epochs: Número de épocas de entrenamiento
            use_existing_optimizers: Si True, usa los optimizadores del pipeline (con LR ajustado)
            learning_rate: Learning rate para optimizadores nuevos (default: 1e-4)
        
        Returns:
            Dictionary with comparison results:
            - 'before': Metrics before training
            - 'after': Metrics after training
            - 'improvements': Improvement metrics per text
            - 'training_histories': Training history per cluster
            - 'initial_lora_states': Initial LoRA states
            - 'final_lora_states': Final LoRA states
            - 'lora_changes': LoRA change metrics
            - 'initial_coherence': Initial coherence metrics
            - 'final_coherence': Final coherence metrics
            - 'initial_domain_purity': Initial domain purity
            - 'final_domain_purity': Final domain purity
        """
        if self.verbose:
            print("=" * 70)
            print("🔬 EVALUACIÓN DE APRENDIZAJE: Antes vs Después del Entrenamiento")
            print("=" * 70)
        
        # Extract components from pipeline
        knowledge_nodes = pipeline_result['knowledge_nodes']
        embeddings = pipeline_result['embeddings']
        labels = pipeline_result['labels']
        texts = pipeline_result.get('texts', [])
        text_embeddings = pipeline_result.get('text_embeddings')
        optimizers = pipeline_result.get('optimizers', {})
        
        # Use pipeline texts if test texts not provided
        if test_texts is None:
            test_texts = texts
            test_embeddings = embeddings
        else:
            # Generate embeddings for test texts
            if text_embeddings is None:
                # Import TextPreprocessor if needed
                from ..core import TextPreprocessor
                text_preprocessor = TextPreprocessor()
                test_embeddings_tensor = text_preprocessor.encode_texts(test_texts)
                test_embeddings = test_embeddings_tensor.numpy()
            else:
                # Use existing text_embeddings
                test_embeddings = text_embeddings.encode_texts(test_texts).numpy()
        
        # CAPTURE INITIAL STATE OF LoRA AND COHERENCE
        if self.verbose:
            print("\n📸 Capturando estado inicial...")
            print("-" * 70)
        
        initial_lora_states = {}
        initial_coherence = {}
        initial_domain_purity = {}
        
        # Try to get domain_mapping from dataset if available
        domain_mapping = {}
        if 'domain_mapping' in pipeline_result:
            domain_mapping = pipeline_result['domain_mapping']
        else:
            # Try to get from dataset_path if available
            dataset_path = pipeline_result.get('dataset_path')
            if dataset_path and os.path.exists(dataset_path):
                try:
                    import pandas as pd
                    df_check = pd.read_csv(dataset_path, sep=",", quotechar='"', dtype=str, nrows=1)
                    if 'category' in df_check.columns:
                        df_full = pd.read_csv(dataset_path, sep=",", quotechar='"', dtype=str)
                        if 'category' in df_full.columns:
                            for i, cat in enumerate(df_full['category'].dropna()):
                                if i < len(labels):
                                    domain_mapping[i] = cat
                except:
                    pass
        
        for cluster_id, node in knowledge_nodes.items():
            # Capture LoRA state
            initial_lora_states[cluster_id] = self.lora_auditor.capture_state(node)
            num_lora_params = sum(p.numel() for p in node.get_lora_params())
            
            # Capture initial coherence
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            coherence_metrics = calculate_cluster_coherence(cluster_embeddings, metric='cosine')
            # Map to old format for compatibility
            coherence_metrics = {
                'coherence': coherence_metrics['coherence'],
                'cv': coherence_metrics['cv'],
                'mean_dist': coherence_metrics['mean_distance'],
                'std_dist': coherence_metrics['std_distance']
            }
            initial_coherence[cluster_id] = coherence_metrics
            
            # Capture initial domain_purity
            cluster_indices = np.where(cluster_mask)[0]
            purity = calculate_domain_purity(cluster_indices, domain_mapping) if domain_mapping else 1.0
            initial_domain_purity[cluster_id] = purity
            
            if self.verbose:
                print(f"  Cluster {cluster_id}:")
                print(f"    - Parámetros LoRA: {num_lora_params:,}")
                print(f"    - Coherencia inicial: {coherence_metrics['coherence']:.3f} (CV={coherence_metrics['cv']:.3f})")
                print(f"    - Domain purity inicial: {purity:.3f}")
        
        # EVALUATION BEFORE TRAINING
        if self.verbose:
            print("\n[*] EVALUACIÓN INICIAL (Antes del entrenamiento):")
            print("-" * 70)
        
        before_metrics = {}
        
        # Assign test texts to clusters using similarity with centroids
        cluster_centroids = {}
        for cluster_id in knowledge_nodes.keys():
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            if len(cluster_embeddings) > 0:
                cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        test_assignments = {}  # cluster_id -> [(text_idx, embedding)]
        
        for i, (text, emb) in enumerate(zip(test_texts, test_embeddings)):
            # Find most similar cluster
            best_cluster_id = None
            best_similarity = -1.0
            
            for cluster_id, centroid in cluster_centroids.items():
                similarity = cosine_similarity(
                    emb.reshape(1, -1),
                    centroid.reshape(1, -1)
                )[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            
            if best_cluster_id and best_similarity >= 0.3:  # Minimum threshold
                if best_cluster_id not in test_assignments:
                    test_assignments[best_cluster_id] = []
                test_assignments[best_cluster_id].append((i, emb))
                
                # Evaluate performance
                node = knowledge_nodes[best_cluster_id]
                metrics = self.evaluate_node_performance(node, [emb], f"Cluster_{best_cluster_id}")
                before_metrics[i] = metrics
        
        if self.verbose:
            print(f"  Textos asignados a clusters: {sum(len(v) for v in test_assignments.values())}/{len(test_texts)}")
        
        # Create optimizers if they don't exist or if not using existing ones
        if not use_existing_optimizers or not optimizers:
            for cluster_id, node in knowledge_nodes.items():
                if cluster_id not in optimizers:
                    lora_params = node.get_lora_params()
                    if len(lora_params) > 0:
                        optimizers[cluster_id] = torch.optim.Adam(lora_params, lr=learning_rate)
                    else:
                        optimizers[cluster_id] = torch.optim.Adam(node.parameters(), lr=learning_rate)
        
        # TRAINING
        if self.verbose:
            print(f"\n[*] Entrenando KnowledgeNodes ({train_epochs} épocas)...")
            print("-" * 70)
        
        training_histories = {}
        criterion = nn.MSELoss()
        
        for cluster_id, node in knowledge_nodes.items():
            if cluster_id in optimizers and cluster_id in test_assignments:
                # Get embeddings assigned to this cluster (already numpy arrays)
                cluster_embeddings = [emb for _, emb in test_assignments[cluster_id]]
                
                if len(cluster_embeddings) > 0:
                    # Get LR from optimizer to report
                    current_lr = optimizers[cluster_id].param_groups[0]['lr']
                    
                    # Use unified function for training
                    training_result = fine_tune_cluster_with_lora(
                        node=node,
                        embeddings=cluster_embeddings,
                        optimizer=optimizers[cluster_id],
                        criterion=criterion,
                        num_epochs=train_epochs,
                        device=self.device
                    )
                    
                    # Calculate similarities for each epoch (extend unified function)
                    history = {
                        'losses': training_result.get('epoch_losses', []),
                        'similarities': []
                    }
                    
                    # Calculate similarities by running forward passes
                    node.eval()
                    with torch.no_grad():
                        for epoch in range(train_epochs):
                            epoch_similarities = []
                            
                            for emb in cluster_embeddings:
                                # Convert to tensor if needed
                                if isinstance(emb, np.ndarray):
                                    emb = torch.tensor(emb, dtype=torch.float32)
                                emb_input = emb.unsqueeze(0).unsqueeze(0).to(self.device)
                                
                                # Forward
                                refined_output, _ = node(emb_input)
                                
                                # Target: projected embedding
                                if node.input_proj:
                                    target = node.input_proj(emb_input).mean(dim=1)
                                else:
                                    target = emb_input.mean(dim=1)
                                
                                # Calculate similarity
                                input_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
                                output_norm = refined_output / (refined_output.norm(dim=1, keepdim=True) + 1e-8)
                                similarity = (input_norm * output_norm).sum().item()
                                
                                epoch_similarities.append(similarity)
                            
                            history['similarities'].append(np.mean(epoch_similarities) if epoch_similarities else 0.0)
                    
                    training_histories[cluster_id] = history
                    
                    # Get cluster size
                    cluster_mask = labels == cluster_id
                    cluster_size = cluster_mask.sum()
                    
                    if self.verbose:
                        print(f"  Cluster {cluster_id} ({cluster_size} textos, LR={current_lr:.6f}):")
                        print(f"    - Loss inicial → final: {history['losses'][0]:.6f} → {history['losses'][-1]:.6f}")
                        print(f"    - Similitud inicial → final: {history['similarities'][0]:.4f} → {history['similarities'][-1]:.4f}")
                        if len(history['losses']) > 1:
                            loss_improvement = ((history['losses'][0] - history['losses'][-1]) / history['losses'][0]) * 100
                            print(f"    - Mejora en loss: {loss_improvement:+.2f}%")
        
        # CAPTURE FINAL STATE OF LoRA AND COHERENCE
        if self.verbose:
            print("\n📸 Capturando estado final...")
            print("-" * 70)
        
        final_lora_states = {}
        final_coherence = {}
        final_domain_purity = {}
        lora_changes = {}
        
        for cluster_id, node in knowledge_nodes.items():
            # Capture final LoRA state
            final_lora_states[cluster_id] = self.lora_auditor.capture_state(node)
            if cluster_id in initial_lora_states:
                changes = self.lora_auditor.compute_changes(initial_lora_states[cluster_id], final_lora_states[cluster_id])
                lora_changes[cluster_id] = changes
            
            # Capture final coherence
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            coherence_metrics = calculate_cluster_coherence(cluster_embeddings, metric='cosine')
            # Map to old format for compatibility
            coherence_metrics = {
                'coherence': coherence_metrics['coherence'],
                'cv': coherence_metrics['cv'],
                'mean_dist': coherence_metrics['mean_distance'],
                'std_dist': coherence_metrics['std_distance']
            }
            final_coherence[cluster_id] = coherence_metrics
            
            # Capture final domain_purity
            cluster_indices = np.where(cluster_mask)[0]
            purity = calculate_domain_purity(cluster_indices, domain_mapping) if domain_mapping else 1.0
            final_domain_purity[cluster_id] = purity
            
            if self.verbose:
                print(f"  Cluster {cluster_id}:")
                if cluster_id in lora_changes:
                    changes = lora_changes[cluster_id]
                    print(f"    - Cambio LoRA promedio: {changes['mean_absolute_change']:.6f}")
                    print(f"    - Cambio LoRA relativo: {changes['mean_relative_change']:.4f}")
                
                coherence_change = coherence_metrics['coherence'] - initial_coherence[cluster_id]['coherence']
                purity_change = purity - initial_domain_purity[cluster_id]
                
                print(f"    - Coherencia: {initial_coherence[cluster_id]['coherence']:.3f} → {coherence_metrics['coherence']:.3f} ({coherence_change:+.3f})")
                print(f"    - Domain purity: {initial_domain_purity[cluster_id]:.3f} → {purity:.3f} ({purity_change:+.3f})")
        
        # EVALUATION AFTER TRAINING
        if self.verbose:
            print("\n[*] EVALUACIÓN FINAL (Después del entrenamiento):")
            print("-" * 70)
        
        after_metrics = {}
        improvements = {}
        
        for cluster_id, assignments in test_assignments.items():
            if cluster_id not in knowledge_nodes:
                continue
            
            node = knowledge_nodes[cluster_id]
            
            for text_idx, emb in assignments:
                if text_idx in before_metrics:
                    metrics = self.evaluate_node_performance(node, [emb], f"Cluster_{cluster_id}")
                    after_metrics[text_idx] = metrics
                    
                    # Calculate improvements
                    before = before_metrics[text_idx]
                    after = metrics
                    
                    similarity_improvement = after['avg_similarity'] - before['avg_similarity']
                    error_reduction = before['avg_reconstruction_error'] - after['avg_reconstruction_error']
                    
                    improvements[text_idx] = {
                        'similarity_improvement': similarity_improvement,
                        'error_reduction': error_reduction,
                        'improvement_percent': (similarity_improvement / (before['avg_similarity'] + 1e-8)) * 100,
                        'cluster_id': cluster_id
                    }
                    
                    if self.verbose:
                        print(f"  Texto {text_idx+1} (Cluster {cluster_id}):")
                        print(f"    - Similitud: {before['avg_similarity']:.4f} → {after['avg_similarity']:.4f} "
                              f"({similarity_improvement:+.4f})")
                        print(f"    - Error: {before['avg_reconstruction_error']:.6f} → {after['avg_reconstruction_error']:.6f} "
                              f"({error_reduction:+.6f})")
                        print(f"    - Mejora: {improvements[text_idx]['improvement_percent']:+.2f}%")
        
        return {
            'before': before_metrics,
            'after': after_metrics,
            'improvements': improvements,
            'training_histories': training_histories,
            'initial_lora_states': initial_lora_states,
            'final_lora_states': final_lora_states,
            'lora_changes': lora_changes,
            'initial_coherence': initial_coherence,
            'final_coherence': final_coherence,
            'initial_domain_purity': initial_domain_purity,
            'final_domain_purity': final_domain_purity
        }
    
    def visualize_learning_progress(
        self,
        training_histories: Dict[int, Dict[str, List[float]]],
        lora_changes: Optional[Dict[int, Dict[str, Any]]] = None,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualiza el progreso del aprendizaje durante el entrenamiento.
        Incluye visualización de cambios en LoRA.
        
        Args:
            training_histories: Dictionary mapping cluster_id to training history
            lora_changes: Dictionary mapping cluster_id to LoRA change metrics (optional)
            output_path: Path to save visualization (default: auto-generate)
        """
        if not training_histories:
            if self.verbose:
                print("  [WARNING]  No hay historiales de entrenamiento para visualizar")
            return
        
        # Determine layout: 2x2 if LoRA data available, 1x2 if not
        has_lora = lora_changes and len(lora_changes) > 0
        
        if has_lora:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        for cluster_id, history in training_histories.items():
            epochs = range(1, len(history['losses']) + 1)
            ax1.plot(epochs, history['losses'], marker='o', label=f'Cluster {cluster_id}')
        
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Evolución del Loss durante el Entrenamiento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similarity plot
        for cluster_id, history in training_histories.items():
            epochs = range(1, len(history['similarities']) + 1)
            ax2.plot(epochs, history['similarities'], marker='s', label=f'Cluster {cluster_id}')
        
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Similitud Coseno')
        ax2.set_title('Evolución de la Similitud Input/Output')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # LoRA visualizations if available
        if has_lora:
            # Plot of changes in LoRA parameters (Frobenius norm)
            cluster_ids = []
            frobenius_norms = []
            for cluster_id, changes in lora_changes.items():
                if changes['frobenius_norm_changes']:
                    cluster_ids.append(f'Cluster {cluster_id}')
                    # Average changes per layer
                    avg_frobenius = np.mean(changes['frobenius_norm_changes'])
                    frobenius_norms.append(avg_frobenius)
            
            if cluster_ids:
                bars = ax3.bar(cluster_ids, frobenius_norms, color='steelblue', alpha=0.7)
                ax3.set_ylabel('Norma de Frobenius del Cambio')
                ax3.set_title('Magnitud de Cambios en Parámetros LoRA')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add values on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Plot of change distribution (histogram)
            all_changes = []
            for changes in lora_changes.values():
                # Get all absolute changes
                for change_dict in changes.get('lora_A_changes', []):
                    all_changes.append(change_dict['mean'])
                for change_dict in changes.get('lora_B_changes', []):
                    all_changes.append(change_dict['mean'])
            
            if all_changes:
                ax4.hist(all_changes, bins=20, color='coral', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Cambio Absoluto Promedio')
                ax4.set_ylabel('Frecuencia (Capas)')
                ax4.set_title('Distribución de Cambios en Parámetros LoRA')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.axvline(np.mean(all_changes), color='red', linestyle='--', 
                           label=f'Media: {np.mean(all_changes):.6f}')
                ax4.legend()
        
        plt.tight_layout()
        
        if output_path is None:
            # Auto-generate path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
            output_path = os.path.join(project_root, "notebooks", "learning_progress.png")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"  [*] Gráfico guardado en: {output_path}")
        
        plt.close()
    
    def generate_learning_report(
        self,
        pipeline_result: Dict[str, Any],
        test_texts: Optional[List[str]] = None,
        train_epochs: int = 5,
        use_existing_optimizers: bool = True,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Genera un reporte completo de aprendizaje con auditoría detallada de LoRA, coherencia y domain_purity.
        
        Args:
            pipeline_result: Resultado de dynamic_clustering_pipeline
            test_texts: Lista de textos de prueba (opcional)
            train_epochs: Número de épocas de entrenamiento
            use_existing_optimizers: Si True, usa los optimizadores del pipeline (con LR ajustado)
            learning_rate: Learning rate para optimizadores nuevos (default: 1e-4)
        
        Returns:
            Dictionary with complete learning report (same as compare_before_after_training)
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] REPORTE DE APRENDIZAJE CON AUDITORÍA COMPLETA")
            print("=" * 70)
        
        # Compare before/after
        comparison = self.compare_before_after_training(
            pipeline_result, test_texts, train_epochs, use_existing_optimizers, learning_rate
        )
        
        # Visualize progress (includes LoRA if available)
        if comparison['training_histories']:
            self.visualize_learning_progress(
                comparison['training_histories'], 
                lora_changes=comparison.get('lora_changes')
            )
        
        # Specific visualization of LoRA changes
        if comparison.get('lora_changes'):
            self.lora_auditor.visualize_changes(comparison['lora_changes'])
        
        # DETAILED LoRA REPORT
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] AUDITORÍA DETALLADA DE LoRA")
            print("=" * 70)
        
        if comparison.get('lora_changes'):
            for cluster_id, changes in comparison['lora_changes'].items():
                if self.verbose:
                    print(f"\n[*] Cluster {cluster_id}:")
                    print(f"  • Parámetros LoRA totales: {changes['total_params']:,}")
                    if changes['total_params'] > 0:
                        pct_modified = (changes['changed_params'] / changes['total_params']) * 100
                        print(f"  • Parámetros modificados: {changes['changed_params']:,} ({pct_modified:.1f}%)")
                    else:
                        print(f"  • Parámetros modificados: {changes['changed_params']:,} (N/A)")
                    print(f"  • Cambio absoluto promedio: {changes['mean_absolute_change']:.8f}")
                    print(f"  • Cambio absoluto máximo: {changes['max_absolute_change']:.8f}")
                    print(f"  • Cambio relativo promedio: {changes['mean_relative_change']:.4f} ({changes['mean_relative_change']*100:.2f}%)")
                    
                    # Statistics by parameter type
                    if changes['lora_A_changes']:
                        avg_A_change = np.mean([c['mean'] for c in changes['lora_A_changes']])
                        max_A_change = max([c['max'] for c in changes['lora_A_changes']])
                        print(f"  • LoRA-A: promedio={avg_A_change:.8f}, máximo={max_A_change:.8f}")
                    
                    if changes['lora_B_changes']:
                        avg_B_change = np.mean([c['mean'] for c in changes['lora_B_changes']])
                        max_B_change = max([c['max'] for c in changes['lora_B_changes']])
                        print(f"  • LoRA-B: promedio={avg_B_change:.8f}, máximo={max_B_change:.8f}")
                    
                    # Frobenius norms
                    if changes['frobenius_norm_changes']:
                        avg_frobenius = np.mean(changes['frobenius_norm_changes'])
                        max_frobenius = max(changes['frobenius_norm_changes'])
                        print(f"  • Norma de Frobenius: promedio={avg_frobenius:.6f}, máximo={max_frobenius:.6f}")
        else:
            if self.verbose:
                print("  [WARNING]  No se detectaron cambios en parámetros LoRA.")
                print("  [TIP] Verifica que use_lora=True en los KnowledgeNodes.")
        
        # Summary and recommendations
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] RESUMEN Y RECOMENDACIONES")
            print("=" * 70)
        
        if comparison['improvements']:
            avg_improvement = np.mean([imp['similarity_improvement'] for imp in comparison['improvements'].values()])
            avg_error_reduction = np.mean([imp['error_reduction'] for imp in comparison['improvements'].values()])
            
            if self.verbose:
                print(f"\n[OK] Mejora promedio en similitud: {avg_improvement:+.4f}")
                print(f"[OK] Reducción promedio en error: {avg_error_reduction:+.6f}")
                
                # Recommendations
                print("\n[TIP] RECOMENDACIONES:")
                
                if avg_improvement > 0.05:
                    print("  [OK] El aprendizaje es EFECTIVO. Los nodos están mejorando significativamente.")
                    print("  [OK] Puedes continuar añadiendo más datos de entrenamiento.")
                elif avg_improvement > 0.01:
                    print("  [WARNING]  El aprendizaje muestra mejoras MODERADAS.")
                    print("  [TIP] Considera:")
                    print("     - Aumentar el número de épocas")
                    print("     - Ajustar el learning rate")
                    print("     - Verificar la calidad de los datos")
                elif avg_improvement > 0:
                    print("  [WARNING]  El aprendizaje muestra mejoras MÍNIMAS.")
                    print("  [TIP] Considera:")
                    print("     - Revisar si los datos son apropiados para estos clusters")
                    print("     - Aumentar significativamente las épocas")
                    print("     - Verificar que LoRA esté habilitado correctamente")
                else:
                    print("  [ERROR] El aprendizaje NO muestra mejoras.")
                    print("  [TIP] Acciones recomendadas:")
                    print("     - Verificar que el entrenamiento se esté ejecutando")
                    print("     - Revisar los hiperparámetros (LR, épocas)")
                    print("     - Considerar si los datos son apropiados")
                    print("     - Verificar que los optimizadores estén actualizando parámetros")
                
                if avg_error_reduction > 0:
                    print(f"\n  [OK] El error de reconstrucción se redujo en promedio {avg_error_reduction:.6f}")
                else:
                    print(f"\n  [WARNING]  El error de reconstrucción no mejoró")
        
        # Specific recommendations based on LoRA
        if comparison.get('lora_changes'):
            if self.verbose:
                print("\n[TIP] ANÁLISIS DE CONTRIBUCIÓN LoRA:")
            avg_lora_change = np.mean([c['mean_absolute_change'] 
                                      for c in comparison['lora_changes'].values()])
            
            if self.verbose:
                if avg_lora_change > 1e-4:
                    print("  [OK] LoRA está ajustando significativamente los parámetros.")
                    print(f"  [OK] Cambio promedio en parámetros LoRA: {avg_lora_change:.8f}")
                    print("  [OK] El aprendizaje incremental está funcionando correctamente.")
                elif avg_lora_change > 1e-6:
                    print("  [WARNING]  LoRA está ajustando parámetros, pero los cambios son moderados.")
                    print(f"  [WARNING]  Cambio promedio: {avg_lora_change:.8f}")
                    print("  [TIP] Considera aumentar el learning rate o las épocas.")
                else:
                    print("  [WARNING]  Los cambios en LoRA son muy pequeños.")
                    print(f"  [WARNING]  Cambio promedio: {avg_lora_change:.8f}")
                    print("  [TIP] Verifica que:")
                    print("     - Los parámetros LoRA estén siendo optimizados")
                    print("     - El learning rate sea apropiado")
                    print("     - Los datos sean relevantes para los clusters")
        
        # Coherence analysis pre/post training
        if comparison.get('initial_coherence') and comparison.get('final_coherence'):
            if self.verbose:
                print("\n[TIP] ANÁLISIS DE COHERENCIA PRE/POST ENTRENAMIENTO:")
            coherence_improvements = {}
            for cluster_id in comparison['initial_coherence'].keys():
                if cluster_id in comparison['final_coherence']:
                    init_coherence = comparison['initial_coherence'][cluster_id]['coherence']
                    final_coherence = comparison['final_coherence'][cluster_id]['coherence']
                    improvement = final_coherence - init_coherence
                    coherence_improvements[cluster_id] = improvement
                    
                    if self.verbose:
                        if improvement > 0.05:
                            print(f"  [OK] Cluster {cluster_id}: Coherencia mejoró significativamente ({init_coherence:.3f} → {final_coherence:.3f}, +{improvement:.3f})")
                        elif improvement > 0:
                            print(f"  [WARNING]  Cluster {cluster_id}: Coherencia mejoró ligeramente ({init_coherence:.3f} → {final_coherence:.3f}, +{improvement:.3f})")
                        elif improvement < -0.05:
                            print(f"  [ERROR] Cluster {cluster_id}: Coherencia empeoró ({init_coherence:.3f} → {final_coherence:.3f}, {improvement:.3f})")
                            print(f"     → Considerar revisar datos o configuración del nodo")
            
            avg_coherence_improvement = np.mean(list(coherence_improvements.values())) if coherence_improvements else 0
            if self.verbose:
                print(f"\n  [*] Mejora promedio en coherencia: {avg_coherence_improvement:+.3f}")
        
        # Domain purity analysis
        if comparison.get('initial_domain_purity') and comparison.get('final_domain_purity'):
            if self.verbose:
                print("\n[TIP] ANÁLISIS DE DOMAIN PURITY:")
            for cluster_id in comparison['initial_domain_purity'].keys():
                if cluster_id in comparison['final_domain_purity']:
                    init_purity = comparison['initial_domain_purity'][cluster_id]
                    final_purity = comparison['final_domain_purity'][cluster_id]
                    
                    if self.verbose:
                        if init_purity < 0.6:
                            print(f"  [WARNING]  Cluster {cluster_id}: Domain purity baja ({init_purity:.3f}) → posible cluster heterogéneo")
                            if final_purity < init_purity:
                                print(f"     → Purity empeoró después del entrenamiento ({final_purity:.3f})")
                                print(f"     → Considerar sub-clustering o revisar asignación de textos")
        
        return comparison

