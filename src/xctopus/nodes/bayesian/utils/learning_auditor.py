"""
Learning Auditor Module

Provides LearningAuditor class for auditing and evaluating learning in KnowledgeNodes.
"""

import os
from typing import Optional, List, Dict, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import from local modules
from ..bayesian_filter import FilterBayesianNode
from ..bayesian_node import KnowledgeNode
from ..core import TextPreprocessor
from .cluster_utils import (
    analyze_cluster_quality_per_cluster,
    extract_embeddings_from_nodes
)


class LearningAuditor:
    """
    Auditoría y evaluación del aprendizaje en KnowledgeNodes.
    
    Esta clase proporciona métodos para:
    - Evaluar respuestas de KnowledgeNodes
    - Comparar estados antes/después del entrenamiento
    - Analizar calidad de clusters y memoria
    - Visualizar embeddings y aprendizaje
    - Generar reportes completos de auditoría
    
    Example:
        >>> from xctopus.nodes.bayesian.utils import LearningAuditor
        >>> auditor = LearningAuditor(device=torch.device("cpu"), d_model=128)
        >>> report = auditor.generate_audit_report(
        ...     nodes_dict, test_texts, text_preprocessor, filter_node
        ... )
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        d_model: int = 128,
        verbose: bool = True
    ):
        """
        Initialize LearningAuditor.
        
        Args:
            device: PyTorch device (default: cpu)
            d_model: Model dimension (default: 128)
            verbose: Whether to print progress messages (default: True)
        """
        self.device = device or torch.device("cpu")
        self.d_model = d_model
        self.verbose = verbose
    
    def evaluate_node_response(
        self,
        node: KnowledgeNode,
        embedding: torch.Tensor,
        node_id: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Evalúa cómo responde un KnowledgeNode a un embedding.
        
        Args:
            node: KnowledgeNode to evaluate
            embedding: Input embedding tensor
            node_id: Identifier for the node (default: "Unknown")
        
        Returns:
            Dictionary with response metrics:
            - node_id: Node identifier
            - input_shape: Shape of input tensor
            - output_shape: Shape of output tensor
            - similarity: Cosine similarity between input and output
            - memory_size: Size of node's memory
            - refined_output: Refined output tensor
            - feedback: Feedback tensor (if available)
        """
        node.eval()
        with torch.no_grad():
            emb_input = embedding.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Forward pass
            refined_output, feedback = node(emb_input)
            
            # Calculate similarity between input and output
            if node.input_proj:
                input_projected = node.input_proj(emb_input).mean(dim=1)
            else:
                input_projected = emb_input.mean(dim=1)
            
            # Cosine similarity
            input_norm = input_projected / (input_projected.norm(dim=1, keepdim=True) + 1e-8)
            output_norm = refined_output / (refined_output.norm(dim=1, keepdim=True) + 1e-8)
            similarity = (input_norm * output_norm).sum(dim=1).item()
            
            # Memory statistics
            memory_size = len(node.filter.memory) if node.filter else 0
            
            return {
                'node_id': node_id,
                'input_shape': tuple(emb_input.shape),
                'output_shape': tuple(refined_output.shape),
                'similarity': similarity,
                'memory_size': memory_size,
                'refined_output': refined_output.cpu(),
                'feedback': feedback.cpu() if isinstance(feedback, torch.Tensor) else None
            }
    
    def compare_before_after(
        self,
        nodes_dict: Dict[int, KnowledgeNode],
        test_texts: List[str],
        text_preprocessor: TextPreprocessor
    ) -> Dict[str, Any]:
        """
        Compara las respuestas de los nodos antes y después de entrenar con nuevos datos.
        
        Args:
            nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
            test_texts: List of test texts to evaluate
            text_preprocessor: TextPreprocessor instance for encoding texts
        
        Returns:
            Dictionary with comparison results for each text
        """
        if self.verbose:
            print("=" * 70)
            print("== AUDITORÍA DE APRENDIZAJE: Comparación Antes/Después")
            print("=" * 70)
        
        # Convert texts to embeddings
        test_embeddings = text_preprocessor.encode_texts(test_texts)
        
        results = {}
        
        for text, emb in zip(test_texts, test_embeddings):
            if self.verbose:
                print(f"\n== Texto de prueba: '{text[:50]}...'")
                print("-" * 70)
            
            # Find the closest cluster
            best_cluster = None
            best_similarity = -1
            cluster_similarities = {}
            
            for cluster_id, node in nodes_dict.items():
                if node.filter:
                    cluster_id_detected, confidence = node.filter.evaluate(emb.unsqueeze(0))
                    cluster_similarities[cluster_id] = confidence
                    if confidence > best_similarity:
                        best_similarity = confidence
                        best_cluster = cluster_id
            
            if self.verbose:
                print(f"  == Cluster asignado: {best_cluster} (confianza: {best_similarity:.4f})")
                print(f"  == Similitudes por cluster: {cluster_similarities}")
            
            # Evaluate response of assigned node
            if best_cluster in nodes_dict:
                node = nodes_dict[best_cluster]
                response = self.evaluate_node_response(node, emb, f"Cluster_{best_cluster}")
                
                if self.verbose:
                    print(f"  == Respuesta del KnowledgeNode:")
                    print(f"     - Similitud input/output: {response['similarity']:.4f}")
                    print(f"     - Memoria del nodo: {response['memory_size']} embeddings")
                    print(f"     - Output shape: {response['output_shape']}")
                
                results[text] = {
                    'cluster': best_cluster,
                    'confidence': best_similarity,
                    'response': response,
                    'cluster_similarities': cluster_similarities
                }
            else:
                if self.verbose:
                    print(f"  **  No se encontró nodo para el cluster {best_cluster}")
        
        return results
    
    def analyze_cluster_quality(
        self,
        nodes_dict: Dict[int, KnowledgeNode]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analiza la calidad de los clusters y la distribución de la memoria.
        
        This method uses the unified analyze_cluster_quality_per_cluster()
        from cluster_utils to avoid code duplication.
        
        Args:
            nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
        
        Returns:
            Dictionary with stats per cluster:
            {
                cluster_id: {
                    'size': int,
                    'centroid': torch.Tensor,
                    'variance': float,
                    'avg_internal_similarity': float
                }
            }
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("** ANÁLISIS DE CALIDAD DE CLUSTERS")
            print("=" * 70)
        
        # Extract embeddings and labels from nodes using unified function
        embeddings, labels, raw_embeddings_dict = extract_embeddings_from_nodes(nodes_dict)
        
        if len(embeddings) == 0:
            if self.verbose:
                print("  [WARNING]  No hay embeddings para analizar")
            return {}
        
        # Use unified function to analyze cluster quality per cluster
        cluster_stats_np = analyze_cluster_quality_per_cluster(
            embeddings=embeddings,
            labels=labels,
            return_centroids=True,
            return_variance=True,
            return_internal_similarity=True,
            metric='cosine'
        )
        
        # Convert centroids back to torch tensors for backward compatibility
        cluster_stats = {}
        for cluster_id, stats in cluster_stats_np.items():
            cluster_stats[cluster_id] = {
                'size': stats['size'],
                'centroid': torch.from_numpy(stats['centroid']) if 'centroid' in stats and stats['centroid'] is not None else None,
                'variance': stats.get('variance', 0.0),
                'avg_internal_similarity': stats.get('avg_internal_similarity', 0.0)
            }
            
            if self.verbose:
                print(f"\n  Cluster {cluster_id}:")
                print(f"    - Tamaño: {stats['size']} embeddings")
                print(f"    - Varianza promedio: {stats.get('variance', 0.0):.6f}")
                print(f"    - Similitud interna promedio: {stats.get('avg_internal_similarity', 0.0):.4f}")
        
        # Check for clusters in nodes_dict that weren't found in memory
        for cluster_id in nodes_dict.keys():
            if cluster_id not in cluster_stats:
                if self.verbose:
                    print(f"\n  Cluster {cluster_id}: Vacío o no encontrado en memoria")
        
        return cluster_stats
    
    def visualize_embeddings_space(
        self,
        nodes_dict: Dict[int, KnowledgeNode],
        test_texts: List[str],
        text_preprocessor: TextPreprocessor,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualiza los embeddings en el espacio y cómo se agrupan por cluster.
        
        Args:
            nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
            test_texts: List of test texts to visualize
            text_preprocessor: TextPreprocessor instance for encoding texts
            output_path: Path to save visualization (default: auto-generate)
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] VISUALIZACIÓN DEL ESPACIO DE EMBEDDINGS")
            print("=" * 70)
        
        # Get test embeddings
        test_embeddings = text_preprocessor.encode_texts(test_texts)
        
        # Collect all embeddings from memory
        cluster_embeddings = {}
        for cluster_id, node in nodes_dict.items():
            if node.filter and hasattr(node.filter, 'memory'):
                memory = node.filter.memory
                if isinstance(memory, dict):
                    if cluster_id in memory and len(memory[cluster_id]) > 0:
                        cluster_embeddings[cluster_id] = torch.stack(memory[cluster_id]).cpu().numpy()
                elif isinstance(memory, list) and len(memory) > 0:
                    cluster_embeddings[cluster_id] = torch.stack(memory).cpu().numpy()
        
        # Reduce dimensionality for visualization (PCA)
        all_embeddings = []
        labels = []
        
        # Add cluster embeddings
        for cluster_id, embs in cluster_embeddings.items():
            all_embeddings.append(embs)
            labels.extend([f"Cluster_{cluster_id}"] * len(embs))
        
        # Add test embeddings
        all_embeddings.append(test_embeddings.cpu().numpy())
        labels.extend(["Test"] * len(test_texts))
        
        if len(all_embeddings) == 0:
            if self.verbose:
                print("  [WARNING]  No hay embeddings para visualizar")
            return
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Reduce to 2D for visualization
        if all_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(all_embeddings)
            if self.verbose:
                print(f"  [*] PCA: {all_embeddings.shape[1]}D → 2D (varianza explicada: {pca.explained_variance_ratio_.sum():.2%})")
        else:
            embeddings_2d = all_embeddings
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors by cluster
        colors = plt.cm.tab10
        unique_labels = list(set(labels))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            if "Cluster" in label:
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          label=label, alpha=0.6, s=50, c=[colors(i)])
            else:
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          label=label, alpha=1.0, s=100, marker='*', c='red', edgecolors='black')
        
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_title('Visualización de Embeddings por Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            # Auto-generate path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
            output_path = os.path.join(project_root, "notebooks", "embeddings_visualization.png")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"  [*] Visualización guardada en: {output_path}")
        
        plt.close()
    
    def test_learning_effectiveness(
        self,
        nodes_dict: Dict[int, KnowledgeNode],
        test_texts: List[str],
        text_preprocessor: TextPreprocessor,
        filter_node: FilterBayesianNode,
        num_iterations: int = 3
    ) -> Dict[int, Dict[str, Any]]:
        """
        Prueba la efectividad del aprendizaje comparando outputs en múltiples iteraciones.
        
        Args:
            nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
            test_texts: List of test texts to evaluate
            text_preprocessor: TextPreprocessor instance for encoding texts
            filter_node: FilterBayesianNode for cluster assignment
            num_iterations: Number of training iterations to simulate (default: 3)
        
        Returns:
            Dictionary with learning metrics for each test text
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("🧪 TEST DE EFECTIVIDAD DEL APRENDIZAJE")
            print("=" * 70)
        
        test_embeddings = text_preprocessor.encode_texts(test_texts)
        
        # Store initial outputs
        initial_outputs = {}
        
        if self.verbose:
            print("\n[*] Iteración 0 (Estado inicial):")
        
        for i, (text, emb) in enumerate(zip(test_texts, test_embeddings)):
            cluster_id, _ = filter_node.evaluate(emb.unsqueeze(0))
            if cluster_id in nodes_dict:
                node = nodes_dict[cluster_id]
                response = self.evaluate_node_response(node, emb, f"Cluster_{cluster_id}")
                initial_outputs[i] = {
                    'output': response['refined_output'].clone(),
                    'similarity': response['similarity'],
                    'cluster': cluster_id
                }
                if self.verbose:
                    print(f"  Texto {i+1}: Cluster {cluster_id}, Similitud: {response['similarity']:.4f}")
        
        # Simulate additional training
        if self.verbose:
            print(f"\n[*] Simulando {num_iterations} iteraciones de entrenamiento...")
        
        for iteration in range(num_iterations):
            for emb in test_embeddings:
                cluster_id, _ = filter_node.evaluate(emb.unsqueeze(0))
                if cluster_id in nodes_dict:
                    node = nodes_dict[cluster_id]
                    # Forward pass (simulates training)
                    emb_input = emb.unsqueeze(0).unsqueeze(0).to(self.device)
                    node(emb_input)
        
        # Compare final outputs
        if self.verbose:
            print("\n[*] Estado después del entrenamiento:")
        
        learning_metrics = {}
        
        for i, (text, emb) in enumerate(zip(test_texts, test_embeddings)):
            cluster_id, _ = filter_node.evaluate(emb.unsqueeze(0))
            if cluster_id in nodes_dict and i in initial_outputs:
                node = nodes_dict[cluster_id]
                response = self.evaluate_node_response(node, emb, f"Cluster_{cluster_id}")
                final_output = response['refined_output']
                initial_output = initial_outputs[i]['output']
                
                # Calculate change in output
                output_change = torch.nn.functional.cosine_similarity(
                    initial_output, final_output
                ).item()
                
                # Change in similarity input/output
                similarity_change = response['similarity'] - initial_outputs[i]['similarity']
                
                learning_metrics[i] = {
                    'output_change': output_change,
                    'similarity_change': similarity_change,
                    'initial_similarity': initial_outputs[i]['similarity'],
                    'final_similarity': response['similarity']
                }
                
                if self.verbose:
                    print(f"  Texto {i+1}:")
                    print(f"    - Cambio en output (cosine sim): {output_change:.4f}")
                    print(f"    - Cambio en similitud: {similarity_change:+.4f}")
                    print(f"    - Similitud inicial → final: {initial_outputs[i]['similarity']:.4f} → {response['similarity']:.4f}")
        
        return learning_metrics
    
    def generate_audit_report(
        self,
        nodes_dict: Dict[int, KnowledgeNode],
        test_texts: List[str],
        text_preprocessor: TextPreprocessor,
        filter_node: FilterBayesianNode,
        include_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Genera un reporte completo de auditoría.
        
        Args:
            nodes_dict: Dictionary mapping cluster_id to KnowledgeNode
            test_texts: List of test texts to evaluate
            text_preprocessor: TextPreprocessor instance for encoding texts
            filter_node: FilterBayesianNode for cluster assignment
            include_visualization: Whether to include visualization (default: True)
        
        Returns:
            Dictionary with complete audit report:
            {
                'cluster_stats': Dict with cluster quality stats,
                'comparison_results': Dict with before/after comparison,
                'learning_metrics': Dict with learning effectiveness metrics
            }
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] REPORTE DE AUDITORÍA COMPLETO")
            print("=" * 70)
        
        # 1. Cluster analysis
        cluster_stats = self.analyze_cluster_quality(nodes_dict)
        
        # 2. Before/after comparison
        comparison_results = self.compare_before_after(nodes_dict, test_texts, text_preprocessor)
        
        # 3. Learning effectiveness test
        learning_metrics = self.test_learning_effectiveness(
            nodes_dict, test_texts, text_preprocessor, filter_node
        )
        
        # 4. Visualization
        if include_visualization:
            self.visualize_embeddings_space(nodes_dict, test_texts, text_preprocessor)
        
        # 5. Summary
        if self.verbose:
            print("\n" + "=" * 70)
            print("[*] RESUMEN DE AUDITORÍA")
            print("=" * 70)
            
            print(f"\n[OK] KnowledgeNodes activos: {len(nodes_dict)}")
            print(f"[OK] Textos evaluados: {len(test_texts)}")
            
            if learning_metrics:
                avg_similarity_change = np.mean([m['similarity_change'] for m in learning_metrics.values()])
                print(f"[OK] Cambio promedio en similitud: {avg_similarity_change:+.4f}")
                
                if avg_similarity_change > 0:
                    print("  [*] El aprendizaje está mejorando la capacidad de reconstrucción")
                else:
                    print("  [WARNING]  El aprendizaje no muestra mejora clara (puede necesitar más datos)")
            
            if cluster_stats:
                avg_internal_sim = np.mean([s['avg_internal_similarity'] for s in cluster_stats.values()])
                print(f"[OK] Similitud interna promedio de clusters: {avg_internal_sim:.4f}")
                
                if avg_internal_sim > 0.7:
                    print("  [*] Los clusters son cohesivos (buena agrupación)")
                elif avg_internal_sim > 0.5:
                    print("  [WARNING]  Los clusters tienen cohesión moderada")
                else:
                    print("  [WARNING]  Los clusters pueden necesitar ajuste de umbral")
        
        return {
            'cluster_stats': cluster_stats,
            'comparison_results': comparison_results,
            'learning_metrics': learning_metrics
        }

