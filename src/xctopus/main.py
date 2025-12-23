"""
Main para Capa Clustering.

Punto de entrada del sistema: carga dataset, inicializa componentes y procesa embeddings.
Integra todas las fases anteriores en un flujo completo.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("rich is not installed. Tables will be displayed in simple format.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .repository import KNRepository
from .filter_bayesian import FilterBayesian
from .orchestrator import Orchestrator
from .fusion import fuse_knowledge_nodes
from .settings import (
    DB_PATH,
    SAVE_BATCH_SIZE,
    DEVICE,
    DTYPE,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)
console = Console() if RICH_AVAILABLE else None


def load_embeddings(dataset_path: str) -> List[torch.Tensor]:
    """
    Load embeddings from dataset.
    
    Supports formats:
    - CSV: archivo con embeddings separados por comas
    - NPY: archivo numpy (.npy)
    - NPZ: archivo numpy comprimido (.npz)
    
    Args:
        dataset_path: Ruta al archivo del dataset
    
    Returns:
        Lista de tensores FP16 en DEVICE configurado
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato no es soportado o embeddings son inválidos
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    
    logger.info(f"Loading embeddings from: {dataset_path}")
    
    # Detect format by extension
    extension = dataset_path.suffix.lower()
    
    try:
        if extension == ".csv":
            # CSV: assume each row is an embedding
            data = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        elif extension == ".npy":
            # NPY: numpy array
            data = np.load(dataset_path).astype(np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        elif extension == ".npz":
            # NPZ: search for 'embeddings' key or first available key
            npz_file = np.load(dataset_path)
            keys = list(npz_file.keys())
            if "embeddings" in keys:
                data = npz_file["embeddings"].astype(np.float32)
            elif len(keys) > 0:
                data = npz_file[keys[0]].astype(np.float32)
            else:
                raise ValueError("NPZ file does not contain valid data")
            if data.ndim == 1:
                data = data.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported format: {extension}. Use .csv, .npy or .npz")
        
        # Validate dimensions
        if data.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (N, {EMBEDDING_DIM}), received shape: {data.shape}")
        
        num_embeddings, dim = data.shape
        
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"Incorrect embedding dimension: expected {EMBEDDING_DIM}, "
                f"received {dim}"
            )
        
        # Convert to FP16 tensors on DEVICE
        embeddings = []
        for i in range(num_embeddings):
            tensor = torch.from_numpy(data[i]).to(device=DEVICE, dtype=DTYPE)
            embeddings.append(tensor)
        
        logger.info(f"Embeddings loaded: {num_embeddings} embeddings of dimension {dim}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise


def initialize_components() -> Tuple[KNRepository, FilterBayesian, Orchestrator]:
    """
    Initialize all system components.
    
    OPTIMIZED: Loads signatures once at startup (warmup).
    This avoids refreshing on each iteration of the main loop.
    
    Returns:
        Tuple with (Repository, FilterBayesian, Orchestrator)
    """
    logger.info("Initializing components...")
    
    # Create Repository
    repository = KNRepository()
    logger.debug("Repository initialized")
    
    # Create FilterBayesian
    filter_bayesian = FilterBayesian()
    logger.debug("FilterBayesian initialized")
    
    # ========================================================================
    # Warmup: Load existing signatures once
    # ========================================================================
    # This avoids refreshing on each iteration of the main loop.
    # Orchestrator will handle intelligent refreshes (every REFRESH_INTERVAL).
    signatures = repository.get_all_signatures()
    filter_bayesian.refresh_signatures(signatures)
    logger.debug(f"FilterBayesian initialized with {len(signatures)} signatures (warmup)")
    
    # Create Orchestrator (with Repository and FilterBayesian as dependencies)
    orchestrator = Orchestrator(repository, filter_bayesian)
    logger.debug("Orchestrator initialized")
    
    logger.info("Components initialized correctly")
    
    return repository, filter_bayesian, orchestrator


def process_dataset(
    embeddings: List[torch.Tensor],
    repository: KNRepository,
    filter_bayesian: FilterBayesian,
    orchestrator: Orchestrator,
    progress_interval: int = 100,
) -> None:
    """
    Procesa el dataset completo de embeddings.
    
    Flujo para cada embedding:
    1. Ruteo: FilterBayesian decide ruteo (usa firmas en memoria)
    2. Ejecución: Orchestrator ejecuta decisión
       - Refrescos inteligentes cada REFRESH_INTERVAL
    3. Batch commits: Commit periódico
    
    OPTIMIZADO: Las firmas se cargan una vez al inicio (warmup).
    El Orchestrator maneja refrescos inteligentes, evitando 18,233 consultas SQL.
    
    Args:
        embeddings: Lista de embeddings a procesar
        repository: Instancia de KNRepository
        filter_bayesian: Instancia de FilterBayesian
        orchestrator: Instancia de Orchestrator
        progress_interval: Cada cuántos embeddings mostrar progreso (default: 100)
    """
    total_embeddings = len(embeddings)
    logger.info(f"Iniciando procesamiento de {total_embeddings} embeddings")
    
    # Detectar si estamos en un notebook
    in_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    
    # Elegir método de progreso según el entorno
    if in_notebook and TQDM_AVAILABLE:
        # En notebooks: usar tqdm (compatible con Jupyter, no duplica)
        embeddings_iter = tqdm(
            enumerate(embeddings),
            total=total_embeddings,
            desc="Procesando embeddings",
            unit="embedding",
            ncols=100  # Ancho fijo para mejor visualización en notebooks
        )
        
        for i, embedding in embeddings_iter:
            _process_single_embedding(
                embedding, repository, filter_bayesian, orchestrator, i
            )
            
            # Actualizar postfix con contadores de KNs y Buffers
            counts = orchestrator.get_counts()
            embeddings_iter.set_postfix({
                "KNs": counts["kn_count"],
                "Buffers": counts["buffer_count"]
            })
            
            # Logging periódico
            if (i + 1) % progress_interval == 0:
                logger.info(
                    f"Procesados {i + 1}/{total_embeddings} embeddings "
                    f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                    f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                )
    elif RICH_AVAILABLE and console and not in_notebook:
        # Fuera de notebooks: usar rich Progress (más elegante)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Procesando embeddings...",
                total=total_embeddings
            )
            
            for i, embedding in enumerate(embeddings):
                _process_single_embedding(
                    embedding, repository, filter_bayesian, orchestrator, i
                )
                
                # Actualizar progreso con información de KNs y Buffers
                counts = orchestrator.get_counts()
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Procesando embeddings... [dim]KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}[/dim]"
                )
                
                # Logging periódico
                if (i + 1) % progress_interval == 0:
                    logger.info(
                        f"Procesados {i + 1}/{total_embeddings} embeddings "
                        f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                        f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                    )
    else:
        # Fallback: usar logging simple
        for i, embedding in enumerate(embeddings):
            _process_single_embedding(
                embedding, repository, filter_bayesian, orchestrator, i
            )
            
            # Logging periódico con contadores
            if (i + 1) % progress_interval == 0:
                counts = orchestrator.get_counts()
                logger.info(
                    f"Procesados {i + 1}/{total_embeddings} embeddings "
                    f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                    f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                )
    
    logger.info(f"Procesamiento completado: {total_embeddings} embeddings procesados")


def _process_single_embedding(
    embedding: torch.Tensor,
    repository: KNRepository,
    filter_bayesian: FilterBayesian,
    orchestrator: Orchestrator,
    index: int,
) -> None:
    """
    Process a single embedding.
    
    OPTIMIZED: Doesn't refresh signatures on each iteration.
    - Signatures are loaded once in initialize_components() (warmup)
    - Orchestrator handles intelligent refreshes (every REFRESH_INTERVAL)
    - This avoids 18,233 unnecessary SQL queries
    
    Args:
        embedding: Embedding tensor
        repository: KNRepository instance
        filter_bayesian: FilterBayesian instance
        orchestrator: Orchestrator instance
        index: Embedding index (for batch commits)
    """
    # ========================================================================
    # Routing: FilterBayesian decides routing (uses signatures in memory)
    # ========================================================================
    # Note: Signatures are loaded once in initialize_components() (warmup)
    # and are automatically refreshed by Orchestrator when REFRESH_INTERVAL is reached.
    # This allows 99% of iterations to occur in GPU/RAM without touching disk.
    decision = filter_bayesian.route(embedding)
    
    # ========================================================================
    # Execution: Orchestrator executes decision
    # ========================================================================
    # Orchestrator handles:
    # - Intelligent refreshes (every REFRESH_INTERVAL embeddings)
    # - Similar buffer grouping
    # - Signature updates in Repository
    orchestrator.process_decision(decision, embedding)
    
    # ========================================================================
    # Batch Commits: Periodic commit (handled internally by Repository)
    # ========================================================================
    # Repository handles batch commits internally with _maybe_commit()
    # We don't need to manually commit here


def finalize(
    repository: KNRepository,
    orchestrator: Orchestrator,
) -> None:
    """
    Finalize processing: final commit, close and summary.
    
    Args:
        repository: KNRepository instance
        orchestrator: Orchestrator instance
    """
    logger.info("Finalizing processing...")
    
    # Final commit in Repository
    repository.close()
    logger.debug("Final commit performed in Repository")
    
    # Get final statistics
    signatures = repository.get_all_signatures()
    active_nodes_count = orchestrator.get_active_node_count()
    
    # Show summary with rich
    if RICH_AVAILABLE and console:
        _show_summary_rich(signatures, active_nodes_count)
    else:
        _show_summary_simple(signatures, active_nodes_count)
    
    logger.info("Processing finalized correctly")


def _show_summary_rich(signatures: List[dict], active_nodes_count: int) -> None:
    """
    Show final summary using rich (well-formatted tables).
    
    Args:
        signatures: List of KnowledgeNode signatures
        active_nodes_count: Number of active nodes
    """
    console.print("\n[bold cyan]📊 Processing Summary[/bold cyan]\n")
    
    # General statistics table
    stats_table = Table(title="General Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Métrica", style="cyan")
    stats_table.add_column("Valor", style="green", justify="right")
    
    total_mass = sum(sig["mass"] for sig in signatures)
    avg_mass = total_mass / len(signatures) if signatures else 0
    avg_variance = sum(sig["variance"] for sig in signatures) / len(signatures) if signatures else 0
    
    stats_table.add_row("Total KnowledgeNodes", str(len(signatures)))
    stats_table.add_row("Nodos Activos", str(active_nodes_count))
    stats_table.add_row("Total Embeddings", str(total_mass))
    stats_table.add_row("Masa Promedio", f"{avg_mass:.1f}")
    stats_table.add_row("Varianza Promedio", f"{avg_variance:.4f}")
    
    console.print(stats_table)
    
    # Tabla de KnowledgeNodes (top 10 por masa)
    if signatures:
        console.print("\n[bold cyan]🔝 Top 10 KnowledgeNodes por Masa[/bold cyan]\n")
        
        kn_table = Table(show_header=True, header_style="bold magenta")
        kn_table.add_column("Node ID", style="cyan", width=40)
        kn_table.add_column("Masa", style="green", justify="right")
        kn_table.add_column("Varianza", style="yellow", justify="right")
        
        # Ordenar por masa (descendente)
        sorted_signatures = sorted(signatures, key=lambda x: x["mass"], reverse=True)
        
        for sig in sorted_signatures[:10]:
            kn_table.add_row(
                sig["node_id"][:37] + "..." if len(sig["node_id"]) > 40 else sig["node_id"],
                str(sig["mass"]),
                f"{sig['variance']:.4f}"
            )
        
        console.print(kn_table)
    
    console.print()


def _show_summary_simple(signatures: List[dict], active_nodes_count: int) -> None:
    """
    Muestra resumen final en formato simple (sin rich).
    
    Args:
        signatures: Lista de firmas de KnowledgeNodes
        active_nodes_count: Cantidad de nodos activos
    """
    logger.info("=" * 60)
    logger.info("RESUMEN DEL PROCESAMIENTO")
    logger.info("=" * 60)
    
    total_mass = sum(sig["mass"] for sig in signatures)
    avg_mass = total_mass / len(signatures) if signatures else 0
    avg_variance = sum(sig["variance"] for sig in signatures) / len(signatures) if signatures else 0
    
    logger.info(f"Total KnowledgeNodes: {len(signatures)}")
    logger.info(f"Nodos Activos: {active_nodes_count}")
    logger.info(f"Total Embeddings: {total_mass}")
    logger.info(f"Masa Promedio: {avg_mass:.1f}")
    logger.info(f"Varianza Promedio: {avg_variance:.4f}")
    
    if signatures:
        logger.info("\nTop 10 KnowledgeNodes por Masa:")
        sorted_signatures = sorted(signatures, key=lambda x: x["mass"], reverse=True)
        for i, sig in enumerate(sorted_signatures[:10], 1):
            logger.info(
                f"  {i}. {sig['node_id'][:50]}: "
                f"masa={sig['mass']}, varianza={sig['variance']:.4f}"
            )
    
    logger.info("=" * 60)


def main(dataset_path: str) -> None:
    """
    Función principal: punto de entrada del sistema.
    
    Flujo completo:
    1. Cargar embeddings del dataset
    2. Inicializar componentes
    3. Procesar dataset
    4. Finalizar y mostrar resumen
    
    Args:
        dataset_path: Ruta al archivo del dataset
    """
    try:
        logger.info("=" * 60)
        logger.info("XCTOPUS - Capa Clustering")
        logger.info("=" * 60)
        
        # 1. Cargar embeddings
        embeddings = load_embeddings(dataset_path)
        
        # 2. Inicializar componentes
        repository, filter_bayesian, orchestrator = initialize_components()
        
        # 3. Procesar dataset
        process_dataset(embeddings, repository, filter_bayesian, orchestrator)
        
        # 4. Fusionar Knowledge Nodes (post-clustering)
        logger.info("Iniciando proceso de fusión de Knowledge Nodes...")
        fusion_stats = fuse_knowledge_nodes(repository, orchestrator)
        logger.info(
            f"Fusión completada: {fusion_stats['fusions_performed']} fusiones realizadas, "
            f"{fusion_stats['initial_kns']} -> {fusion_stats['final_kns']} KNs"
        )
        
        # 5. Finalizar
        finalize(repository, orchestrator)
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en procesamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Punto de entrada cuando se ejecuta como script
    if len(sys.argv) < 2:
        # Usar sys.stderr.write para mensajes de error en consola
        # (no usar print para cumplir Regla 2)
        sys.stderr.write("Error: Se requiere ruta al dataset\n")
        sys.stderr.write("Uso: python -m xctopus.main <dataset_path>\n")
        sys.stderr.write("Ejemplo: python -m xctopus.main data/embeddings.csv\n")
        logger.error("Uso: python -m xctopus.main <dataset_path>")
        logger.error("Ejemplo: python -m xctopus.main data/embeddings.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    main(dataset_path)
