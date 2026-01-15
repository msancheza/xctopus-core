"""
Main for Clustering Layer.

System entry point: loads dataset, initializes components, and processes embeddings.
Integrates all previous phases into a complete flow.
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
    # Try tqdm.notebook first (better for Jupyter), fallback to regular tqdm
    try:
        from tqdm.notebook import tqdm
        TQDM_AVAILABLE = True
        TQDM_NOTEBOOK = True
    except ImportError:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
        TQDM_NOTEBOOK = False
except ImportError:
    TQDM_AVAILABLE = False
    TQDM_NOTEBOOK = False

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
    - CSV: file with comma-separated embeddings
    - NPY: numpy file (.npy)
    - NPZ: compressed numpy file (.npz)
    
    Args:
        dataset_path: Path to dataset file
    
    Returns:
        List of FP16 tensors on configured DEVICE
    
    Raises:
        FileNotFoundError: If file exists
        ValueError: If format is unsupervised or embeddings are invalid
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
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


def initialize_components(
    dataset_paths: Optional[dict] = None
) -> Tuple[KNRepository, FilterBayesian, Orchestrator]:
    """
    Initialize all system components.
    
    OPTIMIZED: Loads signatures once at startup (warmup).
    This avoids refreshing on each iteration of the main loop.
    
    Phase 2: Accepts dataset_paths for DataManager initialization.
    
    Args:
        dataset_paths: Optional dictionary mapping dataset names to file paths
            Example: {
                'arxiv': '/path/to/arxiv_data.csv',
                '20newsgroups': '/path/to/20newsgroups.csv'
            }
            If None, DataManager will be initialized but won't have datasets
    
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
    # Orchestrator will handle immediate signature updates after each embedding is accepted.
    signatures = repository.get_all_signatures()
    filter_bayesian.refresh_signatures(signatures)
    logger.debug(f"FilterBayesian initialized with {len(signatures)} signatures (warmup)")
    
    # Create Orchestrator (with Repository and FilterBayesian as dependencies)
    # Phase 2: Pass dataset_paths for DataManager
    orchestrator = Orchestrator(repository, filter_bayesian, dataset_paths=dataset_paths)
    logger.debug("Orchestrator initialized")
    
    if dataset_paths:
        logger.debug(f"DataManager initialized with {len(dataset_paths)} dataset paths")
    
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
    Processes the complete dataset of embeddings.
    
    Flow for each embedding:
    1. Routing: FilterBayesian decides route (uses in-memory signatures)
    2. Execution: Orchestrator executes decision
       - Immediate signature updates after each embedding is accepted
    3. Batch commits: Periodic commit
    
    OPTIMIZED: Signatures are loaded once at startup (warmup).
    Orchestrator handles intelligent refreshes, avoiding 18,233 SQL queries.
    
    Args:
        embeddings: List of embeddings to process
        repository: KNRepository instance
        filter_bayesian: FilterBayesian instance
        orchestrator: Orchestrator instance
        progress_interval: How many embeddings to show progress (default: 100)
    """
    total_embeddings = len(embeddings)
    logger.info(f"Starting processing of {total_embeddings} embeddings")
    
    # Detect if we are in a notebook
    in_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    
    # Choose progress method based on environment
    if in_notebook and TQDM_AVAILABLE:
        # In notebooks: use tqdm (compatible with Jupyter, no duplication)
        # Use file=sys.stdout to avoid stderr capture by logger
        tqdm_kwargs = {
            "total": total_embeddings,
            "desc": "Processing embeddings",
            "unit": "embedding",
            "mininterval": 0.5,  # Update at most every 0.5 seconds to reduce spam
            "maxinterval": 1.0   # Force update every 1 second max
        }
        # For tqdm.notebook, don't set ncols (auto-adjusts to notebook width)
        # For regular tqdm, use None to auto-adjust or a reasonable width
        if not TQDM_NOTEBOOK:
            tqdm_kwargs["file"] = sys.stdout
            tqdm_kwargs["ncols"] = None  # Auto-adjust width
        else:
            # tqdm.notebook handles width automatically
            tqdm_kwargs["ncols"] = None
        
        embeddings_iter = tqdm(enumerate(embeddings), **tqdm_kwargs)
        
        for i, embedding in embeddings_iter:
            _process_single_embedding(
                embedding, repository, filter_bayesian, orchestrator, i
            )
            
            # Update postfix with KN and Buffer counts
            counts = orchestrator.get_counts()
            postfix_dict = {
                "KNs": counts["kn_count"],
                "Buffers": counts["buffer_count"]
            }
            # Add "T" if training is active (simple indicator, no RAM overhead)
            if orchestrator.has_active_training():
                postfix_dict["T"] = "T"
            embeddings_iter.set_postfix(postfix_dict)
            
            # Periodic logging (DEBUG level to avoid console spam)
            if (i + 1) % progress_interval == 0:
                logger.debug(
                    f"Processed {i + 1}/{total_embeddings} embeddings "
                    f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                    f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                )
    elif RICH_AVAILABLE and console and not in_notebook:
        # Outside notebooks: use rich Progress (more elegant)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing embeddings...",
                total=total_embeddings
            )
            
            for i, embedding in enumerate(embeddings):
                _process_single_embedding(
                    embedding, repository, filter_bayesian, orchestrator, i
                )
                
                # Update progress with KN and Buffer info
                counts = orchestrator.get_counts()
                training_indicator = " T" if orchestrator.has_active_training() else ""
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Processing embeddings... [dim]KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}{training_indicator}[/dim]"
                )
                
                # Periodic logging (DEBUG level to avoid console spam)
                if (i + 1) % progress_interval == 0:
                    logger.debug(
                        f"Processed {i + 1}/{total_embeddings} embeddings "
                        f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                        f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                    )
    else:
        # Fallback: use simple logging
        for i, embedding in enumerate(embeddings):
            _process_single_embedding(
                embedding, repository, filter_bayesian, orchestrator, i
            )
            
            # Periodic logging with counts (DEBUG level to avoid console spam)
            if (i + 1) % progress_interval == 0:
                counts = orchestrator.get_counts()
                logger.debug(
                    f"Processed {i + 1}/{total_embeddings} embeddings "
                    f"({(i + 1) / total_embeddings * 100:.1f}%) | "
                    f"KNs: {counts['kn_count']}, Buffers: {counts['buffer_count']}"
                )
    
    logger.info(f"Processing completed: {total_embeddings} embeddings processed")


def _process_single_embedding(
    embedding: torch.Tensor,
    repository: KNRepository,
    filter_bayesian: FilterBayesian,
    orchestrator: Orchestrator,
    index: int,
    source_id: Optional[str] = None,
) -> None:
    """
    Process a single embedding.
    
    OPTIMIZED: Doesn't refresh signatures on each iteration.
    - Signatures are loaded once in initialize_components() (warmup)
    - Orchestrator handles immediate signature updates after each embedding is accepted
    - This avoids 18,233 unnecessary SQL queries
    
    Phase 2: Accepts source_id for data provenance (pointer to original dataset).
    
    Args:
        embedding: Embedding tensor
        repository: KNRepository instance
        filter_bayesian: FilterBayesian instance
        orchestrator: Orchestrator instance
        index: Embedding index (for batch commits and source_id generation)
        source_id: Optional source ID (pointer to original dataset)
            If None, uses str(index) as source_id
    """
    try:
        # ========================================================================
        # Phase 2: Generate source_id if not provided
        # ========================================================================
        # Use index as source_id for data provenance
        # This allows DataManager to retrieve original texts for training
        actual_source_id = source_id if source_id is not None else str(index)
        
        # ========================================================================
        # Reactive Flow: Orchestrator handles Route -> Act -> Judge -> Learn
        # ========================================================================
        # Phase 2: Pass source_id for data provenance
        # This replaces separate route() and process_decision() calls
        orchestrator.process_embedding(embedding, source_id=actual_source_id)
        
        # ========================================================================
        # Batch Commits: Periodic commit (handled internally by Repository)
        # ========================================================================
        # Repository handles batch commits internally with _maybe_commit()
        # We don't need to manually commit here
    except Exception as e:
        logger.error(f"Error processing embedding at index {index}: {e}", exc_info=True)
        # Continue processing next embedding instead of stopping entire process


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
        _show_summary_rich(signatures, active_nodes_count, repository)
    else:
        _show_summary_simple(signatures, active_nodes_count, repository)
    
    logger.info("Processing finalized correctly")


def _show_summary_rich(signatures: List[dict], active_nodes_count: int, repository: KNRepository) -> None:
    """
    Show final summary using rich (well-formatted tables).
    
    Args:
        signatures: List of KnowledgeNode signatures
        active_nodes_count: Number of active nodes
        repository: KNRepository instance to check training status
    """
    console.print("\n[bold cyan]📊 Processing Summary[/bold cyan]\n")
    
    # Calculate training statistics
    trained_nodes = sum(1 for sig in signatures if repository.is_trained(sig["node_id"]))
    candidates_potential = sum(1 for sig in signatures if sig["mass"] > 5)
    
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
    stats_table.add_row("Average Mass", f"{avg_mass:.1f}")
    stats_table.add_row("Average Variance", f"{avg_variance:.4f}")
    
    console.print(stats_table)
    
    # Training statistics table
    training_table = Table(title="Training Status", show_header=True, header_style="bold yellow")
    training_table.add_column("Métrica", style="cyan")
    training_table.add_column("Valor", style="green", justify="right")
    
    training_table.add_row("Nodos Entrenados", str(trained_nodes))
    training_table.add_row("Candidatos Potenciales (masa > 5)", str(candidates_potential))
    
    console.print("\n")
    console.print(training_table)
    
    # Tabla de KnowledgeNodes (top 10 por masa)
    if signatures:
        console.print("\n[bold cyan]🔝 Top 10 KnowledgeNodes by Mass[/bold cyan]\n")
        
        kn_table = Table(show_header=True, header_style="bold magenta")
        kn_table.add_column("Node ID", style="cyan", width=40)
        kn_table.add_column("Mass", style="green", justify="right")
        kn_table.add_column("Variance", style="yellow", justify="right")
        
        # Sort by mass (descending)
        sorted_signatures = sorted(signatures, key=lambda x: x["mass"], reverse=True)
        
        for sig in sorted_signatures[:10]:
            kn_table.add_row(
                sig["node_id"][:37] + "..." if len(sig["node_id"]) > 40 else sig["node_id"],
                str(sig["mass"]),
                f"{sig['variance']:.4f}"
            )
        
        console.print(kn_table)
    
    console.print()


def _show_summary_simple(signatures: List[dict], active_nodes_count: int, repository: KNRepository) -> None:
    """
    Shows final summary in simple format (without rich).
    
    Args:
        signatures: List of KnowledgeNode signatures
        active_nodes_count: Number of active nodes
        repository: KNRepository instance to check training status
    """
    logger.info("=" * 60)
    
    # Calculate training statistics
    trained_nodes = sum(1 for sig in signatures if repository.is_trained(sig["node_id"]))
    candidates_potential = sum(1 for sig in signatures if sig["mass"] > 5)
    
    logger.info("📊 Análisis de Fin de Jornada:")
    logger.info(f"   Nodos entrenados: {trained_nodes}")
    logger.info(f"   Candidatos potenciales con masa > 5: {candidates_potential}")
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    
    total_mass = sum(sig["mass"] for sig in signatures)
    avg_mass = total_mass / len(signatures) if signatures else 0
    avg_variance = sum(sig["variance"] for sig in signatures) / len(signatures) if signatures else 0
    
    logger.info(f"Total KnowledgeNodes: {len(signatures)}")
    logger.info(f"Active Nodes: {active_nodes_count}")
    logger.info(f"Total Embeddings: {total_mass}")
    logger.info(f"Average Mass: {avg_mass:.1f}")
    logger.info(f"Average Variance: {avg_variance:.4f}")
    
    if signatures:
        logger.info("\nTop 10 KnowledgeNodes by Mass:")
        sorted_signatures = sorted(signatures, key=lambda x: x["mass"], reverse=True)
        for i, sig in enumerate(sorted_signatures[:10], 1):
            logger.info(
                f"  {i}. {sig['node_id'][:50]}: "
                f"mass={sig['mass']}, variance={sig['variance']:.4f}"
            )
    
    logger.info("=" * 60)


def main(dataset_path: str) -> None:
    """
    Main function: system entry point.
    
    Complete flow:
    1. Load embeddings from dataset
    2. Initialize components
    3. Process dataset (Layer 1: Clustering)
    4. Fuse Knowledge Nodes (post-clustering)
    5. Run deferred training (Layer 2: Specialization)
    6. Finalize and show summary
    
    Args:
        dataset_path: Path to dataset file
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
        logger.info("Starting Knowledge Nodes fusion process...")
        fusion_stats = fuse_knowledge_nodes(repository, orchestrator)
        logger.info(
            f"Fusion completed: {fusion_stats['fusions_performed']} fusions performed, "
            f"{fusion_stats['initial_kns']} -> {fusion_stats['final_kns']} KNs"
        )
        
        # 5. Run deferred training (Layer 2: Specialization)
        logger.info("=" * 60)
        logger.info("Starting deferred training phase...")
        logger.info("=" * 60)
        training_stats = orchestrator.run_deferred_training()
        logger.info(
            f"Training completed: {training_stats['trained']} nodes trained, "
            f"{training_stats['failed']} failed (out of {training_stats['total_pending']} pending)"
        )
        
        # 6. Finalize
        finalize(repository, orchestrator)
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Entry point when run as script
    if len(sys.argv) < 2:
        # Use sys.stderr.write for error messages in console
        # (do not use print to comply with Rule 2)
        sys.stderr.write("Error: Dataset path required\n")
        sys.stderr.write("Usage: python -m xctopus.main <dataset_path>\n")
        sys.stderr.write("Example: python -m xctopus.main data/embeddings.csv\n")
        logger.error("Usage: python -m xctopus.main <dataset_path>")
        logger.error("Example: python -m xctopus.main data/embeddings.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    main(dataset_path)
