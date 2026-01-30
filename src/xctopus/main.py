"""
Main for Clustering Layer.

System entry point: loads dataset, initializes components, and processes embeddings.
Integrates all previous phases into a complete flow.
"""

import sys
from pathlib import Path

# Ensure parent package is in path for relative imports
# This handles cases where the module is executed directly or imported
_current_file = Path(__file__).resolve()
_parent_dir = _current_file.parent  # xctopus/src/xctopus/
_grandparent_dir = _parent_dir.parent  # xctopus/src/

# Check if we're being executed directly (not as a module)
_is_main_module = __name__ == "__main__" or not __package__

# Add parent directories to sys.path if not already there
# This allows both relative and absolute imports to work
if _is_main_module:
    # When executed directly, add directories to path
    if str(_grandparent_dir) not in sys.path:
        sys.path.insert(0, str(_grandparent_dir))
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional

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

# Handle both relative and absolute imports
try:
    # Try relative imports first (when used as a module)
    from .repository import KNRepository
    from .filter_bayesian import FilterBayesian
    from .orchestrator import Orchestrator
    from .fusion import fuse_knowledge_nodes
    from .settings import (
        DB_PATH,
        SAVE_BATCH_SIZE,
        DEVICE,
        DTYPE,
        PROCESS_BATCH_SIZE,
        EMBEDDING_DIM,
    )
except ImportError:
    # Fallback to absolute imports (when executed directly or package not found)
    try:
        from xctopus.repository import KNRepository
        from xctopus.filter_bayesian import FilterBayesian
        from xctopus.orchestrator import Orchestrator
        from xctopus.fusion import fuse_knowledge_nodes
        from xctopus.settings import (
            DB_PATH,
            SAVE_BATCH_SIZE,
            DEVICE,
            DTYPE,
            PROCESS_BATCH_SIZE,
            EMBEDDING_DIM,
        )
    except ImportError:
        # Last resort: try direct imports from same directory
        import sys
        from pathlib import Path
        # Add parent directory to path if not already there
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from repository import KNRepository
        from filter_bayesian import FilterBayesian
        from orchestrator import Orchestrator
        from fusion import fuse_knowledge_nodes
        from settings import (
            DB_PATH,
            SAVE_BATCH_SIZE,
            DEVICE,
            DTYPE,
            PROCESS_BATCH_SIZE,
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
            # CSV: handle headers and different formats
            # Try pandas first (more robust for CSV with headers)
            try:
                import pandas as pd
                df = pd.read_csv(dataset_path)
                # Check if first row contains non-numeric data (likely header)
                # If so, skip it and use the rest
                if df.shape[1] > 0:
                    # Try to convert to numeric, if fails, first row is header
                    try:
                        df_numeric = df.apply(pd.to_numeric, errors='coerce')
                        # If first row has NaN values, it's likely a header
                        if df_numeric.iloc[0].isna().any():
                            logger.info("Detected CSV header, skipping first row")
                            df = df.iloc[1:].reset_index(drop=True)
                            df_numeric = df.apply(pd.to_numeric, errors='coerce')
                        data = df_numeric.values.astype(np.float32)
                    except Exception:
                        # Fallback: try to read as numeric directly
                        data = pd.to_numeric(df.values.flatten(), errors='coerce').reshape(-1, df.shape[1]).astype(np.float32)
                else:
                    raise ValueError("CSV file appears to be empty")
            except ImportError:
                # Fallback to numpy if pandas not available
                logger.warning("pandas not available, using numpy.loadtxt (may fail with headers)")
                # Try to detect header by reading first line
                with open(dataset_path, 'r') as f:
                    first_line = f.readline().strip()
                    # Check if first line contains non-numeric values
                    try:
                        np.fromstring(first_line, sep=',', dtype=np.float32)
                        skip_rows = 0  # No header
                    except ValueError:
                        skip_rows = 1  # Has header
                        logger.info("Detected CSV header, skipping first row")
                
                data = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32, skiprows=skip_rows)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
        elif extension == ".npy":
            # NPY: numpy array
            try:
                # Prior check: verify physical size
                if dataset_path.exists():
                    actual_size = dataset_path.stat().st_size
                    if actual_size == 0:
                        raise ValueError("File exists but is empty (0 bytes).")
                
                # Attempt 1: Direct load
                try:
                    data = np.load(dataset_path).astype(np.float32)
                except (OSError, ValueError) as e:
                    if isinstance(e, OSError) and e.errno == 5:
                        logger.warning(f"Direct read failure (Errno 5). Retrying with mmap_mode='r'...")
                        # Attempt 2: Load via Memory Mapping (sometimes bypasses kernel buffer issues)
                        data = np.load(dataset_path, mmap_mode='r').copy().astype(np.float32)
                    else:
                        raise e
                        
            except (OSError, ValueError) as e:
                # If we get here, both attempts failed or file is unreadable
                error_msg = f"ERROR: Critical failure reading embeddings '{dataset_path}'."
                
                if isinstance(e, OSError) and e.errno == 5:
                    error_msg += " (Errno 5: Input/output error - Possible disk or network failure)"
                elif "0 bytes" in str(e):
                    error_msg += " (File is truncated/empty)"
                
                logger.error(error_msg)
                print(f"\n[bold red]{error_msg}[/bold red]")
                print(f"[yellow]Suggestion: Lightning.ai environment might have disk latency. Delete '{dataset_path.name}' and retry.[/yellow]\n")
                raise ValueError(error_msg) from e
                
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
    Processes the complete dataset of embeddings using BATCH processing.
    
    OPTIMIZED for High Throughput (>50 emb/s):
    1. Iterates in chunks of PROCESS_BATCH_SIZE (default 64)
    2. Sends batches to Orchestrator.process_batch()
    3. Reduces python overhead and GPU latency
    
    Args:
        embeddings: List of embeddings to process
        repository: KNRepository instance
        filter_bayesian: FilterBayesian instance
        orchestrator: Orchestrator instance
        progress_interval: How many embeddings to show progress (default: 100)
    """
    total_embeddings = len(embeddings)
    logger.info(f"Starting processing of {total_embeddings} embeddings (Batch Size: {PROCESS_BATCH_SIZE})")
    
    # Detect if we are in a notebook
    in_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    
    # Stack all embeddings into a single tensor for easy slicing
    # [N, DIM]
    if len(embeddings) > 0:
        all_embeddings_tensor = torch.stack(embeddings).to(DEVICE, dtype=DTYPE)
        # Generate Source IDs (indices) with dataset prefix if available
        # This allows DataManager to infer the dataset when retrieving texts for training
        dataset_name = None
        if hasattr(orchestrator, 'data_manager') and orchestrator.data_manager.dataset_paths:
            # If only one dataset is loaded, use it as prefix
            if len(orchestrator.data_manager.dataset_paths) == 1:
                dataset_name = list(orchestrator.data_manager.dataset_paths.keys())[0]
                logger.debug(f"Using dataset prefix '{dataset_name}' for source_ids")
            else:
                # Multiple datasets: use first one as default (could be improved)
                dataset_name = list(orchestrator.data_manager.dataset_paths.keys())[0]
                logger.debug(f"Multiple datasets available, using '{dataset_name}' as prefix for source_ids")
        
        if dataset_name:
            # Format: "dataset_name:index" (e.g., "20newsgroups:8293")
            all_source_ids = [f"{dataset_name}:{i}" for i in range(total_embeddings)]
        else:
            # Fallback: just use indices (will fail to retrieve texts, but won't break)
            logger.warning(
                "No dataset paths configured in DataManager. "
                "source_ids will be numeric only and may fail to retrieve texts for training. "
                "Consider passing dataset_paths to Orchestrator initialization."
            )
            all_source_ids = [str(i) for i in range(total_embeddings)]
    else:
        logger.warning("Empty embeddings list.")
        return
        
    num_batches = (total_embeddings + PROCESS_BATCH_SIZE - 1) // PROCESS_BATCH_SIZE
    
    # Choose progress method based on environment
    use_tqdm = False
    tqdm_kwargs = None
    
    # Always try to use tqdm if available (not just in notebooks)
    if TQDM_AVAILABLE:
        # Force regular tqdm (not notebook) for better console compatibility
        # Lightning AI Studio and other environments work better with regular tqdm
        use_notebook_tqdm = False  # Force False for better compatibility
        
        # Check if we're actually in a real Jupyter notebook with proper display
        if in_notebook and TQDM_NOTEBOOK:
            try:
                # Test if notebook display actually works
                from IPython.display import display, clear_output
                # If we can import, we might be in a real notebook
                # But still prefer regular tqdm for console output
                use_notebook_tqdm = False  # Force regular tqdm for visibility
            except ImportError:
                use_notebook_tqdm = False
        
        # Check if we're in a real terminal (not just a pipe)
        is_terminal = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
        
        tqdm_kwargs = {
            "total": total_embeddings,
            "desc": "Processing (Batch)",
            "unit": "emb",
            "mininterval": 0.1,   # Update every 0.1 seconds minimum
            "maxinterval": 0.5,   # Maximum 0.5 seconds between updates
            "file": sys.stderr,   # Use stderr for better visibility (doesn't interfere with stdout)
            "ncols": 100,         # Fixed width for better visibility
            "dynamic_ncols": False,  # Disable dynamic width
            "disable": False,     # Ensure tqdm is enabled
            "ascii": False,       # Use Unicode characters for better display
            "leave": True,        # Keep progress bar after completion
            "smoothing": 0.1,     # Smooth progress updates
        }
        
        # Only use position and custom format if we're in a real terminal
        if is_terminal:
            tqdm_kwargs["position"] = 0  # Position 0 to avoid multiple lines
            tqdm_kwargs["bar_format"] = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        else:
            # If not a terminal, use simpler format and disable position
            tqdm_kwargs["bar_format"] = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        
        if use_notebook_tqdm:
            # Use notebook-specific settings only if explicitly enabled
            tqdm_kwargs["ncols"] = None
            tqdm_kwargs["file"] = None  # Notebook handles output
        else:
            # Force stdout/stderr for console visibility
            tqdm_kwargs["file"] = sys.stderr  # stderr is better for progress bars
        
        use_tqdm = True
        logger.debug(f"Using tqdm for progress (notebook={in_notebook}, notebook_tqdm={TQDM_NOTEBOOK}, using_notebook={use_notebook_tqdm})")
    else:
        logger.debug("tqdm not available, using logging fallback")
    
    # Function to process a single batch
    def process_chunk(batch_idx):
        start_idx = batch_idx * PROCESS_BATCH_SIZE
        end_idx = min(start_idx + PROCESS_BATCH_SIZE, total_embeddings)
        
        # Slice batch
        batch_embeddings = all_embeddings_tensor[start_idx:end_idx]
        batch_source_ids = all_source_ids[start_idx:end_idx]
        
        # Execute Batch (The Magic happens here)
        orchestrator.process_batch(
            batch_embeddings, 
            source_ids=batch_source_ids
        )
        
        return end_idx - start_idx
        
    if use_tqdm:
        try:
            logger.info(f"Starting batch processing: {num_batches} batches of ~{PROCESS_BATCH_SIZE} embeddings")
            # Print message before creating progress bar
            print(f"Procesando {num_batches} batches de ~{PROCESS_BATCH_SIZE} embeddings cada uno...", file=sys.stdout)
            sys.stdout.flush()
            
            # Import regular tqdm if notebook tqdm was imported but we want regular
            if TQDM_NOTEBOOK and not use_notebook_tqdm:
                from tqdm import tqdm as regular_tqdm
                pbar = regular_tqdm(**tqdm_kwargs)
            else:
                pbar = tqdm(**tqdm_kwargs)
            
            sys.stderr.flush()  # Also flush stderr
            
            try:
                # ========================================================================
                # OPTIMIZATION: Block Processing (2025-01-25) - FIXED
                # ========================================================================
                # PROBLEM: Micro-management causing 143 DB calls + 286 I/O ops
                # SOLUTION: Deliver all batches as block to orchestrator
                # FIX: Ensure proper flow continuation
                # ========================================================================
                
                # NEW BLOCK PROCESSING (FIXED):
                try:
                    all_batches = [(i, 
                        all_embeddings_tensor[i*PROCESS_BATCH_SIZE:(i+1)*PROCESS_BATCH_SIZE],
                        all_source_ids[i*PROCESS_BATCH_SIZE:(i+1)*PROCESS_BATCH_SIZE]
                    ) for i in range(num_batches)]
                    
                    orchestrator.process_all_batches(all_batches, pbar)
                    
                    # CRITICAL: Ensure we continue to the next phase
                    logger.info("Block processing completed successfully")
                    
                except Exception as e:
                    logger.error(f"Block processing failed: {e}")
                    logger.info("Falling back to original processing...")
                    
                    # Fallback to original processing
                    for b_idx in range(num_batches):
                        # Log start of batch (first few batches for debugging)
                        if b_idx < 3 or (b_idx + 1) % 50 == 0:
                            logger.info(f"Processing batch {b_idx + 1}/{num_batches}...")
                        
                        # Process batch
                        processed_count = process_chunk(b_idx)
                        
                        # Update Stats every batch (before progress update)
                        counts = orchestrator.get_counts()
                        postfix_dict = {
                            "KNs": counts["kn_count"],
                            "Buf": counts["buffer_count"]
                        }
                        if orchestrator.has_active_training():
                            postfix_dict["T"] = "ON"
                        
                        # Update Progress immediately (force update)
                        pbar.set_postfix(postfix_dict, refresh=False)
                        pbar.update(n=processed_count)
                        
                        # Force flush after each batch to ensure visibility
                        sys.stderr.flush()
                        sys.stdout.flush()
                        
                        # Log progress every 10 batches for debugging
                        if (b_idx + 1) % 10 == 0:
                            current_processed = min((b_idx + 1) * PROCESS_BATCH_SIZE, total_embeddings)
                            logger.info(f"Progress: {b_idx + 1}/{num_batches} batches ({current_processed}/{total_embeddings} embeddings, {current_processed/total_embeddings*100:.1f}%)")
                            pbar.write(f" == Progress: {b_idx + 1}/{num_batches} batches ({current_processed/total_embeddings*100:.1f}%)")
                
                # ========================================================================
                    
            finally:
                pbar.close()
                logger.info(f"Batch processing completed: {total_embeddings} embeddings processed")
                print(f"\n   Process completed: {total_embeddings} embeddings processed")
                sys.stdout.flush()
                sys.stderr.flush()
                    
        except Exception as e:
            logger.warning(f"tqdm error: {e}. Fallback to logging.")
            import traceback
            logger.debug(f"tqdm error details: {traceback.format_exc()}")
            use_tqdm = False
            
    if not use_tqdm:
        # Fallback loop
        for b_idx in range(num_batches):
            processed = process_chunk(b_idx)
            current_total = (b_idx * PROCESS_BATCH_SIZE) + processed
            
            if (current_total) % progress_interval < PROCESS_BATCH_SIZE: # Approx interval
                counts = orchestrator.get_counts()
                logger.debug(
                    f"Processed {current_total}/{total_embeddings} "
                    f"({current_total/total_embeddings*100:.1f}%) | "
                    f"KNs: {counts['kn_count']}, Buf: {counts['buffer_count']}"
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
    console.print("\n[bold cyan]Processing Summary[/bold cyan]\n")
    
    # Calculate training statistics
    trained_nodes = sum(1 for sig in signatures if repository.is_trained(sig["node_id"]))
    candidates_potential = sum(1 for sig in signatures if sig["mass"] > 5)
    
    # General statistics table
    stats_table = Table(title="General Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green", justify="right")
    
    total_mass = sum(sig["mass"] for sig in signatures)
    avg_mass = total_mass / len(signatures) if signatures else 0
    avg_variance = sum(sig["variance"] for sig in signatures) / len(signatures) if signatures else 0
    
    stats_table.add_row("Total KnowledgeNodes", str(len(signatures)))
    stats_table.add_row("Active Nodes", str(active_nodes_count))
    stats_table.add_row("Total Embeddings", str(total_mass))
    stats_table.add_row("Average Mass", f"{avg_mass:.1f}")
    stats_table.add_row("Average Variance", f"{avg_variance:.4f}")
    
    console.print(stats_table)
    
    # Training statistics table
    training_table = Table(title="Training Status", show_header=True, header_style="bold yellow")
    training_table.add_column("Metric", style="cyan")
    training_table.add_column("Value", style="green", justify="right")
    
    training_table.add_row("Trained Nodes", str(trained_nodes))
    training_table.add_row("Potential Candidates (mass > 5)", str(candidates_potential))
    
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
    
    logger.info("End of Day Analysis:")
    logger.info(f"   Trained nodes: {trained_nodes}")
    logger.info(f"   Potential candidates with mass > 5: {candidates_potential}")
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
        logger.info("XCTOPUS - Clustering Layer")
        logger.info("=" * 60)
        
        # 1. Load embeddings
        embeddings = load_embeddings(dataset_path)
        
        # 2. Initialize components
        repository, filter_bayesian, orchestrator = initialize_components()
        
        # 3. Process dataset
        process_dataset(embeddings, repository, filter_bayesian, orchestrator)
        
        # 4. Fuse Knowledge Nodes (post-clustering)
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
