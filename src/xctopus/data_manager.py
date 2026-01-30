"""
DataManager - Bridge between Repository and Original Datasets

Phase 2: Training Support
Purpose: Translate source_ids (pointers) to original texts from datasets
for training adapters.

Added: Phase 2 (2025-01-XX)

Memory Efficiency:
- For massive datasets (e.g., full arXiv corpus), uses SQLite index or memmap
- Avoids loading entire dataset into RAM
- Lazy loading with on-demand access

Data Provenance:
- Ensures training uses exactly the data that FilterBayesian assigned to each node
- Maintains strict mapping between source_ids and original texts

Format Flexibility:
- Supports CSV, JSON, JSONL
- Supports folder-based formats (20newsgroups style)
- Auto-detects format and uses appropriate access method
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Threshold for using memory-efficient methods (in MB)
LARGE_DATASET_THRESHOLD_MB = 500  # 500 MB


class DataManager:
    """
    Manages access to original datasets for training.
    
    Translates source_ids (pointers) to original texts from datasets.
    Supports multiple formats: CSV, JSON, JSONL, folder-based (20newsgroups).
    
    Features:
    - Memory Efficiency: Uses SQLite index or memmap for massive datasets
    - Lazy loading: Only loads datasets when needed
    - Data Provenance: Ensures exact mapping between source_ids and texts
    - Format Flexibility: Auto-detects and handles multiple formats
    - Null safety: Handles missing values gracefully
    - Optimized logging: Uses DEBUG level to avoid console spam
    """
    
    def __init__(
        self, 
        dataset_paths: Optional[Dict[str, str]] = None,
        use_sqlite_index: bool = True,
        sqlite_index_dir: Optional[str] = None,
        repository: Optional[Any] = None
    ):
        """
        Initialize DataManager.
        
        Args:
            dataset_paths: Dictionary mapping dataset names to file paths
            use_sqlite_index: If True, creates SQLite index for large datasets
            sqlite_index_dir: Directory for SQLite index files (default: same as dataset)
            repository: Optional KNRepository instance for session resolution
        """
        self.dataset_paths = dataset_paths or {}
        self.use_sqlite_index = use_sqlite_index
        self.sqlite_index_dir = sqlite_index_dir
        self.repository = repository
        
        # Cache for loaded datasets (lazy loading) - only for small datasets
        # Format: {dataset_name: {source_id: text}}
        self._dataset_cache: Dict[str, Dict[str, str]] = {}
        
        # SQLite connections for large datasets
        # Format: {dataset_name: sqlite3.Connection}
        self._sqlite_connections: Dict[str, sqlite3.Connection] = {}
        
        # Cache for dataset metadata (format, columns, access method, etc.)
        self._dataset_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(
            f"DataManager initialized with {len(self.dataset_paths)} dataset paths "
            f"(sqlite_index={use_sqlite_index})"
        )
    
    def _detect_dataset_type(self, dataset_path: str) -> str:
        """
        Detect dataset type from path (file extension or directory).
        
        Args:
            dataset_path: Path to dataset file or directory
            
        Returns:
            'csv', 'json', 'jsonl', or 'folder'
        """
        path = Path(dataset_path)
        
        # Check if it's a directory (folder-based format)
        if path.is_dir():
            return 'folder'
        
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return 'csv'
        elif ext == '.json':
            # Check if it's JSONL by reading first line
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') and not first_line.startswith('[{'):
                        return 'jsonl'
            except Exception:
                pass
            return 'json'
        elif ext == '.jsonl':
            return 'jsonl'
        else:
            # Default to CSV for unknown extensions
            logger.debug(f"Unknown extension {ext}, defaulting to CSV")
            return 'csv'
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0
    
    def _should_use_sqlite_index(self, dataset_path: str, dataset_type: str) -> bool:
        """
        Determine if dataset is large enough to warrant SQLite index.
        
        Args:
            dataset_path: Path to dataset
            dataset_type: Type of dataset ('csv', 'json', 'jsonl', 'folder')
            
        Returns:
            True if should use SQLite index, False otherwise
        """
        if not self.use_sqlite_index:
            return False
        
        # For folders, check total size of all files
        if dataset_type == 'folder':
            try:
                total_size = sum(
                    f.stat().st_size 
                    for f in Path(dataset_path).rglob('*') 
                    if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                return size_mb > LARGE_DATASET_THRESHOLD_MB
            except Exception:
                return False
        
        # For files, check file size
        size_mb = self._get_file_size_mb(dataset_path)
        return size_mb > LARGE_DATASET_THRESHOLD_MB
    
    def _create_sqlite_index(
        self, 
        dataset_path: str, 
        dataset_name: str, 
        dataset_type: str
    ) -> bool:
        """
        Create SQLite index for large dataset (memory-efficient access).
        
        Args:
            dataset_path: Path to dataset file or directory
            dataset_name: Name identifier
            dataset_type: Type of dataset ('csv', 'json', 'jsonl', 'folder')
            
        Returns:
            True if index created successfully, False otherwise
        """
        # Determine index file path
        if self.sqlite_index_dir:
            index_dir = Path(self.sqlite_index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use same directory as dataset
            index_dir = Path(dataset_path).parent
        
        index_file = index_dir / f"{dataset_name}_index.db"
        
        # Check if index already exists
        if index_file.exists():
            logger.debug(f"SQLite index already exists: {index_file}")
            return True
        
        try:
            conn = sqlite3.connect(str(index_file))
            cursor = conn.cursor()
            
            # Create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS texts (
                    source_id TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_id ON texts(source_id)')
            
            # Load data based on type
            if dataset_type == 'csv':
                self._index_csv_to_sqlite(dataset_path, cursor)
            elif dataset_type in ['json', 'jsonl']:
                self._index_json_to_sqlite(dataset_path, cursor, dataset_type)
            elif dataset_type == 'folder':
                self._index_folder_to_sqlite(dataset_path, cursor)
            else:
                logger.debug(f"Unsupported dataset type for SQLite index: {dataset_type}")
                conn.close()
                return False
            
            conn.commit()
            conn.close()
            
            logger.debug(f"SQLite index created: {index_file}")
            return True
            
        except Exception as e:
            logger.debug(f"Error creating SQLite index: {e}")
            return False
    
    def _index_csv_to_sqlite(self, dataset_path: str, cursor: sqlite3.Cursor) -> None:
        """Index CSV file into SQLite."""
        try:
            import pandas as pd
        except ImportError:
            logger.debug("pandas not available for CSV indexing")
            return
        
        try:
            # Load CSV in chunks to avoid memory issues
            chunk_size = 10000
            text_column = None
            
            for chunk_idx, chunk in enumerate(pd.read_csv(
                dataset_path,
                sep=",",
                quotechar='"',
                dtype=str,
                on_bad_lines='skip',
                engine='python',
                chunksize=chunk_size
            )):
                # Detect text column on first chunk
                if text_column is None:
                    for col in ['text', 'summaries', 'titles', 'abstract', 'content', 'body']:
                        if col in chunk.columns:
                            text_column = col
                            break
                    if text_column is None:
                        text_column = chunk.columns[0] if len(chunk.columns) > 0 else None
                
                if text_column is None:
                    continue
                
                # Insert rows
                for idx, row in chunk.iterrows():
                    text = row.get(text_column)
                    if pd.isna(text) or text is None or str(text).strip() == '':
                        continue
                    
                    source_id = str(chunk_idx * chunk_size + idx)
                    cursor.execute(
                        "INSERT OR REPLACE INTO texts (source_id, text_content) VALUES (?, ?)",
                        (source_id, str(text).strip())
                    )
                
                # Commit periodically
                if chunk_idx % 10 == 0:
                    cursor.connection.commit()
            
            cursor.connection.commit()
            
        except Exception as e:
            logger.debug(f"Error indexing CSV to SQLite: {e}")
    
    def _index_json_to_sqlite(
        self, 
        dataset_path: str, 
        cursor: sqlite3.Cursor, 
        dataset_type: str
    ) -> None:
        """Index JSON/JSONL file into SQLite."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                if dataset_type == 'jsonl':
                    # JSONL format
                    for idx, line in enumerate(f):
                        try:
                            obj = json.loads(line.strip())
                            source_id, text = self._extract_text_from_json_object(obj, idx)
                            if source_id and text:
                                cursor.execute(
                                    "INSERT OR REPLACE INTO texts (source_id, text_content) VALUES (?, ?)",
                                    (source_id, text)
                                )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
                else:
                    # Standard JSON
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, obj in enumerate(data):
                            source_id, text = self._extract_text_from_json_object(obj, idx)
                            if source_id and text:
                                cursor.execute(
                                    "INSERT OR REPLACE INTO texts (source_id, text_content) VALUES (?, ?)",
                                    (source_id, text)
                                )
            
            cursor.connection.commit()
            
        except Exception as e:
            logger.debug(f"Error indexing JSON to SQLite: {e}")
    
    def _index_folder_to_sqlite(self, dataset_path: str, cursor: sqlite3.Cursor) -> None:
        """Index folder-based dataset (20newsgroups style) into SQLite."""
        try:
            folder_path = Path(dataset_path)
            source_id_counter = 0
            
            # Recursively find all text files
            for text_file in folder_path.rglob('*.txt'):
                try:
                    with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if text:
                            # Use relative path as source_id for provenance
                            relative_path = text_file.relative_to(folder_path)
                            source_id = str(relative_path).replace(os.sep, '/')
                            
                            cursor.execute(
                                "INSERT OR REPLACE INTO texts (source_id, text_content) VALUES (?, ?)",
                                (source_id, text)
                            )
                            source_id_counter += 1
                except Exception:
                    continue
                
                # Commit periodically
                if source_id_counter % 1000 == 0:
                    cursor.connection.commit()
            
            cursor.connection.commit()
            
        except Exception as e:
            logger.debug(f"Error indexing folder to SQLite: {e}")
    
    def _load_csv_dataset(self, dataset_path: str, dataset_name: str) -> Dict[str, str]:
        """
        Load CSV dataset and create mapping {source_id: text}.
        
        Args:
            dataset_path: Path to CSV file
            dataset_name: Name identifier for caching
            
        Returns:
            Dictionary mapping source_id (row index as string) to text
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not available, cannot load CSV datasets")
            return {}
        
        if not os.path.exists(dataset_path):
            logger.debug(f"CSV file not found: {dataset_path}")
            return {}
        
        try:
            # Load CSV with error handling
            df = pd.read_csv(
                dataset_path,
                sep=",",
                quotechar='"',
                dtype=str,
                on_bad_lines='skip',  # Skip malformed lines
                engine='python'  # More lenient parsing
            )
            
            # Detect text column (common names)
            text_column = None
            for col in ['text', 'summaries', 'titles', 'abstract', 'content', 'body']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                # Use first column as fallback
                text_column = df.columns[0] if len(df.columns) > 0 else None
                logger.debug(f"No standard text column found, using first column: {text_column}")
            
            if text_column is None:
                logger.debug(f"No columns found in CSV: {dataset_path}")
                return {}
            
            # Create mapping: {row_index: text}
            # source_id is the row index (0-based) as string
            mapping = {}
            for idx, row in df.iterrows():
                text = row.get(text_column)
                
                # Handle null/NaN values
                if pd.isna(text) or text is None or str(text).strip() == '':
                    continue  # Skip null/empty texts
                
                # Use row index as source_id
                source_id = str(idx)
                mapping[source_id] = str(text).strip()
            
            # Store metadata for future reference
            self._dataset_metadata[dataset_name] = {
                'type': 'csv',
                'text_column': text_column,
                'total_rows': len(df),
                'valid_texts': len(mapping)
            }
            
            logger.debug(
                f"Loaded CSV dataset '{dataset_name}': "
                f"{len(mapping)} valid texts from {len(df)} rows"
            )
            
            return mapping
            
        except Exception as e:
            logger.debug(f"Error loading CSV dataset {dataset_path}: {e}")
            return {}
    
    def _load_json_dataset(self, dataset_path: str, dataset_name: str) -> Dict[str, str]:
        """
        Load JSON dataset and create mapping {source_id: text}.
        
        Supports:
        - JSON array of objects
        - JSON lines (one object per line)
        - Single JSON object with array field
        
        Args:
            dataset_path: Path to JSON file
            dataset_name: Name identifier for caching
            
        Returns:
            Dictionary mapping source_id to text
        """
        if not os.path.exists(dataset_path):
            logger.debug(f"JSON file not found: {dataset_path}")
            return {}
        
        try:
            mapping = {}
            
            # Try to detect format
            with open(dataset_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset to beginning
                
                # Check if it's JSONL (one object per line)
                if first_line.startswith('{') and not first_line.startswith('[{'):
                    # JSONL format
                    for idx, line in enumerate(f):
                        try:
                            obj = json.loads(line.strip())
                            source_id, text = self._extract_text_from_json_object(obj, idx)
                            if source_id and text:
                                mapping[source_id] = text
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue  # Skip malformed lines
                else:
                    # Standard JSON (array or object)
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        # Array of objects
                        for idx, obj in enumerate(data):
                            source_id, text = self._extract_text_from_json_object(obj, idx)
                            if source_id and text:
                                mapping[source_id] = text
                    elif isinstance(data, dict):
                        # Single object - check for common array fields
                        for key in ['items', 'data', 'papers', 'documents']:
                            if key in data and isinstance(data[key], list):
                                for idx, obj in enumerate(data[key]):
                                    source_id, text = self._extract_text_from_json_object(obj, idx)
                                    if source_id and text:
                                        mapping[source_id] = text
                                break
            
            # Store metadata
            self._dataset_metadata[dataset_name] = {
                'type': 'json',
                'total_items': len(mapping),
                'valid_texts': len(mapping)
            }
            
            logger.debug(
                f"Loaded JSON dataset '{dataset_name}': "
                f"{len(mapping)} valid texts"
            )
            
            return mapping
            
        except Exception as e:
            logger.debug(f"Error loading JSON dataset {dataset_path}: {e}")
            return {}
    
    def _extract_text_from_json_object(
        self, 
        obj: Dict[str, Any], 
        fallback_idx: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract source_id and text from JSON object.
        
        Args:
            obj: JSON object (dictionary)
            fallback_idx: Index to use as source_id if no ID field found
            
        Returns:
            Tuple of (source_id, text) or (None, None) if extraction fails
        """
        # Try to find ID field (common names)
        source_id = None
        for id_field in ['id', 'paper_id', 'arxiv_id', 'source_id', 'doc_id', '_id']:
            if id_field in obj and obj[id_field] is not None:
                source_id = str(obj[id_field])
                break
        
        # Fallback to index if no ID field
        if source_id is None:
            source_id = str(fallback_idx)
        
        # Try to find text field (common names)
        text = None
        for text_field in ['abstract', 'summary', 'text', 'content', 'title', 'body']:
            if text_field in obj and obj[text_field] is not None:
                text_value = obj[text_field]
                # Handle nested structures
                if isinstance(text_value, str):
                    text = text_value.strip()
                elif isinstance(text_value, list):
                    # Join list items
                    text = ' '.join(str(item) for item in text_value if item).strip()
                elif isinstance(text_value, dict):
                    # Try to extract text from nested dict
                    for nested_field in ['text', 'content', 'abstract']:
                        if nested_field in text_value and isinstance(text_value[nested_field], str):
                            text = text_value[nested_field].strip()
                            break
                
                if text and text != '':
                    break
        
        # Handle null/empty text
        if not text or text == '':
            return None, None
        
        return source_id, text
    
    def _get_dataset_name_from_source_id(self, source_id: str) -> Optional[str]:
        """
        Infer dataset name from source_id format.
        
        Args:
            source_id: Source ID (e.g., "arxiv:1234.5678", "20newsgroups:42", "42")
            
        Returns:
            Dataset name or None if cannot infer
        """
        # Check if source_id has prefix (e.g., "arxiv:1234.5678" or "sA9x2:42")
        if ':' in source_id:
            prefix = source_id.split(':', 1)[0]
            if prefix in self.dataset_paths:
                return prefix
            
            # Phase 2 (Session Support): Try to resolve session prefix via Repository
            if self.repository and hasattr(self.repository, 'get_session_info'):
                session_info = self.repository.get_session_info(prefix)
                if session_info:
                    dataset_name = session_info['dataset_name']
                    dataset_path = session_info['dataset_path']
                    
                    # Register this dataset path dynamically
                    if dataset_name not in self.dataset_paths:
                        self.dataset_paths[dataset_name] = dataset_path
                        logger.info(f"Resolved session-based dataset: {prefix} -> {dataset_name} ({dataset_path})")
                    
                    # Also register the prefix itself as a valid dataset name pointing to the same file
                    # This allows subsequent lookups for the same prefix to be fast
                    if prefix not in self.dataset_paths:
                        self.dataset_paths[prefix] = dataset_path
                    
                    return dataset_name
        
        # Try to match by checking all datasets
        # This is a fallback - ideally source_id should have prefix
        for dataset_name in self.dataset_paths.keys():
            if dataset_name.lower() in source_id.lower():
                return dataset_name
        
        # If no match and only one dataset available, use it as fallback
        # This handles cases where source_ids are just numeric indices (e.g., "8293")
        # and we're processing a single dataset
        if len(self.dataset_paths) == 1:
            dataset_name = list(self.dataset_paths.keys())[0]
            logger.debug(
                f"Cannot infer dataset from source_id '{source_id}', "
                f"using single available dataset '{dataset_name}' as fallback"
            )
            return dataset_name
        
        # Multiple datasets but no prefix - cannot determine which one
        logger.debug(
            f"Cannot infer dataset from source_id '{source_id}': "
            f"no prefix and {len(self.dataset_paths)} datasets available"
        )
        return None
    
    def _load_folder_dataset(self, dataset_path: str, dataset_name: str) -> Dict[str, str]:
        """
        Load folder-based dataset (20newsgroups style).
        
        Args:
            dataset_path: Path to folder
            dataset_name: Name identifier
            
        Returns:
            Dictionary mapping source_id (relative path) to text
        """
        folder_path = Path(dataset_path)
        
        if not folder_path.is_dir():
            logger.debug(f"Folder not found: {dataset_path}")
            return {}
        
        mapping = {}
        file_count = 0
        
        try:
            # Recursively find all text files
            for text_file in folder_path.rglob('*.txt'):
                try:
                    with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        if text:
                            # Use relative path as source_id for data provenance
                            relative_path = text_file.relative_to(folder_path)
                            source_id = str(relative_path).replace(os.sep, '/')
                            mapping[source_id] = text
                            file_count += 1
                except Exception as e:
                    logger.debug(f"Error reading file {text_file}: {e}")
                    continue
            
            # Store metadata
            self._dataset_metadata[dataset_name] = {
                'type': 'folder',
                'total_files': file_count,
                'valid_texts': len(mapping)
            }
            
            logger.debug(
                f"Loaded folder dataset '{dataset_name}': "
                f"{len(mapping)} valid texts from {file_count} files"
            )
            
            return mapping
            
        except Exception as e:
            logger.debug(f"Error loading folder dataset {dataset_path}: {e}")
            return {}
    
    def _get_sqlite_connection(self, dataset_name: str) -> Optional[sqlite3.Connection]:
        """Get or create SQLite connection for dataset."""
        if dataset_name in self._sqlite_connections:
            return self._sqlite_connections[dataset_name]
        
        # Determine index file path
        if self.sqlite_index_dir:
            index_dir = Path(self.sqlite_index_dir)
        else:
            dataset_path = self.dataset_paths.get(dataset_name)
            if not dataset_path:
                return None
            index_dir = Path(dataset_path).parent
        
        index_file = index_dir / f"{dataset_name}_index.db"
        
        if not index_file.exists():
            return None
        
        try:
            conn = sqlite3.connect(str(index_file))
            conn.row_factory = sqlite3.Row
            self._sqlite_connections[dataset_name] = conn
            return conn
        except Exception as e:
            logger.debug(f"Error opening SQLite index: {e}")
            return None
    
    def _get_text_from_sqlite(self, dataset_name: str, source_id: str) -> Optional[str]:
        """Get text from SQLite index."""
        conn = self._get_sqlite_connection(dataset_name)
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text_content FROM texts WHERE source_id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            return row["text_content"] if row else None
        except Exception as e:
            logger.debug(f"Error querying SQLite index: {e}")
            return None
    
    def _ensure_dataset_loaded(self, dataset_name: str) -> bool:
        """
        Ensure dataset is loaded or indexed (lazy loading).
        
        For large datasets, creates SQLite index instead of loading into memory.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            True if dataset loaded/indexed successfully, False otherwise
        """
        # Check if already loaded or indexed
        if dataset_name in self._dataset_cache:
            return True
        
        if dataset_name in self._sqlite_connections:
            return True
        
        # Check if SQLite index exists
        if self.use_sqlite_index:
            if self.sqlite_index_dir:
                index_dir = Path(self.sqlite_index_dir)
            else:
                dataset_path = self.dataset_paths.get(dataset_name)
                if not dataset_path:
                    return False
                index_dir = Path(dataset_path).parent
            
            index_file = index_dir / f"{dataset_name}_index.db"
            if index_file.exists():
                # Index exists, just open connection
                conn = self._get_sqlite_connection(dataset_name)
                if conn:
                    self._dataset_metadata[dataset_name] = {
                        'type': 'sqlite_index',
                        'access_method': 'sqlite'
                    }
                    return True
        
        # Check if dataset path exists
        if dataset_name not in self.dataset_paths:
            logger.debug(f"Dataset '{dataset_name}' not in dataset_paths")
            return False
        
        dataset_path = self.dataset_paths[dataset_name]
        
        # Check if path exists
        if not os.path.exists(dataset_path):
            logger.debug(f"Dataset path not found: {dataset_path}")
            return False
        
        # Detect dataset type
        dataset_type = self._detect_dataset_type(dataset_path)
        
        # Check if should use SQLite index (for large datasets)
        if self._should_use_sqlite_index(dataset_path, dataset_type):
            # Create SQLite index
            if self._create_sqlite_index(dataset_path, dataset_name, dataset_type):
                self._dataset_metadata[dataset_name] = {
                    'type': dataset_type,
                    'access_method': 'sqlite',
                    'indexed': True
                }
                # Open connection
                conn = self._get_sqlite_connection(dataset_name)
                return conn is not None
            else:
                # Fallback to memory loading if index creation fails
                logger.debug(f"SQLite index creation failed, falling back to memory loading")
        
        # Load into memory (for small datasets)
        if dataset_type == 'csv':
            mapping = self._load_csv_dataset(dataset_path, dataset_name)
        elif dataset_type in ['json', 'jsonl']:
            mapping = self._load_json_dataset(dataset_path, dataset_name)
        elif dataset_type == 'folder':
            mapping = self._load_folder_dataset(dataset_path, dataset_name)
        else:
            logger.debug(f"Unsupported dataset type: {dataset_type}")
            return False
        
        # Store in cache
        self._dataset_cache[dataset_name] = mapping
        self._dataset_metadata[dataset_name] = {
            'type': dataset_type,
            'access_method': 'memory',
            'cached_texts': len(mapping)
        }
        
        return True
    
    def get_texts_from_pointers(self, source_ids: List[str]) -> List[str]:
        """
        Retrieve texts from original datasets using source_ids (pointers).
        
        Args:
            source_ids: List of source IDs (pointers to original dataset)
                Format can be:
                - Simple index: "42"
                - With prefix: "arxiv:1234.5678", "20newsgroups:42"
                - Mixed formats
        
        Returns:
            List of texts corresponding to source_ids
            Returns empty list if no texts found or all source_ids invalid
        
        Example:
            source_ids = ["0", "1", "2"]
            texts = data_manager.get_texts_from_pointers(source_ids)
            # Returns: ["First text", "Second text", "Third text"]
        """
        if not source_ids:
            return []
        
        texts = []
        failed_count = 0
        
        for source_id in source_ids:
            # Handle null/empty source_id
            if not source_id or not isinstance(source_id, str):
                failed_count += 1
                continue
            
            # Infer dataset name from source_id
            dataset_name = self._get_dataset_name_from_source_id(source_id)
            
            if dataset_name is None:
                logger.debug(f"Cannot infer dataset from source_id: {source_id}")
                failed_count += 1
                continue
            
            # Ensure dataset is loaded or indexed
            if not self._ensure_dataset_loaded(dataset_name):
                failed_count += 1
                continue
            
            # Extract actual source_id (remove prefix if present)
            actual_source_id = source_id
            if ':' in source_id:
                actual_source_id = source_id.split(':', 1)[1]
            
            # Get text based on access method
            metadata = self._dataset_metadata.get(dataset_name, {})
            access_method = metadata.get('access_method', 'memory')
            
            text = None
            if access_method == 'sqlite':
                # Query from SQLite index (memory-efficient)
                text = self._get_text_from_sqlite(dataset_name, actual_source_id)
            else:
                # Get from memory cache
                dataset_cache = self._dataset_cache.get(dataset_name, {})
                text = dataset_cache.get(actual_source_id)
            
            if text:
                texts.append(text)
            else:
                failed_count += 1
                logger.debug(f"Text not found for source_id: {source_id}")
        
        if failed_count > 0:
            logger.debug(
                f"Failed to retrieve {failed_count}/{len(source_ids)} texts "
                f"(success: {len(texts)})"
            )
        
        return texts
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """
        Clear dataset cache and close SQLite connections to free memory.
        
        Args:
            dataset_name: Specific dataset to clear, or None to clear all
        """
        if dataset_name:
            # Clear memory cache
            self._dataset_cache.pop(dataset_name, None)
            self._dataset_metadata.pop(dataset_name, None)
            
            # Close SQLite connection
            if dataset_name in self._sqlite_connections:
                try:
                    self._sqlite_connections[dataset_name].close()
                except Exception:
                    pass
                del self._sqlite_connections[dataset_name]
            
            logger.debug(f"Cache cleared for dataset: {dataset_name}")
        else:
            # Clear all memory caches
            self._dataset_cache.clear()
            self._dataset_metadata.clear()
            
            # Close all SQLite connections
            for conn in self._sqlite_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._sqlite_connections.clear()
            
            logger.debug("All caches cleared")
    
    def close(self) -> None:
        """Close all SQLite connections and clear caches."""
        self.clear_cache()
        logger.debug("DataManager closed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded/indexed datasets.
        
        Returns:
            Dictionary with cache and index statistics
        """
        stats = {
            'memory_datasets': len(self._dataset_cache),
            'sqlite_indexed_datasets': len(self._sqlite_connections),
            'total_datasets': len(self._dataset_cache) + len(self._sqlite_connections),
            'datasets': {}
        }
        
        # Memory-cached datasets
        for dataset_name, cache in self._dataset_cache.items():
            metadata = self._dataset_metadata.get(dataset_name, {})
            stats['datasets'][dataset_name] = {
                'cached_texts': len(cache),
                'access_method': 'memory',
                **metadata
            }
        
        # SQLite-indexed datasets
        for dataset_name in self._sqlite_connections.keys():
            metadata = self._dataset_metadata.get(dataset_name, {})
            # Count texts in SQLite index
            conn = self._get_sqlite_connection(dataset_name)
            text_count = 0
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM texts")
                    row = cursor.fetchone()
                    text_count = row[0] if row else 0
                except Exception:
                    pass
            
            stats['datasets'][dataset_name] = {
                'indexed_texts': text_count,
                'access_method': 'sqlite',
                **metadata
            }
        
        return stats
