"""
Core text preprocessing and embedding component.

This module contains the TextPreprocessor class, which is a core component
required for pipeline execution. It provides text preprocessing and embedding
generation functionality.
"""

import os
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict

import warnings 
warnings.filterwarnings("ignore", category=UserWarning)

class TextPreprocessor(nn.Module):
    """
    Flexible text preprocessing and embedding pipeline.
    
    This class provides a unified interface for preparing textual datasets for
    clustering, training, or incremental learning. It supports two operation modes:
    
    1. Dataset Mode:
       - Provide a dataset path (path_dataset) and the names of text-related columns.
       - The class handles loading, cleaning, combining columns, tokenization, and
         optional embedding generation.
       - Use process_dataset() for standard preprocessing or 
         analyze_and_prepare_dataset() for full statistical inspection and validation.
    
    2. Direct Input Mode:
       - Provide a list of raw texts or a DataFrame directly.
       - Use encode_texts() to obtain embeddings without requiring a dataset file.
    
    The class standardizes text cleaning, column selection, tokenization, and 
    embedding generation using SentenceTransformer, making it suitable for 
    downstream tasks such as clustering, knowledge node creation, and training 
    within the Xctopus pipeline.
    
    Args:
        path_dataset (str, optional): Path to CSV dataset file. Must be provided explicitly
            if you want to use process_dataset() or analyze_and_prepare_dataset(). If None, 
            only encode_texts() can be used. No default path is assumed. Defaults to None.
        embedding_dim (int): Embedding dimension (determined by the model, this parameter
            is kept for compatibility). Defaults to 128.
        text_columns (list[str], optional): List of column names to use for embeddings when processing
            datasets. If None, defaults to ['text']. These columns will be combined with join_with.
            Defaults to None.
        join_with (str): Separator to use when combining multiple text columns from the dataset.
            Defaults to "\\n" (newline character).
        meta_columns (list, optional): List of column names to keep as metadata (not used for embeddings).
            Defaults to None.
        model_name (str): Name of the SentenceTransformer model to use for embeddings.
            Defaults to 'distiluse-base-multilingual-cased-v1'.
        max_length (int): Maximum number of tokens to encode. Longer texts will be truncated.
            Defaults to 512.
        normalize (bool): Whether to normalize embeddings to unit norm (L2 normalization).
            Defaults to True.
        drop_empty (bool): Whether to remove rows with no valid text when processing datasets.
            Defaults to False.
        label_column (str, optional): Column name containing labels for supervised learning.
            If provided, indicates the dataset is supervised. If None, dataset is unsupervised.
            Examples: "sentiment", "category", None. Defaults to None.
        id_column (str, optional): Column name containing unique identifiers for tracking and mapping.
            Useful for datasets with millions of rows or when you need to map predictions back.
            Examples: "id", "uuid", "review_id". If not provided, no ID tracking is performed.
            Defaults to None.
    
    Example:
        # Without dataset (framework mode - use encode_texts directly)
        text_emb = TextPreprocessor(path_dataset=None)
        embeddings = text_emb.encode_texts(["Hello world", "Machine learning"])
        
        # With dataset and custom columns (supervised learning)
        text_emb = TextPreprocessor(
            path_dataset="data/articles.csv",
            text_columns=["title", "abstract", "notes"],
            join_with="\n",
            meta_columns=["date", "source", "author"],
            model_name="all-MiniLM-L6-v2",
            max_length=256,
            normalize=True,
            drop_empty=True,
            label_column="sentiment",  # Supervised learning
            id_column="uuid"  # For tracking
        )
        result = text_emb.analyze_and_prepare_dataset()
        embeddings = text_emb.encode_texts(result['texts'])
    """
    def __init__(self, path_dataset=None, embedding_dim=128, text_columns=None, 
                 join_with="\n", meta_columns=None, model_name='distiluse-base-multilingual-cased-v1',
                 max_length=512, normalize=True, drop_empty=False, label_column=None, id_column=None): 
        super().__init__()   
        self.path_dataset = path_dataset
        self.vocab = None
        self.embedding_dim=embedding_dim 
        self.embedding_layer = None
        self.embeddings = None
        # Column configuration for dataset processing
        self.text_columns = text_columns
        self.join_with = join_with
        self.meta_columns = meta_columns
        self.label_column = label_column
        self.id_column = id_column
        # Model and encoding configuration
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.drop_empty = drop_empty
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(model_name)
    
    def has_dataset(self):
        """
        Check if a dataset path is configured and the file exists.
        
        Returns:
            bool: True if dataset path is set and file exists, False otherwise
        """
        if self.path_dataset is None:
            return False
        return os.path.exists(self.path_dataset)
    
    def process_dataset(self, max_len=20, n_clusters=2):
        """
        Process the dataset from the configured path.
        
        Args:
            max_len: Maximum length (not currently used)
            n_clusters: Number of clusters for KMeans
        
        Returns:
            tuple: (DataFrame, embeddings, assigned_clusters) or None if dataset not available
        
        Raises:
            ValueError: If path_dataset is None or file doesn't exist
        """
        # Validate that path_dataset is provided
        if self.path_dataset is None:
            raise ValueError(
                "path_dataset is None. To use process_dataset(), you must provide a dataset path "
                "when initializing TextPreprocessor. If you don't have a dataset file, use "
                "encode_texts() method directly with a list of texts instead."
            )
        
        # Validate that the file exists
        if not os.path.exists(self.path_dataset):
            raise FileNotFoundError(
                f"Dataset file not found: {self.path_dataset}. "
                f"Please check that the path is correct and the file exists."
            )

        ds = pd.read_csv(self.path_dataset, sep=",", quotechar='"', dtype=str)
        ds = ds[['text', 'category']]
        sentences = ds['text'].tolist()

        self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        print(f"\n=== Dataset ===")
        print(f"Rows: {len(ds)}, Columns: {list(ds.columns)}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        assigned_clusters = kmeans.fit_predict(self.embeddings.cpu().numpy())

        cluster_sentences = defaultdict(list)
        for cluster, sentence in zip(assigned_clusters, sentences):
            cluster_sentences[cluster].append(sentence)

        print("\n=== Conteo de frases por cluster ===")
        for cluster, sents in cluster_sentences.items():
            print(f"Cluster {cluster}: {len(sents)} frases")

        for cluster, sents in cluster_sentences.items():
            print(f"\n=== Cluster {cluster} (primeras 5 frases) ===")
            for sentence in sents[:5]:
                print("-", sentence)

        plt.figure(figsize=(6,4))
        cluster_counts = {k: len(v) for k, v in cluster_sentences.items()}
        plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
        plt.xlabel("Cluster")
        plt.ylabel("Cantidad de frases")
        plt.title("Distribución de frases por cluster")
        plt.show()

        return ds, self.embeddings, assigned_clusters
    
    def analyze_and_prepare_dataset(self, dataframe=None, embedding_text_columns=None, 
                                    meta_columns=None, separator=None, validate=True, drop_empty=None,
                                    label_column=None, id_column=None):
        """
        Analyze and prepare dataset with flexible column configuration.
        
        This method:
        1. Validates the dataset structure
        2. Assembles text from multiple columns for embeddings
        3. Provides warnings about data quality
        4. Returns prepared texts and metadata
        
        Args:
            dataframe (pd.DataFrame, optional): The complete dataset (already cleaned).
                If None, loads from path_dataset. If provided, uses this DataFrame instead.
            embedding_text_columns (list, optional): List of column names to use for embeddings.
                If None, uses text_columns from __init__ or defaults to ['text'].
                Columns are combined with separator.
            meta_columns (list, optional): List of column names to keep as metadata.
                If None, uses meta_columns from __init__. These columns are not used for 
                embeddings but are returned for reference.
            separator (str, optional): Separator to use when combining multiple text columns.
                If None, uses join_with from __init__ or defaults to "\\n" (newline).
            validate (bool): Whether to perform validation and show warnings.
                Defaults to True.
            drop_empty (bool, optional): Whether to remove rows with no valid text.
                If None, uses drop_empty from __init__. Defaults to None.
            label_column (str, optional): Column name containing labels for supervised learning.
                If None, uses label_column from __init__. Defaults to None.
            id_column (str, optional): Column name containing unique identifiers.
                If None, uses id_column from __init__. Defaults to None.
        
        Returns:
            dict: Dictionary with keys:
                - 'texts': List of assembled text strings ready for embedding
                - 'metadata': DataFrame with meta_columns (if provided)
                - 'labels': Series or array with labels (if label_column provided)
                - 'ids': Series or array with unique IDs (if id_column provided)
                - 'df': Original DataFrame (or filtered if drop_empty=True)
                - 'used_columns': List of columns actually used for embeddings
                - 'warnings': List of warning messages generated during validation
                - 'is_supervised': bool indicating if dataset has labels
        
        Raises:
            ValueError: If path_dataset is None and dataframe is not provided
            FileNotFoundError: If path_dataset file doesn't exist
            KeyError: If specified columns don't exist in the dataset
        """
        # Use provided dataframe or load from path
        if dataframe is not None:
            df = dataframe.copy()
        elif self.path_dataset is not None:
            # Validate that the file exists
            if not os.path.exists(self.path_dataset):
                raise FileNotFoundError(
                    f"Dataset file not found: {self.path_dataset}. "
                    f"Please check that the path is correct and the file exists."
                )
            df = pd.read_csv(self.path_dataset, sep=",", quotechar='"', dtype=str)
        else:
            raise ValueError(
                "Either path_dataset must be provided in __init__ or dataframe must be passed "
                "to analyze_and_prepare_dataset(). If you don't have a dataset, use "
                "encode_texts() method directly with a list of texts instead."
            )
        
        # Use drop_empty parameter (from method or instance)
        if drop_empty is None:
            drop_empty = self.drop_empty
        
        # Use instance attributes if parameters not provided
        if embedding_text_columns is None:
            embedding_text_columns = self.text_columns
        if meta_columns is None:
            meta_columns = self.meta_columns
        if separator is None:
            separator = self.join_with if self.join_with is not None else " "
        if label_column is None:
            label_column = self.label_column
        if id_column is None:
            id_column = self.id_column
        
        # Default columns if not specified
        using_default = False
        if embedding_text_columns is None or len(embedding_text_columns) == 0:
            embedding_text_columns = ['text']
            using_default = True
        
        warnings_list = []
        used_columns = []
        
        # Validate and prepare text columns
        if validate:
            # Check if default column exists
            if using_default:
                if 'text' not in df.columns:
                    warnings_list.append(
                        "** Warning: No embedding text columns specified. Using default 'text' column, "
                        "but it doesn't exist in the dataset. Available columns: "
                        f"{list(df.columns)}"
                    )
                    # Try to find a reasonable default
                    possible_defaults = ['content', 'body', 'description', 'summary']
                    for default_col in possible_defaults:
                        if default_col in df.columns:
                            warnings_list.append(
                                f"[TIP] Suggestion: Consider using '{default_col}' column instead."
                            )
                            break
                else:
                    warnings_list.append(
                        "** Warning: No embedding text columns specified. Using default 'text' column."
                    )
                used_columns = ['text'] if 'text' in df.columns else []
            else:
                # Validate specified columns exist
                missing_columns = [col for col in embedding_text_columns if col not in df.columns]
                if missing_columns:
                    raise KeyError(
                        f"Specified embedding columns not found in dataset: {missing_columns}. "
                        f"Available columns: {list(df.columns)}"
                    )
                
                # Validate each column
                for col in embedding_text_columns:
                    if col not in df.columns:
                        continue
                    
                    # Check if column is mostly empty
                    non_empty = df[col].notna() & (df[col].str.strip() != '')
                    empty_pct = (1 - non_empty.sum() / len(df)) * 100
                    
                    if empty_pct > 90:
                        warnings_list.append(
                            f"** Warning: Column '{col}' is present but {empty_pct:.1f}% empty."
                        )
                    elif empty_pct > 50:
                        warnings_list.append(
                            f"** Warning: Column '{col}' is {empty_pct:.1f}% empty."
                        )
                    
                    # Check average text length
                    if non_empty.sum() > 0:
                        avg_length = df[col][non_empty].str.len().mean()
                        if avg_length < 10:
                            warnings_list.append(
                                f"** Warning: Column '{col}' exists but has only {avg_length:.1f} "
                                f"characters average."
                            )
                    
                    # Check variety (unique values)
                    if non_empty.sum() > 0:
                        unique_pct = (df[col][non_empty].nunique() / non_empty.sum()) * 100
                        if unique_pct < 5:
                            warnings_list.append(
                                f"** Warning: Column '{col}' has low variety ({unique_pct:.1f}% unique values)."
                            )
                    
                    used_columns.append(col)
        else:
            # Without validation, just use specified columns
            used_columns = [col for col in embedding_text_columns if col in df.columns]
            if not used_columns:
                raise KeyError(
                    f"None of the specified columns found in dataset. "
                    f"Available columns: {list(df.columns)}"
                )
        
        # Assemble texts from multiple columns
        texts = []
        valid_indices = []
        for idx, row in df.iterrows():
            text_parts = []
            for col in used_columns:
                if col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        text_parts.append(str(value).strip())
            
            # Combine parts with separator
            combined_text = separator.join(text_parts)
            
            # Apply drop_empty logic
            if drop_empty and not combined_text.strip():
                continue  # Skip empty rows
            
            texts.append(combined_text if combined_text else "")
            valid_indices.append(idx)
        
        # Prepare metadata (only for valid indices if drop_empty was used)
        metadata = None
        if meta_columns:
            available_meta = [col for col in meta_columns if col in df.columns]
            if available_meta:
                if drop_empty and valid_indices:
                    metadata = df.loc[valid_indices, available_meta].copy()
                else:
                    metadata = df[available_meta].copy()
                missing_meta = [col for col in meta_columns if col not in df.columns]
                if missing_meta and validate:
                    warnings_list.append(
                        f"** Warning: Meta columns not found: {missing_meta}"
                    )
        
        # Validate and extract label_column (for supervised learning)
        labels = None
        is_supervised = False
        if label_column:
            if label_column not in df.columns:
                if validate:
                    warnings_list.append(
                        f"** Warning: Label column '{label_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}"
                    )
            else:
                is_supervised = True
                if drop_empty and valid_indices:
                    labels = df.loc[valid_indices, label_column].copy()
                else:
                    labels = df[label_column].copy()
                if validate:
                    # Check for missing labels
                    missing_labels = labels.isna().sum()
                    if missing_labels > 0:
                        warnings_list.append(
                            f"** Warning: {missing_labels} rows have missing labels in '{label_column}'"
                        )
        
        # Validate and extract id_column (for tracking)
        ids = None
        if id_column:
            if id_column not in df.columns:
                if validate:
                    warnings_list.append(
                        f"** Warning: ID column '{id_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}"
                    )
            else:
                if drop_empty and valid_indices:
                    ids = df.loc[valid_indices, id_column].copy()
                else:
                    ids = df[id_column].copy()
                if validate:
                    # Check for duplicate IDs
                    duplicate_ids = ids.duplicated().sum()
                    if duplicate_ids > 0:
                        warnings_list.append(
                            f"** Warning: {duplicate_ids} duplicate IDs found in '{id_column}'"
                        )
                    # Check for missing IDs
                    missing_ids = ids.isna().sum()
                    if missing_ids > 0:
                        warnings_list.append(
                            f"** Warning: {missing_ids} rows have missing IDs in '{id_column}'"
                        )
        
        # Final validation: check if we have any valid texts
        non_empty_texts = [t for t in texts if t.strip()]
        if len(non_empty_texts) == 0:
            warnings_list.append(
                "[WARNING] Warning: No valid texts found after assembling columns. "
                "All rows resulted in empty strings."
            )
        elif len(non_empty_texts) < len(texts) * 0.1:
            empty_pct = (1 - len(non_empty_texts) / len(texts)) * 100
            warnings_list.append(
                f"[WARNING] Warning: {empty_pct:.1f}% of rows resulted in empty texts after assembly."
            )
        
        # Print warnings if validation is enabled
        if validate and warnings_list:
            print("\n" + "=" * 70)
            print("[*] DATASET VALIDATION WARNINGS")
            print("=" * 70)
            for warning in warnings_list:
                print(warning)
            print("=" * 70 + "\n")
        
        # Update df if rows were dropped
        if drop_empty and valid_indices and len(valid_indices) < len(df):
            df = df.loc[valid_indices].copy().reset_index(drop=True)
        
        return {
            'texts': texts,
            'metadata': metadata,
            'labels': labels,
            'ids': ids,
            'df': df,
            'used_columns': used_columns,
            'warnings': warnings_list,
            'is_supervised': is_supervised
        }
    
    def encode_texts(self, texts, max_length=None, normalize=None):
        """
        Encode a list of texts or a DataFrame into embeddings.
        
        Args:
            texts: List of strings to encode, or pd.DataFrame (if DataFrame, uses text_columns
                and join_with from __init__ to assemble texts)
            max_length (int, optional): Maximum number of tokens to encode. If None, uses
                max_length from __init__. Defaults to None.
            normalize (bool, optional): Whether to normalize embeddings. If None, uses
                normalize from __init__. Defaults to None.
            
        Returns:
            torch.Tensor: Embeddings [num_texts, embedding_dim] (normalized if normalize=True)
        """
        # Handle DataFrame input
        if isinstance(texts, pd.DataFrame):
            # Use analyze_and_prepare_dataset to process DataFrame
            result = self.analyze_and_prepare_dataset(dataframe=texts, validate=False)
            texts = result['texts']
        
        if not texts or len(texts) == 0:
            return torch.empty(0, self.embedding_dim)
        
        # Use parameters from method or fall back to instance attributes
        max_len = max_length if max_length is not None else self.max_length
        do_normalize = normalize if normalize is not None else self.normalize
        
        # Set max_seq_length on the model if different from current
        if hasattr(self.model, 'max_seq_length') and self.model.max_seq_length != max_len:
            self.model.max_seq_length = max_len
        
        # Encode texts using SentenceTransformer
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=do_normalize
        )
        
        # Manual normalization fallback (if normalize_embeddings parameter didn't work)
        if do_normalize:
            # Check if embeddings are already normalized (some models do this automatically)
            # If not, normalize manually
            norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Ensure embeddings are on CPU to avoid device issues
        # (especially if SentenceTransformer model is on MPS/GPU)
        embeddings = embeddings.cpu()
        
        return embeddings

