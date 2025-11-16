#!/usr/bin/env python3
"""
Basic Validation Script

This script runs a quick validation of the Xctopus pipeline to verify
that everything works correctly. Ideal for CI/CD or quick verification.

Usage:
    python examples/validate_basic.py [--dataset PATH] [--quick]
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / 'src'))

try:
    from xctopus import XctopusPipeline
except ImportError as e:
    print(f"[ERROR] Error importing XctopusPipeline: {e}")
    print("[TIP] Make sure you're in the correct directory")
    sys.exit(1)


def validate_basic(dataset_path=None, quick=False):
    """
    Runs basic pipeline validation
    
    Args:
        dataset_path: Path to dataset (optional, searches for a default one)
        quick: If True, uses minimal configuration for quick validation
    """
    print("=" * 80)
    print("[*] XCTOPUS PIPELINE BASIC VALIDATION")
    print("=" * 80)
    print()
    
    # Search for dataset if not specified
    if dataset_path is None:
        datasets_dir = script_dir / 'datasets'
        possible_datasets = [
            '20newsgroups_sample_200.csv',
            '20newsgroups_sample_500.csv',
            '20newsgroups.csv'
        ]
        
        for dataset_name in possible_datasets:
            candidate = datasets_dir / dataset_name
            if candidate.exists():
                dataset_path = str(candidate)
                print(f"[*] Using found dataset: {dataset_path}")
                break
        
        if dataset_path is None:
            print("[ERROR] No dataset found")
            print(f"[TIP] Searching in: {datasets_dir}")
            print("[TIP] Create a dataset or specify one with --dataset PATH")
            return False
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return False
    
    try:
        # Initialize pipeline
        print("[*] Initializing pipeline...")
        pipeline = XctopusPipeline(
            dataset_path=dataset_path,
            text_columns=['text'] if '20newsgroups' in dataset_path else None,
            auto_detect_text_columns=True
        )
        print("[OK] Pipeline initialized")
        print()
        
        # Configuration for quick validation
        if quick:
            print("[*] Quick mode: minimal configuration")
            epochs = 1
            initial_threshold = 0.7
        else:
            epochs = 3
            initial_threshold = 0.6
        
        # Run basic clustering
        print(f"[*] Running clustering (epochs={epochs})...")
        result = pipeline.run(
            step='clustering',
            epochs=epochs,
            enable_training=True,
            enable_merge=True,
            initial_threshold=initial_threshold,
            min_threshold=0.4,
            max_threshold=0.7,
            adaptive_threshold=False
        )
        print("[OK] Clustering completed")
        print()
        
        # Verify results
        nodes = pipeline.get_nodes()
        if not nodes:
            print("[WARNING] No knowledge nodes were created")
            return False
        
        print(f"[*] Knowledge Nodes created: {len(nodes)}")
        
        # Count embeddings
        total_embeddings = 0
        for cluster_id, node in nodes.items():
            if hasattr(node, 'filter') and hasattr(node.filter, 'memory'):
                memory = node.filter.memory
                if isinstance(memory, dict) and cluster_id in memory:
                    total_embeddings += len(memory[cluster_id])
        
        print(f"[*] Total embeddings processed: {total_embeddings}")
        print()
        
        # Successful validation
        print("=" * 80)
        print("[OK] VALIDATION SUCCESSFUL")
        print("=" * 80)
        print()
        print("[*] Summary:")
        print(f"   - Dataset: {dataset_path}")
        print(f"   - Clusters created: {len(nodes)}")
        print(f"   - Embeddings processed: {total_embeddings}")
        print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Basic Xctopus pipeline validation'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to CSV dataset (optional, searches automatically)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: minimal configuration (1 epoch)'
    )
    
    args = parser.parse_args()
    
    success = validate_basic(
        dataset_path=args.dataset,
        quick=args.quick
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
