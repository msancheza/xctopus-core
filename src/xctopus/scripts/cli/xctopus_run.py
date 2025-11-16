#!/usr/bin/env python3
"""
Xctopus CLI - Command Line Interface for XctopusPipeline

This script provides a command-line interface to run the Xctopus pipeline.

Usage:
    xctopus-run data.csv
    xctopus-run data.csv --config config.yaml
    xctopus-run data.csv --step clustering --epochs 10
"""

import sys
import os
import argparse

try:
    from xctopus.pipeline import XctopusPipeline, PipelineConfig
except ImportError as e:
    print(f"[ERROR] Error importing XctopusPipeline: {e}")
    print("[TIP] Make sure xctopus is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Xctopus Pipeline CLI - Run the complete Xctopus workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  xctopus-run data.csv
  
  # Run with custom config
  xctopus-run data.csv --config config.yaml
  
  # Run specific step
  xctopus-run data.csv --step clustering --epochs 10
  
  # Run full pipeline skipping some steps
  xctopus-run data.csv --skip-analysis --skip-evaluation
        """
    )
    
    # Required arguments
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to dataset file (CSV or JSON)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    # Step selection
    parser.add_argument(
        '--step',
        type=str,
        default=None,
        choices=['analysis', 'clustering', 'config_update', 'optimize', 
                 'fine_tune', 'audit', 'evaluation'],
        help='Run a specific step instead of full pipeline'
    )
    
    # Skip options for full pipeline
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis step in full pipeline'
    )
    parser.add_argument(
        '--skip-config-update',
        action='store_true',
        help='Skip config_update step in full pipeline'
    )
    parser.add_argument(
        '--skip-fine-tune',
        action='store_true',
        help='Skip fine_tune step in full pipeline'
    )
    parser.add_argument(
        '--skip-optimize',
        action='store_true',
        help='Skip optimize step in full pipeline'
    )
    parser.add_argument(
        '--skip-audit',
        action='store_true',
        help='Skip audit step in full pipeline'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step in full pipeline'
    )
    
    # Common options
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs for clustering (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: current directory)'
    )
    
    # Advanced options
    parser.add_argument(
        '--save-checkpoint',
        type=str,
        default=None,
        help='Save checkpoint to specified path after execution'
    )
    
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Load checkpoint from specified path before execution'
    )
    
    parser.add_argument(
        '--export-graph',
        type=str,
        default=None,
        help='Export pipeline graph to Mermaid file (e.g., graph.mmd)'
    )
    
    parser.add_argument(
        '--text-columns',
        type=str,
        nargs='+',
        default=None,
        help='Text columns to use (overrides config and auto-detection)'
    )
    
    parser.add_argument(
        '--no-auto-detect',
        action='store_true',
        help='Disable automatic text column detection'
    )
    
    return parser.parse_args()


def validate_dataset_path(dataset_path: str) -> str:
    """Validate that dataset file exists"""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    if not dataset_path.endswith(('.csv', '.json')):
        print(f"[WARNING]  Warning: Dataset file should be CSV or JSON, got: {dataset_path}")
    
    return dataset_path


def validate_config_path(config_path: str) -> str:
    """Validate that config file exists"""
    if not os.path.exists(config_path):
        print(f"[ERROR] Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if not config_path.endswith(('.yaml', '.yml')):
        print(f"[WARNING]  Warning: Config file should be YAML, got: {config_path}")
    
    return config_path


def main():
    """Main CLI entry point"""
    args = parse_args()
    
    # Validate dataset path
    dataset_path = validate_dataset_path(args.dataset)
    
    # Validate config path if provided
    config_path = None
    if args.config:
        config_path = validate_config_path(args.config)
    
    print("=" * 70)
    print("[XCTOPUS] XCTOPUS PIPELINE CLI")
    print("=" * 70)
    print(f"[*] Dataset: {dataset_path}")
    if config_path:
        print(f"[*]  Config: {config_path}")
    print("=" * 70)
    
    try:
        # Load checkpoint if specified
        if args.load_checkpoint:
            if not os.path.exists(args.load_checkpoint):
                print(f"[ERROR] Error: Checkpoint file not found: {args.load_checkpoint}")
                sys.exit(1)
            
            print(f"[*] Loading checkpoint from: {args.load_checkpoint}")
            # Create minimal pipeline first
            pipeline = XctopusPipeline()
            metadata = pipeline.load_state(args.load_checkpoint)
            print(f"[OK] Checkpoint loaded (version: {metadata.get('version', 'unknown')})")
            
            # Override dataset_path if provided
            if dataset_path:
                pipeline.dataset_path = dataset_path
        else:
            # Create pipeline
            pipeline_kwargs = {}
            if config_path:
                pipeline_kwargs['config'] = config_path
            if args.text_columns:
                pipeline_kwargs['text_columns'] = args.text_columns
            if args.no_auto_detect:
                pipeline_kwargs['auto_detect_text_columns'] = False
            
            pipeline = XctopusPipeline(dataset_path, **pipeline_kwargs)
        
        # Prepare kwargs for steps
        step_kwargs = {}
        if args.epochs:
            step_kwargs['epochs'] = args.epochs
        
        # Run specific step or full pipeline
        if args.step:
            # Run specific step
            print(f"\n[*] Running step: {args.step}")
            result = pipeline.run(step=args.step, **step_kwargs)
            print(f"\n[OK] Step '{args.step}' completed successfully")
        else:
            # Run full pipeline
            print(f"\n[*] Running full pipeline...")
            result = pipeline.run_full_pipeline(
                skip_analysis=args.skip_analysis,
                skip_config_update=args.skip_config_update,
                skip_fine_tune=args.skip_fine_tune,
                skip_optimize=args.skip_optimize,
                skip_audit=args.skip_audit,
                skip_evaluation=args.skip_evaluation,
                **step_kwargs
            )
            print(f"\n[OK] Full pipeline completed successfully")
        
        # Print summary
        if isinstance(result, dict) and 'summary' in result:
            summary = result['summary']
            print(f"\n[*] Summary:")
            print(f"   Executed steps: {summary.get('total_steps', 0)}")
            print(f"   Clusters created: {summary.get('clusters_created', 0)}")
        
        # Export graph if requested
        if args.export_graph:
            print(f"\n[*] Exporting pipeline graph...")
            pipeline.export_graph_mermaid(args.export_graph)
        
        # Save checkpoint if specified
        if args.save_checkpoint:
            pipeline.save_state(args.save_checkpoint)
            print(f"\n[*] Checkpoint saved to: {args.save_checkpoint}")
        elif args.output_dir:
            # Auto-save checkpoint to output_dir
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, 'pipeline_checkpoint.ckpt')
            pipeline.save_state(checkpoint_path)
            print(f"\n[*] Checkpoint saved to: {checkpoint_path}")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Pipeline execution completed!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n[WARNING]  Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Error during pipeline execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

