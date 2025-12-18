#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE BENCHMARK WITH CUSTOM OUTPUT PATH

This script runs comprehensive benchmarking of all RAG methods on all datasets.
Methods: naive_rag, node_rag_sota, lightrag_sota, graphrag_sota (excluding HyDE and LP-RAG)
Datasets: hotpotqa_full, musique, 2wikimultihop
Allows custom output directory path to avoid overwriting existing results.
"""

import sys
import argparse
from pathlib import Path
import time
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run the final comprehensive benchmark."""
    parser = argparse.ArgumentParser(description='Run final comprehensive RAG benchmark')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['naive_rag', 'node_rag_sota', 'lightrag_sota', 'graphrag_sota'],
        help='Methods to benchmark (excluding HyDE and LP-RAG as requested)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['hotpotqa_full', 'musique', '2wikimultihop'],
        help='Datasets to use'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples per dataset (None = full dataset)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file to write progress to (default: auto-generated based on output-dir)'
    )
    
    args = parser.parse_args()

    # Generate default output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"data/results/final_benchmark_{timestamp}"

    print("="*80)
    print("  FINAL COMPREHENSIVE RAG BENCHMARK")
    print("="*80)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output Directory: {args.output_dir}")
    if args.samples:
        print(f"Samples per dataset: {args.samples}")
    else:
        print("Using full datasets")
    print("="*80)

    # Run the extended benchmark (no confirmation when using nohup)
    print("\nStarting comprehensive benchmark (this may take several hours)...")
    
    from scripts.run_multi_dataset_benchmark_extended import main_from_args
    
    main_from_args(
        output_dir=args.output_dir,
        log_file=args.log_file,
        methods=args.methods,
        datasets=args.datasets,
        samples_per_dataset=args.samples
    )
    
    print(f"\nBenchmark completed! Results saved to: {args.output_dir}")
    print(f"Key files created:")
    print(f"  - {args.output_dir}/MULTI_DATASET_REPORT.md")
    print(f"  - {args.output_dir}/multi_dataset_results.json")
    print(f"  - {args.output_dir}/final_comprehensive_results.csv")

if __name__ == "__main__":
    main()