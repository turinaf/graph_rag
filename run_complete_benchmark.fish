#!/usr/bin/env fish
# Final Comprehensive Benchmark Runner
# This script runs all SOTA methods on all datasets

cd /home/viture-ai/Documents/Training-Free-LP-RAG

# Activate virtual environment
source venv/bin/activate.fish

# Run the comprehensive benchmark
# Default: 30 samples per dataset, all methods, all datasets
python3 scripts/run_final_benchmark.py \
    --datasets hotpotqa_full musique 2wikimultihop \
    --methods naive_rag node_rag_sota lightrag_sota hyde_sota graphrag_sota \
    --samples 30

echo ""
echo "✅ Benchmark complete! Results saved to data/results/final_comprehensive_benchmark.json"
