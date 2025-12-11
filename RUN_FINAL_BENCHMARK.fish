#!/usr/bin/env fish

# FINAL COMPREHENSIVE BENCHMARK
# This is the verified working command to run all methods on all datasets

cd /home/viture-ai/Documents/Training-Free-LP-RAG

echo "================================================================================"
echo "  FINAL COMPREHENSIVE RAG BENCHMARK - ENTIRE DATASETS"
echo "================================================================================"
echo ""
echo "📊 Datasets (Full - No Sampling):"
echo "  • HotpotQA Full    : ~90,000 questions (multi-hop QA)"
echo "  • MuSiQue          : ~2,400 questions (complex reasoning)"
echo "  • 2WikiMultiHop    : ~12,500 questions (Wikipedia multi-hop)"
echo ""
echo "🔬 Methods (State-of-the-Art):"
echo "  • naive_rag        : Baseline similarity search"
echo "  • node_rag_sota    : Entity-aware heterogeneous graphs"
echo "  • lightrag_sota    : Lightweight with communities"
echo "  • hyde_sota        : Hypothetical documents (3 hypotheses)"
echo "  • graphrag_sota    : Hierarchical community RAG"
echo ""
echo "⏱️  Estimated Time: 18-24 hours (all datasets, all methods)"
echo ""
echo "📝 Progress: Verbose logging enabled - you'll see:"
echo "  • Dataset loading progress"
echo "  • Method-by-method status (X/5)"
echo "  • Question progress (every 10 questions)"
echo "  • Accuracy/F1 scores after each method"
echo ""
echo "💾 Results will be saved to:"
echo "  • data/results/multi_dataset/MULTI_DATASET_REPORT.md"
echo "  • data/results/multi_dataset/multi_dataset_results.json"
echo ""
echo "================================================================================"
echo ""
read -P "Press ENTER to start the benchmark, or Ctrl+C to cancel..."
echo ""

# Activate venv
source venv/bin/activate.fish

echo "🚀 Starting benchmark..."
echo ""

# Run the benchmark (using the working multi-dataset script)
# Optional first arg: samples (e.g., 500 or 1000). If not provided, the entire dataset is used.
set SAMPLES_ARG
# Accept numeric first arg OR --samples N OR env var BENCH_SAMPLES
if test (count $argv) -ge 1
    # First arg numeric e.g., ./RUN_FINAL_BENCHMARK.fish 500
    if string match -r '^[0-9]+$' "$argv[1]"
        set SAMPLES_ARG --samples $argv[1]
    else if test "$argv[1]" = "--samples"
        if test (count $argv) -ge 2
            if string match -r '^[0-9]+$' "$argv[2]"
                set SAMPLES_ARG --samples $argv[2]
            end
        end
    end
end

# Fallback to BENCH_SAMPLES env var (if provided and not otherwise set)
if test -z "$SAMPLES_ARG" -a -n "$BENCH_SAMPLES"
    set SAMPLES_ARG --samples $BENCH_SAMPLES
end

python3 scripts/run_multi_dataset_benchmark.py $SAMPLES_ARG \
    --datasets hotpotqa_full musique 2wikimultihop \
    --methods naive_rag node_rag_sota lightrag_sota hyde_sota graphrag_sota

echo ""
echo "========================================="
echo "✅ BENCHMARK COMPLETE!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - data/results/final_comprehensive_results.json"
echo "  - data/results/final_comprehensive_results.csv"
echo ""
