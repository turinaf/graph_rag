# Training-Free RAG Benchmarking System

A comprehensive benchmarking system for evaluating training-free Retrieval-Augmented Generation (RAG) methods on HotpotQA-S dataset.

## 🎯 Overview

This project implements and benchmarks multiple state-of-the-art RAG methods **without requiring any training**:

- **NaiveRAG**: Simple embedding similarity retrieval (baseline)
- **HyDE**: Hypothetical Document Embeddings
- **GraphRAG**: Community-based graph retrieval
- **LightRAG**: Lightweight graph-based retrieval
- **NodeRAG**: SOTA baseline (86.90% accuracy on HotpotQA-S)
- **LP-RAG**: Link Prediction-based RAG (with modular link predictors)

### Target: Replicate NodeRAG Baseline

**NodeRAG Performance (from paper):**
- **Accuracy**: 86.90%
- **F1 Score**: 69.50%
- **Tokens**: ~4.0k
- **Dataset**: HotpotQA-S (20 documents, multi-hop questions)

## 📁 Project Structure

```
Training-Free-LP-RAG/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── .env.example             # Environment variables template
│   └── prompts.yaml             # LLM prompts
├── data/
│   ├── raw/                     # Downloaded HotpotQA data
│   ├── processed/               # Processed chunks, embeddings, graphs
│   └── results/                 # Benchmark results
├── src/
│   ├── data/                    # Data loading and chunking
│   ├── embeddings/              # Embedding encoder
│   ├── generation/              # LLM client
│   ├── graph/                   # Graph construction
│   ├── retrieval/               # RAG methods
│   ├── link_prediction/         # Link prediction methods
│   ├── evaluation/              # Metrics and benchmarking
│   └── utils/                   # Utilities
├── scripts/
│   ├── download_data.py         # Download and prepare HotpotQA-S
│   ├── prepare_dataset.py       # Chunk documents and build graph
│   └── run_benchmark.py         # Run benchmarks
├── logs/                        # Log files
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp config/.env.example config/.env
# Edit config/.env with your API keys and endpoints
```

### 2. Configure Services

**Edit `config/config.yaml`:**

```yaml
# LLM Configuration (Local endpoint on port 8078)
llm:
  api_base_url: "http://localhost:8078/v1"
  model_name: "gpt-4o-mini"

# Embedding Configuration (Docker container)
embedding:
  api_url: "http://localhost:8080/embed"
```

### 3. Download and Prepare Data

```bash
# Download HotpotQA and create HotpotQA-S (20 documents)
python scripts/download_data.py

# Chunk documents and build graph
python scripts/prepare_dataset.py

# Or use simple chunker (faster, no LLM needed)
python scripts/prepare_dataset.py --use-simple-chunker
```

### 4. Run Benchmarks

```bash
# Run all methods
python scripts/run_benchmark.py

# Run specific methods
python scripts/run_benchmark.py --methods naive_rag node_rag lp_rag

# Quick test with 10 questions
python scripts/run_benchmark.py --quick-test
```

## 📊 Benchmark Metrics

The system evaluates each RAG method on:

1. **Accuracy (%)**: Percentage of questions answered exactly correctly
2. **F1 Score (%)**: Token-level F1 between prediction and ground truth
3. **Avg Tokens**: Average number of tokens in retrieved context
4. **Avg Time (s)**: Average retrieval time per question

Results are saved in:
- `data/results/benchmark_summary.json` - Summary table
- `data/results/RESULTS.md` - Markdown report
- `data/results/{method}_results.json` - Detailed results per method

## 🔧 Configuration Guide

### LLM Configuration

The system uses OpenAI-style API endpoints. Configure in `config/config.yaml`:

```yaml
llm:
  provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0
  max_tokens: 4096
  api_base_url: "http://localhost:8078/v1"  # Your LLM endpoint
  timeout: 120
```

Works with:
- **Local LLMs**: vLLM, Ollama, LM Studio, Text Generation WebUI
- **Cloud APIs**: OpenAI, Groq, Anthropic, etc.

### Embedding Configuration

Configure embedding service in `config/config.yaml`:

```yaml
embedding:
  provider: "http"
  api_url: "http://localhost:8080/embed"  # Your embedding endpoint
  batch_size: 32
  max_length: 512
```

The embedding service should accept POST requests:

```python
# Request format
{
  "texts": ["text1", "text2", ...]
}

# Response format
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

### Dataset Configuration

```yaml
dataset:
  sample_size: 20           # Number of documents for HotpotQA-S
  random_seed: 42           # For reproducibility
  max_questions: 500        # Limit questions for faster testing
```

### Graph Configuration

```yaml
graph:
  k_neighbors: 5            # k for k-NN graph
  similarity_threshold: 0.7 # Minimum edge similarity
  mutual_knn: true          # Use mutual k-NN
```

### Link Prediction (for LP-RAG)

```yaml
link_prediction:
  method: "common_neighbors"  # Options: common_neighbors, jaccard, adamic_adar
  normalize_scores: true
```

## 📚 RAG Methods Explained

### NaiveRAG
Simple baseline that retrieves top-k chunks by embedding similarity.

### HyDE (Hypothetical Document Embeddings)
1. Generates a hypothetical answer to the query
2. Encodes the hypothetical answer
3. Retrieves chunks similar to the hypothetical answer

### GraphRAG
1. Detects communities in the chunk graph
2. Scores communities by relevance to query
3. Retrieves chunks from top communities

### LightRAG
1. Finds initial relevant chunks by similarity
2. Expands via graph neighbors (1-2 hops)
3. Re-ranks all candidates

### NodeRAG (SOTA Baseline)
1. Finds initial relevant nodes via similarity
2. Multi-hop graph expansion with score propagation
3. Combines graph structure with embedding similarity
4. **Target: 86.90% accuracy on HotpotQA-S**

### LP-RAG (Link Prediction-based)
1. Formulates retrieval as link prediction
2. Adds query as temporary node to graph
3. Predicts links between query and chunks
4. Modular design for different link predictors:
   - **Common Neighbors**: Number of shared neighbors
   - **Jaccard Coefficient**: Normalized common neighbors
   - **Adamic-Adar**: Weighted common neighbors
   - **CH3-L3 (Placeholder)**: Your custom method here!

## 🔬 HotpotQA-S Dataset

**HotpotQA-S** is a sampled version of HotpotQA designed for focused benchmarking:

- **20 documents** randomly sampled from HotpotQA fullwiki
- **Multi-hop questions** requiring reasoning across multiple documents
- **Consistent sampling** using random seed 42
- Questions filtered to only use the sampled documents

The dataset tests:
- Multi-hop reasoning capability
- Graph-based retrieval effectiveness
- Context aggregation from multiple sources

## 🎨 Customization

### Adding a Custom RAG Method

1. Create a new file in `src/retrieval/my_method.py`
2. Inherit from `BaseRetriever`
3. Implement `index()` and `retrieve()` methods
4. Add to `scripts/run_benchmark.py`

Example:

```python
from src.retrieval.base import BaseRetriever

class MyRAG(BaseRetriever):
    def index(self, chunks, embeddings, graph=None):
        # Index your data
        pass
    
    def retrieve(self, query):
        # Implement retrieval logic
        # Return list of chunks with scores
        pass
```

### Adding a Custom Link Predictor

1. Create a new file in `src/link_prediction/my_predictor.py`
2. Inherit from `BaseLinkPredictor`
3. Implement `predict_links()` method

Example:

```python
from src.link_prediction.base import BaseLinkPredictor

class MyPredictor(BaseLinkPredictor):
    def predict_links(self, graph, query_node, candidate_nodes):
        # Compute link scores
        # Return list of (node_id, score) tuples
        pass
```

### CH3-L3 Placeholder

The LP-RAG implementation includes a placeholder for your custom CH3-L3 link prediction method in `src/link_prediction/placeholder.py`. Simply replace the implementation with your logic!

## 📈 Expected Results

Based on the NodeRAG paper, expected results on HotpotQA-S:

| Method | Accuracy (%) | F1 (%) | Avg Tokens | Notes |
|--------|-------------|--------|------------|-------|
| NaiveRAG | ~67.5 | ~22.5 | ~3.5k | Simple baseline |
| HyDE | ~72.2 | ~37.0 | ~3.8k | Better query representation |
| LightRAG | ~79.8 | ~65.0 | ~4.2k | Graph expansion |
| GraphRAG | ~87.2 | ~65.0 | ~4.5k | Community-based |
| **NodeRAG** | **86.9** | **69.5** | **~4.0k** | **SOTA baseline** |
| LP-RAG | ? | ? | ? | To be determined |

## 🐛 Troubleshooting

### LLM Connection Issues

```bash
# Test LLM endpoint
curl http://localhost:8078/v1/models

# Check logs
tail -f logs/benchmark.log
```

### Embedding Service Issues

```bash
# Test embedding endpoint
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test"]}'
```

### Memory Issues

If you encounter memory issues:

1. Reduce `max_questions` in config
2. Use `--quick-test` flag
3. Reduce `batch_size` for embeddings
4. Use `--use-simple-chunker` to skip LLM chunking

### Dataset Download Issues

If HuggingFace download fails:

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or use cache
export HF_DATASETS_CACHE=./data/raw
```

## 📝 Citation

If you use this code, please cite the relevant papers:

**NodeRAG:**
```
Xu et al. (2025). NodeRAG: Graph-based Retrieval-Augmented Generation without Training.
```

**LP-RAG:**
```
Your LP-RAG paper citation here
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Benchmarking! 🚀**
