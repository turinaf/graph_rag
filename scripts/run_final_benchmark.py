#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE BENCHMARK
Runs all SOTA methods on all datasets and generates complete report.

Usage:
    python scripts/run_final_benchmark.py

This will:
1. Load HotpotQA Full, MuSiQue, and 2WikiMultiHop datasets
2. Run naive_rag, hyde_sota, graphrag_sota, node_rag_sota, lightrag_sota
3. Test on 30 samples per dataset
4. Generate comprehensive result tables
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multi_dataset_loader import MultiDatasetLoader
from src.embeddings.encoder import EmbeddingEncoder
from src.graph.builder import GraphBuilder
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.hyde_sota import HyDESOTA
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag_sota import NodeRAGSOTA
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.generation.llm import LLMClient
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(documents, config):
    """Chunk, embed, and build graph."""
    # Simple text-based chunking
    chunks = []
    for doc in documents:
        text = doc.get('text', '')
        title = doc.get('title', '')
        chunk_size = config['chunk_size']
        overlap = config['chunk_overlap']
        
        # Split into chunks
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if len(chunk_text) < 100:  # Skip very small chunks
                continue
            chunks.append({
                'text': chunk_text,
                'doc_id': doc.get('doc_id', ''),
                'title': title,
                'chunk_id': f"{doc.get('doc_id', '')}_{len(chunks)}"
            })
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Create encoder config
    encoder_config = {
        'embedding': {
            'api_url': config['embedding_api_base'] + '/embeddings',
            'model_name': config['embedding_model'],
            'batch_size': 32,
            'provider': 'vllm'
        }
    }
    encoder = EmbeddingEncoder(encoder_config)
    embeddings = encoder.encode([c['text'] for c in chunks])
    
    builder = GraphBuilder(similarity_threshold=config.get('similarity_threshold', 0.7))
    graph = builder.build_graph(chunks, embeddings)
    
    logger.info(f"Prepared: {len(chunks)} chunks, {graph.number_of_edges()} edges")
    return chunks, embeddings, graph


def benchmark_dataset(dataset_name, methods, config, max_samples=30):
    """Benchmark all methods on one dataset."""
    logger.info(f"\n{'='*80}\nBENCHMARKING: {dataset_name.upper()}\n{'='*80}")
    
    # Load data
    loader = MultiDatasetLoader()
    try:
        documents, questions = loader.load_dataset(dataset_name, split='dev', max_samples=max_samples)
        if len(questions) == 0:
            logger.warning(f"No questions in {dataset_name}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return {}
    
    # Prepare
    try:
        chunks, embeddings, graph = prepare_data(documents, config)
    except Exception as e:
        logger.error(f"Failed to prepare {dataset_name}: {e}")
        return {}
    
    # LLM
    llm_config = {
        'llm': {
            'api_url': config['llm_api_base'] + '/chat/completions',
            'model_name': config['llm_model'],
            'temperature': 0.1
        }
    }
    llm = LLMClient(llm_config)
    
    # Prompts
    try:
        import yaml
        with open('config/prompts.yaml') as f:
            prompts = yaml.safe_load(f)
    except:
        prompts = {}
    
    # Methods
    retrievers = {}
    for method in methods:
        try:
            if method == 'naive_rag':
                retrievers[method] = NaiveRAG(chunks, embeddings, llm, config)
            elif method == 'hyde_sota':
                retrievers[method] = HyDESOTA(chunks, embeddings, llm, config, prompts)
            elif method == 'graphrag_sota':
                retrievers[method] = GraphRAGSOTA(chunks, embeddings, graph, llm, config, prompts)
            elif method == 'node_rag_sota':
                retrievers[method] = NodeRAGSOTA(chunks, embeddings, graph, llm, config)
            elif method == 'lightrag_sota':
                retrievers[method] = LightRAGSOTA(chunks, embeddings, graph, llm, config)
            
            if hasattr(retrievers[method], 'index'):
                retrievers[method].index()
        except Exception as e:
            logger.error(f"Failed to init {method}: {e}")
    
    # Benchmark
    results = {}
    for method, retriever in retrievers.items():
        logger.info(f"\nRunning {method}...")
        predictions, references, times = [], [], []
        
        for i, q in enumerate(questions):
            try:
                t0 = time.time()
                ans = retriever.query(q['question'])
                predictions.append(ans)
                references.append(q.get('answer', ''))
                times.append(time.time() - t0)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  {i + 1}/{len(questions)} done")
            except Exception as e:
                logger.error(f"Error on Q{i}: {e}")
                predictions.append('')
                references.append(q.get('answer', ''))
                times.append(0)
        
        metrics = compute_metrics(predictions, references)
        metrics['avg_time'] = sum(times) / len(times) if times else 0
        metrics['num_q'] = len(questions)
        results[method] = metrics
        
        logger.info(f"{method}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}, Time={metrics['avg_time']:.3f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa_full', 'musique', '2wikimultihop'])
    parser.add_argument('--methods', nargs='+', default=['naive_rag', 'node_rag_sota', 'lightrag_sota', 'hyde_sota', 'graphrag_sota'])
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    
    config = {
        'chunk_size': 2048,
        'chunk_overlap': 200,
        'embedding_model': 'qwen-embedding',
        'embedding_api_base': 'http://localhost:8071/v1',
        'llm_model': 'Qwen-VLM',
        'llm_api_base': 'http://localhost:8078/v1',
        'similarity_threshold': 0.7,
        'top_k': 20,
        'max_tokens': 7200,
    }
    
    logger.info("="*80)
    logger.info("FINAL COMPREHENSIVE BENCHMARK")
    logger.info("="*80)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Samples: {args.samples} per dataset")
    
    # Run all benchmarks
    all_results = {}
    for dataset in args.datasets:
        results = benchmark_dataset(dataset, args.methods, config, args.samples)
        all_results[dataset] = results
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80 + "\n")
    
    # Create table
    print(f"\n{'Dataset':<20} {'Method':<20} {'Accuracy':<12} {'F1 Score':<12} {'Avg Time (s)':<15} {'Questions':<10}")
    print("="*100)
    
    for dataset, methods_results in all_results.items():
        for method, metrics in methods_results.items():
            print(f"{dataset:<20} {method:<20} {metrics.get('accuracy', 0)*100:>10.1f}%  {metrics.get('f1', 0)*100:>10.1f}%  {metrics.get('avg_time', 0):>13.3f}  {metrics.get('num_q', 0):>10}")
    
    print("\n" + "="*100)
    
    # Save results
    output_file = Path('data/results/final_comprehensive_benchmark.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Create comparison matrices
    print("\n\nACCURACY COMPARISON (%):")
    print(f"{'Method':<20}", end='')
    for dataset in args.datasets:
        print(f"{dataset:<20}", end='')
    print()
    print("="*80)
    
    for method in args.methods:
        print(f"{method:<20}", end='')
        for dataset in args.datasets:
            acc = all_results.get(dataset, {}).get(method, {}).get('accuracy', 0) * 100
            print(f"{acc:>18.1f}  ", end='')
        print()
    
    print("\n\nF1 SCORE COMPARISON (%):")
    print(f"{'Method':<20}", end='')
    for dataset in args.datasets:
        print(f"{dataset:<20}", end='')
    print()
    print("="*80)
    
    for method in args.methods:
        print(f"{method:<20}", end='')
        for dataset in args.datasets:
            f1 = all_results.get(dataset, {}).get(method, {}).get('f1', 0) * 100
            print(f"{f1:>18.1f}  ", end='')
        print()
    
    logger.info("\n✅ BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
