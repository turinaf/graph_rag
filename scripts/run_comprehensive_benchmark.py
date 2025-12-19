"""
Comprehensive Multi-Dataset Benchmark Script
Runs all SOTA RAG methods on HotpotQA, MuSiQue, 2WikiMultiHop, and RAG-QA Arena.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess.multi_dataset_loader import MultiDatasetLoader
from src.preprocess.chunker import Chunker
from src.embeddings.encoder import Encoder
from src.graph.builder import GraphBuilder
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.hyde_sota import HyDESOTA
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag_sota import NodeRAGSOTA
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.generation.llm import LLM
from src.evaluation.metrics import compute_metrics, compute_recall_at_k
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_dataset(documents: List[Dict], config: Dict) -> tuple:
    """Prepare dataset: chunk, embed, build graph."""
    logger.info("Chunking documents...")
    chunker = Chunker(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    logger.info("Generating embeddings...")
    encoder = Encoder(
        model_name=config['embedding_model'],
        api_base=config['embedding_api_base']
    )
    embeddings = encoder.encode([c['text'] for c in chunks])
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    logger.info("Building graph...")
    graph_builder = GraphBuilder(
        similarity_threshold=config.get('similarity_threshold', 0.7)
    )
    graph = graph_builder.build_graph(chunks, embeddings)
    logger.info(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return chunks, embeddings, graph


def run_benchmark_on_dataset(
    dataset_name: str,
    methods: List[str],
    config: Dict,
    max_samples: int = 50
) -> Dict:
    """Run benchmark on a single dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARKING {dataset_name.upper()}")
    logger.info(f"{'='*80}\n")
    
    # Load dataset
    loader = MultiDatasetLoader()
    try:
        documents, questions = loader.load_dataset(dataset_name, split='dev', max_samples=max_samples)
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return {}
    
    if len(questions) == 0:
        logger.warning(f"No questions found for {dataset_name}, skipping")
        return {}
    
    # Prepare dataset
    try:
        chunks, embeddings, graph = prepare_dataset(documents, config)
    except Exception as e:
        logger.error(f"Failed to prepare {dataset_name}: {e}")
        return {}
    
    # Initialize LLM
    llm = LLM(
        model_name=config['llm_model'],
        api_base=config['llm_api_base'],
        temperature=config.get('temperature', 0.1)
    )
    
    # Load prompts
    prompts_file = Path("config/prompts.yaml")
    if prompts_file.exists():
        import yaml
        with open(prompts_file) as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = {}
    
    # Initialize methods
    method_instances = {}
    for method_name in methods:
        try:
            if method_name == 'naive_rag':
                method_instances[method_name] = NaiveRAG(chunks, embeddings, llm, config)
            elif method_name == 'hyde_sota':
                method_instances[method_name] = HyDESOTA(chunks, embeddings, llm, config, prompts)
            elif method_name == 'graphrag_sota':
                method_instances[method_name] = GraphRAGSOTA(chunks, embeddings, graph, llm, config, prompts)
            elif method_name == 'node_rag_sota':
                method_instances[method_name] = NodeRAGSOTA(chunks, embeddings, graph, llm, config)
            elif method_name == 'lightrag_sota':
                method_instances[method_name] = LightRAGSOTA(chunks, embeddings, graph, llm, config)
            
            # Index the method
            if hasattr(method_instances[method_name], 'index'):
                method_instances[method_name].index()
                logger.info(f"Indexed {method_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {method_name}: {e}")
    
    # Run benchmark for each method
    results = {}
    for method_name, retriever in method_instances.items():
        logger.info(f"\nRunning {method_name} on {dataset_name}...")
        
        predictions = []
        references = []
        query_times = []
        
        for idx, question in enumerate(questions):
            try:
                start_time = time.time()
                answer = retriever.query(question['question'])
                query_time = time.time() - start_time
                
                predictions.append(answer)
                references.append(question.get('answer', ''))
                query_times.append(query_time)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(questions)} questions")
            except Exception as e:
                logger.error(f"Error processing question {idx}: {e}")
                predictions.append('')
                references.append(question.get('answer', ''))
                query_times.append(0)
        
        # Compute metrics
        try:
            metrics = compute_metrics(predictions, references)
            metrics['avg_query_time'] = sum(query_times) / len(query_times) if query_times else 0
            metrics['num_questions'] = len(questions)
            
            results[method_name] = metrics
            logger.info(f"{method_name} - Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}")
        except Exception as e:
            logger.error(f"Failed to compute metrics for {method_name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset RAG Benchmark')
    parser.add_argument('--datasets', nargs='+', 
                       default=['hotpotqa_full', 'musique', '2wikimultihop'],
                       help='Datasets to benchmark')
    parser.add_argument('--methods', nargs='+',
                       default=['naive_rag', 'hyde_sota', 'graphrag_sota', 'node_rag_sota', 'lightrag_sota'],
                       help='Methods to benchmark')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per dataset')
    parser.add_argument('--output', type=str, default='data/results/multi_dataset_benchmark.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = {
        'chunk_size': 2048,
        'chunk_overlap': 200,
        'embedding_model': 'qwen-embedding',
        'embedding_api_base': 'http://localhost:8071/v1',
        'llm_model': 'Qwen-VLM',
        'llm_api_base': 'http://localhost:8078/v1',
        'temperature': 0.1,
        'similarity_threshold': 0.7,
        'top_k': 20,
        'max_tokens': 7200,
    }
    
    logger.info("="*80)
    logger.info("MULTI-DATASET RAG BENCHMARK")
    logger.info("="*80)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Samples per dataset: {args.samples}")
    
    # Run benchmarks
    all_results = {}
    for dataset in args.datasets:
        results = run_benchmark_on_dataset(dataset, args.methods, config, args.samples)
        all_results[dataset] = results
    
    # Create summary table
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80 + "\n")
    
    summary_data = []
    for dataset, methods_results in all_results.items():
        for method, metrics in methods_results.items():
            summary_data.append({
                'Dataset': dataset,
                'Method': method,
                'Accuracy (%)': f"{metrics.get('accuracy', 0) * 100:.1f}",
                'F1 (%)': f"{metrics.get('f1', 0) * 100:.1f}",
                'Avg Query Time (s)': f"{metrics.get('avg_query_time', 0):.3f}",
                'Num Questions': metrics.get('num_questions', 0)
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
        
        # Create pivot tables for easier comparison
        print("\n\nACCURACY COMPARISON (%):")
        pivot_acc = df.pivot(index='Method', columns='Dataset', values='Accuracy (%)')
        print(pivot_acc.to_string())
        
        print("\n\nF1 SCORE COMPARISON (%):")
        pivot_f1 = df.pivot(index='Method', columns='Dataset', values='F1 (%)')
        print(pivot_f1.to_string())
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")
    
    # Save summary table
    if summary_data:
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary table saved to: {csv_path}")
    
    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
