"""
Multi-dataset benchmark script.

Evaluates all RAG methods on multiple datasets with R@K metrics.
"""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
from typing import Dict, List
import yaml
import numpy as np

from src.data.multi_dataset_loader import MultiDatasetLoader
from src.data.chunker import LLMChunker
from src.embeddings.encoder import EmbeddingEncoder
from src.graph.builder import GraphBuilder
from src.generation.llm import LLMClient
from src.evaluation.metrics import RAGEvaluator
from src.utils.logger import setup_logger

# Import all retrieval methods
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.hyde import HyDE
from src.retrieval.hyde_sota import HyDESOTA
from src.retrieval.lightrag import LightRAG
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.retrieval.graphrag import GraphRAG
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag import NodeRAG
from src.retrieval.node_rag_sota import NodeRAGSOTA

logger = logging.getLogger('multi_dataset_benchmark')


class MultiDatasetBenchmark:
    """Benchmark RAG methods across multiple datasets."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize benchmark with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.encoder = EmbeddingEncoder(self.config)
        self.llm_client = LLMClient(self.config)
        self.evaluator = RAGEvaluator()
        self.multi_loader = MultiDatasetLoader(self.config.get('data_dir', 'data'))
        
        # Storage
        self.results_dir = Path("data/results/multi_dataset")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset(self, dataset_name: str, max_samples: int = 50) -> Dict:
        """
        Prepare dataset for benchmarking.
        
        Returns dict with: chunks, embeddings, graph, questions
        """
        logger.info(f"Preparing {dataset_name} dataset...")
        
        # Load dataset
        documents, questions = self.multi_loader.load_dataset(
            dataset_name, 
            split='validation',
            max_samples=max_samples
        )
        
        # Chunk documents
        chunker = LLMChunker(chunk_size=self.config['chunking']['chunk_size'])
        chunks = []
        for doc in documents:
            doc_chunks = chunker.chunk_text(
                doc['text'],
                doc_id=doc['doc_id'],
                doc_title=doc['title']
            )
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Embed chunks
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.encoder.encode(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Build graph
        graph_builder = GraphBuilder(self.config['graph'])
        graph = graph_builder.build_graph(chunks, embeddings)
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'graph': graph,
            'questions': questions
        }
    
    def benchmark_method(
        self,
        method_name: str,
        dataset_data: Dict,
        enable_recall: bool = False
    ) -> Dict:
        """
        Benchmark single method on dataset.
        
        Args:
            method_name: Name of RAG method
            dataset_data: Dict with chunks, embeddings, graph, questions
            enable_recall: Whether to compute R@K metrics (requires supporting facts)
            
        Returns:
            Dict with aggregated results
        """
        logger.info(f"Benchmarking {method_name}...")
        
        # Initialize method
        method_class = self._get_method_class(method_name)
        if method_class is None:
            logger.error(f"Unknown method: {method_name}")
            return {}
        
        retriever = method_class(self.config, self.encoder, self.llm_client)
        
        # Index
        start_time = time.time()
        retriever.index(
            dataset_data['chunks'],
            dataset_data['embeddings'],
            dataset_data['graph']
        )
        index_time = time.time() - start_time
        
        # Evaluate on questions
        results = []
        for question_data in dataset_data['questions']:
            try:
                question = question_data['question']
                answer = question_data['answer']
                
                # Retrieve
                retrieval_start = time.time()
                retrieved = retriever.retrieve(question)
                retrieval_time = time.time() - retrieval_start
                
                # Generate answer
                context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved)])
                generation_start = time.time()
                prediction = self.llm_client.generate_answer(question, context)
                generation_time = time.time() - generation_start
                
                # Evaluate
                eval_result = self.evaluator.evaluate_answer(prediction, answer, question)
                
                # Count tokens
                token_count = sum(self.evaluator.count_tokens(c['text']) for c in retrieved)
                
                result = {
                    'question_id': question_data.get('question_id'),
                    'exact_match': eval_result['exact_match'],
                    'f1': eval_result['f1'],
                    'token_count': token_count,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time
                }
                
                # Compute R@K if supporting facts available
                if enable_recall and 'supporting_facts' in question_data:
                    supporting_facts = question_data['supporting_facts']
                    # Extract relevant doc titles/ids
                    relevant_docs = set()
                    for fact in supporting_facts:
                        if isinstance(fact, list) and len(fact) > 0:
                            relevant_docs.add(fact[0])  # doc title
                        elif isinstance(fact, str):
                            relevant_docs.add(fact)
                    
                    if relevant_docs:
                        recall_scores = self.evaluator.compute_recall_at_k(
                            retrieved,
                            relevant_docs,
                            k_values=[2, 5, 10]
                        )
                        result.update(recall_scores)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error on question {question_data.get('question_id')}: {e}")
                continue
        
        # Aggregate
        aggregated = self.evaluator.aggregate_results(results)
        aggregated['indexing_time'] = index_time
        aggregated['method'] = method_name
        
        logger.info(
            f"{method_name}: Accuracy={aggregated['accuracy']:.1f}%, "
            f"F1={aggregated['f1']:.1f}%, "
            f"Tokens={aggregated['avg_tokens']:.1f}"
        )
        
        return aggregated
    
    def run_multi_dataset_benchmark(
        self,
        methods: List[str],
        datasets: List[str] = ['hotpotqa', 'musique', 'multihop', 'ragqa', 'iclr'],
        samples_per_dataset: int = 50
    ):
        """
        Run benchmark on all datasets and methods.
        
        Args:
            methods: List of method names to benchmark
            datasets: List of dataset names
            samples_per_dataset: Number of samples per dataset
        """
        logger.info("=" * 80)
        logger.info("MULTI-DATASET BENCHMARK")
        logger.info("=" * 80)
        logger.info(f"Methods: {methods}")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Samples per dataset: {samples_per_dataset}")
        
        all_results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n{'='*80}")
            logger.info(f"DATASET: {dataset_name.upper()}")
            logger.info(f"{'='*80}")
            
            try:
                # Prepare dataset
                dataset_data = self.prepare_dataset(dataset_name, samples_per_dataset)
                
                # Benchmark each method
                dataset_results = {}
                for method_name in methods:
                    result = self.benchmark_method(method_name, dataset_data)
                    if result:
                        dataset_results[method_name] = result
                
                all_results[dataset_name] = dataset_results
                
                # Save intermediate results
                self._save_results(all_results)
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        # Generate summary report
        self._generate_report(all_results, methods, datasets)
        
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-DATASET BENCHMARK COMPLETE")
        logger.info("=" * 80)
    
    def _get_method_class(self, method_name: str):
        """Get retriever class by name."""
        method_map = {
            'naive_rag': NaiveRAG,
            'hyde': HyDE,
            'hyde_sota': HyDESOTA,
            'lightrag': LightRAG,
            'lightrag_sota': LightRAGSOTA,
            'graphrag': GraphRAG,
            'graphrag_sota': GraphRAGSOTA,
            'node_rag': NodeRAG,
            'node_rag_sota': NodeRAGSOTA,
        }
        return method_map.get(method_name)
    
    def _save_results(self, results: Dict):
        """Save results to JSON."""
        output_file = self.results_dir / "multi_dataset_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def _generate_report(self, results: Dict, methods: List[str], datasets: List[str]):
        """Generate markdown report with summary table."""
        report_file = self.results_dir / "MULTI_DATASET_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# Multi-Dataset Benchmark Results\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Methods**: {', '.join(methods)}\n\n")
            f.write(f"**Datasets**: {', '.join(datasets)}\n\n")
            
            # Overall summary table
            f.write("## Overall Summary\n\n")
            f.write("| Method | Avg Accuracy | Avg F1 | Avg Tokens | Avg Time (s) |\n")
            f.write("|--------|--------------|--------|------------|-------------|\n")
            
            for method in methods:
                # Compute average across datasets
                accuracies = []
                f1s = []
                tokens = []
                times = []
                
                for dataset in datasets:
                    if dataset in results and method in results[dataset]:
                        r = results[dataset][method]
                        accuracies.append(r['accuracy'])
                        f1s.append(r['f1'])
                        tokens.append(r['avg_tokens'])
                        times.append(r['avg_total_time'])
                
                if accuracies:
                    f.write(
                        f"| {method} | {np.mean(accuracies):.1f}% | "
                        f"{np.mean(f1s):.1f}% | {np.mean(tokens):.1f} | "
                        f"{np.mean(times):.3f} |\n"
                    )
            
            # Per-dataset tables
            for dataset in datasets:
                if dataset not in results:
                    continue
                
                f.write(f"\n## {dataset.upper()} Results\n\n")
                f.write("| Method | Accuracy | F1 | Tokens | Time (s) |\n")
                f.write("|--------|----------|-----|--------|----------|\n")
                
                for method in methods:
                    if method in results[dataset]:
                        r = results[dataset][method]
                        f.write(
                            f"| {method} | {r['accuracy']:.1f}% | "
                            f"{r['f1']:.1f}% | {r['avg_tokens']:.1f} | "
                            f"{r['avg_total_time']:.3f} |\n"
                        )
        
        logger.info(f"Report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset RAG benchmark")
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['naive_rag', 'hyde_sota', 'lightrag_sota', 'graphrag_sota', 'node_rag_sota'],
        help='Methods to benchmark'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['hotpotqa', 'musique', 'multihop', 'ragqa', 'iclr'],
        help='Datasets to use'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Samples per dataset'
    )
    
    args = parser.parse_args()
    
    setup_logger('INFO')
    
    benchmark = MultiDatasetBenchmark()
    benchmark.run_multi_dataset_benchmark(
        methods=args.methods,
        datasets=args.datasets,
        samples_per_dataset=args.samples
    )


if __name__ == "__main__":
    main()
