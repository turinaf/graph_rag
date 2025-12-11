"""
Full HotpotQA benchmark with Recall@K metrics.

Evaluates on complete HotpotQA validation set with supporting facts.
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
from tqdm import tqdm

from src.data.multi_dataset_loader import MultiDatasetLoader
from src.data.chunker import LLMChunker
from src.embeddings.encoder import EmbeddingEncoder
from src.graph.builder import GraphBuilder
from src.generation.llm import LLMClient
from src.evaluation.metrics import RAGEvaluator
from src.utils.logger import setup_logger

# Import retrieval methods
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.hyde_sota import HyDESOTA
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag_sota import NodeRAGSOTA

logger = logging.getLogger('hotpotqa_full_benchmark')


class HotpotQAFullBenchmark:
    """Full HotpotQA benchmark with R@K metrics."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize benchmark."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.encoder = EmbeddingEncoder(self.config)
        self.llm_client = LLMClient(self.config)
        self.evaluator = RAGEvaluator()
        self.loader = MultiDatasetLoader(self.config.get('data_dir', 'data'))
        
        self.results_dir = Path("data/results/hotpotqa_full")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(
        self,
        methods: List[str],
        max_questions: int = None
    ):
        """
        Run full HotpotQA benchmark.
        
        Args:
            methods: List of method names
            max_questions: Limit number of questions (None = all)
        """
        logger.info("=" * 80)
        logger.info("FULL HOTPOTQA BENCHMARK WITH RECALL@K")
        logger.info("=" * 80)
        
        # Load full dataset
        logger.info("Loading full HotpotQA dataset...")
        documents, questions = self.loader.load_dataset(
            'hotpotqa_full',
            split='validation',
            max_samples=max_questions
        )
        
        logger.info(f"Loaded {len(documents)} documents, {len(questions)} questions")
        
        # Prepare data
        logger.info("Preparing dataset...")
        chunks, embeddings, graph = self._prepare_data(documents)
        
        # Benchmark each method
        all_results = {}
        for method_name in methods:
            logger.info(f"\n{'='*80}")
            logger.info(f"METHOD: {method_name.upper()}")
            logger.info(f"{'='*80}")
            
            result = self._benchmark_method(
                method_name,
                chunks,
                embeddings,
                graph,
                questions
            )
            
            if result:
                all_results[method_name] = result
                self._save_results(all_results)
        
        # Generate report
        self._generate_report(all_results, methods)
        
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
    
    def _prepare_data(self, documents: List[Dict]):
        """Prepare chunks, embeddings, and graph."""
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
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Embed chunks
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.encoder.encode(chunk_texts)
        logger.info(f"Generated embeddings")
        
        # Build graph
        graph_builder = GraphBuilder(self.config['graph'])
        graph = graph_builder.build_graph(chunks, embeddings)
        logger.info(f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return chunks, embeddings, graph
    
    def _benchmark_method(
        self,
        method_name: str,
        chunks: List[Dict],
        embeddings,
        graph,
        questions: List[Dict]
    ) -> Dict:
        """Benchmark single method."""
        # Get method class
        method_class = self._get_method_class(method_name)
        if not method_class:
            logger.error(f"Unknown method: {method_name}")
            return {}
        
        # Initialize
        retriever = method_class(self.config, self.encoder, self.llm_client)
        
        # Index
        logger.info("Indexing...")
        start_time = time.time()
        retriever.index(chunks, embeddings, graph)
        index_time = time.time() - start_time
        logger.info(f"Indexing completed in {index_time:.2f}s")
        
        # Evaluate
        results = []
        for question_data in tqdm(questions, desc=method_name):
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
                
                # Evaluate answer
                eval_result = self.evaluator.evaluate_answer(prediction, answer, question)
                
                # Token count
                token_count = sum(self.evaluator.count_tokens(c['text']) for c in retrieved)
                
                # Compute Recall@K using supporting facts
                supporting_facts = question_data.get('supporting_facts', [])
                relevant_docs = set()
                for fact in supporting_facts:
                    if isinstance(fact, list) and len(fact) > 0:
                        relevant_docs.add(fact[0])  # doc title
                
                recall_scores = {}
                if relevant_docs:
                    recall_scores = self.evaluator.compute_recall_at_k(
                        retrieved,
                        relevant_docs,
                        k_values=[2, 5, 10]
                    )
                
                result = {
                    'question_id': question_data.get('question_id'),
                    'exact_match': eval_result['exact_match'],
                    'f1': eval_result['f1'],
                    'token_count': token_count,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    **recall_scores
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error on question {question_data.get('question_id')}: {e}")
                continue
        
        # Aggregate
        aggregated = self.evaluator.aggregate_results(results)
        aggregated['indexing_time'] = index_time
        aggregated['method'] = method_name
        
        logger.info(
            f"\nResults for {method_name}:"
            f"\n  Accuracy: {aggregated['accuracy']:.2f}%"
            f"\n  F1: {aggregated['f1']:.2f}%"
            f"\n  Recall@2: {aggregated['recall_at_2']:.2f}%"
            f"\n  Recall@5: {aggregated['recall_at_5']:.2f}%"
            f"\n  Recall@10: {aggregated['recall_at_10']:.2f}%"
            f"\n  Avg Tokens: {aggregated['avg_tokens']:.1f}"
            f"\n  Avg Time: {aggregated['avg_total_time']:.3f}s"
        )
        
        return aggregated
    
    def _get_method_class(self, method_name: str):
        """Get retriever class."""
        method_map = {
            'naive_rag': NaiveRAG,
            'hyde_sota': HyDESOTA,
            'lightrag_sota': LightRAGSOTA,
            'graphrag_sota': GraphRAGSOTA,
            'node_rag_sota': NodeRAGSOTA,
        }
        return method_map.get(method_name)
    
    def _save_results(self, results: Dict):
        """Save results to JSON."""
        output_file = self.results_dir / "full_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def _generate_report(self, results: Dict, methods: List[str]):
        """Generate markdown report."""
        report_file = self.results_dir / "FULL_HOTPOTQA_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# Full HotpotQA Benchmark Results\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main results table
            f.write("## Results Summary\n\n")
            f.write("| Method | Accuracy (%) | F1 (%) | R@2 (%) | R@5 (%) | R@10 (%) | Avg Tokens | Avg Time (s) |\n")
            f.write("|--------|--------------|--------|---------|---------|----------|------------|-------------|\n")
            
            for method in methods:
                if method in results:
                    r = results[method]
                    f.write(
                        f"| {method} | {r['accuracy']:.2f} | {r['f1']:.2f} | "
                        f"{r['recall_at_2']:.2f} | {r['recall_at_5']:.2f} | "
                        f"{r['recall_at_10']:.2f} | {r['avg_tokens']:.1f} | "
                        f"{r['avg_total_time']:.3f} |\n"
                    )
            
            # Detailed breakdown
            f.write("\n## Detailed Metrics\n\n")
            for method in methods:
                if method in results:
                    r = results[method]
                    f.write(f"### {method.upper()}\n\n")
                    f.write(f"- **Accuracy**: {r['accuracy']:.2f}%\n")
                    f.write(f"- **F1 Score**: {r['f1']:.2f}%\n")
                    f.write(f"- **Recall@2**: {r['recall_at_2']:.2f}%\n")
                    f.write(f"- **Recall@5**: {r['recall_at_5']:.2f}%\n")
                    f.write(f"- **Recall@10**: {r['recall_at_10']:.2f}%\n")
                    f.write(f"- **Avg Tokens**: {r['avg_tokens']:.1f}\n")
                    f.write(f"- **Avg Retrieval Time**: {r['avg_retrieval_time']:.3f}s\n")
                    f.write(f"- **Avg Generation Time**: {r['avg_generation_time']:.3f}s\n")
                    f.write(f"- **Total Time**: {r['avg_total_time']:.3f}s\n")
                    f.write(f"- **Indexing Time**: {r['indexing_time']:.2f}s\n")
                    f.write(f"- **Questions Evaluated**: {r['num_questions']}\n\n")
        
        logger.info(f"Report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Full HotpotQA benchmark")
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['naive_rag', 'hyde_sota', 'lightrag_sota', 'graphrag_sota', 'node_rag_sota'],
        help='Methods to benchmark'
    )
    parser.add_argument(
        '--max-questions',
        type=int,
        default=None,
        help='Limit number of questions'
    )
    
    args = parser.parse_args()
    
    setup_logger('INFO')
    
    benchmark = HotpotQAFullBenchmark()
    benchmark.run_benchmark(
        methods=args.methods,
        max_questions=args.max_questions
    )


if __name__ == "__main__":
    main()
