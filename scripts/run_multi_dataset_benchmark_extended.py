"""
Extended Multi-dataset benchmark script with custom output paths.

Evaluates all RAG methods on multiple datasets with R@K metrics.
Allows saving results to custom output directory to avoid overwriting existing data.
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

from src.preprocess.multi_dataset_loader import MultiDatasetLoader
from src.embeddings.encoder import EmbeddingEncoder
from src.graph.builder import GraphBuilder
from src.generation.llm import LLMClient
from src.evaluation.metrics import RAGEvaluator
from src.utils.logger import setup_logger

# Import all retrieval methods
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag_sota import NodeRAGSOTA

logger = logging.getLogger('multi_dataset_benchmark_extended')


class ExtendedMultiDatasetBenchmark:
    """Extended benchmark with custom output path support."""

    def __init__(self, config_path: str = "config/config.yaml", prompts_path: str = "config/prompts.yaml", output_dir: str = "data/results/multi_dataset"):
        """Initialize benchmark with configuration and custom output path."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)

        # Initialize components
        self.encoder = EmbeddingEncoder(self.config)
        self.llm_client = LLMClient(self.config)
        self.evaluator = RAGEvaluator()
        self.multi_loader = MultiDatasetLoader(self.config.get('data_dir', 'data'))

        # Storage - allow custom output directory
        self.results_dir = Path(output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, dataset_name: str, max_samples: int = None) -> Dict:
        """
        Prepare dataset for benchmarking.

        Returns dict with: chunks, embeddings, graph, questions
        """
        logger.info(f"Preparing {dataset_name} dataset...")

        # Load dataset - use full dataset if max_samples is None
        documents, questions = self.multi_loader.load_dataset(
            dataset_name,
            split='validation',
            max_samples=max_samples
        )

        # Chunk documents using simple text splitting
        chunk_size = 2048  # Max chars per chunk
        chunk_overlap = 200  # Overlap between chunks

        chunks = []
        for doc in documents:
            text = doc['text']
            doc_id = doc['doc_id']
            doc_title = doc.get('title', '')

            # Simple sliding window chunking
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                if chunk_text.strip():  # Skip empty chunks
                    chunks.append({
                        'text': chunk_text,
                        'doc_id': doc_id,
                        'doc_title': doc_title,
                        'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                        'metadata': doc.get('metadata', {})
                    })
                    chunk_id += 1

                start = end - chunk_overlap  # Move forward with overlap
                if start >= len(text):
                    break

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Embed chunks
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.encoder.encode(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Build graph (GraphBuilder expects full config, not just graph section)
        graph_builder = GraphBuilder(self.config)
        graph = graph_builder.build_chunk_graph(chunks, embeddings)
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
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting benchmark for: {method_name.upper()}")
        logger.info(f"Questions to process: {len(dataset_data['questions'])}")
        logger.info(f"{'='*60}")

        # Initialize method
        method_class = self._get_method_class(method_name)
        if method_class is None:
            logger.error(f"Unknown method: {method_name}")
            return {}

        # Some methods need prompts parameter
        # Only pass `self.prompts` to methods that accept prompts to avoid constructor mismatch
        if method_name in ['graphrag_sota']:
            retriever = method_class(self.config, self.encoder, self.llm_client, self.prompts)
        else:
            retriever = method_class(self.config, self.encoder, self.llm_client)

        # Index
        logger.info(f"Indexing {len(dataset_data['chunks'])} chunks...")
        start_time = time.time()
        retriever.index(
            dataset_data['chunks'],
            dataset_data['embeddings'],
            dataset_data['graph']
        )
        index_time = time.time() - start_time
        logger.info(f"✓ Indexing completed in {index_time:.2f}s")

        # Evaluate on questions
        results = []
        total_questions = len(dataset_data['questions'])
        for idx, question_data in enumerate(dataset_data['questions'], 1):
            try:
                question = question_data['question']
                answer = question_data['answer']

                # Log progress every 10 questions or every question if total is small
                if idx % 10 == 0 or idx == 1 or total_questions <= 20:
                    logger.info(f"Processing question {idx}/{total_questions}...")

                # Retrieve
                retrieval_start = time.time()
                retrieved = retriever.retrieve(question)
                retrieval_time = time.time() - retrieval_start

                # Generate answer using qa_prompt
                chunks_text = "\n\n".join([f"[{i+1}] {c['text'][:500]}" for i, c in enumerate(retrieved)])  # Limit text length to prevent prompt overflow
                prompt = self.prompts['qa_prompt'].format(chunks=chunks_text, query=question)

                generation_start = time.time()
                prediction = self.llm_client.generate(prompt, max_tokens=256)
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
                logger.error(f"Error on question {idx}/{total_questions} ({question_data.get('question_id')}): {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

        # Aggregate
        logger.info(f"\n✓ Completed {len(results)}/{total_questions} questions for {method_name}")
        aggregated = self.evaluator.aggregate_results(results)
        aggregated['indexing_time'] = index_time
        aggregated['method'] = method_name
        aggregated['method_specific_config'] = {
            'ppr_alpha': getattr(self.config.get('noderag', {}), 'get', lambda x, y: y)('ppr_alpha', 0.85),
            'k_neighbors': getattr(self.config.get('graph', {}), 'get', lambda x, y: y)('k_neighbors', 10),
            'leiden_resolution': getattr(self.config.get('graph', {}), 'get', lambda x, y: y)('leiden_resolution', 1.0)
        }

        logger.info(
            f"{method_name}: Accuracy={aggregated['accuracy']:.1f}%, "
            f"F1={aggregated['f1']:.1f}%, "
            f"Tokens={aggregated['avg_tokens']:.1f}"
        )

        return aggregated

    def run_multi_dataset_benchmark(
        self,
        methods: List[str],
        datasets: List[str] = ['hotpotqa_full', 'musique', '2wikimultihop'],  # Updated to match the requested datasets
        samples_per_dataset: int = None
    ):
        """
        Run comprehensive benchmark on all datasets and methods.

        Args:
            methods: List of method names to benchmark
            datasets: List of dataset names
            samples_per_dataset: Number of samples per dataset (None = entire dataset)
        """
        logger.info("=" * 100)
        logger.info("EXTENDED MULTI-DATASET RAG BENCHMARK - COMPREHENSIVE EVALUATION")
        logger.info("=" * 100)
        logger.info(f"Methods: {methods}")
        logger.info(f"Datasets: {datasets}")
        if samples_per_dataset:
            logger.info(f"Samples per dataset: {samples_per_dataset}")
        else:
            logger.info(f"Using ENTIRE dataset for each (no sampling)")
        logger.info(f"Output directory: {self.results_dir}")
        logger.info("=" * 100 + "\n")

        all_results = {}

        for dataset_idx, dataset_name in enumerate(datasets, 1):
            logger.info(f"\n{'#'*100}")
            logger.info(f"DATASET {dataset_idx}/{len(datasets)}: {dataset_name.upper()}")
            logger.info(f"{'#'*100}")

            try:
                # Prepare dataset
                dataset_data = self.prepare_dataset(dataset_name, samples_per_dataset)
                logger.info(f"Dataset loaded: {len(dataset_data['questions'])} questions, {len(dataset_data['chunks'])} chunks\n")

                # Benchmark each method
                dataset_results = {}
                for method_idx, method_name in enumerate(methods, 1):
                    logger.info(f"--- Method {method_idx}/{len(methods)}: {method_name} ---")
                    result = self.benchmark_method(method_name, dataset_data)
                    if result:
                        dataset_results[method_name] = result
                        logger.info(f"✓ {method_name}: Accuracy={result['accuracy']:.1f}%, F1={result['f1']:.1f}%\n")

                all_results[dataset_name] = dataset_results
                all_results[dataset_name]['num_questions'] = len(dataset_data['questions'])

                # Save intermediate results to custom directory
                self._save_results(all_results)

            except Exception as e:
                import traceback
                logger.error(f"Error processing {dataset_name}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                continue

        # Generate summary report in custom directory
        self._generate_report(all_results, methods, datasets)

        logger.info("\n" + "=" * 100)
        logger.info("EXTENDED MULTI-DATASET BENCHMARK COMPLETE")
        logger.info("=" * 100)

    def _get_method_class(self, method_name: str):
        """Get retriever class by name."""
        method_map = {
            'naive_rag': NaiveRAG,
            'lightrag_sota': LightRAGSOTA,
            'graphrag_sota': GraphRAGSOTA,
            'node_rag_sota': NodeRAGSOTA,
        }
        return method_map.get(method_name)

    def _save_results(self, results: Dict, custom_output_dir: str = None):
        """Save results to JSON in custom directory."""
        if custom_output_dir:
            output_dir = Path(custom_output_dir)
        else:
            output_dir = self.results_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "multi_dataset_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

        # Write a simple CSV summary: dataset, method, accuracy, f1, avg_tokens, avg_total_time, num_questions
        csv_file = output_dir / 'final_comprehensive_results.csv'
        import csv
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'method', 'accuracy', 'f1', 'avg_tokens', 'avg_total_time', 'num_questions'])
            for d, mr in results.items():
                num_q = mr.get('num_questions', '')
                for m, r in mr.items():
                    if not isinstance(r, dict):
                        continue
                    writer.writerow([
                        d,
                        m,
                        r.get('accuracy', ''),
                        r.get('f1', ''),
                        r.get('avg_tokens', ''),
                        r.get('avg_total_time', ''),
                        num_q
                    ])
        logger.info(f"CSV summary saved to {csv_file}")

    def _generate_report(self, results: Dict, methods: List[str], datasets: List[str], custom_output_dir: str = None):
        """Generate markdown report with summary table in custom directory."""
        if custom_output_dir:
            output_dir = Path(custom_output_dir)
        else:
            output_dir = self.results_dir
            
        report_file = output_dir / "MULTI_DATASET_REPORT.md"

        with open(report_file, 'w') as f:
            f.write("# Extended Multi-Dataset Benchmark Results\n\n")
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
                        if isinstance(r, dict):
                            accuracies.append(r['accuracy'])
                            f1s.append(r['f1'])
                            tokens.append(r['avg_tokens'])
                            times.append(r['avg_total_time'])

                if accuracies:
                    avg_acc = np.mean(accuracies)
                    avg_f1 = np.mean(f1s)
                    avg_tokens = np.mean(tokens)
                    avg_time = np.mean(times)
                    f.write(
                        f"| {method} | {avg_acc:.1f}% | "
                        f"{avg_f1:.1f}% | {avg_tokens:.1f} | "
                        f"{avg_time:.3f} |\n"
                    )

            # Per-dataset tables
            for dataset in datasets:
                if dataset not in results:
                    continue

                num_questions = results[dataset].get('num_questions', 'N/A')
                f.write(f"\n## {dataset.upper()} Results\n\n")
                f.write(f"**Number of Questions**: {num_questions}\n\n")
                f.write("| Method | Accuracy | F1 | Tokens | Time (s) |\n")
                f.write("|--------|----------|-----|--------|----------|\n")

                # Determine best accuracy and f1 for highlighting
                accuracies = []
                f1s = []
                for method in methods:
                    if method in results[dataset] and isinstance(results[dataset][method], dict):
                        accuracies.append(results[dataset][method]['accuracy'])
                        f1s.append(results[dataset][method]['f1'])
                best_acc = max(accuracies) if accuracies else None
                best_f1 = max(f1s) if f1s else None

                for method in methods:
                    if method in results[dataset] and isinstance(results[dataset][method], dict):
                        r = results[dataset][method]
                        acc_text = f"{r['accuracy']:.1f}%"
                        f1_text = f"{r['f1']:.1f}%"
                        if best_acc is not None and r['accuracy'] == best_acc:
                            acc_text = f"**{acc_text}**"
                        if best_f1 is not None and r['f1'] == best_f1:
                            f1_text = f"**{f1_text}**"
                        f.write(
                            f"| {method} | {acc_text} | {f1_text} | {r['avg_tokens']:.1f} | "
                            f"{r['avg_total_time']:.3f} |\n"
                        )

        logger.info(f"Report generated: {report_file}")


def main_from_args(output_dir=None, log_file=None, methods=None, datasets=None, samples_per_dataset=None):
    """
    Main function that can be called with parameters instead of command line args
    """
    import logging
    if log_file:
        actual_log_file = log_file
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_log_file = f"benchmark_progress_{timestamp}.log"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(actual_log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    # Also setup the utility logger
    setup_logger('INFO')
    
    print(f"Logging progress to: {actual_log_file}")
    
    benchmark = ExtendedMultiDatasetBenchmark(output_dir=output_dir)
    benchmark.run_multi_dataset_benchmark(
        methods=methods or ['naive_rag', 'lightrag_sota', 'graphrag_sota', 'node_rag_sota'],
        datasets=datasets or ['hotpotqa_full', 'musique', '2wikimultihop'],
        samples_per_dataset=samples_per_dataset
    )


def main():
    parser = argparse.ArgumentParser(description="Extended Multi-dataset RAG benchmark with custom output path")
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['naive_rag', 'lightrag_sota', 'graphrag_sota', 'node_rag_sota'],
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
        help='Samples per dataset (None = use entire dataset). Pass e.g., 500 or 1000 for larger sample sizes.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results/extended_benchmark',
        help='Custom output directory for results (default: data/results/extended_benchmark)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file to write progress to (default: auto-generated based on output-dir)'
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    main_from_args(
        output_dir=args.output_dir,
        log_file=args.log_file,
        methods=args.methods,
        datasets=args.datasets,
        samples_per_dataset=args.samples
    )


if __name__ == "__main__":
    main()