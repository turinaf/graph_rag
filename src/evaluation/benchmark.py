"""
Benchmarking orchestration.

Runs full evaluation pipeline for multiple RAG methods.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from .metrics import RAGEvaluator

logger = logging.getLogger(__name__)


class RAGBenchmark:
    """Orchestrates RAG benchmarking."""
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        """
        Initialize benchmark.
        
        Args:
            config: Configuration dictionary
            llm_client: LLM client for judge evaluation
        """
        self.config = config
        use_llm_judge = config.get('benchmark', {}).get('use_llm_judge', False)
        self.evaluator = RAGEvaluator(llm_client=llm_client, use_llm_judge=use_llm_judge)
        self.output_dir = Path(config['benchmark']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = Path(config['dataset']['cache_dir']) / 'processed'
        self.storage_mb = self._compute_storage_mb(processed_dir)
    
    def run_benchmark(
        self,
        retriever_dict: Dict[str, Any],
        questions: List[Dict],
        method_name: str
    ) -> Dict[str, Any]:
        """
        Run benchmark for a single RAG method.
        
        Args:
            retriever_dict: Dictionary with 'retriever' and 'prompts'
            questions: List of questions
            method_name: Name of the RAG method
            
        Returns:
            Results dictionary with metrics and details
        """
        logger.info(f"Running benchmark for {method_name}...")
        
        retriever = retriever_dict['retriever']
        prompts = retriever_dict.get('prompts', {})
        
        results = []
        
        for i, question_data in enumerate(tqdm(questions, desc=method_name)):
            question = question_data['question']
            ground_truth = question_data['answer']
            
            try:
                # Retrieve
                start_time = time.time()
                retrieved_chunks = retriever.retrieve(question)
                retrieval_time = time.time() - start_time
                
                # Generate answer
                gen_start = time.time()
                prompt_template = prompts.get('qa_prompt') if prompts else None
                answer = retriever.generate_answer(question, retrieved_chunks, prompt_template)
                generation_time = time.time() - gen_start

                logger.info(
                    f"[{method_name}] q#{i}: retrieved={len(retrieved_chunks)} chunks in {retrieval_time:.3f}s, "
                    f"generated in {generation_time:.3f}s"
                )
                
                # Count tokens
                total_text = ' '.join([c['text'] for c in retrieved_chunks])
                token_count = self.evaluator.count_tokens(total_text)
                
                # Evaluate (pass question for LLM judge)
                eval_metrics = self.evaluator.evaluate_answer(answer, ground_truth, question=question)
                
                # Store result
                result = {
                    'question_id': i,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': answer,
                    'exact_match': eval_metrics['exact_match'],
                    'f1': eval_metrics['f1'],
                    'token_count': token_count,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'num_chunks': len(retrieved_chunks)
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                results.append({
                    'question_id': i,
                    'question': question,
                    'error': str(e),
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'token_count': 0,
                    'retrieval_time': 0.0,
                    'generation_time': 0.0
                })
        
        # Aggregate metrics
        aggregated = self.evaluator.aggregate_results(results)
        
        # Build final result
        final_result = {
            'method': method_name,
            'metrics': aggregated,
            'details': results,
            'index_time_sec': retriever_dict.get('index_time_sec', 0.0),
            'storage_mb': self.storage_mb
        }
        
        # Save results
        self._save_results(final_result, method_name)
        
        logger.info(
            f"{method_name} - Acc: {aggregated['accuracy']:.2f}%, "
            f"F1: {aggregated['f1']:.2f}%, "
            f"Tokens: {aggregated['avg_tokens']:.1f}, "
            f"Query Time: {aggregated['avg_total_time']:.3f}s"
        )
        
        return final_result
    
    def run_all_benchmarks(
        self,
        retrievers: Dict[str, Any],
        questions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run benchmarks for all methods.
        
        Args:
            retrievers: Dictionary mapping method names to retriever configs
            questions: List of questions
            
        Returns:
            Combined results for all methods
        """
        logger.info("Starting benchmarks for all methods...")
        
        all_results = {}
        
        for method_name, retriever_dict in retrievers.items():
            try:
                result = self.run_benchmark(retriever_dict, questions, method_name)
                all_results[method_name] = result
            except Exception as e:
                logger.error(f"Failed to run benchmark for {method_name}: {e}")
        
        # Create summary
        summary = self._create_summary(all_results)
        
        # Save summary
        self._save_summary(summary)
        
        logger.info("All benchmarks completed!")
        
        return {
            'summary': summary,
            'detailed_results': all_results
        }
    
    def _save_results(self, result: Dict, method_name: str) -> None:
        """Save results for a method."""
        filepath = self.output_dir / f"{method_name}_results.json"
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved results to {filepath}")

    def _compute_storage_mb(self, directory: Path) -> float:
        """Compute total storage usage (MB) for processed artifacts."""
        if not directory.exists():
            return 0.0
        total_bytes = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_bytes += path.stat().st_size
        return total_bytes / (1024 * 1024)
    
    def _create_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary table of results."""
        summary = []
        
        for method_name, result in all_results.items():
            metrics = result['metrics']
            index_time = result.get('index_time_sec', 0.0) / 60.0
            summary.append({
                'Method': method_name,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                'F1 (%)': f"{metrics['f1']:.2f}",
                'Avg Tokens': f"{metrics['avg_tokens']:.1f}",
                'Avg Query Time (s)': f"{metrics.get('avg_total_time', 0.0):.3f}",
                'Avg Retrieval Time (s)': f"{metrics.get('avg_retrieval_time', 0.0):.3f}",
                'Avg Generation Time (s)': f"{metrics.get('avg_generation_time', 0.0):.3f}",
                'Indexing Time (min)': f"{index_time:.2f}",
                'Storage (MB)': f"{self.storage_mb:.1f}",
                'Num Questions': metrics['num_questions']
            })
        
        # Sort by accuracy
        summary.sort(key=lambda x: float(x['Accuracy (%)']), reverse=True)
        
        return summary
    
    def _save_summary(self, summary: List[Dict]) -> None:
        """Save summary table."""
        filepath = self.output_dir / "benchmark_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as markdown table
        md_filepath = self.output_dir / "RESULTS.md"
        with open(md_filepath, 'w') as f:
            f.write("# RAG Benchmark Results\n\n")
            f.write("## Summary\n\n")
            
            # Table header
            f.write("| Method | Accuracy (%) | F1 (%) | Avg Tokens | Avg Query Time (s) | Avg Retrieval Time (s) | Avg Generation Time (s) | Indexing Time (min) | Storage (MB) |\n")
            f.write("|--------|-------------|--------|------------|-------------------|------------------------|-------------------------|---------------------|-------------|\n")
            
            # Table rows
            for row in summary:
                f.write(
                    f"| {row['Method']} | {row['Accuracy (%)']} | {row['F1 (%)']} | {row['Avg Tokens']} | "
                    f"{row['Avg Query Time (s)']} | {row['Avg Retrieval Time (s)']} | {row['Avg Generation Time (s)']} | "
                    f"{row['Indexing Time (min)']} | {row['Storage (MB)']} |\n"
                )
        
        logger.info(f"Saved summary to {filepath} and {md_filepath}")
