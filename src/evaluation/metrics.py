"""
Evaluation metrics for RAG benchmarking.

Metrics:
- Macro Accuracy: % of questions answered correctly
- F1 Score: Token-level F1 between prediction and ground truth
- Token Count: Average number of retrieved tokens
- Retrieval Time: Average time per retrieval
- Recall@K: % of questions with relevant docs in top-K retrieved
"""

import re
import string
from typing import Dict, List, Optional, Set
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self, llm_client=None, use_llm_judge: bool = False):
        """
        Initialize evaluator.
        
        Args:
            llm_client: LLM client for judge evaluation
            use_llm_judge: Whether to use LLM as judge (more lenient than exact match)
        """
        self.llm_client = llm_client
        self.use_llm_judge = use_llm_judge and llm_client is not None
    
    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: str,
        question: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single answer.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            question: The question (needed for LLM judge)
            
        Returns:
            Dictionary with metrics: exact_match, f1
        """
        # Try LLM judge first if enabled
        if self.use_llm_judge and question and self.llm_client:
            llm_score = self._llm_judge(question, prediction, ground_truth)
            if llm_score is not None:
                # LLM judge provides more nuanced evaluation
                return {
                    'exact_match': llm_score,
                    'f1': llm_score
                }
        
        # Fallback to traditional metrics
        # Normalize texts
        pred_norm = self._normalize_answer(prediction)
        truth_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        exact_match = float(pred_norm == truth_norm)
        
        # Also check if ground truth is contained in prediction (common case)
        if exact_match == 0 and truth_norm in pred_norm:
            exact_match = 1.0
        
        # F1 score
        f1 = self._compute_f1(pred_norm, truth_norm)
        
        return {
            'exact_match': exact_match,
            'f1': f1
        }
    
    def _llm_judge(
        self,
        question: str,
        predicted: str,
        ground_truth: str
    ) -> Optional[float]:
        """
        Use LLM as judge to evaluate if answer is correct.
        
        Returns:
            1.0 if correct, 0.0 if incorrect, None if error
        """
        judge_prompt = f"""You are an expert judge evaluating question-answering systems.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Does the predicted answer correctly answer the question based on the ground truth?
Consider the answer CORRECT if:
- It contains the key information from the ground truth
- It's semantically equivalent even if worded differently  
- For yes/no questions, it matches the ground truth intent
- For factual questions, the core fact is present (even if in a sentence)

Respond with ONLY one word: "CORRECT" or "INCORRECT"""
        
        try:
            response = self.llm_client.generate(
                judge_prompt,
                temperature=0,
                max_tokens=10
            )
            
            response_lower = response.strip().lower()
            if 'correct' in response_lower and 'incorrect' not in response_lower:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"LLM judge error: {e}, falling back to exact match")
            return None
    
    def _normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Remove extra whitespace
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score.
        
        Args:
            prediction: Predicted answer (normalized)
            ground_truth: Ground truth answer (normalized)
            
        Returns:
            F1 score
        """
        pred_tokens = prediction.split()
        truth_tokens = ground_truth.split()
        
        if len(pred_tokens) == 0 and len(truth_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        # Count common tokens
        common = set(pred_tokens) & set(truth_tokens)
        num_common = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count.
        
        Uses simple heuristic: 4 characters ≈ 1 token
        """
        return len(text) // 4
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across multiple questions."""
        if not results:
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'avg_tokens': 0.0,
                'avg_retrieval_time': 0.0,
                'avg_generation_time': 0.0,
                'avg_total_time': 0.0,
                'recall_at_2': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
                'num_questions': 0
            }
        
        # Calculate averages
        accuracies = [r['exact_match'] for r in results if 'exact_match' in r]
        f1_scores = [r['f1'] for r in results if 'f1' in r]
        token_counts = [r['token_count'] for r in results if 'token_count' in r]
        retrieval_times = [r['retrieval_time'] for r in results if 'retrieval_time' in r]
        generation_times = [r['generation_time'] for r in results if 'generation_time' in r]
        total_times = [r['retrieval_time'] + r.get('generation_time', 0.0) for r in results]
        
        # Recall@K metrics
        recall_at_2 = [r.get('recall_at_2', 0.0) for r in results if 'recall_at_2' in r]
        recall_at_5 = [r.get('recall_at_5', 0.0) for r in results if 'recall_at_5' in r]
        recall_at_10 = [r.get('recall_at_10', 0.0) for r in results if 'recall_at_10' in r]
        
        aggregated = {
            'accuracy': np.mean(accuracies) * 100 if accuracies else 0.0,  # percentage
            'f1': np.mean(f1_scores) * 100 if f1_scores else 0.0,          # percentage
            'avg_tokens': np.mean(token_counts) if token_counts else 0.0,
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0,
            'avg_generation_time': np.mean(generation_times) if generation_times else 0.0,
            'avg_total_time': np.mean(total_times) if total_times else 0.0,
            'recall_at_2': np.mean(recall_at_2) * 100 if recall_at_2 else 0.0,
            'recall_at_5': np.mean(recall_at_5) * 100 if recall_at_5 else 0.0,
            'recall_at_10': np.mean(recall_at_10) * 100 if recall_at_10 else 0.0,
            'num_questions': len(results)
        }
        
        return aggregated
    
    def compute_recall_at_k(
        self,
        retrieved_chunks: List[Dict],
        relevant_doc_ids: Set[str],
        k_values: List[int] = [2, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@K metrics.
        
        Args:
            retrieved_chunks: List of retrieved chunks (ordered by score)
            relevant_doc_ids: Set of relevant document IDs (from supporting facts)
            k_values: List of K values to compute recall for
            
        Returns:
            Dictionary with recall_at_k for each k
        """
        if not relevant_doc_ids:
            # No ground truth, return 1.0 for all (perfect recall)
            return {f'recall_at_{k}': 1.0 for k in k_values}
        
        # Extract doc_ids from retrieved chunks
        retrieved_doc_ids = [
            chunk.get('doc_id') or chunk.get('doc_title', '')
            for chunk in retrieved_chunks
        ]
        
        recall_scores = {}
        for k in k_values:
            # Get top-k retrieved doc_ids
            top_k_docs = set(retrieved_doc_ids[:k])
            
            # Compute recall: how many relevant docs are in top-k
            relevant_retrieved = top_k_docs & relevant_doc_ids
            recall = len(relevant_retrieved) / len(relevant_doc_ids)
            
            recall_scores[f'recall_at_{k}'] = recall
        
        return recall_scores
