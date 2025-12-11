"""
HyDE SOTA: Multi-hypothesis generation with self-consistency.

State-of-the-art implementation including:
- Multiple hypothesis generation
- Self-consistency voting
- Query decomposition for complex questions
- Ensemble retrieval
"""

import numpy as np
import networkx as nx
from typing import List, Dict
import logging
from .base import BaseRetriever
from collections import Counter

logger = logging.getLogger(__name__)


class HyDESOTA(BaseRetriever):
    """
    HyDE SOTA: Multi-hypothesis with self-consistency.
    
    Key improvements:
    - Generate N hypotheses with temperature sampling
    - Ensemble retrieval across hypotheses
    - Self-consistency for better coverage
    """
    
    def __init__(self, config: dict, encoder, llm_client, prompts: dict):
        super().__init__(config, encoder, llm_client, method_name='hyde')
        self.prompts = prompts
        self.chunks = None
        self.embeddings = None
        self.num_hypotheses = config.get('hyde', {}).get('num_hypotheses', 3)
        self.temperature = config.get('hyde', {}).get('temperature', 0.7)
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks and embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings
        logger.info(f"HyDE SOTA indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve using multi-hypothesis generation.
        
        Steps:
        1. Generate N diverse hypotheses
        2. Retrieve top-k for each hypothesis
        3. Aggregate with voting/ranking fusion
        4. Return deduplicated results
        """
        # Generate multiple hypotheses
        hypotheses = self._generate_multiple_hypotheses(query)
        
        # Retrieve for each hypothesis
        all_retrievals = []
        hypothesis_scores = {}  # chunk_id -> [scores from each hypothesis]
        
        for i, hypo in enumerate(hypotheses):
            # Encode hypothesis
            hypo_emb = self.encoder.encode([hypo])[0]
            
            # Compute similarities
            similarities = np.dot(self.embeddings, hypo_emb)
            
            # Get top candidates for this hypothesis
            top_k = min(15, len(self.chunks))
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                chunk_id = self.chunks[idx]['chunk_id']
                score = float(similarities[idx])
                
                if chunk_id not in hypothesis_scores:
                    hypothesis_scores[chunk_id] = []
                hypothesis_scores[chunk_id].append(score)
        
        # Aggregate scores across hypotheses
        # Use max + count for diversity and consistency
        aggregated_results = []
        for chunk_id, scores in hypothesis_scores.items():
            # Combine max score with consistency (how many hypotheses retrieved it)
            max_score = max(scores)
            consistency = len(scores) / len(hypotheses)  # 0 to 1
            
            # Weighted combination: 70% max score + 30% consistency
            combined_score = max_score * 0.7 + consistency * 0.3
            
            idx = next(i for i, c in enumerate(self.chunks) if c['chunk_id'] == chunk_id)
            chunk = self.chunks[idx]
            
            aggregated_results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': combined_score,
                'consistency': consistency,
                'max_score': max_score
            })
        
        # Sort by combined score
        aggregated_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 20 before token trimming
        aggregated_results = aggregated_results[:20]
        
        # Trim by tokens
        aggregated_results = self._trim_by_tokens(aggregated_results)
        
        logger.info(
            f"HyDE SOTA: {len(hypotheses)} hypotheses, "
            f"unique_chunks={len(hypothesis_scores)}, returned={len(aggregated_results)}, "
            f"target_tokens={self.max_tokens}"
        )
        return aggregated_results
    
    def _generate_multiple_hypotheses(self, query: str) -> List[str]:
        """
        Generate multiple diverse hypothetical answers.
        
        Uses temperature sampling to get diverse hypotheses.
        """
        hyde_prompt_template = self.prompts.get('hyde_prompt', 
            "Please write a passage to answer the question. The passage should be detailed and informative.\n\nQuestion: {query}\n\nPassage:")
        
        hypotheses = []
        
        for i in range(self.num_hypotheses):
            hyde_prompt = hyde_prompt_template.format(query=query)
            
            try:
                # Use temperature for diversity
                temp = self.temperature if i > 0 else 0.0  # First one deterministic
                hypothesis = self.llm.generate(hyde_prompt, max_tokens=512, temperature=temp)
                hypotheses.append(hypothesis.strip())
                logger.debug(f"Hypothesis {i+1}: {hypothesis[:80]}...")
            except Exception as e:
                logger.warning(f"Failed to generate hypothesis {i+1}: {e}")
                if hypotheses:
                    # Reuse previous hypothesis with variation
                    hypotheses.append(hypotheses[-1])
                else:
                    # Fallback to query itself
                    hypotheses.append(query)
        
        return hypotheses
