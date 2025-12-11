"""
NaiveRAG: Simple embedding similarity retrieval.

Baseline method that retrieves top-k chunks by embedding similarity.
"""

import numpy as np
import networkx as nx
from typing import List, Dict
import logging
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class NaiveRAG(BaseRetriever):
    """
    NaiveRAG: Simple top-k retrieval by embedding similarity.
    
    Baseline method from the paper.
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='naive_rag')
        self.chunks = None
        self.embeddings = None
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks and embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings
        logger.info(f"NaiveRAG indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve chunks by embedding similarity.
        
        Steps:
        1. Encode query
        2. Compute similarity to all chunks
        3. Return top-k
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get top-k indices (retrieve more initially, trim by tokens)
        initial_k = min(20, len(self.chunks))  # Retrieve 20 chunks to fill ~7.2k token budget
        top_indices = np.argsort(similarities)[-initial_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': self.chunks[idx]['chunk_id'],
                'text': self.chunks[idx]['text'],
                'doc_id': self.chunks[idx]['doc_id'],
                'doc_title': self.chunks[idx]['doc_title'],
                'score': float(similarities[idx])
            })
        
        # Trim by tokens
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"NaiveRAG retrieval: considered={len(self.chunks)}, initial={len(top_indices)}, "
            f"returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results
