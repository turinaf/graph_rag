"""
HyDE: Hypothetical Document Embeddings.

Generates a hypothetical answer first, then retrieves based on that.
Reference: Gao et al. (2022)
"""

import numpy as np
import networkx as nx
from typing import List, Dict
import logging
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class HyDE(BaseRetriever):
    """
    HyDE: Hypothetical Document Embeddings.
    
    Steps:
    1. Generate hypothetical answer to query
    2. Encode hypothetical answer
    3. Retrieve chunks similar to hypothetical answer
    """
    
    def __init__(self, config: dict, encoder, llm_client, prompts: dict):
        super().__init__(config, encoder, llm_client, method_name='hyde')
        self.prompts = prompts
        self.chunks = None
        self.embeddings = None
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks and embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings
        logger.info(f"HyDE indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve chunks using hypothetical document.
        
        Steps:
        1. Generate hypothetical answer
        2. Encode it
        3. Retrieve similar chunks
        """
        # Generate hypothetical answer
        hyde_prompt = self.prompts.get('hyde_prompt', 
            "Please write a passage to answer the question. The passage should be detailed and informative.\n\nQuestion: {query}\n\nPassage:")
        hyde_prompt = hyde_prompt.format(query=query)
        
        hypothetical_doc = self.llm.generate(hyde_prompt, max_tokens=512)
        logger.debug(f"Generated hypothetical doc: {hypothetical_doc[:100]}...")
        
        # Encode hypothetical document
        hyde_emb = self.encoder.encode([hypothetical_doc])[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, hyde_emb)
        
        # Get top-k indices (retrieve more initially, trim by tokens)
        initial_k = min(20, len(self.chunks))  # Retrieve 20 chunks to fill ~7.3k token budget
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
            f"HyDE retrieval: hypo_len={len(hypothetical_doc)}, initial={len(top_indices)}, "
            f"returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results
