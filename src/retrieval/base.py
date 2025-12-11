"""
Base interface for RAG retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Base class for RAG retrieval methods."""
    
    def __init__(self, config: dict, encoder, llm_client, method_name: str = None):
        """
        Initialize retriever.
        
        Args:
            config: Configuration dictionary
            encoder: Embedding encoder
            llm_client: LLM client for generation
            method_name: Name of method for loading specific token budget
        """
        self.config = config
        self.encoder = encoder
        self.llm = llm_client
        self.top_k = config['retrieval']['top_k']
        self.method_name = method_name
        
        # Get method-specific token budget if available
        method_budgets = config['retrieval'].get('method_token_budgets', {})
        if method_name and method_name in method_budgets:
            self.max_tokens = method_budgets[method_name]
        else:
            self.max_tokens = config['retrieval']['max_tokens']
        
    @abstractmethod
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """
        Index chunks for retrieval.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Chunk embeddings
            graph: Optional graph structure
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved chunks with scores
        """
        pass
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], prompt_template: str = None) -> str:
        """
        Generate answer using LLM with retrieved chunks.
        
        Args:
            query: Query string
            retrieved_chunks: List of retrieved chunks
            prompt_template: Optional custom prompt template
            
        Returns:
            Generated answer
        """
        # Format chunks as context
        context = "\n\n".join([
            f"[Chunk {i+1}] {chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Build prompt
        if prompt_template:
            prompt = prompt_template.format(chunks=context, query=query)
        else:
            prompt = f"""You are a thorough assistant responding to questions based on retrieved information.
Below, you'll find a bag of chunks and a query. Your job is to answer the query using only the information contained in the chunks. If the answer isn't in the chunks, simply answer: "I don't know."

[Bag of chunks]
{context}

[Query]
{query}

Please provide a clear, concise answer:"""
        
        logger.info(
            f"Generating answer with {len(retrieved_chunks)} chunks, approx_tokens={self._estimate_tokens(context)}"
        )
        # Generate
        answer = self.llm.generate(prompt)
        return answer
    
    def _trim_by_tokens(self, chunks: List[Dict]) -> List[Dict]:
        """Trim chunks to fit within token budget."""
        # Simple approximation: 4 chars ≈ 1 token
        total_tokens = 0
        trimmed = []
        
        for chunk in chunks:
            chunk_tokens = len(chunk['text']) // 4
            if total_tokens + chunk_tokens <= self.max_tokens:
                trimmed.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return trimmed

    def _estimate_tokens(self, text: str) -> int:
        """Rudimentary token estimate for logging."""
        return len(text) // 4
