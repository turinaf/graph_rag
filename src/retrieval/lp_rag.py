"""
LP-RAG: Link Prediction-based RAG.

Uses link prediction to retrieve relevant chunks.
Modular design for plugging in different link predictors.
"""

import networkx as nx
import numpy as np
from typing import List, Dict
import logging
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class LPRAG(BaseRetriever):
    """
    LP-RAG: Link Prediction-based RAG.
    
    Formulates retrieval as link prediction problem.
    Supports multiple link prediction methods.
    """
    
    def __init__(
        self,
        config: dict,
        encoder,
        llm_client,
        link_predictor
    ):
        """
        Initialize LP-RAG.
        
        Args:
            config: Configuration dictionary
            encoder: Embedding encoder
            llm_client: LLM client
            link_predictor: Link prediction method
        """
        super().__init__(config, encoder, llm_client)
        self.link_predictor = link_predictor
        self.confidence_threshold = config['retrieval']['confidence_threshold']
        self.chunks = None
        self.embeddings = None
        self.graph = None
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        # Validate embedding dimensionality vs graph node embeddings (if present)
        try:
            if embeddings is not None and embeddings.ndim == 2 and graph is not None:
                emb_dim = embeddings.shape[1]
                # Check a sample of graph node embeddings
                mismatched = []
                for n, d in graph.nodes(data=True):
                    e = d.get('embedding')
                    if e is not None and len(e) != emb_dim:
                        mismatched.append((n, len(e)))
                        break
                if mismatched:
                    logger.warning(
                        "Detected embedding dimension mismatch between provided embeddings (%d) and graph node '%s' (%d). "
                        "This can cause dot-product / similarity errors at retrieval. Rebuild graph or embeddings to match.",
                        emb_dim, mismatched[0][0], mismatched[0][1]
                    )
        except Exception:
            # keep indexing but warn
            logger.exception("Error while validating embedding dimensions during indexing.")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve chunks using link prediction.
        
        Steps:
        1. Encode query
        2. Add query as temporary node to graph
        3. Use link predictor to score query-chunk links
        4. Return top-scoring chunks above threshold
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]
        
        # Create temporary graph with query node
        temp_graph = self.graph.copy() if self.graph else nx.Graph()
        query_node_id = 'temp_query'
        
        temp_graph.add_node(
            query_node_id,
            text=query,
            node_type='query',
            embedding=query_emb
        )
        
        # Get all chunk node IDs
        chunk_node_ids = list(self.chunks.keys())
        
        # Predict links using link predictor
        try:
            link_scores = self.link_predictor.predict_links(
                temp_graph,
                query_node_id,
                chunk_node_ids
            )
        except Exception as e:
            logger.warning(f"Link prediction failed: {e}. Falling back to similarity.")
            # Fallback: use embedding similarity
            link_scores = []
            for chunk_id in chunk_node_ids:
                chunk_idx = list(self.chunks.keys()).index(chunk_id)
                similarity = float(np.dot(self.embeddings[chunk_idx], query_emb))
                link_scores.append((chunk_id, similarity))
            link_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by confidence threshold; if empty, keep top-k overall
        filtered_scores = [
            (node_id, score)
            for node_id, score in link_scores
            if score >= self.confidence_threshold
        ]

        if not filtered_scores:
            filtered_scores = link_scores[: self.top_k]
        else:
            filtered_scores = filtered_scores[: self.top_k]
        
        # Build results
        results = []
        for chunk_id, score in filtered_scores:
            chunk = self.chunks[chunk_id]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': float(score)
            })
        
        # Trim by tokens
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"LP-RAG retrieval: considered={len(chunk_node_ids)}, above_threshold={len(filtered_scores)}, returned={len(results)}"
        )
        return results
