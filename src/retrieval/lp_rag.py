"""
LP-RAG: Link Prediction-based RAG.

Uses link prediction to retrieve relevant chunks.
Supports query-to-query similarity for synthetic query bridging.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set
import logging
from .base import BaseRetriever
from collections import defaultdict

logger = logging.getLogger(__name__)


class LPRAG(BaseRetriever):
    """
    LP-RAG: Link Prediction-based RAG.
    
    Enhanced retrieval strategy:
    1. Direct query→chunk via link prediction
    2. Query→synthetic_query→chunks via similarity bridging
    3. Multi-hop chunk expansion via graph neighbors
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
        
        # Enhanced retrieval config
        retrieval_config = config.get('retrieval', {})
        self.use_synthetic_queries = retrieval_config.get('use_synthetic_queries', True)
        self.synthetic_query_weight = retrieval_config.get('synthetic_query_weight', 0.7)
        self.expand_neighbors = retrieval_config.get('expand_neighbors', True)
        self.neighbor_weight_decay = retrieval_config.get('neighbor_weight_decay', 0.5)
        self.max_synthetic_queries = retrieval_config.get('max_synthetic_queries', 5)
        
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self._synthetic_query_cache = None
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        
        # Cache synthetic queries for faster lookup
        if graph is not None:
            self._cache_synthetic_queries()
        
        # Validate embedding dimensionality
        self._validate_embeddings()
    
    def _cache_synthetic_queries(self) -> None:
        """Cache synthetic query nodes and their embeddings for efficient retrieval."""
        if self.graph is None:
            self._synthetic_query_cache = None
            return
            
        synthetic_queries = []
        query_embeddings = []
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'query' and 'embedding' in data:
                synthetic_queries.append({
                    'node_id': node_id,
                    'text': data.get('text', ''),
                    'query_id': data.get('query_id', node_id)
                })
                query_embeddings.append(data['embedding'])
        
        if query_embeddings:
            self._synthetic_query_cache = {
                'queries': synthetic_queries,
                'embeddings': np.array(query_embeddings)
            }
            logger.info(f"Cached {len(synthetic_queries)} synthetic queries for retrieval")
        else:
            self._synthetic_query_cache = None
            logger.warning("No synthetic queries with embeddings found in graph")
    
    def _validate_embeddings(self) -> None:
        """Validate embedding dimensions consistency."""
        try:
            if self.embeddings is not None and self.embeddings.ndim == 2 and self.graph is not None:
                emb_dim = self.embeddings.shape[1]
                # Check graph node embeddings
                mismatched = []
                for n, d in self.graph.nodes(data=True):
                    e = d.get('embedding')
                    if e is not None and len(e) != emb_dim:
                        mismatched.append((n, len(e)))
                        break
                if mismatched:
                    logger.warning(
                        "Embedding dimension mismatch: provided embeddings (%d) vs graph node '%s' (%d)",
                        emb_dim, mismatched[0][0], mismatched[0][1]
                    )
        except Exception:
            logger.exception("Error validating embedding dimensions")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Enhanced retrieval using multi-strategy approach:
        1. Direct link prediction (query→chunks)
        2. Synthetic query bridging (query→synthetic_queries→chunks)  
        3. Neighbor expansion (chunks→neighboring_chunks)
        """
        # Encode user query
        query_emb = self.encoder.encode([query])[0]
        
        # Strategy 1: Direct link prediction
        direct_scores = self._direct_link_prediction(query, query_emb)
        
        # Strategy 2: Synthetic query bridging (if enabled and available)
        bridged_scores = {}
        if self.use_synthetic_queries and self._synthetic_query_cache is not None:
            bridged_scores = self._synthetic_query_bridging(query_emb)
        
        # Strategy 3: Combine scores with weighted fusion
        combined_scores = self._fuse_scores(direct_scores, bridged_scores)
        
        # Strategy 4: Expand with graph neighbors (if enabled)
        if self.expand_neighbors and self.graph is not None:
            combined_scores = self._expand_with_neighbors(combined_scores)
        
        # Filter, rank, and return results
        return self._finalize_results(combined_scores)
    
    def _direct_link_prediction(self, query: str, query_emb: np.ndarray) -> Dict[str, float]:
        """Perform direct link prediction between query and chunks."""
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
            return {chunk_id: score for chunk_id, score in link_scores}
        except Exception as e:
            logger.warning(f"Link prediction failed: {e}. Using fallback similarity.")
            # Fallback: direct embedding similarity
            scores = {}
            for chunk_id in chunk_node_ids:
                if chunk_id in self.chunks:
                    chunk_idx = list(self.chunks.keys()).index(chunk_id)
                    similarity = float(np.dot(self.embeddings[chunk_idx], query_emb))
                    scores[chunk_id] = similarity
            return scores
    
    def _synthetic_query_bridging(self, query_emb: np.ndarray) -> Dict[str, float]:
        """Bridge user query to chunks via most similar synthetic queries."""
        if self._synthetic_query_cache is None:
            return {}
        
        # Find most similar synthetic queries
        syn_embeddings = self._synthetic_query_cache['embeddings']
        syn_queries = self._synthetic_query_cache['queries']
        
        # Compute query→synthetic_query similarities
        similarities = syn_embeddings.dot(query_emb)
        
        # Get top-k most similar synthetic queries
        top_indices = np.argsort(similarities)[::-1][:self.max_synthetic_queries]
        
        chunk_scores = defaultdict(float)
        
        for idx in top_indices:
            if similarities[idx] <= 0:  # Skip negative/zero similarities
                continue
                
            syn_query = syn_queries[idx]
            syn_node_id = syn_query['node_id']
            syn_similarity = similarities[idx]
            
            # Get chunks connected to this synthetic query
            if syn_node_id in self.graph:
                for neighbor in self.graph.neighbors(syn_node_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('node_type') != 'query':  # It's a chunk
                        edge_weight = self.graph[syn_node_id][neighbor].get('weight', 1.0)
                        
                        # Combined score: query→syn_query similarity * syn_query→chunk weight
                        bridged_score = syn_similarity * edge_weight * self.synthetic_query_weight
                        chunk_scores[neighbor] = max(chunk_scores[neighbor], bridged_score)
        
        logger.debug(f"Synthetic query bridging found {len(chunk_scores)} candidate chunks")
        return dict(chunk_scores)
    
    def _fuse_scores(self, direct_scores: Dict[str, float], bridged_scores: Dict[str, float]) -> Dict[str, float]:
        """Fuse direct and bridged scores using weighted combination."""
        all_chunks = set(direct_scores.keys()) | set(bridged_scores.keys())
        fused_scores = {}
        
        direct_weight = 1.0 - self.synthetic_query_weight
        bridge_weight = self.synthetic_query_weight
        
        for chunk_id in all_chunks:
            direct_score = direct_scores.get(chunk_id, 0.0)
            bridged_score = bridged_scores.get(chunk_id, 0.0)
            
            # Weighted fusion
            fused_score = direct_weight * direct_score + bridge_weight * bridged_score
            fused_scores[chunk_id] = fused_score
        
        return fused_scores
    
    def _expand_with_neighbors(self, chunk_scores: Dict[str, float]) -> Dict[str, float]:
        """Expand results by including neighboring chunks in the graph."""
        if not chunk_scores or self.graph is None:
            return chunk_scores
        
        expanded_scores = chunk_scores.copy()
        
        # For each high-scoring chunk, consider its neighbors
        for chunk_id, score in chunk_scores.items():
            if score < self.confidence_threshold:
                continue
                
            if chunk_id not in self.graph:
                continue
                
            # Get chunk neighbors (only chunk nodes, not queries)
            for neighbor in self.graph.neighbors(chunk_id):
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get('node_type') == 'query':
                    continue  # Skip query nodes
                    
                if neighbor in expanded_scores:
                    continue  # Already scored
                    
                # Propagated score: original_score * edge_weight * decay
                edge_weight = self.graph[chunk_id][neighbor].get('weight', 1.0)
                neighbor_score = score * edge_weight * self.neighbor_weight_decay
                
                expanded_scores[neighbor] = max(
                    expanded_scores.get(neighbor, 0.0),
                    neighbor_score
                )
        
        logger.debug(f"Neighbor expansion: {len(chunk_scores)} → {len(expanded_scores)} chunks")
        return expanded_scores
    
    def _finalize_results(self, chunk_scores: Dict[str, float]) -> List[Dict]:
        """Filter, rank, and format final results."""
        # Filter by confidence threshold
        filtered_scores = [
            (chunk_id, score)
            for chunk_id, score in chunk_scores.items()
            if score >= self.confidence_threshold and chunk_id in self.chunks
        ]
        
        # If no chunks pass threshold, take top-k overall
        if not filtered_scores:
            all_scores = [(cid, s) for cid, s in chunk_scores.items() if cid in self.chunks]
            filtered_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)[:self.top_k]
        else:
            filtered_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:self.top_k]
        
        # Build result objects
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
        
        # Trim by token limit
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"LP-RAG retrieval: considered={len(chunk_scores)}, "
            f"above_threshold={len([s for s in chunk_scores.values() if s >= self.confidence_threshold])}, "
            f"returned={len(results)}"
        )
        
        return results
