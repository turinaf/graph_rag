"""
NodeRAG: Graph-based retrieval using heterogeneous graphs.

SOTA baseline from Xu et al. (2025) - 86.90% accuracy on HotpotQA-S.
"""

import networkx as nx
import numpy as np
from typing import List, Dict
import logging
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class NodeRAG(BaseRetriever):
    """
    NodeRAG: Graph-based retrieval with multi-hop expansion.
    
    This is the BASELINE to match: 86.90% accuracy on HotpotQA-S.
    
    Key features:
    - Uses heterogeneous graph structures
    - Combines graph algorithms with retrieval
    - Multi-hop expansion with re-ranking
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='node_rag')
        self.chunks = None
        self.embeddings = None
        self.graph = None
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        logger.info(f"NodeRAG indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks using NodeRAG approach.
        
        Steps:
        1. Encode query
        2. Find initial relevant nodes via similarity
        3. Multi-hop graph expansion
        4. Score propagation
        5. Re-rank and return top-k
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]
        
        # Map chunk IDs to embedding indices
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        
        # Compute similarities to all chunks
        similarities = np.dot(self.embeddings, query_emb)
        max_sim = float(np.max(similarities)) if len(similarities) else 1.0
        
        # Get initial candidates (top-10 for expansion)
        initial_k = min(10, len(self.chunks))
        initial_indices = np.argsort(similarities)[-initial_k:][::-1]
        initial_chunk_ids = [list(self.chunks.keys())[idx] for idx in initial_indices]
        
        # Score dictionary: chunk_id -> score
        scores = {cid: float(similarities[chunk_id_to_idx[cid]]) for cid in initial_chunk_ids}
        
        # Personalized PageRank over the chunk graph to mimic NodeRAG graph walk
        if self.graph and len(self.graph):
            try:
                personalization = {
                    cid: max(float(similarities[idx]), 0.0) + 1e-6
                    for cid, idx in chunk_id_to_idx.items()
                }
                ppr_scores = nx.pagerank(
                    self.graph,
                    alpha=0.85,
                    personalization=personalization,
                    weight='weight'
                )
            except Exception as e:
                logger.warning(f"PPR failed, skipping graph walk: {e}")
                ppr_scores = {}
        else:
            ppr_scores = {}

        # Expand using graph structure with score propagation
        if self.graph:
            expanded_nodes = set(initial_chunk_ids)
            
            # Multi-hop expansion (2 hops)
            for hop in range(2):
                new_nodes = set()
                
                for node_id in list(expanded_nodes):
                    if node_id in self.graph:
                        neighbors = list(self.graph.neighbors(node_id))
                        
                        for neighbor in neighbors:
                            if neighbor not in expanded_nodes:
                                # Propagate score with decay
                                decay_factor = 0.5 ** (hop + 1)
                                edge_weight = self.graph[node_id][neighbor].get('weight', 0.5)
                                
                                # Score = base similarity + propagated score
                                base_score = float(similarities[chunk_id_to_idx[neighbor]]) if neighbor in chunk_id_to_idx else 0.0
                                propagated_score = scores.get(node_id, 0.0) * edge_weight * decay_factor
                                
                                new_score = base_score + propagated_score
                                
                                if neighbor not in scores or new_score > scores[neighbor]:
                                    scores[neighbor] = new_score
                                    new_nodes.add(neighbor)
                
                expanded_nodes.update(new_nodes)
        else:
            expanded_nodes = set(initial_chunk_ids)
        
        # Build results from scored nodes (combine similarity + PPR)
        results = []
        for chunk_id in expanded_nodes:
            if chunk_id not in self.chunks:
                continue
            chunk = self.chunks[chunk_id]
            base = scores.get(chunk_id, 0.0)
            ppr = ppr_scores.get(chunk_id, 0.0)
            combined = 0.6 * base + 0.4 * ppr * max_sim
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': combined
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:15]  # Keep top 15 before token trimming to fill ~4.0k budget (most efficient)
        
        # Trim by tokens
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"NodeRAG retrieval: expanded_nodes={len(expanded_nodes)}, returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results
