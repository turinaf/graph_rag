"""
Classical link prediction heuristics.

Training-free methods:
- Common Neighbors
- Jaccard Coefficient  
- Adamic-Adar Index
"""

import networkx as nx
import numpy as np
from typing import List, Tuple
import logging
from .base import BaseLinkPredictor

logger = logging.getLogger(__name__)


class LinkPredictionHeuristics(BaseLinkPredictor):
    """Classical link prediction methods."""
    
    def __init__(self, method: str = 'common_neighbors', normalize: bool = True):
        """
        Initialize link predictor.
        
        Args:
            method: One of 'common_neighbors', 'jaccard', 'adamic_adar'
            normalize: Whether to normalize scores
        """
        self.method = method
        self.normalize = normalize
        
        self.methods = {
            'common_neighbors': self._common_neighbors,
            'jaccard': self._jaccard,
            'adamic_adar': self._adamic_adar
        }
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.methods.keys())}")
        
        logger.info(f"Initialized link predictor: {method}")
    
    def predict_links(
        self,
        graph: nx.Graph,
        query_node: int,
        candidate_nodes: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Predict link scores between query_node and candidate_nodes.
        
        Args:
            graph: NetworkX graph
            query_node: Source node ID
            candidate_nodes: List of candidate node IDs
            
        Returns:
            List of (node_id, score) tuples, sorted by score descending
        """
        predictor = self.methods[self.method]
        scores = []
        
        for candidate in candidate_nodes:
            score = predictor(graph, query_node, candidate)
            scores.append((candidate, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize if requested
        if self.normalize and scores:
            max_score = max(s for _, s in scores)
            if max_score > 0:
                scores = [(node, score / max_score) for node, score in scores]
        
        return scores
    
    def _common_neighbors(self, G: nx.Graph, u: int, v: int) -> float:
        """
        Common Neighbors: Number of common neighbors.
        
        Score(u,v) = |N(u) ∩ N(v)|
        """
        if u not in G or v not in G:
            return 0.0
        
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        
        return float(len(neighbors_u & neighbors_v))
    
    def _jaccard(self, G: nx.Graph, u: int, v: int) -> float:
        """
        Jaccard Coefficient: Normalized common neighbors.
        
        Score(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
        """
        if u not in G or v not in G:
            return 0.0
        
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        
        union = neighbors_u | neighbors_v
        if len(union) == 0:
            return 0.0
        
        intersection = neighbors_u & neighbors_v
        return len(intersection) / len(union)
    
    def _adamic_adar(self, G: nx.Graph, u: int, v: int) -> float:
        """
        Adamic-Adar Index: Weighted common neighbors.
        
        Score(u,v) = Σ(w∈N(u)∩N(v)) 1/log(|N(w)|)
        
        Gives more weight to common neighbors with fewer connections.
        """
        if u not in G or v not in G:
            return 0.0
        
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = neighbors_u & neighbors_v
        
        score = 0.0
        for w in common:
            degree_w = G.degree(w)
            if degree_w > 1:
                score += 1.0 / np.log(degree_w)
        
        return score
