"""
Base interface for link prediction methods.
"""

from abc import ABC, abstractmethod
import networkx as nx
from typing import List, Tuple


class BaseLinkPredictor(ABC):
    """Base class for link prediction methods."""
    
    @abstractmethod
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
        pass
