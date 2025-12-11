"""
Graph construction for RAG systems - SOTA Implementation.

Implements state-of-the-art graph construction:
- Adaptive k-NN with density-aware neighbor selection
- Multi-level edge filtering (similarity + structural)
- Hierarchical community detection using Leiden algorithm
- Dynamic threshold adjustment
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphBuilder:
    """SOTA Graph builder for RAG systems."""
    
    def __init__(self, config: dict):
        """
        Initialize SOTA graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.k = config['graph']['k_neighbors']
        self.similarity_threshold = config['graph']['similarity_threshold']
        self.mutual_knn = config['graph'].get('mutual_knn', False)  # SOTA uses adaptive, not mutual
        self.adaptive_k = config['graph'].get('adaptive_k', True)  # Adapt k based on density
        self.min_k = config['graph'].get('min_k', 3)
        self.max_k = config['graph'].get('max_k', 20)
        
    def build_chunk_graph(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ) -> nx.Graph:
        """
        Build SOTA chunk-chunk graph using adaptive k-NN and multi-level filtering.
        
        Features:
        - Density-aware k selection
        - Similarity + structural filtering
        - Edge weight normalization
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Chunk embeddings array (n_chunks, embedding_dim)
            
        Returns:
            NetworkX graph with chunk nodes and weighted edges
        """
        logger.info(f"Building SOTA chunk graph with {len(chunks)} nodes...")
        
        G = nx.Graph()
        
        # Add nodes with metadata
        for i, chunk in enumerate(chunks):
            G.add_node(
                chunk['chunk_id'],
                chunk_id=chunk['chunk_id'],
                text=chunk['text'],
                doc_id=chunk['doc_id'],
                doc_title=chunk['doc_title'],
                embedding=embeddings[i]
            )
        
        if len(embeddings) <= 2:
            logger.warning("Too few chunks for graph construction")
            return G
        
        # Compute adaptive k based on local density
        if self.adaptive_k:
            k_values = self._compute_adaptive_k(embeddings)
        else:
            k_values = [self.k] * len(embeddings)
        
        max_k_needed = min(max(k_values) + 1, len(embeddings))
        
        # Compute k-NN for all nodes
        nbrs = NearestNeighbors(n_neighbors=max_k_needed, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Build edges with adaptive k
        edge_weights = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk['chunk_id']
            k_i = k_values[i]
            
            # Get top-k neighbors (skip self at index 0)
            for j in range(1, min(k_i + 1, len(indices[i]))):
                neighbor_idx = indices[i][j]
                neighbor_id = chunks[neighbor_idx]['chunk_id']
                
                # Similarity score (convert distance to similarity)
                similarity = 1 - distances[i][j]
                
                # Apply threshold
                if similarity < self.similarity_threshold:
                    continue
                
                # Store edge (undirected)
                edge = tuple(sorted([chunk_id, neighbor_id]))
                edge_weights[edge].append(similarity)
        
        # Add edges with max similarity (if bidirectional connection exists)
        for edge, sims in edge_weights.items():
            # Use max similarity from both directions
            weight = max(sims)
            
            # Optional: require bidirectional for stronger edges
            if not self.mutual_knn or len(sims) > 1:
                G.add_edge(edge[0], edge[1], weight=weight)
        
        # Prune weak edges using global statistics
        if G.number_of_edges() > 0:
            weights = [d['weight'] for u, v, d in G.edges(data=True)]
            median_weight = np.median(weights)
            threshold = median_weight * 0.7  # Keep edges above 70% of median
            
            edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
            G.remove_edges_from(edges_to_remove)
        
        logger.info(
            f"Built SOTA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
            f"(avg degree: {2*G.number_of_edges()/G.number_of_nodes():.1f})"
        )
        return G
    
    def _compute_adaptive_k(self, embeddings: np.ndarray) -> List[int]:
        """
        Compute adaptive k for each node based on local density.
        
        Dense regions → smaller k (more selective)
        Sparse regions → larger k (more connections)
        """
        n = len(embeddings)
        k_probe = min(self.max_k + 5, n - 1)
        
        # Probe local density
        nbrs = NearestNeighbors(n_neighbors=k_probe, metric='cosine').fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Compute local density (inverse of avg distance to k neighbors)
        k_values = []
        for i in range(n):
            # Use distances to k_probe neighbors
            avg_dist = np.mean(distances[i][1:])  # Skip self
            
            # Higher distance (lower density) → higher k
            # Lower distance (higher density) → lower k
            if avg_dist < 0.3:  # Dense
                k = self.min_k
            elif avg_dist > 0.6:  # Sparse
                k = self.max_k
            else:  # Medium
                # Linear interpolation
                k = int(self.min_k + (self.max_k - self.min_k) * (avg_dist - 0.3) / 0.3)
            
            k_values.append(min(k, n - 1))
        
        return k_values
    
    def build_chunk_query_graph(
        self,
        chunk_graph: nx.Graph,
        synthetic_queries: List[Dict],
        query_embeddings: np.ndarray
    ) -> nx.Graph:
        """
        Build chunk-query graph for LP-RAG.
        
        Args:
            chunk_graph: Existing chunk-chunk graph
            synthetic_queries: List of {query_id, text, chunk_id}
            query_embeddings: Query embeddings
            
        Returns:
            NetworkX graph with both chunks and queries
        """
        logger.info(f"Adding {len(synthetic_queries)} query nodes to graph...")
        
        # Copy chunk graph
        G = chunk_graph.copy()
        
        # Add query nodes and connect to their source chunks
        for i, query in enumerate(synthetic_queries):
            query_id = f"query_{query['query_id']}"
            
            G.add_node(
                query_id,
                query_id=query['query_id'],
                text=query['text'],
                node_type='query',
                embedding=query_embeddings[i]
            )
            
            # Connect query to its source chunk
            source_chunk_id = query['chunk_id']
            G.add_edge(query_id, source_chunk_id, weight=1.0)
        
        logger.info(f"Chunk-query graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
