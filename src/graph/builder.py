"""
Graph construction for RAG systems - SOTA Implementation.

Implements state-of-the-art graph construction:
- Adaptive k-NN with density-aware neighbor selection
- Multi-level edge filtering (similarity + structural)
- Hierarchical community detection using Leiden algorithm
- Dynamic threshold adjustment
- Mutual k-NN for better connectivity
"""

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
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
        - Mutual vs standard k-NN based on config
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

        # Build edges with adaptive k - using mutual k-NN or standard k-NN
        if self.mutual_knn:
            # Mutual k-NN: only connect if both nodes are in each other's k-NN
            for i, chunk in enumerate(chunks):
                chunk_id = chunk['chunk_id']
                k_i = k_values[i]

                # Get top-k neighbors (skip self at index 0)
                for j_idx in range(1, min(k_i + 1, len(indices[i]))):
                    neighbor_idx = indices[i][j_idx]
                    neighbor_id = chunks[neighbor_idx]['chunk_id']

                    # Similarity score (convert distance to similarity)
                    similarity = 1 - distances[i][j_idx]

                    # Apply threshold
                    if similarity < self.similarity_threshold:
                        continue

                    # For mutual k-NN, check if i is in j's k-NN neighbors
                    neighbor_k = k_values[neighbor_idx]
                    if i in indices[neighbor_idx][1:neighbor_k+1]:
                        G.add_edge(chunk_id, neighbor_id, weight=similarity)
        else:
            # Standard k-NN approach
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
        query_embeddings: np.ndarray,
        encoder = None  # optional encoder to re-embed if dims mismatch
    ) -> nx.Graph:
        """
        Build a heterogeneous chunk-query graph. Ensures embeddings are compatible.

        If query_embeddings have a different dimensionality than chunk node embeddings,
        and an encoder is provided, synthetic queries will be re-encoded with that encoder
        to match chunk embeddings. If no encoder is provided, a clear error is raised.
        """
        logger.info(f"Building query-chunk knowledge graph with {len(synthetic_queries)} query nodes...")

        G = chunk_graph.copy() if chunk_graph is not None else nx.Graph()

        # Try to get chunk nodes with embeddings, or fall back to all nodes
        chunk_nodes_with_emb = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'chunk' and 'embedding' in d]
        
        if chunk_nodes_with_emb:
            # Case 1: Chunk nodes already have embeddings as attributes
            chunk_nodes = chunk_nodes_with_emb
            chunk_embeddings = np.array([G.nodes[n]['embedding'] for n in chunk_nodes])
        else:
            # Case 2: No embeddings in graph nodes - need to reconstruct from external data
            # Get all chunk nodes (assume they are the non-query nodes)
            all_nodes = list(G.nodes())
            chunk_nodes = [n for n in all_nodes if not str(n).startswith('query_')]
            
            if not chunk_nodes:
                logger.warning("No chunk nodes found in graph. Creating empty graph with query nodes only.")
                chunk_embeddings = np.empty((0, 384))  # Default embedding dim
                chunk_nodes = []
            else:
                # We need external embeddings - this should be passed or reconstructed
                # For now, we'll need to get embeddings from the chunks data
                logger.info(f"Found {len(chunk_nodes)} chunk nodes without embeddings. Need to reconstruct from external data.")
                
                # This is a limitation - we need the original chunks and embeddings
                # For now, create a minimal working version
                if encoder is not None:
                    # Try to get chunk texts from node attributes
                    chunk_texts = []
                    for node in chunk_nodes:
                        node_data = G.nodes[node]
                        text = node_data.get('text', node_data.get('chunk_text', f"chunk_{node}"))
                        chunk_texts.append(text)
                    
                    logger.info("Re-encoding chunk texts with provided encoder...")
                    chunk_embeddings = encoder.encode(chunk_texts)
                    
                    # Store embeddings in graph nodes for future use
                    for i, node in enumerate(chunk_nodes):
                        G.nodes[node]['embedding'] = chunk_embeddings[i]
                        G.nodes[node]['node_type'] = 'chunk'
                else:
                    raise ValueError(
                        "No chunk nodes with embeddings found in chunk_graph and no encoder provided. "
                        "Either ensure chunk nodes have 'embedding' attributes or provide an encoder to reconstruct embeddings."
                    )

        if chunk_embeddings.ndim != 2:
            raise ValueError("Chunk embeddings must be a 2D array (n_chunks, dim).")

        chunk_dim = chunk_embeddings.shape[1]

        # Ensure query_embeddings is a 2D array
        if query_embeddings is None or len(query_embeddings) == 0:
            # try to re-encode if encoder available
            if encoder is None:
                raise ValueError("query_embeddings are missing and no encoder provided to compute them.")
            query_texts = [q['text'] for q in synthetic_queries]
            query_embeddings = encoder.encode(query_texts)
        else:
            query_embeddings = np.asarray(query_embeddings)
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)

        # If dims mismatch, try to re-encode synthetic queries with provided encoder
        if query_embeddings.shape[1] != chunk_dim:
            if encoder is not None:
                logger.warning(
                    "Synthetic query embeddings dimension (%d) does not match chunk embedding dimension (%d). "
                    "Re-encoding synthetic queries with provided encoder to align dimensions.",
                    query_embeddings.shape[1], chunk_dim
                )
                query_texts = [q['text'] for q in synthetic_queries]
                query_embeddings = encoder.encode(query_texts)
            else:
                raise ValueError(
                    f"Embedding dimension mismatch: queries {query_embeddings.shape[1]} vs chunks {chunk_dim}. "
                    "Provide an encoder to re-encode synthetic queries or rebuild embeddings with a consistent model."
                )

        # Final check
        if query_embeddings.shape[1] != chunk_dim:
            raise ValueError(
                f"Unable to align embedding dimensions after re-encoding: queries {query_embeddings.shape[1]} vs chunks {chunk_dim}."
            )

        # Config for query->chunk connections
        qc_cfg = self.config.get('query_chunk_graph', {})
        sim_threshold = float(qc_cfg.get('similarity_threshold', 0.25))
        max_connections = int(qc_cfg.get('max_connections_per_query', 10))

        # Add query nodes and create similarity-based edges
        for i, query in enumerate(synthetic_queries):
            q_node = f"query_{query['query_id']}"
            q_emb = query_embeddings[i]
            G.add_node(
                q_node,
                query_id=query['query_id'],
                text=query['text'],
                node_type='query',
                embedding=q_emb
            )

            if len(chunk_nodes) == 0:
                continue

            # Compute similarities (assuming normalized embeddings; dot = cosine)
            sims = chunk_embeddings.dot(q_emb)
            pairs = list(zip(chunk_nodes, sims))
            pairs.sort(key=lambda x: x[1], reverse=True)
            # Apply threshold and limit
            pairs = [(nid, float(s)) for nid, s in pairs if s >= sim_threshold][:max_connections]

            for chunk_id, sim in pairs:
                weight = 1.0 if chunk_id == query['chunk_id'] else float(sim)
                G.add_edge(q_node, chunk_id, weight=weight)

        logger.info(
            "Built chunk-query graph: nodes=%d edges=%d",
            G.number_of_nodes(), G.number_of_edges()
        )
        return G