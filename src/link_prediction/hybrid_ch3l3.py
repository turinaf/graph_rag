"""
Hybrid Link Predictor implementation for LP-RAG.

Implements training-free link prediction using a combination of:
1. Content Similarity (Embedding Cosine Similarity)
2. Graph Topology (Adamic-Adar / Common Neighbors)
3. Triadic Closure (Query -> Anchor -> Chunk)
"""

import networkx as nx
import numpy as np
import math
from typing import List, Tuple, Dict, Set
from .base import BaseLinkPredictor

class CH3L3Predictor(BaseLinkPredictor):
    """
    Hybrid Link Predictor that combines content and topological signals.
    
    Logic:
    1. Anchoring: Connects the query node to the graph via top-k similar nodes (Anchors).
    2. Propagation: Scores candidate chunks based on their connectivity to these anchors.
    3. Fusion: Combines direct similarity with topological scores.
    """
    
    def __init__(self, config: dict):
        """
        Initialize predictor with config.
        
        Args:
            config: Configuration dictionary containing 'link_prediction' settings.
        """
        self.config = config
        lp_config = config.get('link_prediction', {})
        
        # Number of anchors to connect the query to (simulating observed links)
        self.anchor_k = lp_config.get('anchor_k', 10)
        
        # Weight for topological score vs content score (0.0 = only content, 1.0 = only topology)
        self.alpha = lp_config.get('alpha', 0.5)
        
        # Heuristic method: 'adamic_adar' or 'common_neighbors'
        self.heuristic = lp_config.get('heuristic', 'adamic_adar')

    def predict_links(
        self,
        graph: nx.Graph,
        query_node: int,
        candidate_nodes: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Predict link scores between query_node and candidate_nodes.
        
        Args:
            graph: NetworkX graph (containing chunks, synthetic queries, and the query_node)
            query_node: ID of the query node (must exist in graph)
            candidate_nodes: List of chunk IDs to score
            
        Returns:
            List of (node_id, score) tuples, sorted by score descending
        """
        if query_node not in graph.nodes:
            return []

        # 1. Get Query Embedding
        query_data = graph.nodes[query_node]
        query_emb = query_data.get('embedding')
        
        if query_emb is None:
            # Fallback if no embedding: return 0 scores
            return [(c, 0.0) for c in candidate_nodes]

        # 2. Identify Anchors (Step 1 of LP-RAG inference)
        # We calculate similarity to all potential anchors in the graph to "ground" the query.
        # Potential anchors = Chunks + Synthetic Queries
        
        # Optimization: In production, use a vector index. Here we iterate for simplicity/compatibility.
        potential_anchors = []
        for node in graph.nodes:
            if node != query_node and 'embedding' in graph.nodes[node]:
                potential_anchors.append(node)
        
        anchor_scores = []
        candidate_set = set(candidate_nodes)
        
        # Store base similarities for candidates to avoid re-calculation
        base_similarities = {}

        for node in potential_anchors:
            node_emb = graph.nodes[node]['embedding']
            # Cosine similarity
            sim = float(np.dot(query_emb, node_emb))
            
            anchor_scores.append((node, sim))
            
            if node in candidate_set:
                base_similarities[node] = sim

        # Select top-k anchors to form the "observed" edges
        anchor_scores.sort(key=lambda x: x[1], reverse=True)
        top_anchors = anchor_scores[:self.anchor_k]
        
        # 3. Update Graph Topology (Virtual Edges)
        # We add edges from Query -> Anchors to enable topological metrics
        # We work on the graph directly (assuming it's a copy or transient)
        for anchor, score in top_anchors:
            if score > 0.0: # Only positive correlations
                graph.add_edge(query_node, anchor, weight=score)

        # 4. Compute Topological Scores (Step 2 of LP-RAG inference)
        # We measure how well each candidate chunk is connected to the query's anchors.
        # This captures multi-hop reasoning (e.g., Query -> SynQuery -> Chunk)
        
        topo_scores = {}
        query_neighbors = set(graph.neighbors(query_node)) # These are the anchors
        
        for candidate in candidate_nodes:
            if candidate not in graph:
                continue
                
            # Skip if candidate is the query itself
            if candidate == query_node:
                continue

            # Get neighbors of the candidate chunk
            candidate_neighbors = set(graph.neighbors(candidate))
            
            # Find shared neighbors (paths of length 2: Query -> Anchor -> Candidate)
            common_neighbors = query_neighbors.intersection(candidate_neighbors)
            
            score = 0.0
            if self.heuristic == 'adamic_adar':
                # Adamic-Adar: sum(1 / log(degree(z))) for z in common neighbors
                for z in common_neighbors:
                    deg = graph.degree(z)
                    if deg > 1:
                        score += 1.0 / math.log(deg)
            else:
                # Common Neighbors: simple count
                score = float(len(common_neighbors))
            
            topo_scores[candidate] = score

        # 5. Fusion (Step 3 of LP-RAG inference)
        # Combine Content (Similarity) + Context (Topology)
        
        # Normalize topological scores to [0, 1] range for fair combination
        max_topo = max(topo_scores.values()) if topo_scores else 1.0
        if max_topo == 0: max_topo = 1.0
        
        final_results = []
        for candidate in candidate_nodes:
            if candidate not in graph:
                continue
                
            # Get components
            sim_score = base_similarities.get(candidate, 0.0)
            # Ensure sim_score is non-negative for combination
            sim_score = max(0.0, sim_score)
            
            raw_topo = topo_scores.get(candidate, 0.0)
            norm_topo = raw_topo / max_topo
            
            # Weighted combination
            # If a chunk is an anchor, it has high sim_score.
            # If a chunk is connected to many anchors (e.g. via synthetic queries), it has high topo_score.
            final_score = (1 - self.alpha) * sim_score + self.alpha * norm_topo
            
            final_results.append((candidate, final_score))

        # Sort by final score descending
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results
