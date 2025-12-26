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

        Improvements:
         - Proper cosine similarity (normalized embeddings).
         - Work on a copy of the graph to avoid mutating original.
         - Add Personalized PageRank seeded by anchors as an extra topological signal.
         - Combine Adamic-Adar + PPR, normalize, then fuse with content score.
        """
        if query_node not in graph.nodes:
            return []

        lp_config = self.config.get('link_prediction', {})
        use_ppr = lp_config.get('use_ppr', True)
        ppr_alpha = lp_config.get('ppr_alpha', 0.85)
        topo_beta = lp_config.get('topo_beta', 0.5)  # balance AA vs PPR in topo fusion
        eps = 1e-12

        # Get query embedding
        query_data = graph.nodes[query_node]
        query_emb = query_data.get('embedding')
        if query_emb is None:
            return [(c, 0.0) for c in candidate_nodes]

        # Precompute norms and normalized embeddings for stability
        norms = {}
        for n, d in graph.nodes(data=True):
            emb = d.get('embedding')
            if emb is None:
                norms[n] = 0.0
            else:
                norms[n] = float(np.linalg.norm(emb)) + eps

        # Build potential anchors (nodes with embeddings)
        potential_anchors = [n for n in graph.nodes if n != query_node and norms.get(n, 0.0) > eps]

        # Compute cosine similarities (stable) between query and potentials
        q_norm = norms[query_node]
        anchor_scores = []
        candidate_set = set(candidate_nodes)
        base_similarities = {}

        for node in potential_anchors:
            node_emb = graph.nodes[node]['embedding']
            sim = float(np.dot(query_emb, node_emb) / (q_norm * norms[node]))
            # Map cosine [-1,1] -> [0,1] and clamp to >=0 (prefer positive alignment)
            sim_pos = max(0.0, (sim + 1.0) / 2.0)
            anchor_scores.append((node, sim_pos))
            if node in candidate_set:
                base_similarities[node] = sim_pos

        # Select top-k anchors (only those with positive sim)
        anchor_scores.sort(key=lambda x: x[1], reverse=True)
        top_anchors = [(n, s) for n, s in anchor_scores if s > 0.0][:max(1, self.anchor_k)]

        # Work on a shallow copy to avoid mutating user's graph
        G = graph.copy()
        # Add virtual edges from query -> anchors with weight=sim
        for anchor, score in top_anchors:
            G.add_edge(query_node, anchor, weight=float(score))

        # Compute Adamic-Adar topo score (as before but normalized later)
        aa_scores = {}
        query_neighbors = set(G.neighbors(query_node))
        for candidate in candidate_nodes:
            if candidate not in G:
                continue
            if candidate == query_node:
                continue
            candidate_neighbors = set(G.neighbors(candidate))
            common_neighbors = query_neighbors.intersection(candidate_neighbors)
            aa = 0.0
            if self.heuristic == 'adamic_adar':
                for z in common_neighbors:
                    deg = G.degree(z)
                    if deg > 1:
                        aa += 1.0 / math.log(deg + eps)
            else:
                aa = float(len(common_neighbors))
            aa_scores[candidate] = aa

        # Optionally compute Personalized PageRank seeded by anchors
        ppr_scores = {}
        if use_ppr:
            # personalization vector: anchor -> normalized anchor weight
            personalization = {}
            total_anchor_weight = sum(s for _, s in top_anchors) + eps
            for n, s in top_anchors:
                personalization[n] = float(s / total_anchor_weight)
            # ensure query_node is present in personalization (small mass) to stabilize
            personalization.setdefault(query_node, 1e-6)
            try:
                pr = nx.pagerank(G, alpha=ppr_alpha, personalization=personalization)
                for candidate in candidate_nodes:
                    ppr_scores[candidate] = pr.get(candidate, 0.0)
            except Exception:
                # fallback to zeros if PageRank fails
                for candidate in candidate_nodes:
                    ppr_scores[candidate] = 0.0
        else:
            for candidate in candidate_nodes:
                ppr_scores[candidate] = 0.0

        # Normalize AA and PPR to [0,1] for fair fusion
        def minmax_norm(d):
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            denom = (hi - lo) if hi > lo else 1.0
            return {k: (v - lo) / denom for k, v in d.items()}

        aa_norm = minmax_norm(aa_scores)
        ppr_norm = minmax_norm(ppr_scores)

        # Combined topo score = topo_beta * AA + (1-topo_beta) * PPR
        topo_scores = {}
        for candidate in candidate_nodes:
            a = aa_norm.get(candidate, 0.0)
            p = ppr_norm.get(candidate, 0.0)
            topo_scores[candidate] = topo_beta * a + (1.0 - topo_beta) * p

        # Final fusion: content (sim) vs topology (topo_scores)
        final_results = []
        for candidate in candidate_nodes:
            if candidate not in G:
                continue
            sim_score = base_similarities.get(candidate, 0.0)  # already in [0,1]
            topo_score = topo_scores.get(candidate, 0.0)      # in [0,1]
            final_score = (1.0 - self.alpha) * sim_score + self.alpha * topo_score
            final_results.append((candidate, final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results
