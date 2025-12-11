"""
LightRAG: Lightweight graph-based retrieval.

Combines entity extraction with graph traversal.
Simplified version without entity extraction (uses chunk graph directly).
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set
import logging
import re
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class LightRAG(BaseRetriever):
    """
    LightRAG: Lightweight graph-based retrieval.
    
    Steps:
    1. Find initial relevant chunks
    2. Expand via graph traversal (1-2 hops)
    3. Re-rank and return
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='lightrag')
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self.expansion_hops = 1  # Number of hops for graph expansion
        self.communities = []
        self.community_embeddings = []
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph

        if graph and len(graph.nodes) > 0:
            self.communities = nx.community.louvain_communities(graph)
            self.community_embeddings = self._compute_community_embeddings()
            logger.info(
                f"LightRAG indexed {len(chunks)} chunks across {len(self.communities)} communities"
            )
        else:
            logger.info(f"LightRAG indexed {len(chunks)} chunks (no graph communities detected)")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve chunks using graph expansion.
        
        Steps:
        1. Find top initial chunks by similarity
        2. Expand via graph neighbors
        3. Re-rank all candidates
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]

        # Map chunk IDs to embedding indices
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}

        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb)

        # Local path: keyword anchors + neighbor expansion
        keywords = self._extract_keywords(query)
        anchor_chunk_ids = self._find_anchor_chunks(keywords)

        initial_k = min(5, len(self.chunks))
        initial_indices = np.argsort(similarities)[-initial_k:][::-1]
        initial_chunk_ids = [list(self.chunks.keys())[idx] for idx in initial_indices]

        expanded_chunks: Set[int] = set(initial_chunk_ids + anchor_chunk_ids)

        if self.graph:
            for chunk_id in list(expanded_chunks):
                if chunk_id in self.graph:
                    neighbors = list(self.graph.neighbors(chunk_id))
                    expanded_chunks.update(neighbors)
                    if self.expansion_hops > 1:
                        for neighbor in neighbors:
                            if neighbor in self.graph:
                                expanded_chunks.update(self.graph.neighbors(neighbor))

        # Global path: community centroids
        global_candidates: Set[int] = set()
        if self.community_embeddings:
            centroid_scores = [
                (idx, float(np.dot(centroid, query_emb)))
                for idx, centroid in enumerate(self.community_embeddings)
            ]
            centroid_scores.sort(key=lambda x: x[1], reverse=True)
            for comm_idx, _ in centroid_scores[:2]:
                community = self.communities[comm_idx]
                # pick top scoring chunk within this community
                best_chunk = None
                best_score = -1e9
                for cid in community:
                    if cid not in chunk_id_to_idx:
                        continue
                    score = float(similarities[chunk_id_to_idx[cid]])
                    if score > best_score:
                        best_score = score
                        best_chunk = cid
                if best_chunk is not None:
                    global_candidates.add(best_chunk)

        candidate_ids = expanded_chunks | global_candidates

        # Re-rank candidates by similarity
        results = []
        for chunk_id in candidate_ids:
            if chunk_id not in chunk_id_to_idx:
                continue
            idx = chunk_id_to_idx[chunk_id]
            similarity = float(similarities[idx])
            chunk = self.chunks[chunk_id]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': similarity
            })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:25]  # Keep top 25 before token trimming to fill ~5.4k budget

        # Trim by tokens
        results = self._trim_by_tokens(results)

        logger.info(
            f"LightRAG retrieval: keywords={keywords}, anchors={len(anchor_chunk_ids)}, "
            f"local_candidates={len(expanded_chunks)}, global_candidates={len(global_candidates)}, "
            f"returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results

    def _extract_keywords(self, query: str) -> List[str]:
        stop = {"the", "and", "with", "what", "when", "where", "which", "who", "how"}
        tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", query) if len(t) > 3]
        return [t for t in tokens if t not in stop]

    def _find_anchor_chunks(self, keywords: List[str]) -> List[int]:
        if not keywords:
            return []
        anchors = []
        for cid, chunk in self.chunks.items():
            text = chunk['text'].lower()
            title = chunk['doc_title'].lower() if chunk.get('doc_title') else ""
            if any(k in text or k in title for k in keywords):
                anchors.append(cid)
        return anchors

    def _compute_community_embeddings(self) -> List[np.ndarray]:
        centroids = []
        if not self.communities:
            return centroids
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        for community in self.communities:
            indices = [chunk_id_to_idx[cid] for cid in community if cid in chunk_id_to_idx]
            if not indices:
                centroids.append(np.zeros(self.embeddings.shape[1]))
                continue
            centroid = np.mean(self.embeddings[indices], axis=0)
            centroids.append(centroid)
        return centroids
