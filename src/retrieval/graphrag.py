"""
GraphRAG: Community-based retrieval using graph structure.

Simplified version inspired by Microsoft GraphRAG.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set
import logging
import re
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class GraphRAG(BaseRetriever):
    """
    GraphRAG: Community-based graph retrieval.
    
    Steps:
    1. Detect communities in chunk graph
    2. Retrieve chunks from most relevant communities
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='graphrag')
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self.communities = None
        self.community_embeddings = []
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        
        # Detect communities using Louvain algorithm
        if graph and len(graph.nodes) > 0:
            self.communities = nx.community.louvain_communities(graph)
            self.community_embeddings = self._compute_community_embeddings()
            logger.info(
                f"GraphRAG detected {len(self.communities)} communities; "
                f"computed {len(self.community_embeddings)} community centroids"
            )
        else:
            self.communities = []
            self.community_embeddings = []
        
        logger.info(f"GraphRAG indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve chunks using graph communities.
        
        Steps:
        1. Encode query
        2. Find most relevant communities
        3. Retrieve top chunks from those communities
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]

        # Map chunk IDs to embeddings indices
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}

        keywords = self._extract_keywords(query)
        anchor_chunks = self._find_anchor_chunks(keywords)
        anchor_comms = self._communities_containing(anchor_chunks)

        # If no anchor communities found, fall back to centroid similarity
        if not anchor_comms and self.community_embeddings:
            centroid_scores = [
                (idx, float(np.dot(centroid, query_emb)))
                for idx, centroid in enumerate(self.community_embeddings)
            ]
            centroid_scores.sort(key=lambda x: x[1], reverse=True)
            anchor_comms = [cid for cid, _ in centroid_scores[:3]]

        results = []
        seen_chunks: Set[int] = set()

        for comm_idx in anchor_comms:
            community = self.communities[comm_idx]
            for chunk_id in community:
                if chunk_id in seen_chunks or chunk_id not in chunk_id_to_idx:
                    continue

                idx = chunk_id_to_idx[chunk_id]
                similarity = float(np.dot(self.embeddings[idx], query_emb))

                # Neighbor boost encourages multi-hop traversal
                neighbor_score = 0.0
                if self.graph and chunk_id in self.graph:
                    neighbor_ids = list(self.graph.neighbors(chunk_id))
                    neighbor_score = sum(
                        np.dot(self.embeddings[chunk_id_to_idx[nid]], query_emb)
                        for nid in neighbor_ids if nid in chunk_id_to_idx
                    )
                    neighbor_score = neighbor_score / max(len(neighbor_ids), 1)

                combined_score = similarity * 0.7 + neighbor_score * 0.3

                chunk = self.chunks[chunk_id]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'doc_id': chunk['doc_id'],
                    'doc_title': chunk['doc_title'],
                    'score': combined_score,
                    'community': comm_idx
                })

                seen_chunks.add(chunk_id)

        # Fallback: if still empty, use similarity top-k
        if not results:
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            for idx in top_indices:
                chunk = list(self.chunks.values())[idx]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'doc_id': chunk['doc_id'],
                    'doc_title': chunk['doc_title'],
                    'score': float(similarities[idx]),
                    'community': None
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:30]  # Keep top 30 before token trimming to fill ~6.1k budget

        # Trim by tokens
        results = self._trim_by_tokens(results)

        logger.info(
            f"GraphRAG retrieval: keywords={keywords}, anchors={len(anchor_chunks)}, "
            f"communities_considered={len(anchor_comms)}, returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract lightweight keywords from query (alphanumeric, >3 chars)."""
        stop = {"the", "and", "with", "what", "when", "where", "which", "who", "how"}
        tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", query) if len(t) > 3]
        return [t for t in tokens if t not in stop]

    def _find_anchor_chunks(self, keywords: List[str]) -> List[int]:
        """Find chunks mentioning any keyword (title or text)."""
        anchors = []
        if not keywords:
            return anchors
        for cid, chunk in self.chunks.items():
            text = chunk['text'].lower()
            title = chunk['doc_title'].lower() if chunk.get('doc_title') else ""
            if any(k in text or k in title for k in keywords):
                anchors.append(cid)
        return anchors

    def _communities_containing(self, chunk_ids: List[int]) -> List[int]:
        """Return community indices that contain any of the chunk_ids."""
        if not self.communities:
            return []
        selected = []
        for idx, comm in enumerate(self.communities):
            if any(cid in comm for cid in chunk_ids):
                selected.append(idx)
        return selected

    def _compute_community_embeddings(self) -> List[np.ndarray]:
        """Compute centroid embedding per community for global retrieval."""
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
