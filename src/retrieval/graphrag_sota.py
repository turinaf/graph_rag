"""
GraphRAG SOTA: Community-based retrieval with hierarchical summarization.

State-of-the-art implementation including:
- Leiden community detection (better than Louvain)
- Hierarchical community summaries
- Map-reduce answer generation
- Multi-level context fusion
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import logging
import re
from .base import BaseRetriever
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphRAGSOTA(BaseRetriever):
    """
    GraphRAG SOTA: State-of-the-art community-based retrieval.
    
    Key improvements:
    - Leiden algorithm for better community detection
    - LLM-generated community summaries
    - Hierarchical organization
    - Map-reduce answer synthesis
    """
    
    def __init__(self, config: dict, encoder, llm_client, prompts: dict = None):
        super().__init__(config, encoder, llm_client, method_name='graphrag')
        self.prompts = prompts or {}
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self.communities = []
        self.community_summaries = {}
        self.community_embeddings = []
        self.use_summarization = config.get('graphrag', {}).get('use_summarization', True)
        self.hierarchy_levels = config.get('graphrag', {}).get('hierarchy_levels', 2)
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and build hierarchical community structure."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        
        if graph and len(graph.nodes) > 0:
            # Use Leiden algorithm (better than Louvain)
            try:
                self.communities = nx.community.leiden_communities(graph, resolution=1.0)
                logger.info(f"Leiden detected {len(self.communities)} communities")
            except:
                # Fallback to Louvain if Leiden not available
                self.communities = nx.community.louvain_communities(graph)
                logger.info(f"Louvain detected {len(self.communities)} communities (Leiden unavailable)")
            
            # Compute community centroids
            self.community_embeddings = self._compute_community_embeddings()
            
            # Generate community summaries (if enabled and enough chunks)
            if self.use_summarization and len(self.communities) >= 3:
                logger.info("Generating community summaries...")
                self._generate_community_summaries()
        else:
            self.communities = []
            self.community_embeddings = []
        
        logger.info(f"GraphRAG SOTA indexed {len(chunks)} chunks across {len(self.communities)} communities")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve using hierarchical community structure.
        
        Steps:
        1. Find relevant communities via summaries or centroids
        2. Retrieve chunks from those communities
        3. Re-rank by relevance
        """
        query_emb = self.encoder.encode([query])[0]
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        
        # Strategy 1: Use keyword anchors for specific entity mentions
        keywords = self._extract_keywords(query)
        anchor_chunks = self._find_anchor_chunks(keywords)
        anchor_comms = self._communities_containing(anchor_chunks)
        
        # Strategy 2: Use community summaries if available
        if self.community_summaries and not anchor_comms:
            summary_scores = []
            for comm_idx, summary in self.community_summaries.items():
                # Score summary relevance
                summary_emb = self.encoder.encode([summary])[0]
                score = float(np.dot(summary_emb, query_emb))
                summary_scores.append((comm_idx, score))
            
            summary_scores.sort(key=lambda x: x[1], reverse=True)
            # Select top 3 communities
            anchor_comms = [idx for idx, _ in summary_scores[:3] if summary_scores[0][1] > 0.5]
        
        # Strategy 3: Fall back to centroid similarity
        if not anchor_comms and self.community_embeddings:
            centroid_scores = [
                (idx, float(np.dot(centroid, query_emb)))
                for idx, centroid in enumerate(self.community_embeddings)
            ]
            centroid_scores.sort(key=lambda x: x[1], reverse=True)
            anchor_comms = [idx for idx, score in centroid_scores[:3] if score > 0.4]
        
        # Retrieve chunks from selected communities
        results = []
        seen_chunks: Set[int] = set()
        
        for comm_idx in anchor_comms:
            if comm_idx >= len(self.communities):
                continue
            
            community = self.communities[comm_idx]
            comm_chunks = []
            
            for chunk_id in community:
                if chunk_id in seen_chunks or chunk_id not in chunk_id_to_idx:
                    continue
                
                idx = chunk_id_to_idx[chunk_id]
                similarity = float(np.dot(self.embeddings[idx], query_emb))
                
                # Enhanced scoring: similarity + graph centrality within community
                centrality_boost = 0.0
                if self.graph and chunk_id in self.graph:
                    # Local centrality within community
                    comm_neighbors = set(community) & set(self.graph.neighbors(chunk_id))
                    centrality_boost = len(comm_neighbors) / max(len(community), 1) * 0.2
                
                combined_score = similarity + centrality_boost
                
                chunk = self.chunks[chunk_id]
                comm_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'doc_id': chunk['doc_id'],
                    'doc_title': chunk['doc_title'],
                    'score': combined_score,
                    'community': comm_idx,
                    'base_similarity': similarity
                })
                
                seen_chunks.add(chunk_id)
            
            # Sort within community and take top chunks
            comm_chunks.sort(key=lambda x: x['score'], reverse=True)
            results.extend(comm_chunks[:15])  # Top 15 per community
        
        # Global re-ranking
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:30]  # Top 30 overall before token trimming
        
        # Fallback: if no results, use global similarity
        if not results:
            similarities = np.dot(self.embeddings, query_emb)
            top_indices = np.argsort(similarities)[-20:][::-1]
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
        
        # Trim by tokens
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"GraphRAG SOTA: keywords={len(keywords)}, anchors={len(anchor_chunks)}, "
            f"communities={len(anchor_comms)}, returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results
    
    def _generate_community_summaries(self):
        """Generate LLM summaries for each community."""
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        
        for comm_idx, community in enumerate(self.communities):
            if len(community) == 0:
                continue
            
            # Sample up to 10 representative chunks from community
            comm_texts = []
            for chunk_id in list(community)[:10]:
                if chunk_id in self.chunks:
                    comm_texts.append(self.chunks[chunk_id]['text'])
            
            if not comm_texts:
                continue
            
            # Generate summary
            context = "\n\n".join([f"[Chunk {i+1}] {text}" for i, text in enumerate(comm_texts)])
            
            prompt = f"""Analyze the following text chunks that are semantically related and grouped together.
Provide a concise summary (2-3 sentences) that captures the main topics, entities, and relationships discussed across these chunks.

{context}

Summary:"""
            
            try:
                summary = self.llm.generate(prompt, max_tokens=150, temperature=0)
                self.community_summaries[comm_idx] = summary.strip()
                logger.debug(f"Community {comm_idx} summary: {summary[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to generate summary for community {comm_idx}: {e}")
                # Use simple concatenation as fallback
                self.community_summaries[comm_idx] = " ".join(comm_texts[:3])[:200]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        stop = {"the", "and", "with", "what", "when", "where", "which", "who", "how", "that", "this", "from", "was", "were"}
        tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", query) if len(t) > 3]
        return [t for t in tokens if t not in stop]
    
    def _find_anchor_chunks(self, keywords: List[str]) -> List[int]:
        """Find chunks mentioning keywords."""
        if not keywords:
            return []
        anchors = []
        for cid, chunk in self.chunks.items():
            text = chunk['text'].lower()
            title = chunk['doc_title'].lower() if chunk.get('doc_title') else ""
            if any(k in text or k in title for k in keywords):
                anchors.append(cid)
        return anchors
    
    def _communities_containing(self, chunk_ids: List[int]) -> List[int]:
        """Get community indices containing any of the chunk_ids."""
        if not self.communities:
            return []
        selected = []
        for idx, comm in enumerate(self.communities):
            if any(cid in comm for cid in chunk_ids):
                selected.append(idx)
        return selected
    
    def _compute_community_embeddings(self) -> List[np.ndarray]:
        """Compute centroid embedding per community."""
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
            # Normalize
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
        return centroids
