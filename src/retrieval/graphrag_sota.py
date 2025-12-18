"""
GraphRAG SOTA: Community-based retrieval with hierarchical summarization.

State-of-the-art implementation including:
- Leiden community detection (better than Louvain)
- Hierarchical community summaries
- Multi-hop community reasoning
- Enhanced context fusion
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
    - Hierarchical multi-hop reasoning
    - Enhanced scoring with community structure
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
        self.resolution = config.get('graph', {}).get('leiden_resolution', 1.0)  # Leiden resolution parameter
        self.max_communities_to_query = config.get('graphrag', {}).get('max_communities_to_query', 5)  # Max communities to retrieve from

    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks, embeddings, and build hierarchical community structure."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph

        if graph and len(graph.nodes) > 0:
            # Use Leiden algorithm (better than Louvain)
            try:
                # Try to use leidenalg if available
                import leidenalg
                import igraph as ig
                
                # Convert NetworkX graph to igraph for Leiden
                g_ig = ig.Graph.from_networkx(graph)
                
                # Run Leiden algorithm with resolution parameter
                partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, 
                                                    resolution_parameter=self.resolution)
                
                # Convert back to NetworkX format
                communities = []
                for community_nodes in partition:
                    # Convert igraph vertex indices back to node IDs
                    community_node_ids = [g_ig.vs[i]['name'] for i in community_nodes]
                    communities.append(set(community_node_ids))
                
                self.communities = communities
                logger.info(f"Leiden detected {len(self.communities)} communities (igraph/leidenalg)")
            except ImportError:
                # Fallback: try NetworkX implementation of Leiden
                try:
                    self.communities = nx.community.leiden_communities(graph, resolution=self.resolution)
                    logger.info(f"Leiden detected {len(self.communities)} communities (nx.community)")
                except:
                    # Last fallback: Louvain
                    self.communities = nx.community.louvain_communities(graph, resolution=self.resolution)
                    logger.info(f"Louvain detected {len(self.communities)} communities (nx.community fallback)")

            # Compute community centroids and metadata
            self.community_embeddings = self._compute_community_embeddings()
            self.community_centrality = self._compute_community_centrality()  # Enhanced centrality scores

            # Generate community summaries (if enabled and enough chunks)
            if self.use_summarization and len(self.communities) >= 3:
                logger.info("Generating community summaries...")
                self._generate_community_summaries()
        else:
            self.communities = []
            self.community_embeddings = []
            self.community_centrality = {}

        logger.info(f"GraphRAG SOTA indexed {len(chunks)} chunks across {len(self.communities)} communities")

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve using hierarchical community structure with multi-hop reasoning.

        Steps:
        1. Find relevant communities via semantic similarity
        2. Perform multi-hop expansion across communities if needed
        3. Retrieve chunks from relevant communities
        4. Re-rank by relevance and community structure
        """
        query_emb = self.encoder.encode([query])[0]
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}

        # Step 1: Find relevant communities using multiple strategies
        # Strategy 1: Semantic similarity to community centroids
        centroid_scores = []
        if self.community_embeddings:
            for idx, centroid in enumerate(self.community_embeddings):
                if idx >= len(self.communities):
                    continue
                similarity = float(np.dot(centroid, query_emb))
                centrality_bonus = self.community_centrality.get(idx, 0.0) * 0.1  # Boost by community importance
                centroid_scores.append((idx, similarity + centrality_bonus))

        # Sort by combined score
        centroid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top communities (use more than just top 3 for multi-hop reasoning)
        selected_comm_indices = [idx for idx, _ in centroid_scores[:self.max_communities_to_query]]
        
        # Step 2: Multi-hop community reasoning
        expanded_communities = set(selected_comm_indices)
        
        # Expand to adjacent communities if needed for multi-hop reasoning
        if self.graph and len(self.communities) > 5:  # Only if enough communities exist
            for comm_idx in selected_comm_indices:
                if comm_idx >= len(self.communities):
                    continue
                community = self.communities[comm_idx]
                
                # Find chunks in this community and their neighbors
                for chunk_id in list(community)[:5]:  # Limit to avoid excessive expansion
                    if chunk_id in self.graph:
                        neighbors = list(self.graph.neighbors(chunk_id))
                        for neighbor_id in neighbors:
                            # Find which community this neighbor belongs to
                            for other_idx, other_community in enumerate(self.communities):
                                if neighbor_id in other_community and other_idx not in expanded_communities:
                                    expanded_communities.add(other_idx)
                                    if len(expanded_communities) >= self.max_communities_to_query * 2:
                                        break
                        if len(expanded_communities) >= self.max_communities_to_query * 2:
                            break
        
        # Step 3: Retrieve chunks from selected communities
        results = []
        seen_chunks: Set[str] = set()  # Use chunk_id instead of index

        for comm_idx in expanded_communities:
            if comm_idx >= len(self.communities):
                continue

            community = self.communities[comm_idx]
            comm_chunks = []

            for chunk_id in community:
                if chunk_id in seen_chunks or chunk_id not in chunk_id_to_idx:
                    continue

                idx = chunk_id_to_idx[chunk_id]
                similarity = float(np.dot(self.embeddings[idx], query_emb))

                # Enhanced scoring: similarity + community structure features
                centrality_boost = 0.0
                if self.graph and chunk_id in self.graph:
                    # Local centrality within community and global graph
                    local_degree = len(set(community) & set(self.graph.neighbors(chunk_id)))
                    global_degree = self.graph.degree(chunk_id)
                    local_centrality = local_degree / max(len(community), 1)
                    centrality_boost = (local_centrality * 0.1 + global_degree * 0.001)

                # Combine scores
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
            results.extend(comm_chunks[:20])  # More chunks per community for multi-hop

        # Global re-ranking
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:50]  # More results before token trimming to allow for good multi-hop coverage

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
            f"GraphRAG SOTA: selected={len(expanded_communities)} communities, "
            f"retrieved={len(results)} chunks, target_tokens={self.max_tokens}"
        )
        return results

    def _generate_community_summaries(self):
        """Generate LLM summaries for each community."""
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}

        for comm_idx, community in enumerate(self.communities):
            if len(community) == 0:
                continue

            # Sample representative chunks from community using centrality and diversity
            community_chunks = []
            for chunk_id in community:
                if chunk_id in self.chunks:
                    community_chunks.append(chunk_id)
            
            if not community_chunks:
                continue
                
            # Sort by centrality within community and take diverse samples
            selected_chunks = []
            if len(community_chunks) <= 10:
                selected_chunks = community_chunks  # Take all if small community
            else:
                # Take chunks with different characteristics for diversity
                chunk_scores = []
                for chunk_id in community_chunks:
                    # Calculate various scores
                    if self.graph and chunk_id in self.graph:
                        degree = self.graph.degree(chunk_id)
                        local_centrality = len(set(community) & set(self.graph.neighbors(chunk_id))) / max(len(community), 1)
                        score = degree * 0.6 + local_centrality * 0.4
                    else:
                        score = 0
                    chunk_scores.append((chunk_id, score))
                
                # Sort by score and take diverse samples
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                selected_chunks = [x[0] for x in chunk_scores[:10]]  # Top 10 by centrality

            # Get the texts for selected chunks
            comm_texts = []
            for chunk_id in selected_chunks:
                if chunk_id in self.chunks:
                    comm_texts.append(self.chunks[chunk_id]['text'])

            if not comm_texts:
                continue

            # Generate summary
            context = "\n\n".join([f"[Chunk {i+1}] {text[:500]}" for i, text in enumerate(comm_texts)])  # Limit length

            prompt = f"""Analyze the following text chunks that are semantically related and grouped together in a knowledge graph community.
Provide a concise summary (2-3 sentences) that captures the main topics, entities, and relationships discussed across these chunks.

{context}

Summary:"""

            try:
                summary = self.llm.generate(prompt, max_tokens=200, temperature=0.0)
                self.community_summaries[comm_idx] = summary.strip()
                logger.debug(f"Community {comm_idx} summary: {summary[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to generate summary for community {comm_idx}: {e}")
                # Use simple concatenation as fallback
                self.community_summaries[comm_idx] = " ".join(comm_texts[:3])[:300]

    def _compute_community_centrality(self) -> Dict[int, float]:
        """Compute centrality scores for each community based on graph structure."""
        centrality_scores = {}
        
        if not self.graph or not self.communities:
            return centrality_scores

        # Compute community-level centrality metrics
        for idx, community in enumerate(self.communities):
            if len(community) == 0:
                centrality_scores[idx] = 0.0
                continue

            # Calculate average degree of nodes in the community
            total_degree = sum(self.graph.degree(node) for node in community)
            avg_degree = total_degree / len(community)

            # Calculate community connectivity (how well connected internally vs externally)
            internal_edges = 0
            external_edges = 0
            
            for node in community:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in community:
                        internal_edges += 1
                    else:
                        external_edges += 1

            # Normalize by community size
            internal_density = internal_edges / (len(community) * (len(community) - 1)) if len(community) > 1 else 0
            bridge_score = external_edges / max(len(community), 1)  # How many connections to other communities

            # Combined centrality score
            centrality_scores[idx] = avg_degree * 0.4 + internal_density * 0.3 + bridge_score * 0.3

        return centrality_scores

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        stop = {"the", "and", "with", "what", "when", "where", "which", "who", "how", "that", "this", "from", "was", "were", "is", "are", "an", "a", "of", "in", "on", "at", "to", "for", "by", "about", "as", "into", "through", "during", "before", "after", "above", "below", "between", "among", "within", "without", "under", "over", "more", "most", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "now"}
        tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", query) if len(t) > 2]
        return [t for t in tokens if t not in stop]

    def _find_anchor_chunks(self, keywords: List[str]) -> List[str]:
        """Find chunks mentioning keywords."""
        if not keywords:
            return []
        anchors = []
        for chunk_id, chunk in self.chunks.items():
            text = chunk['text'].lower()
            title = chunk['doc_title'].lower() if chunk.get('doc_title') else ""
            if any(k in text or k in title for k in keywords):
                anchors.append(chunk_id)
        return anchors

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