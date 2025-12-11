"""
NodeRAG SOTA: Heterogeneous graph traversal with entity extraction.

State-of-the-art implementation including:
- Heterogeneous graph (Document/Chunk/Entity nodes)
- Entity extraction and linking
- Relation-aware traversal
- Advanced PPR with type-specific restart probabilities
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
import logging
import re
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class NodeRAGSOTA(BaseRetriever):
    """
    NodeRAG SOTA: Heterogeneous graph with advanced traversal.
    
    Key improvements:
    - Three node types: Document, Chunk, Entity
    - Entity extraction from chunks
    - Type-aware PPR
    - Smarter score propagation
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='node_rag')
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self.hetero_graph = None  # Heterogeneous version
        self.entities = {}  # entity_id -> entity info
        self.extract_entities = config.get('noderag', {}).get('extract_entities', False)
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Index chunks and build heterogeneous graph."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.graph = graph
        
        # Build heterogeneous graph
        if graph:
            self.hetero_graph = self._build_heterogeneous_graph(chunks, graph)
        
        logger.info(
            f"NodeRAG SOTA indexed {len(chunks)} chunks, "
            f"hetero_graph: {self.hetero_graph.number_of_nodes() if self.hetero_graph else 0} nodes"
        )
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve using heterogeneous graph traversal.
        
        Steps:
        1. Find initial nodes (chunks + entities)
        2. Multi-hop expansion with type-aware scoring
        3. PPR with heterogeneous restart
        4. Re-rank and return
        """
        query_emb = self.encoder.encode([query])[0]
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        
        # Compute chunk similarities
        similarities = np.dot(self.embeddings, query_emb)
        
        # Find initial chunk candidates
        initial_k = min(15, len(self.chunks))
        initial_indices = np.argsort(similarities)[-initial_k:][::-1]
        initial_chunk_ids = [list(self.chunks.keys())[idx] for idx in initial_indices]
        
        # Build score dictionary
        scores = {cid: float(similarities[chunk_id_to_idx[cid]]) for cid in initial_chunk_ids}
        
        # Use heterogeneous graph if available
        if self.hetero_graph and len(self.hetero_graph) > 0:
            # Type-aware PPR
            try:
                # Create personalization vector favoring chunk nodes
                personalization = {}
                for node in self.hetero_graph.nodes():
                    node_type = self.hetero_graph.nodes[node].get('type', 'chunk')
                    if node_type == 'chunk' and node in chunk_id_to_idx:
                        idx = chunk_id_to_idx[node]
                        personalization[node] = max(float(similarities[idx]), 0.0) + 0.1
                    elif node_type == 'entity':
                        # Check if entity matches query keywords
                        entity_name = self.hetero_graph.nodes[node].get('name', '').lower()
                        query_lower = query.lower()
                        if entity_name and entity_name in query_lower:
                            personalization[node] = 1.0
                        else:
                            personalization[node] = 0.1
                    else:  # document
                        personalization[node] = 0.05
                
                ppr_scores = nx.pagerank(
                    self.hetero_graph,
                    alpha=0.85,
                    personalization=personalization,
                    weight='weight'
                )
            except Exception as e:
                logger.warning(f"PPR failed: {e}")
                ppr_scores = {}
        else:
            ppr_scores = {}
        
        # Multi-hop expansion with heterogeneous scoring
        expanded_nodes = set(initial_chunk_ids)
        
        if self.hetero_graph:
            for hop in range(2):  # 2 hops
                new_nodes = set()
                
                for node_id in list(expanded_nodes):
                    if node_id not in self.hetero_graph:
                        continue
                    
                    neighbors = list(self.hetero_graph.neighbors(node_id))
                    
                    for neighbor in neighbors:
                        if neighbor in expanded_nodes:
                            continue
                        
                        # Get neighbor type
                        neighbor_type = self.hetero_graph.nodes[neighbor].get('type', 'chunk')
                        
                        # Only expand to chunks (skip intermediate entities/docs)
                        if neighbor_type != 'chunk':
                            # But use them as bridges - expand their chunk neighbors
                            if hop == 0:  # Only in first hop
                                for nn in self.hetero_graph.neighbors(neighbor):
                                    if self.hetero_graph.nodes[nn].get('type') == 'chunk':
                                        new_nodes.add(nn)
                            continue
                        
                        # Score propagation with decay
                        decay = 0.6 ** (hop + 1)
                        edge_weight = self.hetero_graph[node_id][neighbor].get('weight', 0.5)
                        
                        base_score = float(similarities[chunk_id_to_idx[neighbor]]) if neighbor in chunk_id_to_idx else 0.0
                        prop_score = scores.get(node_id, 0.0) * edge_weight * decay
                        
                        new_score = base_score * 0.7 + prop_score * 0.3
                        
                        if neighbor not in scores or new_score > scores[neighbor]:
                            scores[neighbor] = new_score
                            new_nodes.add(neighbor)
                
                expanded_nodes.update(new_nodes)
        
        # Build results combining all signals
        results = []
        max_sim = float(np.max(similarities)) if len(similarities) > 0 else 1.0
        
        for chunk_id in expanded_nodes:
            if chunk_id not in self.chunks:
                continue
            
            chunk = self.chunks[chunk_id]
            
            # Combine signals
            base = scores.get(chunk_id, 0.0)
            ppr = ppr_scores.get(chunk_id, 0.0) * max_sim * 2.0  # Scale PPR
            
            # Graph centrality bonus
            centrality = 0.0
            if self.hetero_graph and chunk_id in self.hetero_graph:
                degree = self.hetero_graph.degree(chunk_id)
                centrality = min(degree / 20.0, 0.2)  # Cap at 0.2
            
            combined = base * 0.5 + ppr * 0.4 + centrality * 0.1
            
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': combined
            })
        
        # Sort and trim
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:15]  # Top 15 before token trimming (NodeRAG is efficient)
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"NodeRAG SOTA: initial={len(initial_chunk_ids)}, expanded={len(expanded_nodes)}, "
            f"returned={len(results)}, target_tokens={self.max_tokens}"
        )
        return results
    
    def _build_heterogeneous_graph(self, chunks: List[Dict], chunk_graph: nx.Graph) -> nx.Graph:
        """
        Build heterogeneous graph with Document, Chunk, and Entity nodes.
        
        Node types:
        - doc_{id}: Document nodes
        - chunk_{id}: Chunk nodes
        - entity_{name}: Entity nodes
        
        Edges:
        - doc -> chunk: containment
        - chunk -> chunk: similarity (from original graph)
        - entity -> chunk: mention
        """
        H = nx.Graph()
        
        # Add chunk nodes (from original graph)
        for node in chunk_graph.nodes():
            H.add_node(node, type='chunk', **chunk_graph.nodes[node])
        
        # Add chunk-chunk edges
        for u, v, data in chunk_graph.edges(data=True):
            H.add_edge(u, v, type='similarity', **data)
        
        # Group chunks by document
        doc_to_chunks = {}
        for chunk in chunks:
            doc_id = chunk['doc_id']
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk['chunk_id'])
        
        # Add document nodes and doc->chunk edges
        for doc_id, chunk_ids in doc_to_chunks.items():
            doc_node = f"doc_{doc_id}"
            H.add_node(doc_node, type='document', doc_id=doc_id)
            
            for chunk_id in chunk_ids:
                H.add_edge(doc_node, chunk_id, type='contains', weight=1.0)
        
        # Extract entities (simple approach: capitalized phrases)
        entity_to_chunks = {}
        for chunk in chunks:
            entities = self._extract_simple_entities(chunk['text'])
            for entity in entities:
                if entity not in entity_to_chunks:
                    entity_to_chunks[entity] = []
                entity_to_chunks[entity].append(chunk['chunk_id'])
        
        # Add entity nodes (only if mentioned in 2+ chunks for noise reduction)
        for entity, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) >= 2:
                entity_node = f"entity_{entity}"
                H.add_node(entity_node, type='entity', name=entity)
                
                for chunk_id in chunk_ids:
                    H.add_edge(entity_node, chunk_id, type='mentions', weight=0.8)
        
        logger.info(
            f"Built heterogeneous graph: {H.number_of_nodes()} nodes "
            f"(chunks: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='chunk')}, "
            f"docs: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='document')}, "
            f"entities: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='entity')}), "
            f"{H.number_of_edges()} edges"
        )
        return H
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction: capitalized phrases.
        
        This is a lightweight alternative to NER.
        For production, use spaCy or similar.
        """
        # Find sequences of capitalized words (2+ words)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        entities = re.findall(pattern, text)
        
        # Also include single capitalized words if 4+ letters
        single_pattern = r'\b([A-Z][a-z]{3,})\b'
        single_entities = re.findall(single_pattern, text)
        
        # Filter common stopwords
        stop = {'This', 'That', 'These', 'Those', 'When', 'Where', 'Which', 'What', 'Who', 'Why', 'How'}
        entities = [e for e in entities if e not in stop]
        single_entities = [e for e in single_entities if e not in stop]
        
        return entities + single_entities
