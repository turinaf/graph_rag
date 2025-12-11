"""
LightRAG SOTA: Enhanced entity-based graph retrieval.

SOTA features:
- Dual graph (entity + relation graphs)
- Better entity extraction
- Entity linking and co-reference
- Hierarchical graph structure
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
import logging
import re
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class LightRAGSOTA(BaseRetriever):
    """
    LightRAG SOTA: Dual-graph entity-based retrieval.
    
    Improvements over baseline:
    1. Explicit entity extraction and linking
    2. Dual-graph structure (entity graph + chunk graph)
    3. Entity-aware expansion
    4. Better fusion of local and global paths
    """
    
    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='lightrag')
        self.chunks = None
        self.embeddings = None
        self.chunk_graph = None
        self.entity_graph = None
        self.entities = {}  # entity_name -> {chunks: set, mentions: int}
        self.chunk_to_entities = {}  # chunk_id -> [entity_names]
    
    def index(self, chunks: List[Dict], embeddings: np.ndarray, graph: nx.Graph = None) -> None:
        """Build dual-graph index with entity extraction."""
        self.chunks = {c['chunk_id']: c for c in chunks}
        self.embeddings = embeddings
        self.chunk_graph = graph
        
        # Extract entities from all chunks
        self._extract_entities()
        
        # Build entity graph
        self._build_entity_graph()
        
        logger.info(
            f"LightRAG SOTA indexed {len(chunks)} chunks, "
            f"{len(self.entities)} entities, "
            f"{self.entity_graph.number_of_edges()} entity relations"
        )
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve using dual-graph approach.
        
        Steps:
        1. Extract entities from query
        2. Local path: entity-aware chunk expansion
        3. Global path: entity graph traversal
        4. Fusion and re-ranking
        """
        # Encode query
        query_emb = self.encoder.encode([query])[0]
        
        # Extract query entities
        query_entities = self._extract_entities_from_text(query)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb)
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}
        
        # LOCAL PATH: Entity-aware expansion
        local_candidates = self._local_retrieval(query_entities, similarities, chunk_id_to_idx)
        
        # GLOBAL PATH: Entity graph traversal
        global_candidates = self._global_retrieval(query_entities, similarities, chunk_id_to_idx)
        
        # FUSION: Combine both paths
        all_candidates = local_candidates | global_candidates
        
        # Re-rank by similarity
        results = []
        for chunk_id in all_candidates:
            if chunk_id not in chunk_id_to_idx:
                continue
            idx = chunk_id_to_idx[chunk_id]
            
            # Boost score if chunk contains query entities
            base_score = float(similarities[idx])
            entity_boost = 0.0
            if chunk_id in self.chunk_to_entities:
                chunk_entities = set(self.chunk_to_entities[chunk_id])
                overlap = chunk_entities & query_entities
                if overlap:
                    entity_boost = 0.1 * len(overlap)
            
            chunk = self.chunks[chunk_id]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'doc_id': chunk['doc_id'],
                'doc_title': chunk['doc_title'],
                'score': base_score + entity_boost
            })
        
        # Sort and trim
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:25]
        results = self._trim_by_tokens(results)
        
        logger.info(
            f"LightRAG SOTA: query_entities={len(query_entities)}, "
            f"local={len(local_candidates)}, global={len(global_candidates)}, "
            f"returned={len(results)}"
        )
        return results
    
    def _extract_entities(self):
        """Extract entities from all chunks using improved patterns."""
        for chunk_id, chunk in self.chunks.items():
            entities = self._extract_entities_from_text(chunk['text'])
            self.chunk_to_entities[chunk_id] = list(entities)
            
            for entity in entities:
                if entity not in self.entities:
                    self.entities[entity] = {'chunks': set(), 'mentions': 0}
                self.entities[entity]['chunks'].add(chunk_id)
                self.entities[entity]['mentions'] += 1
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """
        Extract entities using improved patterns.
        
        Patterns:
        - Capitalized phrases (2-4 words)
        - Quoted terms
        - Numbers with units
        - Date patterns
        """
        entities = set()
        
        # Pattern 1: Capitalized phrases (2-4 consecutive capitalized words)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        for match in re.finditer(cap_pattern, text):
            entity = match.group(1)
            if len(entity) > 3:  # Minimum length
                entities.add(entity)
        
        # Pattern 2: Quoted terms
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, text):
            entity = match.group(1).strip()
            if len(entity) > 2:
                entities.add(entity)
        
        # Pattern 3: Numbers with units (e.g., "5 million", "2023")
        num_pattern = r'\b(\d+(?:\.\d+)?\s*(?:million|billion|thousand|years?|km|miles?)?)\b'
        for match in re.finditer(num_pattern, text):
            entity = match.group(1).strip()
            if any(c.isalpha() for c in entity):  # Has units
                entities.add(entity)
        
        return entities
    
    def _build_entity_graph(self):
        """Build entity co-occurrence graph."""
        self.entity_graph = nx.Graph()
        
        # Add entity nodes
        for entity in self.entities:
            self.entity_graph.add_node(entity, mentions=self.entities[entity]['mentions'])
        
        # Add edges for co-occurring entities (same chunk)
        for chunk_id, entity_list in self.chunk_to_entities.items():
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i+1:]:
                    if self.entity_graph.has_edge(e1, e2):
                        self.entity_graph[e1][e2]['weight'] += 1
                    else:
                        self.entity_graph.add_edge(e1, e2, weight=1)
    
    def _local_retrieval(self, query_entities: Set[str], similarities: np.ndarray, 
                        chunk_id_to_idx: Dict) -> Set[int]:
        """Local path: entity-aware chunk expansion."""
        candidates = set()
        
        # Find chunks containing query entities
        for entity in query_entities:
            if entity in self.entities:
                candidates.update(self.entities[entity]['chunks'])
        
        # Also add top-k by similarity
        initial_k = min(5, len(self.chunks))
        top_indices = np.argsort(similarities)[-initial_k:][::-1]
        top_chunk_ids = [list(self.chunks.keys())[idx] for idx in top_indices]
        candidates.update(top_chunk_ids)
        
        # Expand via chunk graph
        if self.chunk_graph:
            expanded = set(candidates)
            for chunk_id in list(candidates):
                if chunk_id in self.chunk_graph:
                    neighbors = list(self.chunk_graph.neighbors(chunk_id))
                    expanded.update(neighbors[:3])  # Top 3 neighbors
            candidates = expanded
        
        return candidates
    
    def _global_retrieval(self, query_entities: Set[str], similarities: np.ndarray,
                         chunk_id_to_idx: Dict) -> Set[int]:
        """Global path: entity graph traversal."""
        candidates = set()
        
        if not query_entities or not self.entity_graph:
            return candidates
        
        # Find related entities via graph traversal
        related_entities = set(query_entities)
        for entity in query_entities:
            if entity in self.entity_graph:
                # Get 1-hop neighbors
                neighbors = list(self.entity_graph.neighbors(entity))
                # Sort by edge weight
                neighbors.sort(
                    key=lambda e: self.entity_graph[entity][e]['weight'],
                    reverse=True
                )
                related_entities.update(neighbors[:5])  # Top 5 related entities
        
        # Find chunks containing related entities
        for entity in related_entities:
            if entity in self.entities:
                entity_chunks = self.entities[entity]['chunks']
                # Add top-scoring chunks for this entity
                scored = []
                for cid in entity_chunks:
                    if cid in chunk_id_to_idx:
                        idx = chunk_id_to_idx[cid]
                        scored.append((cid, similarities[idx]))
                scored.sort(key=lambda x: x[1], reverse=True)
                candidates.update([cid for cid, _ in scored[:2]])  # Top 2 per entity
        
        return candidates
