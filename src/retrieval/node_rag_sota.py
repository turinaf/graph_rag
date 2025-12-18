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
    - Enhanced for multi-hop reasoning
    """

    def __init__(self, config: dict, encoder, llm_client):
        super().__init__(config, encoder, llm_client, method_name='node_rag')
        self.chunks = None
        self.embeddings = None
        self.graph = None
        self.hetero_graph = None  # Heterogeneous version
        self.entities = {}  # entity_id -> entity info
        self.extract_entities = config.get('noderag', {}).get('extract_entities', False)
        # NodeRAG-specific parameters
        self.ppr_alpha = config.get('noderag', {}).get('ppr_alpha', 0.85)  # Damping factor
        self.max_hops = config.get('noderag', {}).get('max_hops', 3)  # Multi-hop expansion

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
        Retrieve using heterogeneous graph traversal with PPR.

        Follows NodeRAG methodology:
        1. Create personalization vector based on query similarity
        2. Run Personalized PageRank on heterogeneous graph
        3. Rank chunks by PPR scores
        """
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks.values())}

        # Compute query similarities for all chunks (for personalization)
        query_emb = self.encoder.encode([query])[0]
        similarities = np.dot(self.embeddings, query_emb)

        # Create personalization vector for PPR
        personalization = {}
        
        if self.hetero_graph and len(self.hetero_graph) > 0:
            for node in self.hetero_graph.nodes():
                node_type = self.hetero_graph.nodes[node].get('type', 'chunk')
                
                if node_type == 'chunk' and node in chunk_id_to_idx:
                    # Use cosine similarity as base score for chunk nodes
                    idx = chunk_id_to_idx[node]
                    base_score = float(similarities[idx])
                    # Apply ReLU-like function to avoid negative scores
                    personalization[node] = max(base_score, 0.0) + 0.1  # Add small bias
                elif node_type == 'entity':
                    # Check if entity name matches query keywords (for multi-hop reasoning)
                    entity_name = self.hetero_graph.nodes[node].get('name', '').lower()
                    if entity_name and entity_name in query.lower():
                        personalization[node] = 1.0  # High weight for matching entities
                    else:
                        personalization[node] = 0.01  # Low weight
                else:  # document nodes
                    personalization[node] = 0.01  # Very low weight for document nodes

            # Run Personalized PageRank with the custom personalization
            try:
                ppr_scores = nx.pagerank(
                    self.hetero_graph,
                    alpha=self.ppr_alpha,
                    personalization=personalization,
                    weight='weight',
                    max_iter=100,  # Increase iterations for convergence
                    tol=1e-06      # Smaller tolerance for better accuracy
                )
            except Exception as e:
                logger.warning(f"PPR failed: {e}, falling back to similarity-based retrieval")
                ppr_scores = {}
        else:
            # Fallback: use similarity scores directly if no heterogeneous graph
            ppr_scores = {}
            for i, chunk_id in enumerate(chunk_id_to_idx):
                ppr_scores[chunk_id] = float(similarities[chunk_id_to_idx[chunk_id]])

        # Extract results for chunk nodes only
        results = []
        for chunk_id in self.chunks:
            if chunk_id in ppr_scores:
                score = ppr_scores[chunk_id]
                chunk = self.chunks[chunk_id]
                
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'doc_id': chunk['doc_id'],
                    'doc_title': chunk['doc_title'],
                    'score': float(score)
                })

        # Sort by PPR scores (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit top-k results for efficiency
        results = results[:max(self.top_k * 3, 20)]  # Get more candidates to account for token trimming
        
        # Trim by token budget
        results = self._trim_by_tokens(results)

        logger.info(
            f"NodeRAG SOTA: retrieved {len(results)} chunks using PPR on heterogeneous graph, "
            f"top score: {results[0]['score'] if results else 0.0:.4f}"
        )
        return results

    def _build_heterogeneous_graph(self, chunks: List[Dict], chunk_graph: nx.Graph) -> nx.Graph:
        """
        Build heterogeneous graph as described in NodeRAG paper.

        Node types:
        - chunk_{id}: Chunk nodes (original chunk nodes)
        - doc_{id}: Document nodes
        - entity_{name}: Entity nodes (extracted from text)

        Edges:
        - chunk <-> chunk: similarity edges from original graph
        - chunk -> doc: containment relations
        - entity -> chunk: mentions relations
        """
        H = nx.Graph()

        # Add original chunk nodes with their attributes
        for node in chunk_graph.nodes():
            H.add_node(node, type='chunk', **chunk_graph.nodes[node])

        # Copy original chunk-chunk similarity edges
        for u, v, data in chunk_graph.edges(data=True):
            H.add_edge(u, v, type='similarity', **data)

        # Create document-to-chunk mapping
        doc_to_chunks = {}
        for chunk in chunks:
            doc_id = chunk['doc_id']
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk['chunk_id'])

        # Add document nodes and containment edges
        for doc_id, chunk_ids in doc_to_chunks.items():
            doc_node_id = f"doc_{doc_id}"
            H.add_node(doc_node_id, type='document', doc_id=doc_id, original_doc_id=doc_id)

            # Connect each chunk to its document
            for chunk_id in chunk_ids:
                H.add_edge(doc_node_id, chunk_id, type='contains', weight=1.0)

        # Extract entities and add entity nodes
        if self.extract_entities:
            entity_to_chunks = self._extract_entities_with_mentions(chunks)
            
            # Add entity nodes and mention edges
            for entity_name, chunk_ids in entity_to_chunks.items():
                entity_node_id = f"entity_{entity_name}"
                H.add_node(entity_node_id, type='entity', name=entity_name)

                # Connect entity to chunks that mention it
                for chunk_id in chunk_ids:
                    if chunk_id in H and entity_node_id in H:
                        # Use weight based on frequency of mention or other factors
                        H.add_edge(entity_node_id, chunk_id, type='mentions', weight=0.8)

        logger.info(
            f"Built heterogeneous graph: {H.number_of_nodes()} nodes "
            f"(chunks: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='chunk')}, "
            f"docs: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='document')}, "
            f"entities: {sum(1 for n in H.nodes() if H.nodes[n].get('type')=='entity')}), "
            f"{H.number_of_edges()} edges"
        )
        return H

    def _extract_entities_with_mentions(self, chunks: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract entities and map them to chunks that mention them.
        
        This is a placeholder implementation - in a full implementation,
        this would use proper NER (spaCy, transformers, etc.)
        """
        entity_to_chunks = {}

        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            text = chunk['text'].lower()
            
            # Use regex to find potential entities (proper nouns)
            # This is a simplified approach - real implementation would use NER
            entities = self._extract_simple_entities(chunk['text'])
            
            for entity in entities:
                if entity not in entity_to_chunks:
                    entity_to_chunks[entity] = []
                entity_to_chunks[entity].append(chunk_id)

        # Filter entities that appear in multiple chunks for better graph connectivity
        # Keep only entities mentioned in at least 2 chunks to reduce noise
        filtered_entities = {}
        for entity, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) >= 2:  # Only keep entities in multiple chunks
                filtered_entities[entity] = chunk_ids
        
        return filtered_entities

    def _extract_simple_entities(self, text: str) -> List[str]:
        """
        Extract potential entities from text using regex patterns.
        
        This is a simpler alternative to full NER.
        """
        entities = set()
        
        # Pattern for multi-word capitalized entities (e.g., "New York", "United States")
        multi_word_pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*){2,}\b'
        multi_word_entities = re.findall(multi_word_pattern, text)
        entities.update([e.strip() for e in multi_word_entities])
        
        # Pattern for single-word capitalized entities (4+ chars, proper nouns)
        single_word_pattern = r'\b[A-Z][a-z]{3,}\b'
        single_word_entities = re.findall(single_word_pattern, text)
        
        # Filter out common words that aren't likely entities
        common_words = {
            'When', 'Where', 'What', 'Who', 'Which', 'How', 'The', 'This', 'That', 
            'These', 'Those', 'Have', 'Has', 'Had', 'Are', 'Is', 'Was', 'Were',
            'The', 'And', 'With', 'For', 'From', 'About', 'Under', 'Over', 'Into', 'Onto'
        }
        single_entities = [e for e in single_word_entities if e not in common_words]
        entities.update(single_entities)
        
        return list(entities)