"""
LLM-prompted chunking for document processing.

Uses LLM to extract semantic chunks from documents.
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class LLMChunker:
    """LLM-based semantic chunking."""
    
    def __init__(self, llm_client, prompts: dict, config: dict):
        """
        Initialize chunker.
        
        Args:
            llm_client: LLM client for generation
            prompts: Dictionary of prompts
            config: Configuration dictionary
        """
        self.llm = llm_client
        self.prompts = prompts
        self.batch_size = config['chunking']['batch_size']
        self.max_chunk_size = config['chunking'].get('max_chunk_size', 512)
        
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk all documents using LLM.
        
        Args:
            documents: List of {doc_id, title, text}
            
        Returns:
            List of {chunk_id, text, doc_id, doc_title}
        """
        all_chunks = []
        chunk_id = 0
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        for doc in documents:
            doc_text = doc['text']
            
            # If document is too long, split into paragraphs first
            if len(doc_text) > 2000:
                paragraphs = self._split_into_paragraphs(doc_text)
            else:
                paragraphs = [doc_text]
            
            for para in paragraphs:
                if not para.strip():
                    continue
                
                try:
                    # Use LLM to extract chunks
                    chunks = self._extract_chunks_llm(para)
                    
                    # Add metadata
                    for chunk_text in chunks:
                        all_chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text,
                            'doc_id': doc['doc_id'],
                            'doc_title': doc['title']
                        })
                        chunk_id += 1
                        
                except Exception as e:
                    logger.warning(f"Error chunking paragraph: {e}. Using fallback.")
                    # Fallback: use simple sentence splitting
                    fallback_chunks = self._fallback_chunking(para)
                    for chunk_text in fallback_chunks:
                        all_chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text,
                            'doc_id': doc['doc_id'],
                            'doc_title': doc['title']
                        })
                        chunk_id += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or period followed by newline
        paragraphs = re.split(r'\n\n+|\.\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_chunks_llm(self, text: str) -> List[str]:
        """Extract chunks using LLM."""
        prompt = self.prompts['chunking_prompt'].format(text=text)
        
        response = self.llm.generate(prompt, max_tokens=2000)
        
        # Parse chunks from response
        chunks = self._parse_chunks(response)
        
        if not chunks:
            # If parsing failed, use fallback
            chunks = self._fallback_chunking(text)
        
        return chunks
    
    def _parse_chunks(self, llm_response: str) -> List[str]:
        """
        Parse chunks from LLM response.
        
        Looks for patterns like:
        [Extracted Chunks in Sentence X]
        [Chunk text here.]
        [Another chunk here.]
        """
        chunks = []
        
        # Find all text within square brackets that look like chunks
        # Pattern: [text that doesn't start with "Sentence" or "Extracted"]
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for bracketed content that's not a section header
            if line.startswith('[') and line.endswith(']'):
                # Skip headers
                if any(keyword in line.lower() for keyword in ['sentence', 'extracted chunks', 'start', 'end', 'chunk 1', 'chunk 2']):
                    continue
                # Extract chunk text
                chunk_text = line[1:-1].strip()
                if len(chunk_text) > 10:  # Minimum chunk length
                    chunks.append(chunk_text)
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback: simple sentence-based chunking."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # If adding this sentence would exceed max size, save current chunk
            if len(current_chunk) + len(sent) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sent
            else:
                current_chunk += " " + sent if current_chunk else sent
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


def simple_chunker(documents: List[Dict], chunk_size: int = 512) -> List[Dict]:
    """
    Simple chunking without LLM (for testing/fallback).
    
    Args:
        documents: List of documents
        chunk_size: Maximum characters per chunk
        
    Returns:
        List of chunks
    """
    all_chunks = []
    chunk_id = 0
    
    for doc in documents:
        text = doc['text']
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sent
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Current chunk is full, save it
                all_chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'doc_id': doc['doc_id'],
                    'doc_title': doc['title']
                })
                chunk_id += 1
                current_chunk = sent
            else:
                # Add sentence to current chunk
                current_chunk = potential_chunk
        
        # Save last chunk if not empty
        if current_chunk.strip():
            all_chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'doc_id': doc['doc_id'],
                'doc_title': doc['title']
            })
            chunk_id += 1
    
    return all_chunks
