"""
Text encoder using HTTP-based embedding service.

Supports Docker containerized embedding services including vLLM.
"""

import logging
import requests
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Text encoder using HTTP-based embedding service.
    
    Works with Docker containerized embedding models.
    Supports vLLM embedding API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding encoder.
        
        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config
        self.api_url = config['embedding']['api_url']
        self.batch_size = config['embedding']['batch_size']
        self.timeout = config['embedding'].get('timeout', 60)
        self.provider = config['embedding'].get('provider', 'http')
        self.model_name = config['embedding'].get('model_name', 'default')
        
        logger.info(f"Initialized embedding encoder: {self.api_url} (provider: {self.provider})")
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"Could not connect to embedding service: {e}")
    
    def _test_connection(self):
        """Test connection to embedding service."""
        if self.provider == 'vllm':
            response = requests.post(
                self.api_url,
                json={"input": ["test"], "model": self.model_name},
                timeout=10
            )
        else:
            response = requests.post(
                self.api_url,
                json={"texts": ["test"]},
                timeout=10
            )
        response.raise_for_status()
        logger.info("Successfully connected to embedding service")
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # vLLM API format
                if self.provider == 'vllm':
                    response = requests.post(
                        self.api_url,
                        json={
                            "input": batch,
                            "model": self.model_name
                        },
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    # Extract embeddings from vLLM response format
                    data = response.json()
                    batch_embeddings = [item['embedding'] for item in data['data']]
                    embeddings.extend(batch_embeddings)
                else:
                    # Generic HTTP embedding format
                    response = requests.post(
                        self.api_url,
                        json={"texts": batch},
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    batch_embeddings = response.json()['embeddings']
                    embeddings.extend(batch_embeddings)
                
                logger.debug(f"Encoded batch {i // self.batch_size + 1}, {len(batch)} texts")
                
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                raise
        
        embeddings = np.array(embeddings)
        
        if normalize and len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        logger.debug(f"Encoded {len(texts)} texts to shape {embeddings.shape}")
        return embeddings
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            emb1: First embedding array (n, dim)
            emb2: Second embedding array (m, dim)
            
        Returns:
            Similarity matrix (n, m)
        """
        return np.dot(emb1, emb2.T)
