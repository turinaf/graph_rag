"""
Text encoder using ZhipuAI embedding service.

Supports ZhipuAI embedding models via the zai library.
"""

import logging
import os
import numpy as np
from typing import List, Dict, Any

try:
    from zai import ZhipuAiClient
except ImportError:
    raise ImportError("zai library is required. Install with: pip install zai")

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Text encoder using ZhipuAI embedding service.
    
    Uses the zai library for ZhipuAI API integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding encoder.
        
        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config
        embedding_config = config['embedding']
        
        # Get API key from environment or config
        self.api_key = os.getenv('LLM_API_KEY') or os.getenv('ZHIPUAI_API_KEY') or embedding_config.get('api_key', '')
        print("API_KEY: ", self.api_key)
        if not self.api_key:
            raise ValueError(
                "API key not set. Please set it in your .env file:\n"
                "LLM_API_KEY=your_actual_api_key_here\n"
                "or ZHIPUAI_API_KEY=your_actual_api_key_here\n"
                "Get your key from: https://open.bigmodel.cn/"
            )
        
        # self.model_name = embedding_config.get('model_name', 'embedding-3')
        self.batch_size = embedding_config.get('batch_size', 32)
        # self.timeout = config['embedding'].get('timeout', 60)
        # self.provider = config['embedding'].get('provider', 'zhipuai')
        
        # Initialize ZhipuAI client
        try:
            self.client = ZhipuAiClient(api_key=self.api_key)
            logger.info(f"Initialized ZhipuAI embedding encoder (model: embedding-2)")
        except Exception as e:
            logger.error(f"Failed to initialize ZhipuAI client: {e}")
            raise
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"Could not connect to ZhipuAI embedding service: {e}")
            raise
    
    def _test_connection(self):
        """Test connection to ZhipuAI embedding service."""
        try:
            response = self.client.embeddings.create(
                model="embedding-3",
                dimensions=1024,
                input=["test connection"]
            )
            if hasattr(response, 'data') and len(response.data) > 0:
                logger.info("✓ Successfully connected to ZhipuAI embedding service")
            else:
                raise ValueError("Invalid response format from ZhipuAI API")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings using ZhipuAI API.
        
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
                response = self.client.embeddings.create(
                    model="embedding-3",
                    dimensions=1024,
                    input=batch,
                )
                
                # Extract embeddings from response
                if hasattr(response, 'data') and response.data:
                    # Sort by index to maintain order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    batch_embeddings = [item.embedding for item in sorted_data]
                    embeddings.extend(batch_embeddings)
                else:
                    raise ValueError(f"Invalid response format: {response}")
                
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
