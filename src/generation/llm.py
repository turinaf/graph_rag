"""
LLM client for generation tasks.

Supports OpenAI-style API endpoints (local or cloud).
"""

import logging
import os
import requests
from typing import Optional, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client for generation using OpenAI-style API.
    
    Works with:
    - Local LLMs (vLLM, Ollama, LM Studio, etc.)
    - OpenAI API
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client.
        
        Args:
            config: Configuration dictionary with llm settings
        """
        self.config = config
        llm_config = config['llm']
        
        # Get API settings from environment
        api_base = os.getenv('LLM_API_BASE') or llm_config.get('api_base', '')
        api_key = os.getenv('LLM_API_KEY') or llm_config.get('api_key', 'dummy_key')
        model_name = os.getenv('LLM_MODEL_NAME') or llm_config.get('model', 'gpt-4o-mini')
        
        # Replace ${VAR} syntax
        if api_base.startswith('${') and api_base.endswith('}'):
            var_name = api_base[2:-1]
            api_base = os.getenv(var_name, '')
        
        if not api_base:
            raise ValueError(
                "LLM_API_BASE not set. Please set it in your .env file:\n"
                "LLM_API_BASE=http://localhost:8078/v1"
            )
        
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model_name
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=config['llm'].get('timeout', 120)
        )
        
        logger.info(f"Initialized LLM client: {self.model} @ {self.api_base}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> list:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Max tokens per generation
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, temperature, max_tokens)
            results.append(result)
        return results
