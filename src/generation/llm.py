"""
LLM client for generation tasks.

Supports OpenAI-style API endpoints (local or cloud).
"""

import logging
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
        self.model_name = config['llm']['model_name']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        
        # Initialize OpenAI client
        api_base = config['llm'].get('api_base_url')
        api_key = config['llm'].get('api_key', 'dummy_key')  # Local LLMs may not need real key
        
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=config['llm'].get('timeout', 120)
        )
        
        logger.info(f"Initialized LLM client: {self.model_name} @ {api_base}")
    
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
                model=self.model_name,
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
