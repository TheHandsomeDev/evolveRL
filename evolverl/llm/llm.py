"""LLM backend implementation."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import time
import logging
from openai import OpenAI
from anthropic import Anthropic

logger = logging.getLogger(__name__)

@dataclass
class OpenAILLMConfig:
    """Configuration for OpenAI models."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class AnthropicLLMConfig:
    """Configuration for Anthropic models."""
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 4096

class LLMBackend:
    """Backend for interacting with LLM providers."""
    
    def __init__(
        self,
        config: Optional[OpenAILLMConfig | AnthropicLLMConfig] = None,
        provider: str = "openai",
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize LLM backend.
        
        Args:
            config: LLM configuration
            provider: LLM provider ("openai" or "anthropic")
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        if provider not in ["openai", "anthropic"]:
            raise ValueError("Unsupported provider: {}".format(provider))
        
        self.provider = provider
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Set default config if none provided
        if config is None:
            config = OpenAILLMConfig() if provider == "openai" else AnthropicLLMConfig()
        self.config = config
        
        # Initialize client
        if provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI()
            self.config.top_p = config.top_p
            self.config.frequency_penalty = config.frequency_penalty
            self.config.presence_penalty = config.presence_penalty
        else:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic()
    
    def generate(self, prompt: str) -> str:
        """Generate text using the configured LLM."""
        for attempt in range(self.retry_attempts):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt)
                else:
                    return self._generate_anthropic(prompt)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        config = self.config
        response = self.client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate text using Anthropic."""
        config = self.config
        response = self.client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text 