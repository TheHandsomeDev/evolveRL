"""
LLM backend implementation for EvolveRL.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class LLMBackend:
    """Interface to OpenAI's GPT models."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        logger.info(f"Initializing LLM backend with model: {config.model if config else 'default'}")
        self.client = OpenAI(api_key=api_key)
        self.config = config or self.get_default_config()
    
    def get_default_config(self) -> LLMConfig:
        """Get default configuration."""
        return LLMConfig()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, config: Optional[LLMConfig] = None) -> str:
        """Generate text using the LLM."""
        config = config or self.config
        logger.debug(f"Generating with model {config.model}, temperature {config.temperature}")
        
        try:
            response = self.client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty
            )
            
            result = response.choices[0].message.content
            logger.debug(f"Generated {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def mutate_config(self, config: LLMConfig) -> LLMConfig:
        """Create a mutated version of the configuration."""
        import random
        
        # Randomly adjust parameters within reasonable bounds
        new_config = LLMConfig(
            model=config.model,
            temperature=min(1.0, max(0.0, config.temperature + random.uniform(-0.1, 0.1))),
            max_tokens=config.max_tokens,
            top_p=min(1.0, max(0.1, config.top_p + random.uniform(-0.1, 0.1))),
            frequency_penalty=min(2.0, max(-2.0, config.frequency_penalty + random.uniform(-0.2, 0.2))),
            presence_penalty=min(2.0, max(-2.0, config.presence_penalty + random.uniform(-0.2, 0.2)))
        )
        
        logger.debug(f"Mutated config: temp {new_config.temperature:.2f}, top_p {new_config.top_p:.2f}")
        return new_config
    
    def crossover_configs(self, config1: LLMConfig, config2: LLMConfig) -> LLMConfig:
        """Create a new configuration by combining two parent configurations."""
        import random
        
        # Randomly choose parameters from either parent
        new_config = LLMConfig(
            model=random.choice([config1.model, config2.model]),
            temperature=random.choice([config1.temperature, config2.temperature]),
            max_tokens=random.choice([config1.max_tokens, config2.max_tokens]),
            top_p=random.choice([config1.top_p, config2.top_p]),
            frequency_penalty=random.choice([config1.frequency_penalty, config2.frequency_penalty]),
            presence_penalty=random.choice([config1.presence_penalty, config2.presence_penalty])
        )
        
        logger.debug(f"Crossover config: temp {new_config.temperature:.2f}, top_p {new_config.top_p:.2f}")
        return new_config 