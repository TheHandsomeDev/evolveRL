"""
LLM integration module for handling different model backends.
"""
from typing import Dict, Any, Optional, List
import os
import openai
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMConfig:
    """Configuration for LLM interactions."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class LLMBackend:
    """
    Handles interactions with different LLM backends.
    
    Currently supports:
    - OpenAI GPT models
    - (TODO: Add support for other models like Anthropic, local models)
    
    Args:
        config (LLMConfig): Configuration for the LLM
        api_key (Optional[str]): API key for the service
    """
    
    def __init__(
        self,
        config: LLMConfig,
        api_key: Optional[str] = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup API key
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "No API key provided. Either pass it directly or "
                "set the OPENAI_API_KEY environment variable."
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt (str): The prompt to send to the model
            system_prompt (Optional[str]): System prompt for chat models
            
        Returns:
            str: The model's response
        """
        try:
            if "gpt" in self.config.model.lower():
                return self._generate_openai(prompt, system_prompt)
            else:
                raise ValueError(f"Unsupported model: {self.config.model}")
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using OpenAI's API."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = openai.ChatCompletion.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            stop=self.config.stop
        )
        
        return response.choices[0].message.content.strip()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text using the model's embedding API.
        
        Args:
            text (str): Text to get embeddings for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise 