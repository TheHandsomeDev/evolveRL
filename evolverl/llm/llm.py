from dataclasses import dataclass
from typing import Optional, Literal, List, Dict
import openai
import anthropic

ModelType = Literal["openai", "anthropic"]

@dataclass
class LLMConfig:
    """Configuration settings for LLM backends"""
    
    # Common settings
    model_name: str = "gpt-4o-mini"
    model_type: ModelType = "openai"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # OpenAI specific settings
    openai_api_key: Optional[str] = None
    
    # Anthropic specific settings
    anthropic_api_key: Optional[str] = None

class LLMBackend:
    """Backend for interacting with different LLM providers"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate client based on model type"""
        if self.config.model_type == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            openai.api_key = self.config.openai_api_key
            # self._client = openai.OpenAI()
            self._client = openai.AsyncOpenAI() # use asynchronous client
            
        elif self.config.model_type == "anthropic":
            if not self.config.anthropic_api_key:
                raise ValueError("Anthropic API key is required for Anthropic models")
            # self._client = anthropic.Client(api_key=self.config.anthropic_api_key)
            self._client = anthropic.AsyncClient(api_key=self.config.anthropic_api_key) # use asynchronous client   
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured LLM with optional system prompt"""
        if self.config.model_type == "openai":
            messages: List[Dict[str, str]] = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
            
        elif self.config.model_type == "anthropic":
            response = await self._client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt if system_prompt else None,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        raise ValueError(f"Unsupported model type: {self.config.model_type}")

    @classmethod
    def from_config(cls, config: LLMConfig) -> 'LLMBackend':
        """Create an LLMBackend instance from a config object"""
        return cls(config) 