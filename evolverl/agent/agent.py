from dataclasses import dataclass
import json
from typing import Optional
from ..llm import LLMConfig, LLMBackend

@dataclass
class AgentConfig:
    """Configuration settings for Agent"""
    llm_config: LLMConfig
    prompt_template: str = ""
    score: int = 0

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization"""
        llm_dict = {
            "model_name": self.llm_config.model_name,
            "model_type": self.llm_config.model_type,
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature
        }
        return {
            "llm_config": llm_dict,
            "prompt_template": self.prompt_template,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        """Create config from dictionary"""
        llm_config = LLMConfig(**data["llm_config"])
        return cls(
            llm_config=llm_config,
            prompt_template=data["prompt_template"],
            score=data["score"]
        )

    def update_score(self, score: int) -> None:
        """Update the agent's score (0-1000)"""
        if not 0 <= score <= 1000:
            raise ValueError("Score must be between 0 and 1000")
        self.score = score

class Agent:
    """Agent class that manages LLM interactions with configurable settings"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(LLMConfig())
        self.llm_backend = LLMBackend(self.config.llm_config)
        
    def set_default_prompt(self, prompt: str) -> None:
        """Set the default prompt template"""
        self.config.prompt_template = prompt
        
    def update_score(self, score: int) -> None:
        """Update the agent's score"""
        self.config.update_score(score)
        
    async def send_message(self, message: str) -> str:
        """Send a message to the LLM using prompt template as system prompt"""
        return await self.llm_backend.generate(
            prompt=message,
            system_prompt=self.config.prompt_template
        )
    
    def save_config(self, filepath: str) -> None:
        """Save the current configuration to a JSON file"""
        config_dict = self.config.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def load_config(self, filepath: str) -> None:
        """Load configuration from a JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        self.config = AgentConfig.from_dict(config_dict)
        self.llm_backend = LLMBackend(self.config.llm_config) 