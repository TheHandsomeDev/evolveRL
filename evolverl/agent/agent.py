"""Agent implementation."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import logging

from ..llm import LLMBackend, OpenAILLMConfig, AnthropicLLMConfig

logger = logging.getLogger(__name__)

@dataclass
class Agent:
    """LLM-based agent."""
    
    model: str
    provider: str
    prompt_template: str
    fitness: float = 0.0
    llm: Optional[LLMBackend] = None
    
    def __post_init__(self):
        """Initialize LLM backend if not provided."""
        if self.llm is None:
            config = self._load_config()
            self.llm = LLMBackend(
                config=config,
                provider=self.provider
            )
    
    @classmethod
    def from_openai(cls, model: str = "gpt-4o-mini", prompt_template: str = ""):
        """Create agent with OpenAI configuration."""
        return cls(
            model=model,
            provider="openai",
            prompt_template=prompt_template,
            llm=LLMBackend(
                config=OpenAILLMConfig(model=model),
                provider="openai"
            )
        )
    
    @classmethod
    def from_anthropic(cls, model: str = "claude-3-5-sonnet-20241022", prompt_template: str = ""):
        """Create agent with Anthropic configuration."""
        return cls(
            model=model,
            provider="anthropic",
            prompt_template=prompt_template,
            llm=LLMBackend(
                config=AnthropicLLMConfig(model=model),
                provider="anthropic"
            )
        )
    
    def _load_config(self) -> OpenAILLMConfig | AnthropicLLMConfig:
        """Load appropriate LLM configuration."""
        if self.provider == "openai":
            return OpenAILLMConfig(model=self.model)
        elif self.provider == "anthropic":
            return AnthropicLLMConfig(model=self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def run(self, task: str, context: Dict[str, Any]) -> str:
        """Run agent on task."""
        try:
            prompt = self.prompt_template.format(
                task=task,
                context=context
            )
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Agent run failed: {str(e)}")
            return ""
    
    def save_state(self, path: str) -> None:
        """Save agent state to file."""
        state = {
            "model": self.model,
            "provider": self.provider,
            "prompt_template": self.prompt_template,
            "fitness": self.fitness
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
    
    @classmethod
    def load_state(cls, path: str) -> 'Agent':
        """Load agent state from file."""
        with open(path) as f:
            state = json.load(f)
        return cls(**state) 