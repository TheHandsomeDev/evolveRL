"""Base AdversarialTester class for generating test cases."""
from typing import Optional, Dict, Any, List, Tuple, Type
import random
import json
import logging

from ..agent import Agent
from ..llm import LLMBackend, OpenAILLMConfig, AnthropicLLMConfig


class AdversarialTester:
    """Base class for generating challenging test cases."""
    
    def __init__(
        self,
        difficulty: str = "medium",
        domain: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: Optional[str] = None
    ):
        self.difficulty = difficulty
        self.domain = domain
        self.config = config or {}
        self.test_history: List[Dict[str, Any]] = []
        
        # Initialize LLM backend
        if provider == "openai":
            llm_config = OpenAILLMConfig(
                model=model or "gpt-4o-mini",
                temperature=self.config.get("temperature", 0.9),
                max_tokens=self.config.get("max_tokens", 500)
            )
        elif provider == "anthropic":
            llm_config = AnthropicLLMConfig(
                model=model or "claude-3-5-sonnet-20241022",
                temperature=self.config.get("temperature", 0.9),
                max_tokens=self.config.get("max_tokens", 4096)
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        self.llm = LLMBackend(config=llm_config)
        self.logger = logging.getLogger(__name__)

    def generate_test_cases(
        self,
        agent: Agent,
        num_cases: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate test cases for the agent."""
        test_cases = []
        
        for _ in range(num_cases):
            try:
                test_case, metadata = self._generate_single_test(agent)
                test_cases.append((test_case, metadata))
                self.test_history.append({
                    "test_case": test_case,
                    "metadata": metadata,
                    "difficulty": self.difficulty,
                    "domain": self.domain
                })
            except Exception as e:
                self.logger.error(f"Error generating test case: {str(e)}")
                continue
        
        return test_cases

    def _generate_single_test(
        self,
        agent: Agent
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a single test case. Override in subclasses."""
        raise NotImplementedError 