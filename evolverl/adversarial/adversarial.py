"""
Implementation of adversarial testing for EvolveRL.
"""

from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from ..agent import Agent
from ..llm import LLMBackend
from .test_case import AdversarialTestCase

logger = logging.getLogger(__name__)

class AdversarialTester(ABC):
    """Base class for generating test cases."""
    
    VALID_DIFFICULTIES = ["easy", "medium", "hard"]
    
    def __init__(
        self,
        difficulty: str = "medium",
        provider: str = "openai",
        llm_backend: Optional[LLMBackend] = None
    ) -> None:
        """Initialize tester."""
        if difficulty not in self.VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty: {difficulty}. "
                f"Must be one of {self.VALID_DIFFICULTIES}"
            )
        
        if provider not in ["openai", "anthropic"]:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.difficulty = difficulty
        self.provider = provider
        self.llm = llm_backend or LLMBackend(provider=provider)
    
    def generate_test_cases(
        self,
        agent: Agent,
        num_cases: int = 5
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple test cases."""
        return [
            self._generate_single_test(agent)
            for _ in range(num_cases)
        ]
    
    @abstractmethod
    def _generate_single_test(
        self,
        agent: Agent
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a single test case. Override in subclasses."""
        raise NotImplementedError 