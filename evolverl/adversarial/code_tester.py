"""Code-specific test case generator."""
from typing import Tuple, Dict, Any
import logging

from ..agent import Agent
from .adversarial import AdversarialTester

logger = logging.getLogger(__name__)

class CodeAdversarialTester(AdversarialTester):
    """Generates test cases for code tasks."""
    
    def _generate_single_test(self, agent: Agent) -> Tuple[str, Dict[str, Any]]:
        """Generate a single code test case."""
        prompt = self._create_test_prompt(agent)
        
        try:
            # Generate test case using LLM
            scenario = self.llm.generate(prompt)
            
            # Create metadata
            metadata = {
                "domain": "code",
                "difficulty": self.difficulty,
                "expected_elements": [
                    "algorithm_design",
                    "code_quality",
                    "error_handling",
                    "efficiency"
                ]
            }
            
            return scenario, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate test case: {str(e)}")
            return self._get_fallback_test()
    
    def _create_test_prompt(self, agent: Agent) -> str:
        """Create prompt for generating test case."""
        base_prompt = f"""Generate a challenging coding problem with the following characteristics:

Difficulty level: {self.difficulty}
Model capabilities: {agent.model}

The problem should:
1. Test algorithmic thinking
2. Require proper error handling
3. Consider efficiency
4. Have clear requirements
5. Match the specified difficulty level

For {self.difficulty} difficulty:"""

        if self.difficulty == "easy":
            base_prompt += """
- Basic algorithms
- Clear input/output
- Single concept focus
- Minimal edge cases"""
        elif self.difficulty == "medium":
            base_prompt += """
- Intermediate algorithms
- Multiple concepts
- Important edge cases
- Efficiency considerations"""
        else:  # hard
            base_prompt += """
- Complex algorithms
- Multiple optimizations
- Extensive edge cases
- System design aspects"""
            
        base_prompt += "\n\nProvide only the problem description, no solutions."
        
        return base_prompt
    
    def _get_fallback_test(self) -> Tuple[str, Dict[str, Any]]:
        """Get a fallback test case if generation fails."""
        fallback_scenarios = {
            "easy": (
                "Write a function to find the maximum value in an array.",
                {"expected_response": "Implement linear search with error handling"}
            ),
            "medium": (
                "Implement a function to detect cycles in a linked list.",
                {"expected_response": "Use Floyd's cycle detection algorithm"}
            ),
            "hard": (
                "Design a thread-safe cache with LRU eviction policy.",
                {"expected_response": "Implement with proper synchronization"}
            )
        }
        
        scenario, extra_meta = fallback_scenarios[self.difficulty]
        metadata = {
            "domain": "code",
            "difficulty": self.difficulty,
            "expected_elements": [
                "algorithm_design",
                "code_quality",
                "error_handling",
                "efficiency"
            ],
            **extra_meta
        }
        
        return scenario, metadata 