"""Customer support specific test case generator."""
from typing import Tuple, Dict, Any
import logging

from ..agent import Agent
from .adversarial import AdversarialTester

logger = logging.getLogger(__name__)

class CustomerSupportTester(AdversarialTester):
    """Generates test cases for customer support scenarios."""
    
    def _generate_single_test(self, agent: Agent) -> Tuple[str, Dict[str, Any]]:
        """Generate a single customer support test case."""
        # Create prompt for test case generation
        prompt = self._create_test_prompt(agent)
        
        try:
            # Generate test case using LLM
            scenario = self.llm.generate(prompt)
            
            # Create metadata
            metadata = {
                "domain": "customer_support",
                "difficulty": self.difficulty,
                "expected_elements": [
                    "empathy",
                    "clarity",
                    "solution",
                    "professionalism"
                ]
            }
            
            return scenario, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate test case: {str(e)}")
            # Return a fallback test case
            return self._get_fallback_test()
    
    def _create_test_prompt(self, agent: Agent) -> str:
        """Create prompt for generating test case."""
        base_prompt = f"""Generate a challenging customer support scenario with the following characteristics:

Difficulty level: {self.difficulty}
Model capabilities: {agent.model}

The scenario should:
1. Present a realistic customer issue
2. Include relevant context and details
3. Require appropriate empathy and professionalism
4. Have clear success criteria
5. Match the specified difficulty level

For {self.difficulty} difficulty:"""

        if self.difficulty == "easy":
            base_prompt += """
- Common customer issues
- Clear problem statements
- Standard solutions available
- Minimal complexity"""
        elif self.difficulty == "medium":
            base_prompt += """
- More complex issues
- Multiple aspects to address
- Requires careful handling
- Some emotional elements"""
        else:  # hard
            base_prompt += """
- Complex edge cases
- Multiple stakeholders
- High emotional tension
- Policy conflicts to navigate"""
            
        base_prompt += "\n\nProvide only the scenario description, no additional formatting."
        
        return base_prompt
    
    def _get_fallback_test(self) -> Tuple[str, Dict[str, Any]]:
        """Get a fallback test case if generation fails."""
        fallback_scenarios = {
            "easy": (
                "Customer asking about order status for a package ordered 2 days ago.",
                {"expected_response": "Check status, explain shipping times, offer tracking"}
            ),
            "medium": (
                "Customer reporting partial damage to delivered item, outside return window.",
                {"expected_response": "Evaluate damage, consider warranty, find solution"}
            ),
            "hard": (
                "Customer threatening legal action over multiple failed delivery attempts.",
                {"expected_response": "De-escalate, document issues, find resolution"}
            )
        }
        
        scenario, extra_meta = fallback_scenarios[self.difficulty]
        metadata = {
            "domain": "customer_support",
            "difficulty": self.difficulty,
            "expected_elements": [
                "empathy",
                "clarity",
                "solution",
                "professionalism"
            ],
            **extra_meta
        }
        
        return scenario, metadata 