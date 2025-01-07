"""
AdversarialTester for generating challenging test cases.
"""
from typing import Optional, Dict, Any, List, Tuple
import random
import json
import logging

from ..agent import Agent
from ..llm import LLMBackend, LLMConfig


class AdversarialTester:
    """
    Generates challenging test cases to evaluate agent robustness.
    
    This class implements the Adversarial Models component from the paper,
    generating test cases designed to challenge and expose weaknesses in
    the evolutionary models.
    
    Args:
        difficulty (str): Difficulty level ("easy", "medium", "hard")
        domain (Optional[str]): Specific domain to generate tests for
        config (Optional[Dict[str, Any]]): Additional configuration
    """
    
    def __init__(
        self,
        difficulty: str = "medium",
        domain: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.difficulty = difficulty
        self.domain = domain
        self.config = config or {}
        self.test_history: List[Dict[str, Any]] = []
        
        # Initialize LLM backend for test generation
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.9,  # Higher temperature for more diverse tests
            max_tokens=500
        )
        self.llm = LLMBackend(config=llm_config)
        self.logger = logging.getLogger(__name__)
    
    def generate_test_cases(
        self,
        agent: Agent,
        num_cases: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate adversarial test cases for an agent.
        
        This method analyzes the agent's prompt template and configuration
        to generate challenging test cases that might expose weaknesses.
        
        Args:
            agent (Agent): The agent to generate tests for
            num_cases (int): Number of test cases to generate
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of (test_case, metadata) pairs
        """
        test_cases = []
        
        # Generate system prompt based on difficulty and domain
        system_prompt = self._get_system_prompt()
        
        for _ in range(num_cases):
            try:
                # Generate a test case
                test_case, metadata = self._generate_single_test(
                    agent,
                    system_prompt
                )
                test_cases.append((test_case, metadata))
                
                # Record in history
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for test generation."""
        difficulty_prompts = {
            "easy": "Create straightforward but non-trivial test cases.",
            "medium": "Create moderately challenging test cases with some complexity.",
            "hard": "Create very challenging test cases that test edge cases and complex scenarios."
        }
        
        domain_prompts = {
            "math": """Focus on mathematical problems that require:
- Multiple steps or concepts
- Careful attention to units and conversions
- Understanding of mathematical principles
- Edge cases in calculations""",
            
            "code": """Focus on programming challenges that require:
- Proper error handling
- Edge case consideration
- Efficiency considerations
- Clear documentation and structure""",
            
            "defi": """Focus on DeFi scenarios that involve:
- Complex token interactions
- Multiple protocols
- Risk considerations
- Market dynamics""",
            
            None: "Create general-purpose test cases that challenge understanding and reasoning."
        }
        
        return f"""You are an expert at creating challenging test cases.
Your goal is to generate test cases that will help evaluate and improve AI models.

Difficulty Level: {difficulty_prompts.get(self.difficulty, difficulty_prompts["medium"])}

Domain Focus: {domain_prompts.get(self.domain, domain_prompts[None])}

Each test case should:
1. Be clear and unambiguous
2. Have a well-defined correct approach/answer
3. Test important skills and concepts
4. Challenge common assumptions
5. Expose potential weaknesses"""
    
    def _generate_single_test(
        self,
        agent: Agent,
        system_prompt: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a single adversarial test case."""
        if self.domain == "math":
            test = self._generate_math_test()
        elif self.domain == "code":
            test = self._generate_code_test()
        elif self.domain == "defi":
            test = self._generate_defi_test()
        else:
            test = self._generate_general_test()
        
        metadata = {
            "difficulty": self.difficulty,
            "domain": self.domain,
            "expected_behavior": "Provide accurate and relevant information",
            "potential_pitfalls": self._identify_pitfalls(test)
        }
        
        return test, metadata
    
    def _generate_math_test(self) -> str:
        """Generate a math-specific test case."""
        prompt = f"""Create a challenging math problem that requires careful analysis.
Difficulty: {self.difficulty}

The problem should:
1. Require multiple steps
2. Include potential traps or common mistakes
3. Test mathematical reasoning
4. Be clearly stated

Problem:"""
        
        return self.llm.generate(prompt)
    
    def _generate_code_test(self) -> str:
        """Generate a code-specific test case."""
        prompt = f"""Create a programming challenge that tests coding ability.
Difficulty: {self.difficulty}

The challenge should:
1. Require algorithmic thinking
2. Include edge cases to handle
3. Consider efficiency
4. Have clear requirements

Challenge:"""
        
        return self.llm.generate(prompt)
    
    def _generate_defi_test(self) -> str:
        """Generate a DeFi-specific test case."""
        templates = [
            "Analyze the liquidity pool {token1}/{token2} on {dex}",
            "Calculate the impermanent loss for position {position_id}",
            "Evaluate the yield farming strategy on {protocol}"
        ]
        
        tokens = ["ETH", "USDC", "WBTC", "DAI"]
        dexes = ["Uniswap", "Curve", "Balancer"]
        protocols = ["Aave", "Compound", "MakerDAO"]
        
        template = random.choice(templates)
        
        if "{token1}" in template and "{token2}" in template:
            token1, token2 = random.sample(tokens, 2)
            return template.format(
                token1=token1,
                token2=token2,
                dex=random.choice(dexes)
            )
        elif "{protocol}" in template:
            return template.format(protocol=random.choice(protocols))
        else:
            return template.format(position_id=random.randint(1000, 9999))
    
    def _generate_general_test(self) -> str:
        """Generate a general-purpose test case."""
        prompt = f"""Create a challenging test case that requires careful analysis and reasoning.
Difficulty: {self.difficulty}

The test should:
1. Challenge assumptions
2. Require critical thinking
3. Have multiple aspects to consider
4. Be clearly stated

Test case:"""
        
        return self.llm.generate(prompt)
    
    def _identify_pitfalls(self, test: str) -> List[str]:
        """Identify potential pitfalls in a test case."""
        prompt = f"""Analyze this test case and identify potential pitfalls or mistakes that might be made when solving it.
List each pitfall on a new line starting with "- ".

Test case: {test}

Potential pitfalls:"""
        
        try:
            response = self.llm.generate(prompt)
            pitfalls = [
                line[2:].strip()
                for line in response.split("\n")
                if line.startswith("- ")
            ]
            return pitfalls
        except Exception:
            return ["Unknown pitfalls"]
    
    def save_history(self, path: str) -> None:
        """Save test history to a file."""
        with open(path, 'w') as f:
            json.dump(self.test_history, f, indent=2)
    
    @classmethod
    def load_history(cls, path: str) -> List[Dict[str, Any]]:
        """Load test history from a file."""
        with open(path, 'r') as f:
            return json.load(f) 