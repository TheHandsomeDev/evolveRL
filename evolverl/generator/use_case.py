"""Use case generation module."""
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
import json

from ..llm import LLMBackend

logger = logging.getLogger(__name__)

@dataclass
class UseCase:
    """Generated use case configuration."""
    domain: str
    description: str
    train_data: List[Dict[str, Any]]
    val_data: List[Dict[str, Any]]
    base_prompt: str
    judge_prompt: str
    adversary_prompt: str

class UseCaseGenerator:
    """Generates use case configurations from descriptions."""
    
    def __init__(self, provider: str = "openai"):
        """Initialize generator."""
        self.llm = LLMBackend(provider=provider)
    
    def generate(self, description: str) -> UseCase:
        """Generate complete use case configuration."""
        # First determine the domain
        domain = self._determine_domain(description)
        
        # Generate training data
        train_data = self._generate_train_data(domain, description)
        
        # Generate validation data
        val_data = self._generate_val_data(domain, description)
        
        # Generate prompts
        base_prompt = self._generate_base_prompt(domain, description)
        judge_prompt = self._generate_judge_prompt(domain, description)
        adversary_prompt = self._generate_adversary_prompt(domain, description)
        
        return UseCase(
            domain=domain,
            description=description,
            train_data=train_data,
            val_data=val_data,
            base_prompt=base_prompt,
            judge_prompt=judge_prompt,
            adversary_prompt=adversary_prompt
        )
    
    def _determine_domain(self, description: str) -> str:
        """Determine the domain from description."""
        prompt = f"""Given the following use case description, determine the most appropriate domain.
Choose from: code, customer_support, math, or defi.

Description: {description}

Return only the domain name, nothing else."""
        
        try:
            domain = self.llm.generate(prompt).strip().lower()
            if domain not in ["code", "customer_support", "math", "defi"]:
                logger.warning(f"Invalid domain '{domain}', defaulting to 'code'")
                return "code"
            return domain
        except Exception as e:
            logger.error(f"Error determining domain: {str(e)}")
            return "code"
    
    def _generate_train_data(self, domain: str, description: str) -> List[Dict[str, Any]]:
        """Generate training examples."""
        prompt = f"""Generate 5 diverse training examples for the following use case.
Each example should include a task description, context, and ground truth solution.

Domain: {domain}
Use Case: {description}

Format each example as a JSON object with:
- task: Task description
- context: Relevant context
- ground_truth: Ideal solution
- difficulty: "easy", "medium", or "hard"

Return a JSON array of examples."""
        
        response = self.llm.generate(prompt)
        try:
            if response[:7] == "```json":
                response = response[7:]
            if response[-3:] == "```":
                response = response[:-3]
            logger.info(f"Training data loaded: {response}")
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse training data: {e},", "response:", response)
            return []
    
    def _generate_val_data(self, domain: str, description: str) -> List[Dict[str, Any]]:
        """Generate validation examples."""
        prompt = f"""Generate 3 challenging validation examples for the following use case.
These should be more complex than training examples to test generalization.

Domain: {domain}
Use Case: {description}

Format each example as a JSON object with:
- task: Task description
- context: Relevant context
- ground_truth: Ideal solution
- difficulty: "medium" or "hard"

Return a JSON array of examples."""
        
        response = self.llm.generate(prompt)
        try:
            if response[:7] == "```json":
                response = response[7:]
            if response[-3:] == "```":
                response = response[:-3]
            logger.info(f"Validation data loaded: {response}")
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse validation data: {e},", "response:", response)
            return []
    
    def _generate_base_prompt(self, domain: str, description: str) -> str:
        """Generate base agent prompt template."""
        prompt = f"""Create a system prompt template for an AI agent handling the following use case.
The prompt should guide the agent to provide high-quality responses.

Domain: {domain}
Use Case: {description}

The template should:
1. Define the agent's role and capabilities
2. Specify expected response format
3. Include placeholders for {{task}} and {{context}}
4. Emphasize important aspects for this domain

Return only the prompt template."""
        
        return self.llm.generate(prompt)
    
    def _generate_judge_prompt(self, domain: str, description: str) -> str:
        """Generate judge prompt template."""
        prompt = f"""Create a prompt template for a judge evaluating responses to the following use case.
The judge should assess response quality across multiple criteria.

Domain: {domain}
Use Case: {description}

The template should:
1. Define evaluation criteria
2. Specify scoring guidelines
3. Include examples of good/bad responses
4. Focus on domain-specific quality aspects

Return only the prompt template."""
        
        return self.llm.generate(prompt)
    
    def _generate_adversary_prompt(self, domain: str, description: str) -> str:
        """Generate adversary prompt template."""
        prompt = f"""Create a prompt template for generating challenging test cases for the following use case.
The adversary should create diverse and realistic scenarios.

Domain: {domain}
Use Case: {description}

The template should:
1. Define test case generation guidelines
2. Specify complexity levels
3. Include edge case considerations
4. Focus on domain-specific challenges

Return only the prompt template."""
        
        return self.llm.generate(prompt) 