"""
Evolutionary Prompt Writer/Improver for generating and mutating prompts.
"""
from typing import List, Dict, Any, Optional
import random
from dataclasses import dataclass

from ..llm import LLMBackend, LLMConfig


@dataclass
class PromptMutationConfig:
    """Configuration for prompt mutation strategies."""
    mutation_types: List[str] = None  # Types of mutations to apply
    mutation_probability: float = 0.3  # Probability of each mutation type
    max_mutations: int = 2  # Maximum number of mutations per prompt
    
    def __post_init__(self):
        if self.mutation_types is None:
            self.mutation_types = [
                "rewrite_instructions",
                "change_examples",
                "reorder_blocks",
                "add_clarification",
                "change_style",
                "add_constraints"
            ]


class PromptWriter:
    """
    Generates and mutates prompts for evolutionary optimization.
    
    This class implements the Evolutionary Prompt Writer/Improver component
    described in the paper, using various strategies to generate and mutate
    prompts.
    
    Args:
        base_model (str): The LLM to use for prompt generation
        config (Optional[Dict[str, Any]]): Additional configuration
        api_key (Optional[str]): API key for the LLM service
    
    Example:
        >>> writer = PromptWriter(base_model="gpt-4o-mini")
        >>> prompts = writer.generate_initial_population(
        ...     task_description="Create a math problem solver",
        ...     population_size=5
        ... )
    """
    
    def __init__(
        self,
        base_model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ):
        self.base_model = base_model
        self.config = config or {}
        self.mutation_config = PromptMutationConfig()
        
        # Initialize LLM backend for prompt generation
        llm_config = LLMConfig(
            model=base_model,
            temperature=0.9,  # Higher temperature for more creative variations
            max_tokens=1000
        )
        self.llm = LLMBackend(config=llm_config, api_key=api_key)
    
    def generate_initial_population(
        self,
        task_description: str,
        population_size: int = 10
    ) -> List[str]:
        """
        Generate an initial population of prompts for a given task.
        
        Uses the LLM to generate diverse prompt variations based on the task
        description. Each prompt is designed to guide the model in solving
        the task effectively.
        
        Args:
            task_description (str): Description of the task to generate prompts for
            population_size (int): Number of prompts to generate
            
        Returns:
            List[str]: List of generated prompts
            
        Example:
            >>> prompts = writer.generate_initial_population(
            ...     "Create a system that solves math word problems",
            ...     population_size=3
            ... )
        """
        system_prompt = """You are a prompt engineering expert. Your task is to create
diverse and effective prompt templates for an AI model. Each prompt should:
1. Be clear and specific
2. Include step-by-step guidance
3. Encourage thorough and accurate responses
4. Be different from other prompts in approach or style"""
        
        generation_prompt = f"""Create a prompt template for the following task:
{task_description}

The prompt should include placeholders for {{task}} and optional {{context}}.
Generate a unique and effective approach that differs from standard templates.

Prompt template:"""
        
        prompts = []
        while len(prompts) < population_size:
            try:
                prompt = self.llm.generate(
                    prompt=generation_prompt,
                    system_prompt=system_prompt
                )
                if self._is_valid_prompt(prompt):
                    prompts.append(prompt)
            except Exception as e:
                continue
        
        return prompts
    
    def mutate_prompts(
        self,
        prompts: List[str],
        scores: List[float]
    ) -> List[str]:
        """
        Generate mutations of the best-performing prompts.
        
        This method:
        1. Sorts prompts by their scores
        2. Keeps the top performers
        3. Generates mutations of the best prompts
        
        Args:
            prompts (List[str]): Current population of prompts
            scores (List[float]): Performance scores for each prompt
            
        Returns:
            List[str]: New population of mutated prompts
            
        Example:
            >>> new_prompts = writer.mutate_prompts(
            ...     prompts=["Solve: {task}", "Analyze: {task}"],
            ...     scores=[0.8, 0.6]
            ... )
        """
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(
            zip(scores, prompts),
            reverse=True
        )]
        
        # Keep top performers and generate mutations
        top_k = max(1, len(prompts) // 4)
        new_population = sorted_prompts[:top_k]
        
        while len(new_population) < len(prompts):
            parent = random.choice(sorted_prompts[:top_k])
            child = self._apply_mutations(parent)
            new_population.append(child)
        
        return new_population
    
    def _apply_mutations(self, prompt: str) -> str:
        """Apply random mutations to a prompt."""
        num_mutations = random.randint(1, self.mutation_config.max_mutations)
        
        for _ in range(num_mutations):
            if random.random() < self.mutation_config.mutation_probability:
                mutation_type = random.choice(self.mutation_config.mutation_types)
                prompt = self._apply_single_mutation(prompt, mutation_type)
        
        return prompt
    
    def _apply_single_mutation(self, prompt: str, mutation_type: str) -> str:
        """
        Apply a single mutation of the specified type.
        
        Uses the LLM to generate intelligent mutations based on the
        mutation type and the current prompt.
        """
        system_prompt = """You are a prompt engineering expert. Your task is to
modify an existing prompt template to improve its effectiveness."""
        
        mutation_prompts = {
            "rewrite_instructions": """Rewrite the instructions in this prompt to be
more clear and effective, while keeping the same general approach:

{prompt}

Improved prompt:""",
            
            "change_examples": """Add or modify examples in this prompt to better
illustrate the expected approach and format:

{prompt}

Prompt with improved examples:""",
            
            "reorder_blocks": """Reorder and restructure the components of this prompt
to improve its logical flow:

{prompt}

Reordered prompt:""",
            
            "add_clarification": """Add clarifying instructions or notes to this prompt
to prevent common mistakes or misunderstandings:

{prompt}

Clarified prompt:""",
            
            "change_style": """Modify the tone and style of this prompt to be more
{style} while maintaining its effectiveness:

{prompt}

Restyled prompt:""",
            
            "add_constraints": """Add specific constraints or requirements to this prompt
to ensure higher quality responses:

{prompt}

Constrained prompt:"""
        }
        
        try:
            mutation_prompt = mutation_prompts[mutation_type].format(
                prompt=prompt,
                style=random.choice([
                    "professional", "conversational", "technical",
                    "educational", "analytical"
                ])
            )
            
            mutated = self.llm.generate(
                prompt=mutation_prompt,
                system_prompt=system_prompt
            )
            
            return mutated if self._is_valid_prompt(mutated) else prompt
            
        except Exception:
            return prompt
    
    def _is_valid_prompt(self, prompt: str) -> bool:
        """
        Check if a prompt is valid.
        
        A valid prompt must:
        1. Contain the required placeholders
        2. Be properly formatted
        3. Not be too short or too long
        """
        if not prompt or len(prompt) < 10:
            return False
        
        if "{task}" not in prompt:
            return False
        
        if len(prompt) > 2000:  # Avoid extremely long prompts
            return False
        
        return True 