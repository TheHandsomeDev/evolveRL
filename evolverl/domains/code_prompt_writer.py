"""
Code-specific prompt writer implementation for EvolveRL.
"""

from dataclasses import dataclass
from typing import List, Optional
import random

@dataclass
class CodePromptTemplate:
    """Template for code generation prompts."""
    instructions: str
    examples: List[str]
    context: Optional[str] = None
    constraints: Optional[str] = None

class CodePromptWriter:
    """Generates and mutates prompts for code generation tasks."""
    
    def __init__(self):
        self.base_template = """You are an expert programmer. Your task is to:
{instructions}

{context}

{examples}

{constraints}

Please provide a solution that is:
1. Correct and functional
2. Well-documented with docstrings
3. Type-hinted appropriately
4. Efficient and maintainable

Your solution:"""
    
    def generate_initial_prompt(self) -> str:
        """Generate an initial prompt for code generation."""
        template = CodePromptTemplate(
            instructions="Write a Python function that solves the given problem.",
            examples=["Example: def add(a: int, b: int) -> int:\n    return a + b"],
            context="You are writing production-quality code.",
            constraints="Follow PEP 8 style guidelines."
        )
        return self._format_template(template)
    
    def mutate_prompt(self, prompt: str) -> str:
        """Mutate an existing prompt."""
        # Parse the prompt back into a template
        template = self._parse_prompt(prompt)
        
        # Randomly choose mutation operations
        if random.random() < 0.3:
            template.instructions = self._mutate_instructions(template.instructions)
        if random.random() < 0.3:
            template.examples = self._mutate_examples(template.examples)
        if random.random() < 0.3:
            template.constraints = self._mutate_constraints(template.constraints)
            
        return self._format_template(template)
    
    def crossover_prompts(self, prompt1: str, prompt2: str) -> str:
        """Perform crossover between two prompts."""
        template1 = self._parse_prompt(prompt1)
        template2 = self._parse_prompt(prompt2)
        
        # Randomly combine elements from both templates
        new_template = CodePromptTemplate(
            instructions=random.choice([template1.instructions, template2.instructions]),
            examples=random.choice([template1.examples, template2.examples]),
            context=random.choice([template1.context, template2.context]),
            constraints=random.choice([template1.constraints, template2.constraints])
        )
        
        return self._format_template(new_template)
    
    def _format_template(self, template: CodePromptTemplate) -> str:
        """Format a template into a prompt string."""
        return self.base_template.format(
            instructions=template.instructions,
            context=template.context or "",
            examples="\n".join(["Examples:"] + template.examples) if template.examples else "",
            constraints=template.constraints or ""
        )
    
    def _parse_prompt(self, prompt: str) -> CodePromptTemplate:
        """Parse a prompt string back into a template."""
        # Simple parsing implementation
        lines = prompt.split("\n")
        instructions = ""
        examples = []
        context = ""
        constraints = ""
        
        current_section = None
        for line in lines:
            if "Your task is to:" in line:
                current_section = "instructions"
            elif "Examples:" in line:
                current_section = "examples"
            elif "Please provide a solution" in line:
                break
            elif line.strip():
                if current_section == "instructions":
                    instructions += line + "\n"
                elif current_section == "examples":
                    examples.append(line)
                    
        return CodePromptTemplate(
            instructions=instructions.strip(),
            examples=examples,
            context=context.strip() or None,
            constraints=constraints.strip() or None
        )
    
    def _mutate_instructions(self, instructions: str) -> str:
        """Mutate the instructions section."""
        # Add or modify requirements
        additions = [
            "Ensure optimal time complexity.",
            "Include comprehensive error handling.",
            "Make the solution thread-safe.",
            "Optimize for memory usage."
        ]
        return instructions + "\n" + random.choice(additions)
    
    def _mutate_examples(self, examples: List[str]) -> List[str]:
        """Mutate the examples section."""
        # Add or remove examples
        if random.random() < 0.5 and examples:
            return examples[:-1]  # Remove last example
        new_example = "Example: def reverse(s: str) -> str:\n    return s[::-1]"
        return examples + [new_example]
    
    def _mutate_constraints(self, constraints: Optional[str]) -> str:
        """Mutate the constraints section."""
        if not constraints:
            constraints = ""
        # Add or modify constraints
        new_constraints = [
            "Maximum time complexity: O(n)",
            "Maximum space complexity: O(1)",
            "Maximum function length: 20 lines",
            "Must include type hints"
        ]
        return constraints + "\n" + random.choice(new_constraints) 