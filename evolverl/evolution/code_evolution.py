"""Code-specific evolution implementation."""
from typing import List, Tuple, Dict, Any
import logging

from .evolution import Evolution
from ..agent import Agent

logger = logging.getLogger(__name__)

class CodeEvolution(Evolution):
    """Evolution specialized for code generation."""
    
    def _generate_base_prompt(self) -> str:
        """Generate base prompt template."""
        if self.use_case:
            return self.use_case.base_prompt
            
        return """You are an expert programmer. Your task is to write clear, efficient, and well-documented code.

Task: {task}
Context: {context}

Write your solution following these guidelines:
1. Use clear variable and function names
2. Include type hints and docstrings
3. Handle edge cases and errors
4. Consider performance implications
5. Follow language-specific best practices""" 