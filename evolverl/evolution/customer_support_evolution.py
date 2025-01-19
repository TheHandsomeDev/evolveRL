"""Customer support evolution implementation."""
from typing import List, Tuple, Dict, Any
import logging

from .evolution import Evolution
from ..agent import Agent

logger = logging.getLogger(__name__)

class CustomerSupportEvolution(Evolution):
    """Evolution specialized for customer support."""
    
    def _generate_base_prompt(self) -> str:
        """Generate base prompt template."""
        if self.use_case:
            return self.use_case.base_prompt
            
        return """You are a professional customer support agent. Your goal is to provide helpful, empathetic responses.

Task: {task}
Context: {context}

Follow these guidelines in your response:
1. Show empathy and understanding
2. Address the specific issue
3. Provide clear solutions
4. Maintain professional tone
5. Follow up appropriately""" 