"""
Judge module for evaluating LLM responses.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from ..adversarial import TestCase

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class JudgingCriteria:
    """Criteria for judging responses."""
    correctness: float = 1.0
    clarity: float = 0.5
    efficiency: float = 0.7
    robustness: float = 0.6
    completeness: float = 0.5
    consistency: float = 0.4

class Judge:
    """Evaluates LLM responses using multiple criteria."""
    
    def __init__(self, criteria: JudgingCriteria, code_evaluator: Optional[Any] = None):
        self.criteria = criteria
        self.code_evaluator = code_evaluator
        self._metrics = {}
    
    def evaluate(self, test_case: TestCase, response: str) -> float:
        """Evaluate a response using the judging criteria."""
        logger.debug(f"Evaluating response of length {len(response)}")
        self._metrics = {}
        
        # Evaluate each criterion
        self._metrics["correctness"] = self._evaluate_correctness(test_case, response)
        self._metrics["clarity"] = self._evaluate_clarity(response)
        self._metrics["efficiency"] = self._evaluate_efficiency(test_case, response)
        self._metrics["robustness"] = self._evaluate_robustness(test_case, response)
        self._metrics["completeness"] = self._evaluate_completeness(test_case, response)
        self._metrics["consistency"] = self._evaluate_consistency(test_case, response)
        
        # Calculate weighted score
        total_score = sum(
            score * getattr(self.criteria, criterion)
            for criterion, score in self._metrics.items()
        )
        
        # Normalize by sum of weights
        total_weight = sum(
            getattr(self.criteria, criterion)
            for criterion in self._metrics.keys()
        )
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        logger.debug(f"Final score: {final_score:.3f}")
        return final_score
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the last computed metrics."""
        return self._metrics.copy()
    
    def _evaluate_correctness(self, test_case: TestCase, response: str) -> float:
        """Evaluate functional correctness."""
        if self.code_evaluator and test_case.metadata.get("code"):
            return self.code_evaluator.evaluate_correctness(test_case, response)
            
        # Basic correctness check
        if test_case.expected_output:
            return 1.0 if response.strip() == test_case.expected_output.strip() else 0.0
        return 0.5  # No expected output to compare against
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate clarity and readability."""
        if self.code_evaluator:
            return self.code_evaluator.evaluate_style(response)
            
        # Basic clarity metrics
        has_structure = "\n" in response
        has_explanation = any(word in response.lower() for word in ["because", "since", "as"])
        return (has_structure + has_explanation) / 2
    
    def _evaluate_efficiency(self, test_case: TestCase, response: str) -> float:
        """Evaluate solution efficiency."""
        if self.code_evaluator and test_case.metadata.get("code"):
            return self.code_evaluator.evaluate_efficiency(test_case, response)
            
        # Basic efficiency check
        return 0.5  # Default score when efficiency can't be measured
    
    def _evaluate_robustness(self, test_case: TestCase, response: str) -> float:
        """Evaluate solution robustness."""
        # Check for error handling
        has_validation = "if" in response.lower()
        has_error_handling = any(word in response.lower() for word in ["error", "exception", "raise"])
        return (has_validation + has_error_handling) / 2
    
    def _evaluate_completeness(self, test_case: TestCase, response: str) -> float:
        """Evaluate solution completeness."""
        # Check if all parts of the input are addressed
        input_words = set(test_case.input.lower().split())
        response_words = set(response.lower().split())
        coverage = len(input_words.intersection(response_words)) / len(input_words)
        return coverage
    
    def _evaluate_consistency(self, test_case: TestCase, response: str) -> float:
        """Evaluate internal consistency."""
        # Basic consistency checks
        lines = response.strip().split("\n")
        if not lines:
            return 0.0
            
        # Check indentation consistency
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if not indents:
            return 0.0
            
        # Check if all indents are multiples of the same base
        base_indent = min(indent for indent in indents if indent > 0) if any(indent > 0 for indent in indents) else 0
        if base_indent == 0:
            return 0.5
            
        consistent_indents = all(indent % base_indent == 0 for indent in indents)
        return 1.0 if consistent_indents else 0.0 