"""
Judge component for evaluating model responses.
"""
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
import json
import re
import logging

from .llm import LLMBackend, LLMConfig


@dataclass
class JudgingCriteria:
    """Criteria for judging model responses."""
    correctness: float = 1.0  # Weight for answer correctness
    clarity: float = 0.5  # Weight for explanation clarity
    efficiency: float = 0.3  # Weight for solution efficiency
    completeness: float = 0.5  # Weight for addressing all parts
    consistency: float = 0.4  # Weight for internal consistency


class Judge:
    """
    Evaluates model responses based on multiple criteria.
    
    This implements the Judge component from the paper, providing
    sophisticated evaluation of model outputs across multiple dimensions.
    
    Args:
        model (str): The LLM to use for evaluation (if using LLM-based judging)
        criteria (Optional[JudgingCriteria]): Custom judging criteria weights
        config (Optional[Dict[str, Any]]): Additional configuration
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        criteria: Optional[JudgingCriteria] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.criteria = criteria or JudgingCriteria()
        self.config = config or {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Initialize LLM backend for evaluation
        llm_config = LLMConfig(
            model=model,
            temperature=0.2,  # Lower temperature for more consistent evaluation
            max_tokens=500
        )
        self.llm = LLMBackend(config=llm_config)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        task: str,
        response: str,
        ground_truth: Optional[str] = None,
        domain: Optional[str] = None
    ) -> float:
        """
        Evaluate a model's response to a task.
        
        Args:
            task (str): The original task/question
            response (str): The model's response to evaluate
            ground_truth (Optional[str]): Correct answer if available
            domain (Optional[str]): Task domain for specialized evaluation
            
        Returns:
            float: Overall score between 0 and 1
        """
        scores = {}
        
        # Evaluate each criterion
        if ground_truth:
            scores["correctness"] = self._evaluate_correctness(
                response, ground_truth, domain
            )
        else:
            scores["correctness"] = self._evaluate_without_ground_truth(
                task, response, domain
            )
        
        scores["clarity"] = self._evaluate_clarity(response)
        scores["efficiency"] = self._evaluate_efficiency(response)
        scores["completeness"] = self._evaluate_completeness(task, response)
        scores["consistency"] = self._evaluate_consistency(response)
        
        # Calculate weighted average
        weighted_sum = sum(
            getattr(self.criteria, criterion) * score
            for criterion, score in scores.items()
        )
        total_weight = sum(
            getattr(self.criteria, criterion)
            for criterion in scores.keys()
        )
        
        final_score = weighted_sum / total_weight
        
        # Record evaluation
        self.evaluation_history.append({
            "task": task,
            "scores": scores,
            "final_score": final_score,
            "domain": domain
        })
        
        return final_score
    
    def _evaluate_correctness(
        self,
        response: str,
        ground_truth: str,
        domain: Optional[str]
    ) -> float:
        """Evaluate response correctness against ground truth."""
        if domain == "math":
            return self._evaluate_math_correctness(response, ground_truth)
        elif domain == "code":
            return self._evaluate_code_correctness(response, ground_truth)
        
        # Use LLM to evaluate correctness
        prompt = f"""Evaluate if the following response matches the ground truth.
Consider both factual accuracy and completeness.

Ground Truth: {ground_truth}
Response: {response}

Score the response from 0 to 1, where:
0 = Completely incorrect
0.5 = Partially correct
1 = Completely correct

Provide your score in the format:
SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt)
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5  # Default if parsing fails
        except Exception as e:
            self.logger.error(f"Error in correctness evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_without_ground_truth(
        self,
        task: str,
        response: str,
        domain: Optional[str]
    ) -> float:
        """Evaluate response when no ground truth is available."""
        prompt = f"""Evaluate if the following response appropriately answers the task.
Consider accuracy, relevance, and completeness.

Task: {task}
Response: {response}

Score the response from 0 to 1, where:
0 = Completely inappropriate or incorrect
0.5 = Partially appropriate
1 = Fully appropriate and likely correct

Provide your score in the format:
SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt)
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate the clarity and readability of the response."""
        prompt = """Evaluate the clarity and readability of the following response.
Consider:
1. Clear structure and organization
2. Use of explanatory language
3. Logical flow
4. Appropriate detail level

Response: {response}

Score from 0 to 1, where:
0 = Very unclear and hard to follow
0.5 = Moderately clear
1 = Extremely clear and well-structured

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt.format(response=response))
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            
            # Fallback to heuristic evaluation
            indicators = [
                "step by step",
                "first",
                "then",
                "finally",
                "because",
                "therefore"
            ]
            score = sum(1 for ind in indicators if ind in response.lower())
            return min(score / len(indicators), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error in clarity evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_efficiency(self, response: str) -> float:
        """Evaluate the efficiency of the solution."""
        prompt = """Evaluate the efficiency of the following response.
Consider:
1. Directness of approach
2. Absence of unnecessary steps
3. Optimal use of available information
4. Conciseness without sacrificing clarity

Response: {response}

Score from 0 to 1, where:
0 = Very inefficient or roundabout
0.5 = Moderately efficient
1 = Highly efficient and optimal

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt.format(response=response))
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in efficiency evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_completeness(self, task: str, response: str) -> float:
        """Evaluate if the response addresses all parts of the task."""
        prompt = f"""Evaluate if the following response completely addresses all parts of the task.

Task: {task}
Response: {response}

Score from 0 to 1, where:
0 = Many parts unaddressed
0.5 = Some parts addressed
1 = All parts fully addressed

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt)
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in completeness evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_consistency(self, response: str) -> float:
        """Evaluate internal consistency of the response."""
        prompt = """Evaluate the internal consistency of the following response.
Check for:
1. No contradictions
2. Logical flow
3. Consistent terminology
4. Coherent reasoning

Response: {response}

Score from 0 to 1, where:
0 = Many inconsistencies
0.5 = Some minor inconsistencies
1 = Completely consistent

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt.format(response=response))
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in consistency evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_math_correctness(
        self,
        response: str,
        ground_truth: str
    ) -> float:
        """Specialized evaluation for math problems."""
        prompt = f"""Evaluate if the following math solution is correct.
Consider:
1. Final answer correctness
2. Solution process correctness
3. Mathematical reasoning
4. Proper use of formulas/methods

Correct Answer: {ground_truth}
Solution: {response}

Score from 0 to 1, where:
0 = Incorrect answer and process
0.5 = Correct answer but flawed process (or vice versa)
1 = Correct answer and process

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt)
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in math evaluation: {str(e)}")
            return 0.5
    
    def _evaluate_code_correctness(
        self,
        response: str,
        ground_truth: str
    ) -> float:
        """Specialized evaluation for code generation."""
        prompt = f"""Evaluate if the following code solution is correct.
Consider:
1. Functional correctness
2. Code quality and style
3. Efficiency
4. Error handling

Reference Solution: {ground_truth}
Submitted Solution: {response}

Score from 0 to 1, where:
0 = Incorrect and poorly written
0.5 = Partially correct or suboptimal
1 = Correct and well-written

SCORE: [number]
EXPLANATION: [your reasoning]"""
        
        try:
            eval_response = self.llm.generate(prompt)
            score_match = re.search(r"SCORE:\s*(0?\.\d+|1\.0?)", eval_response)
            if score_match:
                return float(score_match.group(1))
            return 0.5
        except Exception as e:
            self.logger.error(f"Error in code evaluation: {str(e)}")
            return 0.5
    
    def save_history(self, path: str) -> None:
        """Save evaluation history to a file."""
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
    
    @classmethod
    def load_history(cls, path: str) -> List[Dict[str, Any]]:
        """Load evaluation history from a file."""
        with open(path, 'r') as f:
            return json.load(f) 