"""Judge module for evaluating agent performance."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from ..llm import LLMBackend, OpenAILLMConfig, AnthropicLLMConfig

logger = logging.getLogger(__name__)

@dataclass
class JudgingCriteria:
    """Criteria for judging agent performance."""
    correctness: float = 1.0  # Weight for solution correctness
    clarity: float = 0.5      # Weight for response clarity
    efficiency: float = 0.7   # Weight for solution efficiency
    robustness: float = 0.6   # Weight for error handling
    completeness: float = 0.5 # Weight for addressing all requirements
    consistency: float = 0.4  # Weight for consistent style/approach

class Judge:
    """Evaluates agent responses using LLM-based judgment."""
    
    def __init__(
        self,
        criteria: JudgingCriteria,
        provider: str = "openai",
        model: Optional[str] = None
    ):
        self.criteria = criteria
        
        # Initialize LLM backend
        if provider == "openai":
            config = OpenAILLMConfig(
                model=model or "gpt-4o-mini",
                temperature=0.3  # Lower temperature for more consistent evaluation
            )
        else:
            config = AnthropicLLMConfig(
                model=model or "claude-3-5-sonnet-20241022",
                temperature=0.3
            )
        
        self.llm = LLMBackend(config=config)
    
    def evaluate(
        self,
        response: str,
        test_case: Any,
        domain: Optional[str] = None
    ) -> float:
        """Evaluate an agent's response to a test case."""
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            response=response,
            test_case=test_case,
            domain=domain
        )
        
        # Get evaluation from LLM
        try:
            eval_response = self.llm.generate(prompt)
            scores = self._parse_evaluation(eval_response)
            
            # Calculate weighted score
            total_score = (
                scores["correctness"] * self.criteria.correctness +
                scores["clarity"] * self.criteria.clarity +
                scores["efficiency"] * self.criteria.efficiency +
                scores["robustness"] * self.criteria.robustness +
                scores["completeness"] * self.criteria.completeness +
                scores["consistency"] * self.criteria.consistency
            )
            
            # Normalize by total weights
            total_weight = sum(vars(self.criteria).values())
            normalized_score = total_score / total_weight
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return 0.0
    
    def _create_evaluation_prompt(
        self,
        response: str,
        test_case: Any,
        domain: Optional[str] = None
    ) -> str:
        """Create prompt for evaluation."""
        base_prompt = f"""Evaluate this AI agent response based on the following criteria:

Test Case:
{test_case.input if hasattr(test_case, 'input') else str(test_case)}

Agent Response:
{response}

For each criterion, provide a score from 0.0 to 1.0 and brief justification:

1. Correctness (solution accuracy)
2. Clarity (response clarity)
3. Efficiency (solution efficiency)
4. Robustness (error handling)
5. Completeness (requirement coverage)
6. Consistency (style/approach)

Format your response as:
criterion: score # justification"""

        if domain:
            base_prompt += f"\n\nDomain-specific considerations for {domain}:"
            if domain == "code":
                base_prompt += """
- Code quality and best practices
- Documentation completeness
- Algorithm efficiency
- Error handling coverage"""
            elif domain == "customer_support":
                base_prompt += """
- Empathy and tone
- Solution clarity
- Follow-up handling
- Policy compliance"""
        
        return base_prompt
    
    def _parse_evaluation(self, eval_response: str) -> Dict[str, float]:
        """Parse evaluation response into scores."""
        scores = {
            "correctness": 0.0,
            "clarity": 0.0,
            "efficiency": 0.0,
            "robustness": 0.0,
            "completeness": 0.0,
            "consistency": 0.0
        }
        
        try:
            # Parse each line for scores
            for line in eval_response.split('\n'):
                if ':' not in line:
                    continue
                    
                criterion, rest = line.split(':', 1)
                criterion = criterion.strip().lower()
                
                # Extract score (assuming format "score # justification")
                score_part = rest.split('#')[0].strip()
                try:
                    score = float(score_part)
                    if criterion in scores:
                        scores[criterion] = max(0.0, min(1.0, score))
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to parse evaluation: {str(e)},", "eval_response:", eval_response)
            
        return scores 