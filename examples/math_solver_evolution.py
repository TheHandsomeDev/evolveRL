"""
Example of using EvolveRL to evolve a math problem solver agent.

This example demonstrates the complete workflow described in the paper,
including prompt evolution, adversarial testing, and performance evaluation.
"""
import logging
from typing import Dict, Any

from evolverl import Agent, Evolution, AdversarialTester, Judge
from evolverl.judge import JudgingCriteria


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_initial_prompt() -> str:
    """Get the initial prompt template for the math solver."""
    return """You are a math problem solver. Given a math problem:

Problem: {task}

Please solve it following these steps:
1. Understand and rewrite the problem in your own words
2. List the key information and unknowns
3. Plan your solution approach
4. Solve step by step
5. Verify your answer

Your solution:"""


def get_evolution_config() -> Dict[str, Any]:
    """Get the configuration for the evolution process."""
    return {
        "population_size": 8,
        "generations": 15,
        "mutation_rate": 0.2,
        "elite_size": 2,
        "fitness_threshold": 0.95,
        "num_adversarial_tests": 5,
        "adversarial_difficulty": "medium",
        "domain": "math"
    }


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize the base agent
    logger.info("Initializing base agent...")
    agent = Agent(
        model="gpt-4o-mini",
        config={
            "temperature": 0.7,
            "max_tokens": 300
        },
        prompt_template=get_initial_prompt()
    )
    
    # Create evolution controller with custom configuration
    logger.info("Setting up evolution process...")
    evolution = Evolution(
        config=get_evolution_config()
    )
    
    # Example math problem for initial testing
    test_problem = (
        "A rectangular pool is 25 meters long and 10 meters wide. "
        "If the pool's depth increases linearly from 1 meter at one end "
        "to 3 meters at the other end, what is the volume of water "
        "needed to fill the pool?"
    )
    
    # Test the initial agent
    logger.info("\nTesting initial agent...")
    initial_response = agent.run(test_problem)
    logger.info(f"Initial response:\n{initial_response}")
    
    # Create a judge with custom criteria weights
    judge = Judge(
        model="gpt-4o-mini",
        criteria=JudgingCriteria(
            correctness=1.0,  # High weight on correctness
            clarity=0.7,      # Value clear explanations
            efficiency=0.4,   # Some focus on efficient solutions
            completeness=0.8, # Important to address all parts
            consistency=0.5   # Moderate weight on consistency
        )
    )
    
    # Initial evaluation
    initial_score = judge.evaluate(
        task=test_problem,
        response=initial_response,
        domain="math"
    )
    logger.info(f"Initial agent score: {initial_score:.3f}")
    
    # Create adversarial tester
    tester = AdversarialTester(
        difficulty="medium",
        domain="math"
    )
    
    # Generate some test cases
    logger.info("\nGenerating adversarial test cases...")
    test_cases = tester.generate_test_cases(agent, num_cases=3)
    
    logger.info("Example test cases:")
    for test, metadata in test_cases:
        logger.info(f"\nTest: {test}")
        logger.info(f"Metadata: {metadata}")
    
    # Train the agent
    logger.info("\nStarting evolution...")
    evolved_agent = evolution.train(
        agent=agent,
        task=test_problem,
        adversarial_difficulty="medium"
    )
    
    logger.info(f"\nEvolution complete!")
    logger.info(f"Best fitness achieved: {evolution.best_fitness:.3f}")
    
    # Test the evolved agent
    logger.info("\nTesting evolved agent...")
    evolved_response = evolved_agent.run(test_problem)
    evolved_score = judge.evaluate(
        task=test_problem,
        response=evolved_response,
        domain="math"
    )
    
    logger.info(f"\nEvolved agent response:\n{evolved_response}")
    logger.info(f"Evolved agent score: {evolved_score:.3f}")
    
    # Compare improvement
    improvement = (evolved_score - initial_score) / initial_score * 100
    logger.info(f"\nImprovement: {improvement:.1f}%")
    
    # Save the evolved agent and evolution history
    evolved_agent.save_state("evolved_math_solver.json")
    evolution.save_state("math_evolution_history.json")
    
    logger.info("\nEvolved agent and history saved to files.")


if __name__ == "__main__":
    main() 