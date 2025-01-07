"""
Example demonstrating the complete evolutionary loop for self-improving LLMs.

This implements the full framework from the paper, including:
1. Evolutionary Prompt Writer/Improver
2. Judge
3. Evolutionary Models
4. Adversarial Models

The system will continue evolving until either:
- Performance threshold is met
- Maximum generations reached
- Convergence detected (no improvement over N generations)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time

from evolverl.agent import Agent
from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.judge import Judge, JudgingCriteria
from evolverl.adversarial import AdversarialTester
from evolverl.prompt_writer import PromptWriter, PromptMutationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionaryLoopConfig:
    """Configuration for the evolutionary loop."""
    # Population settings
    population_size: int = 10
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.1
    
    # Performance thresholds
    target_score: float = 0.95
    convergence_threshold: float = 0.001
    convergence_generations: int = 5
    
    # Domain settings
    domain: str = "math"  # "math", "code", "defi"
    difficulty: str = "medium"
    
    # Model settings
    agent_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    adversarial_model: str = "gpt-4o-mini"
    
    # Save settings
    save_frequency: int = 5
    save_dir: str = "models/evolution"


class EvolutionaryLoop:
    """
    Implements the complete evolutionary loop for self-improving LLMs.
    
    This class orchestrates the interaction between:
    1. Prompt Writer - Generates and mutates prompts
    2. Evolutionary Models - The LLMs being evolved
    3. Adversarial Models - Generate challenging test cases
    4. Judge - Evaluates performance
    """
    
    def __init__(self, config: EvolutionaryLoopConfig):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        
        # Initialize components
        self.prompt_writer = PromptWriter(
            base_model=config.agent_model,
            mutation_config=PromptMutationConfig(
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate
            )
        )
        
        self.judge = Judge(
            model=config.judge_model,
            criteria=JudgingCriteria(
                correctness=1.0,
                clarity=0.7,
                efficiency=0.5,
                completeness=0.8,
                consistency=0.6
            )
        )
        
        self.adversarial = AdversarialTester(
            difficulty=config.difficulty,
            domain=config.domain
        )
        
        self.evolution = Evolution(
            config=EvolutionConfig(
                population_size=config.population_size,
                generations=config.generations,
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate,
                domain=config.domain
            )
        )
    
    def run(self, task: str) -> Agent:
        """
        Run the evolutionary loop until convergence or max generations.
        
        Args:
            task: The task to optimize for (e.g., "Solve math problems")
            
        Returns:
            Agent: The best performing evolved agent
        """
        logger.info(f"Starting evolutionary loop for task: {task}")
        
        # Initialize population
        population = self._initialize_population(task)
        best_agent = None
        best_score = 0.0
        generations_without_improvement = 0
        
        for generation in range(self.config.generations):
            logger.info(f"\nGeneration {generation + 1}/{self.config.generations}")
            
            # Generate adversarial test cases
            test_cases = self.adversarial.generate_test_cases(
                agent=population[0],  # Use best agent from previous gen
                num_cases=5
            )
            
            # Evaluate population
            scores = self._evaluate_population(population, test_cases)
            
            # Track best performer
            gen_best_idx = max(range(len(scores)), key=scores.__getitem__)
            gen_best_score = scores[gen_best_idx]
            
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_agent = population[gen_best_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Log progress
            logger.info(f"Best score: {best_score:.3f}")
            logger.info(f"Generation best: {gen_best_score:.3f}")
            logger.info(f"Population average: {sum(scores)/len(scores):.3f}")
            
            # Save checkpoint
            if (generation + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(best_agent, generation, best_score)
            
            # Check termination conditions
            if best_score >= self.config.target_score:
                logger.info("Target score reached!")
                break
                
            if generations_without_improvement >= self.config.convergence_generations:
                score_range = max(scores) - min(scores)
                if score_range < self.config.convergence_threshold:
                    logger.info("Convergence detected!")
                    break
            
            # Evolve population
            population = self._evolve_population(population, scores, test_cases)
            
            # Record history
            self.history.append({
                "generation": generation,
                "best_score": best_score,
                "generation_best": gen_best_score,
                "average_score": sum(scores)/len(scores),
                "test_cases": [str(test) for test, _ in test_cases]
            })
        
        logger.info("\nEvolution complete!")
        logger.info(f"Final best score: {best_score:.3f}")
        
        return best_agent
    
    def _initialize_population(self, task: str) -> List[Agent]:
        """Initialize the first generation of agents."""
        logger.info("Initializing population...")
        
        # Generate initial prompts
        prompts = self.prompt_writer.generate_initial_population(
            task_description=task,
            population_size=self.config.population_size
        )
        
        # Create agents with different prompts
        return [
            Agent(
                model=self.config.agent_model,
                prompt_template=prompt,
                config={"temperature": 0.7}
            )
            for prompt in prompts
        ]
    
    def _evaluate_population(
        self,
        population: List[Agent],
        test_cases: List[Tuple[str, Dict[str, Any]]]
    ) -> List[float]:
        """Evaluate each agent against the test cases."""
        scores = []
        
        for agent in population:
            agent_scores = []
            for test, metadata in test_cases:
                response = agent.run(task=test)
                score = self.judge.evaluate(
                    task=test,
                    response=response,
                    domain=self.config.domain
                )
                agent_scores.append(score)
            
            # Average score across all test cases
            scores.append(sum(agent_scores) / len(agent_scores))
        
        return scores
    
    def _evolve_population(
        self,
        population: List[Agent],
        scores: List[float],
        test_cases: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Agent]:
        """Evolve the population based on their performance."""
        # Get prompts from current population
        prompts = [agent.prompt_template for agent in population]
        
        # Generate new prompts through mutation and crossover
        new_prompts = self.prompt_writer.mutate_prompts(
            prompts=prompts,
            scores=scores
        )
        
        # Create new agents with evolved prompts
        return [
            Agent(
                model=self.config.agent_model,
                prompt_template=prompt,
                config={"temperature": 0.7}
            )
            for prompt in new_prompts
        ]
    
    def _save_checkpoint(
        self,
        agent: Agent,
        generation: int,
        score: float
    ) -> None:
        """Save evolution checkpoint."""
        checkpoint = {
            "generation": generation,
            "score": score,
            "agent_state": agent.get_state(),
            "history": self.history
        }
        
        path = f"{self.config.save_dir}/checkpoint_gen{generation}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    # Example usage
    config = EvolutionaryLoopConfig(
        population_size=5,
        generations=20,
        domain="math",
        difficulty="medium"
    )
    
    loop = EvolutionaryLoop(config)
    
    # Run evolution
    task = "Solve complex mathematical problems step by step"
    best_agent = loop.run(task=task)
    
    # Save final model
    best_agent.save_state("models/evolved_math_solver_final.json")
    
    # Test the evolved agent
    test_problems = [
        "Solve the quadratic equation: 2x² + 5x - 12 = 0",
        "Find the derivative of f(x) = x³sin(x)",
        "Calculate the area of a circle with radius 5 units"
    ]
    
    print("\nTesting evolved agent:")
    for problem in test_problems:
        print(f"\nProblem: {problem}")
        response = best_agent.run(task=problem)
        print(f"Response: {response}")
        
        score = loop.judge.evaluate(
            task=problem,
            response=response,
            domain="math"
        )
        print(f"Score: {score:.3f}") 