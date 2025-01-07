"""
Evolution controller for managing the evolutionary learning process.
"""
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
import random
import json
import logging
from datetime import datetime

from ..agent import Agent
from ..prompt_writer import PromptWriter
from ..judge import Judge, JudgingCriteria
from ..adversarial import AdversarialTester


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    population_size: int = 10
    generations: int = 100
    mutation_rate: float = 0.1
    elite_size: int = 2
    fitness_threshold: float = 0.95
    num_adversarial_tests: int = 5
    adversarial_difficulty: str = "medium"
    domain: Optional[str] = None


class Evolution:
    """
    Controls the evolutionary learning process for agents.
    
    This class implements the complete evolutionary workflow described in the paper,
    coordinating between the PromptWriter, AdversarialTester, and Judge components.
    
    Args:
        population_size (int): Size of the population in each generation
        generations (int): Number of generations to evolve
        config (Optional[Dict[str, Any]]): Additional configuration parameters
    """
    
    def __init__(
        self,
        population_size: int = 10,
        generations: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = EvolutionConfig(
            population_size=population_size,
            generations=generations,
            **(config or {})
        )
        
        # Initialize components
        self.prompt_writer = PromptWriter()
        self.judge = Judge(criteria=JudgingCriteria(
            correctness=1.0,
            clarity=0.5,
            efficiency=0.3,
            completeness=0.5,
            consistency=0.4
        ))
        self.adversarial = AdversarialTester(
            difficulty=self.config.adversarial_difficulty,
            domain=self.config.domain
        )
        
        # Evolution state
        self.generation = 0
        self.best_fitness = 0.0
        self.population: List[Agent] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        agent: Agent,
        task: str,
        adversarial_difficulty: str = "medium",
        judge_fn: Optional[Callable[[str, str], float]] = None
    ) -> Agent:
        """
        Train an agent through evolutionary optimization.
        
        Args:
            agent (Agent): The base agent to evolve
            task (str): The task to optimize for
            adversarial_difficulty (str): Difficulty level for adversarial testing
            judge_fn (Optional[Callable]): Custom evaluation function
            
        Returns:
            Agent: The best performing agent after evolution
        """
        self.logger.info(f"Starting evolution for task: {task}")
        
        # Initialize population with prompt variations
        self.population = self._initialize_population(agent, task)
        
        for generation in range(self.config.generations):
            self.generation = generation
            self.logger.info(f"\nGeneration {generation + 1}/{self.config.generations}")
            
            # Generate adversarial test cases
            test_cases = self.adversarial.generate_test_cases(
                agent=agent,
                num_cases=self.config.num_adversarial_tests
            )
            
            # Evaluate each agent against test cases
            fitness_scores = self._evaluate_population(
                test_cases=test_cases,
                judge_fn=judge_fn
            )
            
            # Select and record elite agents
            elite_agents, elite_scores = self._select_elite_agents(
                fitness_scores
            )
            
            # Record progress
            self._record_generation(
                test_cases=test_cases,
                fitness_scores=fitness_scores,
                elite_scores=elite_scores
            )
            
            # Check if we've reached our goal
            if max(fitness_scores) >= self.config.fitness_threshold:
                self.logger.info("Reached fitness threshold! Evolution complete.")
                return self.population[fitness_scores.index(max(fitness_scores))]
            
            # Generate next generation
            self.population = self._create_next_generation(
                elite_agents=elite_agents,
                fitness_scores=fitness_scores
            )
        
        # Return best agent after all generations
        best_idx = fitness_scores.index(max(fitness_scores))
        return self.population[best_idx]
    
    def _initialize_population(
        self,
        base_agent: Agent,
        task: str
    ) -> List[Agent]:
        """Initialize population with prompt variations."""
        self.logger.info("Initializing population...")
        
        # Generate initial prompts
        prompts = self.prompt_writer.generate_initial_population(
            task_description=task,
            population_size=self.config.population_size
        )
        
        # Create agent variants
        population = []
        for prompt in prompts:
            agent = Agent(
                model=base_agent.model,
                config=base_agent.config.copy(),
                prompt_template=prompt
            )
            population.append(agent)
        
        return population
    
    def _evaluate_population(
        self,
        test_cases: List[Tuple[str, Dict[str, Any]]],
        judge_fn: Optional[Callable[[str, str], float]] = None
    ) -> List[float]:
        """Evaluate each agent against test cases."""
        fitness_scores = []
        
        for i, agent in enumerate(self.population):
            agent_scores = []
            
            for test, metadata in test_cases:
                response = agent.run(test)
                
                if judge_fn:
                    score = judge_fn(test, response)
                else:
                    score = self.judge.evaluate(
                        task=test,
                        response=response,
                        domain=self.config.domain
                    )
                
                agent_scores.append(score)
            
            # Average score across all test cases
            fitness = sum(agent_scores) / len(agent_scores)
            fitness_scores.append(fitness)
            
            self.logger.info(f"Agent {i + 1}/{len(self.population)} - Fitness: {fitness:.3f}")
        
        return fitness_scores
    
    def _select_elite_agents(
        self,
        fitness_scores: List[float]
    ) -> Tuple[List[Agent], List[float]]:
        """Select the best performing agents."""
        elite_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=True
        )[:self.config.elite_size]
        
        elite_agents = [self.population[i] for i in elite_indices]
        elite_scores = [fitness_scores[i] for i in elite_indices]
        
        best_fitness = elite_scores[0]
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.logger.info(f"New best fitness: {best_fitness:.3f}")
        
        return elite_agents, elite_scores
    
    def _create_next_generation(
        self,
        elite_agents: List[Agent],
        fitness_scores: List[float]
    ) -> List[Agent]:
        """Create the next generation through mutation."""
        # Get all prompts and their scores
        prompts = [agent.prompt_template for agent in self.population]
        
        # Generate new prompts through mutation
        new_prompts = self.prompt_writer.mutate_prompts(
            prompts=prompts,
            scores=fitness_scores
        )
        
        # Create new population
        next_population = elite_agents.copy()  # Keep elite agents unchanged
        
        # Add mutated variants
        while len(next_population) < self.config.population_size:
            parent = random.choice(elite_agents)
            child = Agent(
                model=parent.model,
                config=parent.config.copy(),
                prompt_template=new_prompts[len(next_population)]
            )
            next_population.append(child)
        
        return next_population
    
    def _record_generation(
        self,
        test_cases: List[Tuple[str, Dict[str, Any]]],
        fitness_scores: List[float],
        elite_scores: List[float]
    ) -> None:
        """Record the results of the current generation."""
        generation_data = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "best_fitness": max(fitness_scores),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "elite_scores": elite_scores,
            "test_cases": [test for test, _ in test_cases],
            "population_size": len(self.population)
        }
        
        self.evolution_history.append(generation_data)
    
    def save_state(self, path: str) -> None:
        """Save the evolution state to a file."""
        state = {
            "config": self.config.__dict__,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "evolution_history": self.evolution_history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, path: str) -> 'Evolution':
        """Load an evolution state from a file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        evolution = cls(
            population_size=state["config"]["population_size"],
            generations=state["config"]["generations"],
            config=state["config"]
        )
        evolution.generation = state["generation"]
        evolution.best_fitness = state["best_fitness"]
        evolution.evolution_history = state["evolution_history"]
        return evolution 