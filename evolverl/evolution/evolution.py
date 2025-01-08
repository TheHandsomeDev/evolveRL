"""
Core implementation of the Adversarial Evolutionary Reinforcement Learning (AERL) framework.

This module implements the evolutionary optimization process described in the AERL paper,
combining evolutionary algorithms with adversarial testing for LLM optimization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..llm import LLMBackend, LLMConfig
from ..judge import Judge, JudgingCriteria
from ..adversarial import AdversarialTester
from ..prompt_writer import PromptWriter, PromptMutationConfig

@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary process."""
    population_size: int = 10
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    fitness_weights: Dict[str, float] = None
    adversarial_difficulty: float = 0.5
    min_fitness_threshold: float = 0.7

@dataclass
class Individual:
    """Represents a single individual in the population."""
    prompt: str
    config: LLMConfig
    fitness: float = 0.0
    metrics: Dict[str, float] = None

class Evolution:
    """
    Main class implementing the AERL framework.
    
    This class manages the evolutionary process, including:
    1. Population initialization
    2. Adversarial testing
    3. Scoring and selection
    4. Mutation and crossover
    5. Optional co-evolution of adversaries
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        llm_backend: LLMBackend,
        judge: Judge,
        adversarial: AdversarialTester,
        prompt_writer: PromptWriter
    ):
        self.config = config
        self.llm = llm_backend
        self.judge = judge
        self.adversarial = adversarial
        self.prompt_writer = prompt_writer
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
    def initialize_population(self) -> None:
        """Initialize the first generation of prompts and configs."""
        self.population = []
        for _ in range(self.config.population_size):
            prompt = self.prompt_writer.generate_initial_prompt()
            config = self.llm.get_default_config()
            self.population.append(Individual(prompt=prompt, config=config))
            
    def evaluate_individual(self, individual: Individual) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single individual using adversarial testing and judging."""
        # Generate adversarial test cases
        test_cases = self.adversarial.generate_test_cases(
            difficulty=self.config.adversarial_difficulty
        )
        
        # Run the model on test cases
        responses = []
        for test in test_cases:
            response = self.llm.generate(
                prompt=individual.prompt,
                input_text=test.input,
                config=individual.config
            )
            responses.append(response)
            
        # Judge the responses
        metrics = self.judge.evaluate_batch(
            test_cases=test_cases,
            responses=responses
        )
        
        # Calculate weighted fitness
        if self.config.fitness_weights is None:
            fitness = np.mean(list(metrics.values()))
        else:
            fitness = sum(
                metrics[k] * self.config.fitness_weights[k]
                for k in metrics.keys()
            )
            
        return fitness, metrics
        
    def select_parents(self) -> Tuple[Individual, Individual]:
        """Select parents using tournament selection."""
        def tournament():
            contestants = np.random.choice(
                self.population,
                size=self.config.tournament_size,
                replace=False
            )
            return max(contestants, key=lambda x: x.fitness)
            
        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents."""
        if np.random.random() < self.config.crossover_rate:
            new_prompt = self.prompt_writer.crossover_prompts(
                parent1.prompt,
                parent2.prompt
            )
            new_config = self.llm.crossover_configs(
                parent1.config,
                parent2.config
            )
        else:
            new_prompt = parent1.prompt
            new_config = parent1.config
            
        return Individual(prompt=new_prompt, config=new_config)
        
    def mutate(self, individual: Individual) -> None:
        """Apply mutation to an individual."""
        if np.random.random() < self.config.mutation_rate:
            individual.prompt = self.prompt_writer.mutate_prompt(
                individual.prompt
            )
            individual.config = self.llm.mutate_config(
                individual.config
            )
            
    def evolve_generation(self) -> None:
        """Evolve the population by one generation."""
        # Evaluate current population
        for ind in self.population:
            ind.fitness, ind.metrics = self.evaluate_individual(ind)
            
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if (self.best_individual is None or 
            current_best.fitness > self.best_individual.fitness):
            self.best_individual = current_best
            
        # Create new population
        new_population = []
        
        # Elitism: Keep best individuals
        sorted_pop = sorted(
            self.population,
            key=lambda x: x.fitness,
            reverse=True
        )
        new_population.extend(sorted_pop[:self.config.elite_size])
        
        # Fill rest of population with offspring
        while len(new_population) < self.config.population_size:
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            new_population.append(offspring)
            
        self.population = new_population
        self.generation += 1
        
        # Optional: Co-evolve adversaries
        self.adversarial.update_difficulty(
            generation=self.generation,
            best_fitness=self.best_individual.fitness
        )
        
    def run(self, max_generations: Optional[int] = None) -> Individual:
        """Run the evolutionary process until convergence or max generations."""
        if max_generations is None:
            max_generations = self.config.generations
            
        self.initialize_population()
        
        while (self.generation < max_generations and
               (self.best_individual is None or
                self.best_individual.fitness < self.config.min_fitness_threshold)):
            self.evolve_generation()
            
        return self.best_individual 