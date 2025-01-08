"""
Evolution module for EvolveRL.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import logging

from ..llm import LLMBackend, LLMConfig
from ..judge import Judge
from ..adversarial import AdversarialTester, TestCase

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 10
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_size: int = 1
    min_fitness_threshold: float = 0.8
    tournament_size: int = 2

@dataclass
class Individual:
    """An individual in the population."""
    prompt: str
    config: LLMConfig
    fitness: float = 0.0
    metrics: Dict[str, float] = None

class Evolution:
    """Manages evolutionary optimization process."""
    
    def __init__(
        self,
        config: EvolutionConfig,
        llm_backend: LLMBackend,
        judge: Judge,
        adversarial: AdversarialTester,
        prompt_writer: Any
    ):
        self.config = config
        self.llm = llm_backend
        self.judge = judge
        self.adversarial = adversarial
        self.prompt_writer = prompt_writer
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
    def initialize_population(self):
        """Initialize the population with random individuals."""
        logger.info("Initializing population...")
        self.population = []
        base_config = self.llm.get_default_config()
        
        for i in range(self.config.population_size):
            prompt = self.prompt_writer.generate_initial_prompt()
            config = self.llm.mutate_config(base_config)
            individual = Individual(prompt=prompt, config=config)
            self.population.append(individual)
            logger.debug(f"Created individual {i+1}/{self.config.population_size}")
    
    def evaluate_individual(self, individual: Individual) -> Tuple[float, Dict[str, float]]:
        """Evaluate an individual using test cases."""
        logger.debug(f"Evaluating individual with config: temp={individual.config.temperature:.2f}")
        test_cases = self.adversarial.generate_test_cases(num_cases=3)
        total_score = 0.0
        metrics = {}
        
        for i, test in enumerate(test_cases):
            # Generate response using the individual's prompt and config
            full_prompt = f"{individual.prompt}\n\n{test.input}"
            response = self.llm.generate(prompt=full_prompt, config=individual.config)
            
            # Evaluate response
            score = self.judge.evaluate(test, response)
            total_score += score
            
            # Update metrics
            for key, value in self.judge.get_metrics().items():
                metrics[key] = metrics.get(key, 0.0) + value
            
            logger.debug(f"Test case {i+1}/3: score={score:.3f}")
        
        # Average scores and metrics
        avg_score = total_score / len(test_cases)
        for key in metrics:
            metrics[key] /= len(test_cases)
            
        logger.debug(f"Final score: {avg_score:.3f}")
        return avg_score, metrics
    
    def select_parents(self) -> List[Individual]:
        """Select parents for next generation using tournament selection."""
        tournament_size = min(self.config.tournament_size, len(self.population))
        parents = []
        
        while len(parents) < self.config.population_size:
            # Tournament selection
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
            
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents."""
        if random.random() < self.config.crossover_rate:
            # Crossover prompts
            new_prompt = self.prompt_writer.crossover_prompts(
                parent1.prompt,
                parent2.prompt
            )
            
            # Crossover configs
            new_config = self.llm.crossover_configs(
                parent1.config,
                parent2.config
            )
        else:
            # No crossover, just copy one parent
            new_prompt = parent1.prompt
            new_config = parent1.config
            
        return Individual(prompt=new_prompt, config=new_config)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        if random.random() < self.config.mutation_rate:
            # Mutate prompt
            new_prompt = self.prompt_writer.mutate_prompt(individual.prompt)
            
            # Mutate config
            new_config = self.llm.mutate_config(individual.config)
            
            return Individual(prompt=new_prompt, config=new_config)
            
        return individual
    
    def evolve_generation(self):
        """Evolve one generation."""
        logger.info(f"Generation {self.generation + 1}/{self.config.generations}")
        
        # Evaluate current population
        for i, ind in enumerate(self.population):
            if ind.fitness == 0.0:  # Only evaluate if not already evaluated
                logger.info(f"Evaluating individual {i+1}/{len(self.population)}")
                ind.fitness, ind.metrics = self.evaluate_individual(ind)
        
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if not self.best_individual or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            logger.info(f"New best fitness: {self.best_individual.fitness:.3f}")
            
        # Select parents
        parents = self.select_parents()
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_pop[:self.config.elite_size])
        logger.debug(f"Added {self.config.elite_size} elite individuals")
        
        # Create rest of population
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Create child through crossover
            child = self.crossover(parent1, parent2)
            
            # Mutate child
            child = self.mutate(child)
            
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
        # Update adversarial difficulty based on population performance
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.adversarial.update_difficulty(avg_fitness)
        logger.info(f"Average fitness: {avg_fitness:.3f}")
    
    def run(self) -> Individual:
        """Run the evolutionary process."""
        logger.info("Starting evolution...")
        
        # Initialize population
        self.initialize_population()
        
        # Main evolution loop
        for gen in range(self.config.generations):
            self.evolve_generation()
            
            # Check if we've reached the fitness threshold
            if self.best_individual.fitness >= self.config.min_fitness_threshold:
                logger.info("Reached minimum fitness threshold")
                break
                
        logger.info("Evolution completed")
        return self.best_individual 