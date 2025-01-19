"""Base evolution implementation."""
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Tuple, Dict
import logging
import random

from ..llm import LLMBackend
from ..judge import Judge
from ..adversarial import AdversarialTester
from ..agent import Agent
from ..generator.use_case import UseCaseGenerator, UseCase

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolution."""
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    use_case_description: Optional[str] = None

class Evolution:
    """Base class for evolutionary optimization."""
    
    def __init__(
        self,
        config: EvolutionConfig,
        judge: Judge,
        adversarial: AdversarialTester,
        llm_backend: Optional[LLMBackend] = None,
        save_callback: Optional[Callable[[Agent, int], None]] = None,
        provider: str = "openai"
    ):
        """Initialize evolution."""
        if config.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if config.generations < 1:
            raise ValueError("Number of generations must be at least 1")
        if not (0 <= config.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if not (0 <= config.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if config.tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")
        
        self.config = config
        self.judge = judge
        self.adversarial = adversarial
        self.llm = llm_backend or LLMBackend(provider=provider)
        self.save_callback = save_callback
        self.provider = provider
        self.use_case: Optional[UseCase] = None
        
        if config.use_case_description:
            self._generate_use_case()
    
    def _generate_use_case(self) -> None:
        """Generate use case configuration if needed."""
        logger.info("Generating use case configuration...")
        generator = UseCaseGenerator(provider=self.llm.provider)
        self.use_case = generator.generate(self.config.use_case_description)
        
        # Update components with generated prompts
        if self.use_case:
            logger.info(f"Generated use case for domain: {self.use_case.domain}")
            self.base_prompt = self.use_case.base_prompt
            self.judge.prompt_template = self.use_case.judge_prompt
            # Note: Adversary prompt is handled by specific tester classes
    
    def _create_base_agent(self) -> Agent:
        """Create a base agent."""
        if self.use_case:
            prompt_template = self.use_case.base_prompt
        else:
            prompt_template = self._generate_base_prompt()
            
        if self.provider == "openai":
            return Agent.from_openai(prompt_template=prompt_template)
        else:
            return Agent.from_anthropic(prompt_template=prompt_template)
    
    def _evaluate_agent(self, agent: Agent) -> float:
        """Evaluate an agent's fitness."""
        # Use generated test cases if available
        if self.use_case:
            test_cases = [
                (ex["task"], ex.get("context", ""))
                for ex in self.use_case.train_data
            ]
        else:
            test_cases = self.adversarial.generate_test_cases(agent)
        
        total_score = 0.0
        for task, context in test_cases:
            response = agent.run(task, context)
            score = self.judge.evaluate(response, task, context)
            total_score += score
        
        return total_score / len(test_cases)
    
    def run(self) -> Agent:
        """Run the evolutionary process."""
        logger.info("Starting evolution process...")
        self.population = []
        self.generation = 0
        self.best_fitness = 0.0
        
        # Initialize population
        base_agent = self._create_base_agent()
        for _ in range(self.config.population_size):
            agent = self._mutate_agent(base_agent)
            self.population.append(agent)
        
        # Run generations
        for gen in range(self.config.generations):
            self.generation = gen
            logger.info(f"\nGeneration {gen + 1}/{self.config.generations}")
            
            # Evaluate population
            for agent in self.population:
                agent.fitness = self._evaluate_agent(agent)
            
            # Get best agent
            best_agent = max(self.population, key=lambda x: x.fitness)
            self.best_fitness = best_agent.fitness
            
            # Save checkpoint if callback provided
            if self.save_callback:
                self.save_callback(best_agent, gen)
            
            # Create next generation
            if gen < self.config.generations - 1:
                self._create_next_generation()
        
        return max(self.population, key=lambda x: x.fitness)
    
    def _mutate_agent(self, agent: Agent) -> Agent:
        """Create a mutated copy of an agent."""
        mutation_prompt = f"""Given this prompt template:
{agent.prompt_template}

Create a variation that:
1. Maintains the core functionality
2. Emphasizes different aspects
3. Keeps the basic structure
4. Makes small but meaningful changes

Return only the new prompt template."""
        
        new_prompt = self.llm.generate(mutation_prompt)
        return Agent(
            model=agent.model,
            provider=agent.provider,
            prompt_template=new_prompt
        )
    
    def _crossover_agents(self, agent1: Agent, agent2: Agent) -> Agent:
        """Create new agent by combining prompts."""
        crossover_prompt = f"""Given these two prompt templates:

Template 1:
{agent1.prompt_template}

Template 2:
{agent2.prompt_template}

Create a new template that:
1. Combines the strengths of both
2. Maintains core functionality
3. Has a clear structure
4. Is coherent and well-formed

Return only the new prompt template."""
        
        new_prompt = self.llm.generate(crossover_prompt)
        return Agent(
            model=agent1.model,
            provider=agent1.provider,
            prompt_template=new_prompt
        )
    
    def _create_next_generation(self) -> None:
        """Create the next generation through selection and variation."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep best agents
        elite_size = max(1, self.config.population_size // 5)
        new_population = self.population[:elite_size]
        
        # Fill rest of population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover_agents(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_select()
                child = self._mutate_agent(parent)
            
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_select(self) -> Agent:
        """Select an agent using tournament selection."""
        tournament = random.sample(
            self.population,
            min(self.config.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def _evaluate_population(self, test_cases: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Evaluate all agents in the population."""
        for agent in self.population:
            total_score = 0.0
            for task, context in test_cases:
                response = agent.run(task, context)
                score = self.judge.evaluate(response, task, context)
                total_score += score
            agent.fitness = total_score / len(test_cases)
    
    def _generate_base_prompt(self) -> str:
        """Generate base prompt template for agents.
        
        Returns:
            str: Generated prompt template with {task} and {context} placeholders
        """
        prompt = """Given this task description, generate a base prompt template for an AI agent.
The prompt should:
1. Set clear expectations for the agent's role and behavior
2. Include placeholders for {task} and {context}
3. Be adaptable to different inputs
4. Encourage high-quality, thoughtful responses

Task description: {description}

Return only the prompt template, nothing else."""

        try:
            # Generate base prompt using LLM
            description = self.config.use_case_description or "A general-purpose AI assistant"
            generated = self.llm.generate(prompt.format(description=description))
            
            # Validate prompt has required placeholders
            if "{task}" not in generated or "{context}" not in generated:
                logger.warning("Generated prompt missing placeholders, using default")
                return "Task: {task}\nContext: {context}\n\nProvide a detailed, high-quality response."
            
            return generated.strip()
            
        except Exception as e:
            logger.error(f"Error generating base prompt: {str(e)}")
            # Fallback to default prompt
            return "Task: {task}\nContext: {context}\n\nProvide a detailed, high-quality response."
    
    # ... rest of Evolution class implementation ... 