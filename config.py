"""Configuration management."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class JudgingCriteriaConfig:
    """Configuration for judging criteria."""
    correctness: float = 1.0
    clarity: float = 0.5
    efficiency: float = 0.7
    robustness: float = 0.6
    completeness: float = 0.5
    consistency: float = 0.4

@dataclass
class EvolutionConfig:
    """Configuration for evolution."""
    population_size: int = 2
    generations: int = 2
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 2
    min_fitness_threshold: float = 0.8
    use_case_description: Optional[str] = None

@dataclass
class DomainConfig:
    """Configuration for specific domains."""
    test_cases_per_generation: int
    max_tokens: int

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str
    temperature: float
    max_tokens: int
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0

@dataclass
class Config:
    """Main configuration class."""
    difficulty: str
    JudgingCriteriaConfig: JudgingCriteriaConfig
    EvolutionConfig: EvolutionConfig
    domains: Dict[str, DomainConfig]
    llm: Dict[str, LLMConfig]

def load_config(path: str = "config.json") -> Config:
    """Load configuration from file."""
    with open(path) as f:
        data = json.load(f)
    
    return Config(
        difficulty=data["difficulty"],
        JudgingCriteriaConfig=JudgingCriteriaConfig(**data["judging"]),
        EvolutionConfig=EvolutionConfig(
            population_size=data["evolution"]["population_size"],
            generations=data["evolution"]["generations"],
            mutation_rate=data["evolution"]["mutation_rate"],
            crossover_rate=data["evolution"]["crossover_rate"],
            tournament_size=data["evolution"]["tournament_size"],
            min_fitness_threshold=data["evolution"]["min_fitness_threshold"]
        ),
        domains={
            domain: DomainConfig(**config)
            for domain, config in data["domains"].items()
        },
        llm={
            provider: LLMConfig(**config)
            for provider, config in data["llm"].items()
        }
    )

