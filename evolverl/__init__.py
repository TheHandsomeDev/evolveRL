"""
EvolveRL: Evolutionary Reinforcement Learning for LLMs.

This package provides tools for evolving and optimizing Large Language Models
through evolutionary algorithms and reinforcement learning.
"""

from .agent import Agent
from .evolution import Evolution, EvolutionConfig
from .judge import Judge, JudgingCriteria
from .adversarial import AdversarialTester
from .prompt_writer import PromptWriter, PromptMutationConfig
from .llm import LLMBackend, LLMConfig

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'Agent',
    'Evolution',
    'EvolutionConfig',
    'Judge',
    'JudgingCriteria',
    'AdversarialTester',
    'PromptWriter',
    'PromptMutationConfig',
    'LLMBackend',
    'LLMConfig'
] 