"""
LLM package for interfacing with language models.

This package provides a unified interface for working with different
language model backends (OpenAI GPT, local LLaMA, etc.).
"""

from .llm import LLMBackend, LLMConfig

__all__ = ['LLMBackend', 'LLMConfig'] 