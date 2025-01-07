"""
Judge package for evaluating model responses.

This package provides components for evaluating model responses across
multiple criteria and domains, including specialized evaluation for
math, code, and DeFi tasks.
"""

from .judge import Judge, JudgingCriteria

__all__ = ['Judge', 'JudgingCriteria'] 