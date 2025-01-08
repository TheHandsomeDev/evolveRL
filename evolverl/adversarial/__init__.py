"""
Adversarial package for generating challenging test cases.

This package provides components for generating and managing adversarial
test cases that challenge and expose weaknesses in evolved models.
"""

from .tester import AdversarialTester
from .adversarial import TestCase

__all__ = ['AdversarialTester', 'TestCase']
