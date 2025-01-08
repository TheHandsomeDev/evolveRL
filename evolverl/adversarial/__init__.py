"""
Adversarial testing module for EvolveRL.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TestCase:
    """A test case for adversarial testing."""
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = None

from .adversarial import AdversarialTester

__all__ = ['TestCase', 'AdversarialTester'] 