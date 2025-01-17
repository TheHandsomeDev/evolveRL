"""Test case definition for adversarial testing."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AdversarialTestCase:
    """A test case for adversarial testing."""
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict) 