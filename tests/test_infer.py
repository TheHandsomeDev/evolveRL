"""
Tests for the Judge and AdversarialTester components.
"""
import pytest
from unittest.mock import patch
import tempfile
import os

from evolverl.judge import Judge, JudgingCriteria
from evolverl.adversarial import AdversarialTester
from evolverl.agent import Agent


@pytest.fixture
def mock_llm_response():
    return "Test adversarial case"


@pytest.fixture
def test_judge():
    return Judge(
        model="gpt-4o-mini",
        criteria=JudgingCriteria(
            correctness=1.0,
            clarity=0.7,
            efficiency=0.4,
            completeness=0.8,
            consistency=0.5
        )
    )


@pytest.fixture
def test_tester(mock_llm_response):
    with patch('evolverl.llm.LLMBackend') as mock:
        instance = mock.return_value
        instance.generate.return_value = mock_llm_response
        tester = AdversarialTester(
            difficulty="medium",
            domain="math"
        )
        tester.llm = instance
        yield tester


def test_judge_initialization():
    """Test judge initialization with custom criteria."""
    judge = Judge(model="gpt-4o-mini")
    assert judge.model == "gpt-4o-mini"
    assert isinstance(judge.criteria, JudgingCriteria)
    assert judge.criteria.correctness == 1.0  # Default weight


def test_judge_evaluation(test_judge):
    """Test response evaluation with different criteria."""
    response = """
    Let me solve this step by step:
    1. First, we analyze the problem
    2. Then, we calculate
    3. Finally, we verify
    The answer is 42.
    """
    
    score = test_judge.evaluate(
        task="Solve this math problem",
        response=response,
        domain="math"
    )
    
    assert 0 <= score <= 1
    assert len(test_judge.evaluation_history) == 1


def test_judge_evaluation_with_ground_truth(test_judge):
    """Test evaluation against ground truth."""
    score = test_judge.evaluate(
        task="What is 2+2?",
        response="The answer is 4",
        ground_truth="4",
        domain="math"
    )
    
    assert 0 <= score <= 1
    assert len(test_judge.evaluation_history) == 1


def test_judge_save_load_history(test_judge):
    """Test saving and loading evaluation history."""
    # Generate some evaluation history
    test_judge.evaluate(
        task="Test task",
        response="Test response"
    )
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Save history
        test_judge.save_history(tmp.name)
        
        # Load history
        loaded_history = Judge.load_history(tmp.name)
        
        # Compare
        assert len(loaded_history) == len(test_judge.evaluation_history)
        assert loaded_history[0]["task"] == test_judge.evaluation_history[0]["task"]
        
        # Cleanup
        os.unlink(tmp.name)


def test_adversarial_tester_initialization():
    """Test adversarial tester initialization."""
    tester = AdversarialTester(difficulty="hard", domain="code")
    assert tester.difficulty == "hard"
    assert tester.domain == "code"
    assert len(tester.test_history) == 0


def test_generate_test_cases(test_tester):
    """Test generation of adversarial test cases."""
    agent = Agent(model="gpt-4o-mini")
    test_cases = test_tester.generate_test_cases(
        agent=agent,
        num_cases=3
    )
    
    assert len(test_cases) == 3
    assert all(isinstance(test, tuple) for test in test_cases)
    assert all(isinstance(meta, dict) for _, meta in test_cases)


def test_domain_specific_test_generation(test_tester):
    """Test domain-specific test case generation."""
    # Math domain
    test_tester.domain = "math"
    math_test = test_tester._generate_math_test()
    assert isinstance(math_test, str)
    assert any(term in math_test.lower() for term in ["calculate", "solve", "evaluate"])
    
    # Code domain
    test_tester.domain = "code"
    code_test = test_tester._generate_code_test()
    assert isinstance(code_test, str)
    assert any(term in code_test.lower() for term in ["function", "implement", "write"])


def test_test_history_tracking(test_tester):
    """Test tracking of generated test cases."""
    agent = Agent(model="gpt-4o-mini")
    test_tester.generate_test_cases(agent, num_cases=2)
    
    assert len(test_tester.test_history) == 2
    assert all("difficulty" in h for h in test_tester.test_history)
    assert all("domain" in h for h in test_tester.test_history)


def test_save_load_history(test_tester):
    """Test saving and loading test history."""
    # Generate some test history
    agent = Agent(model="gpt-4o-mini")
    test_tester.generate_test_cases(agent, num_cases=2)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Save history
        test_tester.save_history(tmp.name)
        
        # Load history
        loaded_history = AdversarialTester.load_history(tmp.name)
        
        # Compare
        assert len(loaded_history) == len(test_tester.test_history)
        assert loaded_history[0]["difficulty"] == test_tester.test_history[0]["difficulty"]
        
        # Cleanup
        os.unlink(tmp.name)