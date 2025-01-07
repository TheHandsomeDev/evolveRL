"""
Tests for the Evolution and PromptWriter components.
"""
import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.prompt_writer import PromptWriter, PromptMutationConfig
from evolverl.agent import Agent


@pytest.fixture
def mock_llm_response():
    return "Test prompt template: {task}"


@pytest.fixture
def mock_prompt_writer(mock_llm_response):
    with patch('evolverl.prompt_writer.PromptWriter') as mock:
        instance = mock.return_value
        instance.generate_initial_population.return_value = [
            mock_llm_response for _ in range(3)
        ]
        instance.mutate_prompts.return_value = [
            f"Mutated {i}: {mock_llm_response}"
            for i in range(3)
        ]
        yield instance


@pytest.fixture
def test_evolution(mock_prompt_writer):
    return Evolution(
        population_size=3,
        generations=2,
        config={
            "mutation_rate": 0.2,
            "domain": "test"
        }
    )


def test_evolution_initialization():
    """Test evolution controller initialization."""
    evolution = Evolution(population_size=5, generations=10)
    assert evolution.config.population_size == 5
    assert evolution.config.generations == 10
    assert evolution.generation == 0
    assert evolution.best_fitness == 0.0


def test_evolution_train(test_evolution):
    """Test the training process."""
    agent = Agent(model="gpt-4o-mini")
    
    # Mock the evaluation to always return improving scores
    def mock_evaluate(test_cases, judge_fn):
        return [0.5 + 0.1 * i for i in range(3)]
    
    test_evolution._evaluate_population = mock_evaluate
    
    evolved_agent = test_evolution.train(
        agent=agent,
        task="test task",
        adversarial_difficulty="easy"
    )
    
    assert isinstance(evolved_agent, Agent)
    assert test_evolution.best_fitness > 0.0
    assert len(test_evolution.evolution_history) > 0


def test_evolution_save_load_state(test_evolution):
    """Test saving and loading evolution state."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Set some state
        test_evolution.generation = 5
        test_evolution.best_fitness = 0.8
        
        # Save state
        test_evolution.save_state(tmp.name)
        
        # Load state
        loaded_evolution = Evolution.load_state(tmp.name)
        
        # Compare states
        assert loaded_evolution.generation == test_evolution.generation
        assert loaded_evolution.best_fitness == test_evolution.best_fitness
        assert loaded_evolution.config.population_size == test_evolution.config.population_size
        
        # Cleanup
        os.unlink(tmp.name)


def test_prompt_writer_initialization():
    """Test prompt writer initialization."""
    writer = PromptWriter(base_model="gpt-4o-mini")
    assert writer.base_model == "gpt-4o-mini"
    assert isinstance(writer.mutation_config, PromptMutationConfig)
    assert len(writer.mutation_config.mutation_types) > 0


@patch('evolverl.llm.LLMBackend')
def test_prompt_writer_generate_population(mock_llm):
    """Test initial population generation."""
    mock_llm.return_value.generate.return_value = "Test prompt: {task}"
    
    writer = PromptWriter(base_model="gpt-4o-mini", api_key="test_key")
    prompts = writer.generate_initial_population(
        task_description="Test task",
        population_size=3
    )
    
    assert len(prompts) == 3
    assert all("{task}" in p for p in prompts)


@patch('evolverl.llm.LLMBackend')
def test_prompt_writer_mutations(mock_llm):
    """Test prompt mutation strategies."""
    mock_llm.return_value.generate.return_value = "Mutated prompt: {task}"
    
    writer = PromptWriter(base_model="gpt-4o-mini", api_key="test_key")
    prompts = ["Original prompt: {task}" for _ in range(3)]
    scores = [0.5, 0.8, 0.3]
    
    mutated = writer.mutate_prompts(prompts, scores)
    
    assert len(mutated) == len(prompts)
    assert all("{task}" in p for p in mutated)
    assert any(p != prompts[0] for p in mutated)


def test_prompt_validation():
    """Test prompt validation logic."""
    writer = PromptWriter()
    
    # Valid prompt
    assert writer._is_valid_prompt("Test prompt with {task}")
    
    # Invalid prompts
    assert not writer._is_valid_prompt("")  # Empty
    assert not writer._is_valid_prompt("Test")  # Too short
    assert not writer._is_valid_prompt("Test without placeholder")  # No {task}
    assert not writer._is_valid_prompt("x" * 3000)  # Too long