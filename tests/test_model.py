"""
Tests for the Agent and LLM components.
"""
import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from evolverl.agent import Agent
from evolverl.llm import LLMBackend, LLMConfig


@pytest.fixture
def mock_llm_response():
    return "This is a test response from the LLM."


@pytest.fixture
def mock_llm_backend(mock_llm_response):
    with patch('evolverl.llm.LLMBackend') as mock:
        instance = mock.return_value
        instance.generate.return_value = mock_llm_response
        yield instance


@pytest.fixture
def test_agent(mock_llm_backend):
    return Agent(
        model="gpt-4o-mini",
        config={"temperature": 0.7},
        api_key="test_key"
    )


def test_agent_initialization():
    """Test agent initialization with different configurations."""
    agent = Agent(model="gpt-4o-mini")
    assert agent.model == "gpt-4o-mini"
    assert isinstance(agent.config, dict)
    assert "{task}" in agent.prompt_template
    assert "{context}" in agent.prompt_template


def test_agent_run(test_agent, mock_llm_response):
    """Test agent's run method."""
    response = test_agent.run(
        task="What is 2+2?",
        context="Basic math question"
    )
    assert response == mock_llm_response
    test_agent.llm.generate.assert_called_once()


def test_agent_update_prompt(test_agent):
    """Test prompt updating and history tracking."""
    old_prompt = test_agent.prompt_template
    new_prompt = "New test prompt: {task}"
    
    test_agent.update_prompt(new_prompt)
    assert test_agent.prompt_template == new_prompt
    assert len(test_agent.evolution_history) == 1
    assert test_agent.evolution_history[0]["old_prompt"] == old_prompt
    assert test_agent.evolution_history[0]["new_prompt"] == new_prompt


def test_agent_save_load_state(test_agent):
    """Test saving and loading agent state."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Save state
        test_agent.save_state(tmp.name)
        
        # Load state
        loaded_agent = Agent.load_state(tmp.name)
        
        # Compare states
        assert loaded_agent.model == test_agent.model
        assert loaded_agent.config == test_agent.config
        assert loaded_agent.prompt_template == test_agent.prompt_template
        
        # Cleanup
        os.unlink(tmp.name)


def test_llm_backend_initialization():
    """Test LLM backend initialization."""
    config = LLMConfig(model="gpt-4o-mini")
    
    # Test with direct API key
    backend = LLMBackend(config=config, api_key="test_key")
    assert backend.config == config
    
    # Test with environment variable
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'env_test_key'}):
        backend = LLMBackend(config=config)
        assert backend.config == config
    
    # Test missing API key
    with pytest.raises(ValueError):
        LLMBackend(config=config)


@patch('openai.ChatCompletion.create')
def test_llm_backend_generate(mock_create):
    """Test LLM text generation."""
    mock_create.return_value.choices = [
        Mock(message=Mock(content="Test response"))
    ]
    
    config = LLMConfig(model="gpt-4o-mini")
    backend = LLMBackend(config=config, api_key="test_key")
    
    response = backend.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )
    
    assert response == "Test response"
    mock_create.assert_called_once()


@patch('openai.Embedding.create')
def test_llm_backend_embedding(mock_create):
    """Test LLM embedding generation."""
    mock_embedding = [0.1, 0.2, 0.3]
    mock_create.return_value.data = [Mock(embedding=mock_embedding)]
    
    config = LLMConfig(model="gpt-4o-mini")
    backend = LLMBackend(config=config, api_key="test_key")
    
    embedding = backend.get_embedding("Test text")
    
    assert embedding == mock_embedding
    mock_create.assert_called_once()