"""
Core Agent class for managing LLM interactions and prompt evolution.
"""
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

from ..llm import LLMBackend, LLMConfig


class Agent:
    """
    A flexible AI agent that can evolve its prompts and configurations through training.
    
    This class represents an individual agent in the evolutionary process. Each agent
    has its own prompt template and configuration, which can be mutated and evolved
    over time to improve performance.
    
    Args:
        model (str): The name of the LLM model to use (e.g., "gpt-4o-mini")
        config (Optional[Dict[str, Any]]): Additional configuration parameters
        prompt_template (Optional[str]): Initial prompt template to use
        api_key (Optional[str]): API key for the LLM service
    
    Example:
        >>> agent = Agent(
        ...     model="gpt-4o-mini",
        ...     config={"temperature": 0.7, "max_tokens": 150},
        ...     prompt_template="Solve this math problem: {task}"
        ... )
        >>> response = agent.run("What is 2 + 2?")
        >>> print(response)
        "Let me solve this step by step:
        1. We are adding 2 and 2
        2. 2 + 2 = 4
        Therefore, 2 + 2 equals 4."
    """
    
    def __init__(
        self,
        model: str,
        config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.config = config or {}
        self.prompt_template = prompt_template or self._get_default_prompt()
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Initialize LLM backend
        llm_config = LLMConfig(
            model=model,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 500),
            top_p=self.config.get("top_p", 1.0),
            frequency_penalty=self.config.get("frequency_penalty", 0.0),
            presence_penalty=self.config.get("presence_penalty", 0.0),
            stop=self.config.get("stop", None)
        )
        self.llm = LLMBackend(config=llm_config, api_key=api_key)
    
    def _get_default_prompt(self) -> str:
        """
        Returns the default prompt template.
        
        This template includes basic instructions for the model to:
        1. Consider the context
        2. Analyze the task
        3. Provide a well-structured response
        """
        return """You are an AI assistant. Given the following context and task:
Context: {context}
Task: {task}

Please provide a response that is:
1. Accurate and relevant
2. Well-reasoned
3. Properly formatted

Response:"""

    def run(
        self,
        task: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Execute the agent on a given task.
        
        This method:
        1. Formats the prompt with task and context
        2. Sends it to the LLM
        3. Returns the model's response
        
        Args:
            task (str): The task or query to process
            context (Optional[str]): Additional context for the task
            system_prompt (Optional[str]): System prompt for chat models
            
        Returns:
            str: The agent's response
            
        Example:
            >>> agent = Agent(model="gpt-4o-mini")
            >>> response = agent.run(
            ...     task="Explain quantum computing",
            ...     context="The audience is high school students"
            ... )
        """
        # Format the prompt with task and context
        formatted_prompt = self.prompt_template.format(
            task=task,
            context=context or ""
        )
        
        # Get response from LLM
        return self.llm.generate(
            prompt=formatted_prompt,
            system_prompt=system_prompt
        )
    
    def update_prompt(self, new_prompt: str) -> None:
        """
        Update the agent's prompt template based on evolution results.
        
        This method:
        1. Records the old prompt in evolution history
        2. Updates to the new prompt
        3. Timestamps the change
        
        Args:
            new_prompt (str): The new prompt template to use
            
        Example:
            >>> agent.update_prompt(
            ...     "Solve this math problem step by step: {task}"
            ... )
        """
        self.evolution_history.append({
            "old_prompt": self.prompt_template,
            "new_prompt": new_prompt,
            "timestamp": datetime.now().isoformat()
        })
        self.prompt_template = new_prompt
    
    def save_state(self, path: str) -> None:
        """
        Save the agent's current state to a file.
        
        This saves:
        - Model configuration
        - Current prompt template
        - Evolution history
        
        Args:
            path (str): Path to save the state file
            
        Example:
            >>> agent.save_state("math_solver_state.json")
        """
        state = {
            "model": self.model,
            "config": self.config,
            "prompt_template": self.prompt_template,
            "evolution_history": self.evolution_history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, path: str) -> 'Agent':
        """
        Load an agent's state from a file.
        
        Args:
            path (str): Path to the state file
            
        Returns:
            Agent: A new agent instance with the loaded state
            
        Example:
            >>> agent = Agent.load_state("math_solver_state.json")
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        agent = cls(
            model=state["model"],
            config=state["config"],
            prompt_template=state["prompt_template"]
        )
        agent.evolution_history = state["evolution_history"]
        return agent 