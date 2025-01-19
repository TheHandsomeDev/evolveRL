from .agent import Agent, AgentConfig

__all__ = ['Agent', 'AgentConfig'] 


# # Example usage:
# # Create an agent with default config
# agent = Agent()

# # Set a prompt template
# agent.set_default_prompt("You are a helpful AI assistant.")

# # Send a message
# response = await agent.send_message("What is the capital of France?")

# # Save configuration
# agent.save_config("agent_config.json")

# # Load configuration
# new_agent = Agent()
# new_agent.load_config("agent_config.json")