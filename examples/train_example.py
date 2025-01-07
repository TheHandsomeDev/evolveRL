"""
Example script demonstrating the training process using EvolveRL.

This example shows how to:
1. Initialize and configure the evolution process
2. Train an agent using evolutionary optimization
3. Save and load model states
4. Use different configurations and domains
"""
from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.agent import Agent
from evolverl.judge import Judge, JudgingCriteria
from evolverl.adversarial import AdversarialTester

# Example 1: Using OpenAI GPT-4o-mini
agent = Agent(
    model="gpt-4o-mini",
    config={
        "temperature": 0.7,
        "max_tokens": 500
    }
)

# Example 2: Using Local LLaMA
llama_agent = Agent(
    model="local_llama",
    config={
        "model_path": "models/llama-7b",  # Path to local LLaMA weights
        "temperature": 0.7,
        "max_tokens": 500,
        "device": "cuda"  # or "cpu"
    }
)

# Configure evolution parameters
evolution_config = EvolutionConfig(
    population_size=10,
    generations=5,
    mutation_rate=0.2,
    crossover_rate=0.1,
    domain="math"  # Can be "math", "code", "defi", or None
)

# Initialize evolution controller
evolution = Evolution(config=evolution_config)

# Configure judging criteria
judge = Judge(
    model="gpt-4o-mini",  # Can also use "local_llama"
    criteria=JudgingCriteria(
        correctness=1.0,
        clarity=0.7,
        efficiency=0.5,
        completeness=0.8,
        consistency=0.6
    )
)

# Configure adversarial testing
tester = AdversarialTester(
    difficulty="medium",
    domain="math"
)

# Train the agent (using GPT-4o-mini)
evolved_agent = evolution.train(
    agent=agent,  # or llama_agent
    task="Solve complex mathematical problems step by step",
    judge=judge,
    tester=tester
)

# Save the evolved agent's state
evolved_agent.save_state("models/evolved_math_solver.json")

# Example of loading and using the evolved agent
loaded_agent = Agent.load_state("models/evolved_math_solver.json")
response = loaded_agent.run(
    task="Solve the quadratic equation: x^2 + 5x + 6 = 0",
    context="Show all steps and explain the reasoning."
)
print("Evolved Agent Response:", response)