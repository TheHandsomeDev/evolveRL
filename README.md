# EvolveRL: Evolutionary Reinforcement Learning for LLMs

> Empowering Truly Autonomous AI Agents Through Adversarial Evolutionary Reinforcement Learning

## Links

- [Website](https://www.evolverl.com/)
- [Whitepaper](https://github.com/TheHandsomeDev/evolveRL/blob/main/evolverl-whitepaper.pdf)
- [Techpaper](https://github.com/TheHandsomeDev/evolveRL/blob/main/evolverl-techpaper.pdf)
- [Telegram](https://t.me/evolveRL)
- [Twitter/X](https://x.com/evolveRL_sol)

## 📚 Table of Contents

- 🎯 [Overview](#overview)
- ⭐ [Features](#features)
- 🚀 [Getting Started](#getting-started)
- 🔧 [Installation](#installation)
- 🛠️ [Components](#components)
- 🧬 [Evolutionary Loop](#evolutionary-loop)
- 📊 [Detailed Walkthrough](#detailed-walkthrough)
- 📄 [License](#license)
- 🤝 [Contributing](#contributing)
- 💬 [Citation](#citation)

## Overview

EvolveRL is a groundbreaking framework that enables AI agents to self-improve through evolutionary and adversarial mechanisms. Unlike traditional approaches that rely heavily on manual prompt engineering, EvolveRL allows agents to systematically generate, test, and refine their own prompts and configurations, bridging the gap between theoretical autonomy and actual self-reliance.

### The Challenge

In the emerging AI agent economy, many envision a future where agents run autonomously with minimal human oversight. However, if humans must constantly update AI prompts to handle new tasks or edge cases, the agents aren't truly sovereign. EvolveRL solves this by enabling continuous self-improvement through:

1. **Autonomous Evolution**: Agents detect gaps and update their own prompts
2. **Adversarial Testing**: Robust validation against challenging scenarios
3. **Performance-Based Selection**: Natural emergence of optimal configurations
4. **Continuous Adaptation**: Real-time response to changing conditions

## Features

- **🧬 Evolutionary Optimization**: Evolve prompts and behaviors using genetic algorithms
- **🎯 Domain Agnostic**: Specialization for any domain
- **⚖️ Robust Evaluation**: Comprehensive judging and evaluation
- **🔥 Adversarial Testing**: Generate challenging scenarios to ensure robustness
- **💾 State Management**: Save and load evolved models and their states
- **🔄 Multiple Model Support**: Use OpenAI's GPT or Anthropic's Claude, or run LLaMA locally (coming soon)
- **🤖 Self-Improvement Loop**: Continuous evolution without human intervention

## Installation

```bash
# Basic installation
pip install evolverl

# Install with all dependencies
pip install evolverl[all]
```

## Quick Start

```python
from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.llm import LLMConfig
from evolverl.agent import Agent, AgentConfig

# Configure LLM backend
llm_config = LLMConfig(
    model_name="gpt-4",
    model_type="openai",  # or "anthropic"
    openai_api_key="your-api-key"  # or anthropic_api_key for Claude
)

# Create agent with system prompt
agent_config = AgentConfig(llm_config=llm_config)
agent = Agent(agent_config)
agent.set_default_prompt("""You are an expert AI agent specialized in mathematics.
You break down complex problems step by step and show your work clearly.""")

# Configure evolution process
config = EvolutionConfig(
    population_size=5,
    generations=10,
    mutation_rate=0.1,
    crossover_rate=0.8,
    output_dir="agents"
)

# Create evolution instance
evolution = Evolution(config, experiment_id="math_solver")

# Run evolution process
await evolution.evolve(
    domain="mathematics",
    description="Solve complex math problems with detailed explanations"
)
```

### Direct Agent Usage

You can also use agents directly without evolution:

```python
# Create and configure agent
agent = Agent(AgentConfig(llm_config=llm_config))
agent.set_default_prompt("You are a helpful AI assistant...")

# Send messages
response = await agent.send_message("What is 2+2?")
print(response)
```

## CLI Usage

`train_agent.py` is a single file CLI that runs the evolution process. Be sure to update the config file `default_config.json` first, as well as keep your OpenAI or Anthropic API key as environment variables or in the `.env`. 
```bash
# Basic usage with OpenAI
python train_agent.py --domain math --description "Solve math problems" -v

# Use Anthropic's Claude
python train_agent.py --provider anthropic --domain math --description "Solve math problems"

# Load domain from file
python train_agent.py --domain-file domains/math_solver.json

# Custom output directory
python train_agent.py --domain math --description "..." --output-dir ./my_agents

# Increase verbosity (up to -vvvvv)
python train_agent.py --domain math --description "..." -vvv
```
Current domain examples are in natural language. You can add more details when building your own use cases. In addition, you may include any examples you believe are important for the agent to know. 

## Output Structure

```
agents/
├── {experiment_id}_gen0.json           # Best agent from generation 0
├── {experiment_id}_gen0_full.json      # All variants and scores from generation 0
├── {experiment_id}_gen1.json           # Best agent from generation 1
├── {experiment_id}_gen1_full.json      # All variants and scores from generation 1
└── {experiment_id}_best.json           # Best agent overall
```
The individual `.json` (not the `*_full.json`) contains the `AgentConfig` for the best agent of the generation or overall. You may initiate an agent directly from its `AgentConfig` file by calling `agent.load_config(PATH_TO_CONFIG_FILE)`. Be sure to update the API key as it will not be stored in the `AgentConfig` file.

### Generation Output Format

```json
{
    "population_size": 5,
    "generations": 10,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "min_score_threshold": 0.7,
    "tournament_size": 2,
    "max_interaction_attempts": 5,
    "output_dir": "agents",
    "llm_config": {
        "model_name": "gpt-4o-mini",
        "model_type": "openai",
        "max_tokens": 500,
        "temperature": 0.7
    }
}
```

### Output Structure

```
agents/
├── {experiment_id}_gen0.json           # Best agent from generation 0
├── {experiment_id}_gen0_full.json      # All variants from generation 0
├── {experiment_id}_gen1.json           # Best agent from generation 1
├── {experiment_id}_gen1_full.json      # All variants from generation 1
└── {experiment_id}_best.json           # Best agent overall
```

### Progress Tracking

The evolution process shows real-time progress with nested progress bars:
```
Generation 2/10: 100%|██████████| 6/6 [00:15<00:00, best_score=0875, avg_score=0834]
Overall Progress:  15%|██        | 12/80 [00:30<02:45, generation=2/10, best_overall=0875]
```
This may take a while depending on the number of generations and population size per generation.

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

```bibtex
@software{evolverl2024,
    title={EvolveRL: Evolutionary Reinforcement Learning for LLMs},
    author={TheHandsomeDev},
    year={2025},
    url={https://www.evolverl.com/}
}
```
