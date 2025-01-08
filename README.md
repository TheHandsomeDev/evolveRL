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
- **🎯 Multi-Domain Support**: Specialized components for math, code, and DeFi domains
- **⚖️ Robust Evaluation**: Comprehensive judging system with multiple criteria
- **🔥 Adversarial Testing**: Generate challenging test cases to ensure robustness
- **💾 State Management**: Save and load evolved models and their states
- **🔄 Multiple Model Support**: Use OpenAI's GPT models or run LLaMA locally
- **🤖 Self-Improvement Loop**: Continuous evolution without human intervention
- **📊 Performance Metrics**: Data-driven validation of improvements

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (for local LLaMA support)
- CUDA-capable GPU (recommended for LLaMA)

### Installation

```bash
# Basic installation
pip install evolverl

# For local LLaMA support
pip install evolverl[llama]  # Installs PyTorch and transformers
```

### Model Setup

#### Option 1: OpenAI GPT Models

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set up your environment:

```bash
# Linux/Mac
export OPENAI_API_KEY=your_api_key

# Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key"
```

#### Option 2: Local LLaMA

1. Download LLaMA weights
2. Convert to HuggingFace format:

```bash
python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir path_to_llama_weights \
    --model_size 7B \
    --output_dir models/llama-7b
```

## Components

### 1. Evolution Controller

```python
from evolverl.evolution import Evolution

evolution = Evolution(
    population_size=10,
    generations=5,
    mutation_rate=0.2
)
```

### 2. Agent

```python
from evolverl.agent import Agent

# OpenAI GPT
agent = Agent(
    model="gpt-4o-mini",
    config={"temperature": 0.7}
)

# Local LLaMA
llama_agent = Agent(
    model="local_llama",
    config={
        "model_path": "models/llama-7b",
        "device": "cuda"
    }
)
```

### 3. Judge

```python
from evolverl.judge import Judge, JudgingCriteria

judge = Judge(
    model="gpt-4o-mini",
    criteria=JudgingCriteria(
        correctness=1.0,
        clarity=0.7
    )
)
```

### 4. Adversarial Tester

```python
from evolverl.adversarial import AdversarialTester

tester = AdversarialTester(
    difficulty="hard",
    domain="math"
)
```

## Evolutionary Loop

The core of EvolveRL is its adversarial evolutionary loop:

1. **Generation**: Spawn multiple prompt/configuration variants
2. **Testing**: Challenge variants with adversarial scenarios
3. **Evaluation**: Score performance using the Judge
4. **Selection**: Keep top performers for next generation
5. **Mutation**: Create new variants through controlled changes

```python
# Complete evolution example
evolved_agent = evolution.train(
    agent=agent,
    task="your_task",
    judge=judge,
    tester=tester
)
```

## Detailed Walkthrough

Let's walk through creating an autonomous DeFi trading agent that can analyze market conditions and suggest optimal trading strategies. This agent will continuously evolve to handle new DeFi protocols and market conditions.

### 1. Project Setup

First, create a new project and install dependencies:

```bash
# Create project directory
mkdir defi_agent && cd defi_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install EvolveRL with all dependencies
pip install evolverl[llama,defi]
```

### 2. Initial Configuration

Create a `config.py` file:

```python
from dataclasses import dataclass
from evolverl.evolution import EvolutionConfig
from evolverl.judge import JudgingCriteria

@dataclass
class DeFiAgentConfig:
    # Evolution settings
    evolution_config = EvolutionConfig(
        population_size=10,
        generations=50,
        mutation_rate=0.2,
        crossover_rate=0.1,
        domain="defi"
    )

    # Judging criteria
    judging_criteria = JudgingCriteria(
        correctness=1.0,    # Accurate analysis
        clarity=0.7,        # Clear explanations
        efficiency=0.8,     # Efficient strategies
        completeness=0.9,   # Consider all factors
        consistency=0.6     # Consistent reasoning
    )

    # Agent settings
    agent_config = {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9
    }

    # Initial prompt template
    initial_prompt = """You are a DeFi trading expert. Analyze the following scenario:
Context: {context}
Task: {task}

Consider:
1. Market conditions and trends
2. Protocol-specific risks
3. Gas costs and MEV
4. Impermanent loss potential
5. Alternative strategies

Provide a detailed analysis with:
- Clear reasoning
- Risk assessment
- Specific recommendations
- Expected outcomes

Analysis:"""
```

### 3. Create Training Data

Create `training_data.py`:

```python
import json

defi_scenarios = [
    {
        "task": "Analyze SOL/USDC liquidity provision on Orca",
        "context": "SOL price: $100, Volatility: 55% APR, Current pool TVL: $25M",
        "ground_truth": """
1. Position Analysis:
   - Optimal range: $90-$110 (±10%)
   - Expected IL: -2.5% at range bounds
   - Estimated fees: 0.25% daily volume * 0.04% fee
2. Risk Assessment:
   - High SOL volatility suggests wider range
   - Competition from Raydium pools
3. Recommendation:
   - Start with 25% of capital
   - Set stop-loss at -4% IL
   - Monitor Solana network congestion
""",
        "difficulty": "medium"
    },
    # Add more scenarios...
]

with open('data/train.json', 'w') as f:
    json.dump({"examples": defi_scenarios}, f, indent=2)
```

### 4. Implementation

Create `train_agent.py`:

```python
import logging
from evolverl.agent import Agent
from evolverl.evolution import Evolution
from evolverl.judge import Judge
from evolverl.adversarial import AdversarialTester
from config import DeFiAgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_defi_agent():
    # Load config
    config = DeFiAgentConfig()

    # Initialize components
    agent = Agent(
        model="gpt-4o-mini",
        config=config.agent_config,
        prompt_template=config.initial_prompt
    )

    evolution = Evolution(config=config.evolution_config)

    judge = Judge(
        model="gpt-4o-mini",
        criteria=config.judging_criteria
    )

    tester = AdversarialTester(
        difficulty="medium",
        domain="defi"
    )

    # Start evolution
    logger.info("Starting evolutionary training...")
    evolved_agent = evolution.train(
        agent=agent,
        task="Analyze DeFi trading opportunities and risks",
        judge=judge,
        tester=tester
    )

    # Save the evolved agent
    evolved_agent.save_state("models/evolved_defi_agent.json")

    return evolved_agent

def test_agent(agent):
    # Test scenarios
    test_cases = [
        {
            "task": "Evaluate Solana staking strategy",
            "context": "Validator: Marinade Finance, Staking APY: 6.8%, mSOL premium: 1.02"
        },
        {
            "task": "Analyze arbitrage opportunity",
            "context": "SOL price: Binance $199.5, Orca $200.2, Network fee: 0.000005 SOL"
        }
    ]

    for case in test_cases:
        response = agent.run(
            task=case["task"],
            context=case["context"]
        )
        print(f"\nTest Case: {case['task']}")
        print(f"Response: {response}")

if __name__ == "__main__":
    # Train the agent
    agent = train_defi_agent()

    # Test it
    test_agent(agent)
```

### 5. Running the Evolution

```bash
# Run the training
python train_agent.py
```

During training, you'll see output like:

```
INFO: Starting evolutionary training...
INFO: Generation 1/50
INFO: Generated 5 adversarial test cases
INFO: Best score: 0.723
INFO: Population average: 0.654
...
INFO: Generation 50/50
INFO: Best score: 0.912
INFO: Evolution complete!
```

### 6. Using the Evolved Agent

Create `use_agent.py`:

```python
from evolverl.agent import Agent

# Load the evolved agent
agent = Agent.load_state("models/evolved_defi_agent.json")

# Analyze a new scenario
response = agent.run(
    task="Analyze Raydium pool deposit opportunity",
    context="""
    Pool: SOL-USDC
    TVL: $200M
    Daily volume: $25M
    Current APY: 4.5% + RAY rewards
    Transaction fee: 0.000005 SOL
    """
)

print("Analysis:", response)
```

### 7. Monitoring and Maintenance

The agent will continuously evolve as it encounters new scenarios. You can:

1. **Track Performance**:

```python
# Load evolution history
with open("models/evolved_defi_agent.json") as f:
    history = json.load(f)["evolution_history"]

# Plot performance over time
import matplotlib.pyplot as plt
plt.plot([gen["best_score"] for gen in history])
plt.title("Agent Performance Evolution")
plt.show()
```

2. **Update Training Data**:

```python
# Add new scenarios as DeFi landscape changes
new_scenarios = [
    {
        "task": "Analyze new AMM protocol",
        "context": "..."
    }
]
# Update training data
```

3. **Fine-tune Parameters**:

```python
# Adjust evolution parameters based on performance
config.evolution_config.mutation_rate = 0.3  # Increase exploration
config.judging_criteria.efficiency = 0.9     # Emphasize efficiency
```

This walkthrough demonstrates:

- Complete setup process
- Configuration management
- Training data preparation
- Evolution process
- Testing and validation
- Ongoing maintenance

The resulting agent will:

- Analyze DeFi opportunities
- Adapt to new protocols
- Consider multiple factors
- Provide detailed recommendations
- Continuously improve

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
