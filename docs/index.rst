EvolveRL Documentation
=====================

EvolveRL is a powerful Python framework for training and deploying autonomous AI agents through adversarial evolutionary reinforcement learning. It enables continuous self-improvement through evolutionary algorithms and robust testing mechanisms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   components/index
   examples/index
   api/index
   contributing

Installation
-----------

Install EvolveRL using pip:

.. code-block:: bash

   # Basic installation
   pip install evolverl

   # With LLaMA support
   pip install evolverl[llama]

   # With all features (LLaMA, DeFi tools)
   pip install evolverl[all]

For development installation:

.. code-block:: bash

   git clone https://github.com/TheHandsomeDev/evolverl.git
   cd evolverl
   pip install -e ".[dev]"

Quick Start
----------

Here's a simple example of using EvolveRL with Solana DeFi analysis:

.. code-block:: python

   from evolverl import Agent, Evolution, Judge, AdversarialTester
   from evolverl.judge import JudgingCriteria

   # Initialize components
   agent = Agent(
       model="gpt-4o-mini",
       config={
           "temperature": 0.7,
           "max_tokens": 500
       }
   )

   evolution = Evolution(
       population_size=10,
       generations=50,
       mutation_rate=0.2,
       domain="defi"
   )

   judge = Judge(
       model="gpt-4o-mini",
       criteria=JudgingCriteria(
           correctness=1.0,
           efficiency=0.8,
           completeness=0.9
       )
   )

   tester = AdversarialTester(
       difficulty="medium",
       domain="defi"
   )

   # Train the agent
   evolved_agent = evolution.train(
       agent=agent,
       task="Analyze Solana DeFi opportunities",
       judge=judge,
       tester=tester
   )

   # Use the evolved agent
   response = evolved_agent.run(
       task="Analyze Raydium pool metrics",
       context="""
       Pool: SOL-USDC
       TVL: $200M
       Daily volume: $25M
       Current APY: 4.5% + RAY rewards
       Transaction fee: 0.000005 SOL
       """
   )

Key Features
-----------

- **🧬 Evolutionary Learning**: Automated prompt and configuration optimization
- **🎯 Multi-Domain Support**: Specialized components for math, code, and DeFi domains
- **⚖️ Robust Evaluation**: Comprehensive judging system with multiple criteria
- **🔥 Adversarial Testing**: Generate challenging test cases to ensure robustness
- **💾 State Management**: Save and load evolved models and their states
- **🔄 Multiple Model Support**: Use OpenAI's GPT models or run LLaMA locally
- **🤖 Self-Improvement Loop**: Continuous evolution without human intervention
- **📊 Performance Metrics**: Data-driven validation of improvements

Components
---------

EvolveRL consists of four main components:

1. **Evolution Controller**
   - Manages population evolution and selection
   - Configurable evolution parameters
   - Performance-based selection strategies

2. **Judge**
   - Multi-criteria evaluation system
   - Domain-specific judging strategies
   - Customizable scoring weights

3. **Agent**
   - Flexible LLM backend integration (GPT-4o-mini, LLaMA)
   - State management and checkpointing
   - Configurable inference parameters

4. **Adversarial Tester**
   - Dynamic test case generation
   - Domain-specific testing strategies
   - Difficulty level adjustment

Model Support
------------

EvolveRL supports multiple LLM backends:

1. **OpenAI Models**
   - Set up with API key:
     ``export OPENAI_API_KEY=your_api_key``
   - Supported models: gpt-4o-mini (default)

2. **Local LLaMA**
   - Run models locally with PyTorch
   - Custom model path configuration
   - CPU/GPU support

Contributing
-----------

We welcome contributions! Please check our `Contributing Guidelines <https://github.com/TheHandsomeDev/evolverl/blob/main/CONTRIBUTING.md>`_ for details.

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/TheHandsomeDev/evolverl/blob/main/LICENSE>`_ file for details.

Contact
-------

- GitHub: `@TheHandsomeDev <https://github.com/TheHandsomeDev>`_
- Twitter: `@evolverl <https://x.com/evolverl>`_
- Website: `evolverl.com <https://evolverl.com>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 