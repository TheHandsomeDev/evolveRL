"""
Example script demonstrating inference using evolved agents.

This example shows how to:
1. Load and use evolved agents
2. Run inference with different models and configurations
3. Evaluate responses using the Judge component
4. Generate adversarial test cases
"""
from evolverl.agent import Agent
from evolverl.judge import Judge, JudgingCriteria
from evolverl.adversarial import AdversarialTester

# Example 1: Load GPT-4o-mini evolved agents
math_agent = Agent.load_state("models/evolved_math_solver.json")

# Example 2: Load LLaMA evolved agents
llama_agent = Agent(
    model="local_llama",
    config={
        "model_path": "models/llama-7b",
        "device": "cuda",
        "temperature": 0.7
    }
)
llama_agent.load_state("models/evolved_llama_solver.json")

# Initialize judge for evaluation
judge = Judge(
    model="gpt-4o-mini",  # Can also use "local_llama"
    criteria=JudgingCriteria(
        correctness=1.0,
        clarity=0.8,
        efficiency=0.6,
        completeness=0.7,
        consistency=0.5
    )
)

# Example 1: Math Problem Solving with GPT-4o-mini
math_task = "Solve the system of equations: 2x + y = 7, x - y = 1"
math_response = math_agent.run(
    task=math_task,
    context="Show step-by-step solution using substitution method."
)
print("\nGPT-4o-mini Math Solution:")
print(math_response)

# Example 2: Math Problem Solving with LLaMA
llama_response = llama_agent.run(
    task=math_task,
    context="Show step-by-step solution using substitution method."
)
print("\nLLaMA Math Solution:")
print(llama_response)

# Evaluate responses
gpt_score = judge.evaluate(
    task=math_task,
    response=math_response,
    domain="math"
)
print(f"GPT-4o-mini Score: {gpt_score:.2f}")

llama_score = judge.evaluate(
    task=math_task,
    response=llama_response,
    domain="math"
)
print(f"LLaMA Score: {llama_score:.2f}")

# Generate adversarial test cases
tester = AdversarialTester(
    difficulty="hard",
    domain="math",
    model="gpt-4o-mini"  # Can also use "local_llama"
)
test_cases = tester.generate_test_cases(math_agent, num_cases=2)

print("\nAdversarial Test Cases:")
for test, metadata in test_cases:
    print(f"\nTest: {test}")
    print(f"Metadata: {metadata}")
    
    # Test both models
    gpt_response = math_agent.run(task=test)
    llama_response = llama_agent.run(task=test)
    
    gpt_score = judge.evaluate(task=test, response=gpt_response, domain="math")
    llama_score = judge.evaluate(task=test, response=llama_response, domain="math")
    
    print(f"GPT-4o-mini Score: {gpt_score:.2f}")
    print(f"LLaMA Score: {llama_score:.2f}")