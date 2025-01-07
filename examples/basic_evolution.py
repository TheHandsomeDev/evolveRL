"""
Basic example of using EvolveRL to train an agent through evolutionary learning.
"""
from evolverl import Agent, Evolution, AdversarialTester


def main():
    # Initialize the base agent
    agent = Agent(
        model="gpt-4o-mini",
        config={
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    
    # Create the evolution controller
    evolution = Evolution(
        population_size=5,
        generations=10,
        config={
            "mutation_rate": 0.2,
            "elite_size": 1,
            "fitness_threshold": 0.9
        }
    )
    
    # Create an adversarial tester
    tester = AdversarialTester(
        difficulty="medium",
        domain="defi"
    )
    
    # Generate some test cases
    test_cases = tester.generate_test_cases(agent, num_cases=3)
    
    print("Generated test cases:")
    for test, metadata in test_cases:
        print(f"\nTest: {test}")
        print(f"Metadata: {metadata}")
    
    # Define a simple judge function
    def judge_fn(task: str, response: str) -> float:
        """Simple judge that scores based on response length."""
        # This is just a placeholder - in practice, you'd want a more
        # sophisticated evaluation metric
        return min(len(response) / 100, 1.0)
    
    # Train the agent
    print("\nStarting evolution...")
    evolved_agent = evolution.train(
        agent=agent,
        task=test_cases[0][0],  # Use first test case as training task
        adversarial_difficulty="medium",
        judge_fn=judge_fn
    )
    
    print(f"\nEvolution complete!")
    print(f"Best fitness achieved: {evolution.best_fitness}")
    
    # Test the evolved agent
    print("\nTesting evolved agent...")
    for test, _ in test_cases:
        response = evolved_agent.run(test)
        print(f"\nTest: {test}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main() 