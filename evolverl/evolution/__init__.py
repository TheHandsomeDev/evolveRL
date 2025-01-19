from .evolution import Evolution, EvolutionConfig

__all__ = ['Evolution', 'EvolutionConfig'] 

# Example usage:
# # Create evolution instance
# config = EvolutionConfig(population_size=5, generations=10)
# evolution = Evolution(config, experiment_id="math_solver")

# # Run evolution
# await evolution.evolve(
#     domain="mathematics",
#     description="Solve complex math problems with detailed explanations"
# )