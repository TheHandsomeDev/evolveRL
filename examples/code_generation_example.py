"""
Example script demonstrating the AERL framework for code generation.
"""

import os
import logging
from evolverl.llm import LLMBackend, LLMConfig
from evolverl.judge import Judge, JudgingCriteria
from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.domains.code_generation import CodeAdversarialTester, CodeEvaluator
from evolverl.domains.code_prompt_writer import CodePromptWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing components...")
    
    # Initialize components
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000
    )
    
    logger.info(f"Creating LLM backend with model: {llm_config.model}")
    llm = LLMBackend(config=llm_config)
    
    # Initialize domain-specific components
    logger.info("Creating adversarial tester...")
    adversarial = CodeAdversarialTester(
        initial_difficulty=0.3,
        difficulty_growth_rate=0.1
    )
    
    logger.info("Creating judge...")
    judge = Judge(
        criteria=JudgingCriteria(
            correctness=1.0,
            clarity=0.5,
            efficiency=0.7,
            robustness=0.6,
            completeness=0.5,
            consistency=0.4
        ),
        code_evaluator=CodeEvaluator()
    )
    
    # Use code-specific prompt writer
    logger.info("Creating prompt writer...")
    prompt_writer = CodePromptWriter()
    
    # Initialize evolution with smaller population and fewer generations for testing
    logger.info("Creating evolution controller...")
    evolution = Evolution(
        config=EvolutionConfig(
            population_size=2,  # Reduced for testing
            generations=2,      # Reduced for testing
            mutation_rate=0.2,
            crossover_rate=0.7,
            elite_size=1,
            min_fitness_threshold=0.8,
            tournament_size=2   # Set tournament size equal to population size
        ),
        llm_backend=llm,
        judge=judge,
        adversarial=adversarial,
        prompt_writer=prompt_writer
    )
    
    # Run evolution
    logger.info("Starting evolutionary process...")
    try:
        best_individual = evolution.run()
        
        logger.info("\nEvolution completed successfully!")
        logger.info("\nBest prompt found:")
        print(best_individual.prompt)
        logger.info(f"\nFitness: {best_individual.fitness:.3f}")
        logger.info("\nMetrics:")
        for metric, value in best_individual.metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
            
    except Exception as e:
        logger.error(f"Error during evolution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 