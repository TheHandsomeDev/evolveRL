"""Main CLI script for EvolveRL."""
import argparse
import json
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

from evolverl.agent.agent import Agent
from evolverl.llm import LLMBackend
from evolverl.judge import Judge, JudgingCriteria
from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.evolution.code_evolution import CodeEvolution
from evolverl.evolution.customer_support_evolution import CustomerSupportEvolution
from evolverl.adversarial import AdversarialTester
from evolverl.adversarial.code_tester import CodeAdversarialTester
from evolverl.adversarial.customer_support_tester import CustomerSupportTester
from evolverl.generator.use_case import UseCase, UseCaseGenerator
from config import load_config

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(log_dir: str, verbose: bool = False) -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evolve_{timestamp}.log")
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(output_dir: str, agent: Agent, generation: Optional[int] = None) -> None:
    """Save evolution results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if generation is not None:
        filename = f"agent_gen{generation:03d}_{timestamp}.json"
    else:
        filename = f"agent_final_{timestamp}.json"
    
    output_path = os.path.join(output_dir, filename)
    try:
        agent.save_state(output_path)
        logger.info(f"Saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

def get_tester_for_domain(domain: str, provider: str, llm: LLMBackend) -> AdversarialTester:
    """Get appropriate tester for domain."""
    if domain == "code":
        return CodeAdversarialTester(llm_backend=llm)
    elif domain == "customer_support":
        return CustomerSupportTester(llm_backend=llm)
    else:
        raise ValueError(f"Unsupported domain: {domain}")

def main():
    parser = argparse.ArgumentParser(description="EvolveRL CLI")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Directory for output files")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for log files")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic"],
                       help="LLM provider to use")
    parser.add_argument("--domain", type=str,
                       choices=["code", "customer_support", "auto"],
                       default="auto",
                       help="Domain for evolution")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--save-generations", action="store_true",
                       help="Save agents from each generation")
    parser.add_argument("--use-case", type=str,
                       help="Description of custom use case to generate")
    parser.add_argument("--prompt-template", type=str,
                       help="Prompt template file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir, args.verbose)
    
    # Load config
    config = load_config()
    
    # Load prompt template if provided
    prompt_template = None
    if args.prompt_template:
        with open(args.prompt_template) as f:
            prompt_template = f.read()
    
    # Create evolution config
    evolution_config = EvolutionConfig(
        population_size=config.EvolutionConfig.population_size,
        generations=config.EvolutionConfig.generations,
        mutation_rate=config.EvolutionConfig.mutation_rate,
        crossover_rate=config.EvolutionConfig.crossover_rate,
        tournament_size=config.EvolutionConfig.tournament_size,
        use_case_description=args.use_case or prompt_template
    )
    
    # Initialize LLM with proper config
    llm_config = config.llm[args.provider]
    llm = LLMBackend(
        provider=args.provider,
        config=llm_config
    )
    
    # Create judge with proper config
    judge = Judge(criteria=config.JudgingCriteriaConfig)
    
    # Get domain config
    # domain_config = config.domains[args.domain]
    
    # Determine domain
    domain = args.domain
    if domain == "auto" and args.use_case:
        domain_prompt = f"""Given this use case description, determine the most appropriate domain.
Choose from: code, customer_support

Description: {args.use_case}

Return only the domain name, nothing else."""
        domain = llm.generate(domain_prompt).strip().lower()
        logger.info(f"Auto-detected domain: {domain}")
    elif domain == "auto":
        domain = "code"  # Default to code domain
    
    # Get appropriate evolution class and tester
    # evolution_cls = CodeEvolution if domain == "code" else CustomerSupportEvolution
    evolution_cls = Evolution
    tester = get_tester_for_domain(domain, args.provider, llm)
    
    # Create evolution instance
    evolution = evolution_cls(
        config=evolution_config,
        judge=judge,
        adversarial=tester,
        llm_backend=llm,
        save_callback=lambda agent, gen: save_results(args.output_dir, agent, gen) if args.save_generations else None,
        provider=args.provider
    )
    
    # Run evolution
    try:
        best_agent = evolution.run()
        logger.info("\nEvolution completed successfully!")
        logger.info(f"Best fitness: {best_agent.fitness:.3f}")
        
        # Save final results
        save_results(args.output_dir, best_agent)
        
    except Exception as e:
        logger.error(f"Error during evolution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 