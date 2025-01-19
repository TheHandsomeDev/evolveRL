import argparse
import asyncio
import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv

from evolverl.evolution import Evolution, EvolutionConfig
from evolverl.llm import LLMConfig
from evolverl.utils.logging import setup_logging

load_dotenv(find_dotenv())

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_evolution_config(config_data: Dict[str, Any]) -> EvolutionConfig:
    """Create EvolutionConfig from dictionary data"""
    llm_config = LLMConfig(**config_data.pop("llm_config"))
    return EvolutionConfig(**config_data, llm_config=llm_config)

def load_domain_data(filepath: Optional[str], domain: Optional[str], description: Optional[str]) -> tuple[str, str]:
    """Load domain and description from file or use provided values"""
    if filepath:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data["domain"], data["description"]
    elif domain and description:
        return domain, description
    else:
        raise ValueError("Must provide either filepath or both domain and description")

async def main():
    parser = argparse.ArgumentParser(description='Train an evolveRL agent')
    
    # Config arguments
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config', type=str, help='Path to config JSON file')
    config_group.add_argument('--use-default-config', action='store_true', 
                            help='Use default configuration')
    
    # Domain arguments
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument('--domain-file', type=str, 
                            help='Path to JSON file containing domain and description')
    domain_group.add_argument('--domain', type=str, help='Domain for the agent')
    
    # Optional arguments
    parser.add_argument('--description', type=str, 
                       help='Description of the use case (required if using --domain)')
    parser.add_argument('--experiment-id', type=str, default="default",
                       help='Unique identifier for this experiment')
    parser.add_argument('--provider', type=str,
                       help='select provider: ["openai", "anthropic"] (llama coming soon)')
    
    # Add verbosity argument
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase output verbosity (e.g., -v, -vv, -vvv)')
    
    # Add output directory argument
    parser.add_argument('--output-dir', type=str,
                       help='Directory for saving agent configurations')
    
    args = parser.parse_args()
    
    # Setup logging based on verbosity
    setup_logging(args.verbose)
    
    # Load configuration
    if args.config:
        config_data = load_config(args.config)
    else:
        default_config_path = os.path.join(os.path.dirname(__file__), 
                                         'config', 'default_config.json')
        config_data = load_config(default_config_path)
    
    # Override output directory if provided
    if args.output_dir:
        config_data["output_dir"] = args.output_dir
    
    if not args.provider:
        args.provider = "openai"
    if args.provider == "openai":
        config_data["llm_config"]["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    elif args.provider == "anthropic":
        config_data["llm_config"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Invalid provider: {args.provider}")
    
    # Create evolution config
    config = create_evolution_config(config_data)
    
    # Load domain data
    domain, description = load_domain_data(
        args.domain_file,
        args.domain,
        args.description
    )
    
    if args.domain and not args.description:
        parser.error("--description is required when using --domain")
    
    # Create and run evolution
    evolution = Evolution(config, experiment_id=args.experiment_id)
    
    print(f"Starting evolution process for domain: {domain}")
    print(f"Experiment ID: {args.experiment_id}")
    print("Configuration:", json.dumps(config_data, indent=2))
    
    try:
        await evolution.evolve(domain=domain, description=description)
        print("\nEvolution completed successfully!")
        print(f"Best agent saved to: {evolution.config.output_dir}/{args.experiment_id}_best.json")
    except Exception as e:
        print(f"\nError during evolution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 