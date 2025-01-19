import logging
from dataclasses import dataclass
import json
import os
from typing import List, Optional, Dict, Any
from ..llm import LLMConfig, LLMBackend
from ..agent import Agent, AgentConfig
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration settings for Evolution"""
    population_size: int = 5
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    min_score_threshold: float = 0.7
    tournament_size: int = 2
    max_interaction_attempts: int = 5
    output_dir: str = "agents"
    llm_config: LLMConfig = None

class Evolution:
    """Evolution class that manages agent evolution process"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None, experiment_id: str = "default"):
        self.config = config or EvolutionConfig()
        self.llm_backend = LLMBackend(self.config.llm_config)
        self.experiment_id = experiment_id
        self._ensure_output_dir()
        
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.debug(f"Ensuring output directory exists: {self.config.output_dir}")
    
    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file"""
        return os.path.join(self.config.output_dir, filename)
    
    def build_agent(self, config_path: Optional[str] = None) -> Agent:
        """Build an agent from config file or with default settings"""
        if config_path and os.path.exists(config_path):
            agent = Agent()
            agent.load_config(config_path)
            return agent
        return Agent(AgentConfig(self.config.llm_config))
    
    async def make_base_agent_prompt_template(self, domain: str, description: str) -> str:
        """Generate prompt template for base agent"""
        logger.debug(f"Generating base agent prompt template for domain: {domain}")
        prompt = f"""Create a system prompt template for an AI agent handling the following use case.
The prompt should guide the agent to provide high-quality responses.

Domain: {domain}
Use Case: {description}

The template should:
1. Define the agent's role and capabilities
2. Specify expected response format
3. Include placeholders for {{task}} and {{context}}
4. Emphasize important aspects for this domain

Return only the prompt template."""
        
        template = await self.llm_backend.generate(prompt)
        logger.log(5, f"Generated base prompt template:\n{template}")
        return template
    
    async def mutate_agents(self, base_agent: Agent, count: int) -> List[Agent]:
        """Create mutated variants of the base agent"""
        logger.debug(f"Creating {count} agent variants")
        variants = [base_agent]  # Include unmutated base agent
        
        for i in range(count - 1):
            logger.log(4, f"Creating variant {i+1}/{count-1}")
            new_agent = Agent(AgentConfig(self.config.llm_config))
            mutation_prompt = f"""Given this prompt template:
{base_agent.config.prompt_template}

Create a variation that:
1. Maintains the core functionality
2. Emphasizes different aspects
3. Keeps the basic structure
4. Makes small but meaningful changes

Return only the new prompt template."""
            
            new_prompt = await self.llm_backend.generate(mutation_prompt)
            new_agent.set_default_prompt(new_prompt)
            logger.log(3, f"Variant {i+1} prompt:\n{new_prompt}")
            variants.append(new_agent)
            
        return variants
    
    async def build_adversary(self, domain: str, description: str) -> Agent:
        """Build adversary agent for generating test cases"""
        adversary = Agent(AgentConfig(self.config.llm_config))
        prompt = f"""Create a prompt template for generating challenging test cases for the following use case.
The adversary should create diverse and realistic scenarios.

Domain: {domain}
Use Case: {description}

The template should:
1. Define test case generation guidelines
2. Specify complexity levels
3. Include edge case considerations
4. Focus on domain-specific challenges

Return only the prompt template."""
        
        template = await self.llm_backend.generate(prompt)
        adversary.set_default_prompt(template)
        return adversary
    
    async def build_judge(self, domain: str, description: str) -> Agent:
        """Build judge agent for evaluating responses"""
        judge = Agent(AgentConfig(self.config.llm_config))
        prompt = f"""Create a prompt template for a judge evaluating responses to the following use case.
The judge should assess response quality across multiple criteria.

Domain: {domain}
Use Case: {description}

The template should:
1. Define evaluation criteria
2. Specify scoring guidelines
3. Include examples of good/bad responses
4. Focus on domain-specific quality aspects

Return only the prompt template."""
        
        template = await self.llm_backend.generate(prompt)
        judge.set_default_prompt(template)
        return judge
    
    async def run_interaction(self, variant: Agent, adversary: Agent) -> str:
        """Run interaction between variant and adversary"""
        logger.debug("Starting agent interaction")
        message_history = ""
        attempts = 0
        
        while attempts < self.config.max_interaction_attempts:
            logger.log(4, f"Interaction attempt {attempts + 1}")
            
            # Variant turn
            variant_message = await variant.send_message(message_history)
            message_history += f"\nVariant: {variant_message}"
            logger.log(3, f"Variant response:\n{variant_message}")
            
            if "RESOLVED" in variant_message.upper():
                logger.debug("Interaction resolved by variant")
                break
                
            # Adversary turn
            adversary_message = await adversary.send_message(message_history)
            message_history += f"\nAdversary: {adversary_message}"
            logger.log(3, f"Adversary response:\n{adversary_message}")
            
            if "RESOLVED" in adversary_message.upper():
                logger.debug("Interaction resolved by adversary")
                break
                
            attempts += 1
            
        return message_history
    
    async def run_tournament(self, chat_histories: List[str], judge: Agent) -> int:
        """Run tournament to find best variant"""
        current_round = chat_histories
        
        while len(current_round) > 1:
            next_round = []
            
            # Compare pairs
            for i in range(0, len(current_round), 2):
                if i + 1 >= len(current_round):
                    next_round.append(current_round[i])
                    continue
                    
                comparison_prompt = f"""Compare these two interaction histories and determine which shows better performance:

                History 1:
                {current_round[i]}

                History 2:
                {current_round[i + 1]}

                Return only the number (1 or 2) of the better history."""
                
                result = await judge.send_message(comparison_prompt)
                winner_idx = i if "1" in result else i + 1
                next_round.append(current_round[winner_idx])
            
            current_round = next_round
            
        return chat_histories.index(current_round[0])
    
    async def get_agent_score(self, chat_history: str, judge: Agent) -> int:
        """Get score for a single agent's performance (0-1000)"""
        scoring_prompt = f"""Evaluate this interaction history and score the agent's performance from 0 to 1000.

Consider these aspects and assign precise scores:
- Task completion (0-400 points): Accuracy and completeness of solution
- Response quality (0-300 points): Clarity and depth of explanation
- Problem-solving ability (0-200 points): Method and approach
- Communication clarity (0-100 points): Organization and presentation

Be precise in your scoring. Don't round to nearest 50 or 100.
For example: 934, 801, 767, etc.

History:
{chat_history}

Return only an integer number between 0 and 1000."""
        
        score_str = await judge.send_message(scoring_prompt)
        try:
            score = int(float(score_str.strip()))  # Handle both int and float responses
            return max(0, min(1000, score))  # Clamp between 0 and 1000
        except ValueError:
            logger.warning(f"Invalid score from judge: {score_str}")
            return 0

    def save_generation_results(self, generation: int, variants: List[Agent], 
                              best_idx: int, scores: List[int]) -> None:
        """Save all variants and their scores from a generation"""
        gen_data = {
            "generation": generation,
            "best_variant_index": best_idx,
            "variants": [
                {
                    "config": {
                        "llm_config": {
                            "model_name": variant.config.llm_config.model_name,
                            "model_type": variant.config.llm_config.model_type,
                            "max_tokens": variant.config.llm_config.max_tokens,
                            "temperature": variant.config.llm_config.temperature
                        },
                        "prompt_template": variant.config.prompt_template,
                        "score": scores[i]  # Use the score from the scores list
                    }
                }
                for i, variant in enumerate(variants)
            ]
        }
        
        path = self._get_output_path(f"{self.experiment_id}_gen{generation}_full.json")
        with open(path, 'w') as f:
            json.dump(gen_data, f, indent=2)

    async def evolve(self, domain: str, description: str) -> None:
        """Run the evolution process"""
        logger.info(f"Starting evolution for domain: {domain}")
        
        # Build initial agents
        logger.info("Building initial agents...")
        base_agent = self.build_agent()
        base_prompt = await self.make_base_agent_prompt_template(domain, description)
        base_agent.set_default_prompt(base_prompt)
        
        adversary = await self.build_adversary(domain, description)
        judge = await self.build_judge(domain, description)
        
        # Calculate total steps for overall progress
        total_steps = self.config.generations * (self.config.population_size + 1)  # +1 for saving/setup per generation
        
        # Setup progress bars
        overall_pbar = tqdm(total=total_steps, desc="Overall Progress", position=1, leave=True)
        gen_pbar = tqdm(total=self.config.population_size + 1, desc="Generation Progress", position=0, leave=True)
        
        try:
            for generation in range(self.config.generations):
                # Reset generation progress bar
                gen_pbar.reset()
                gen_pbar.set_description(f"Generation {generation + 1}/{self.config.generations}")
                
                # Create variants
                logger.debug("Creating variants...")
                variants = await self.mutate_agents(base_agent, self.config.population_size)
                gen_pbar.update(1)
                overall_pbar.update(1)
                
                # Run interactions and scoring
                logger.debug("Running interactions...")
                chat_histories = []
                scores = []
                
                for i, variant in enumerate(variants):
                    # Test variant
                    logger.debug(f"Testing variant {i + 1}/{len(variants)}")
                    history = await self.run_interaction(variant, adversary)
                    chat_histories.append(history)
                    
                    # Score variant
                    score = await self.get_agent_score(history, judge)
                    scores.append(score)
                    
                    # Update progress bars
                    gen_pbar.update(1)
                    overall_pbar.update(1)
                    
                    # Update progress bar postfix
                    current_best = max(scores)
                    current_avg = sum(scores)/len(scores) if scores else 0
                    gen_pbar.set_postfix({
                        'best_score': f"{current_best:04f}",
                        'avg_score': f"{current_avg:04f}"
                    })
                
                # Find best variant based on scores
                best_variant_idx = scores.index(max(scores))  # Use highest score instead of tournament
                best_variant = variants[best_variant_idx]
                
                # Update score for best variant
                best_variant.update_score(scores[best_variant_idx])
                
                # Save generation results
                self.save_generation_results(generation, variants, best_variant_idx, scores)
                gen_path = self._get_output_path(f"{self.experiment_id}_gen{generation}.json")
                logger.info(f"Saving best variant to {gen_path}")
                best_variant.save_config(gen_path)
                
                # Update base agent for next generation
                base_agent = best_variant
                
                # Update overall progress bar postfix
                overall_pbar.set_postfix({
                    'generation': f"{generation + 1}/{self.config.generations}",
                    'best_overall': f"{max(best_variant.config.score, max(scores)):04d}"
                })
            
            # Save final best agent
            final_path = self._get_output_path(f"{self.experiment_id}_best.json")
            logger.info(f"Evolution complete. Saving best agent to {final_path}")
            base_agent.save_config(final_path)
            
        finally:
            gen_pbar.close()
            overall_pbar.close() 