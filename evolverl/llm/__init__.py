from .llm import LLMConfig, LLMBackend

__all__ = ['LLMConfig', 'LLMBackend'] 

# # Example usage:
# # Using default config (OpenAI with gpt-4o-mini)
# config = LLMConfig(openai_api_key="your-api-key")
# backend = LLMBackend(config)

# # Using Anthropic Claude
# config = LLMConfig(
#     model_type="anthropic",
#     model_name="claude-3-5-sonnet-20241022",
#     anthropic_api_key="your-anthropic-key"
# )
# backend = LLMBackend(config)

# # Generate text
# response = await backend.generate("Hello, how are you?")