"""
LLM Factory for creating LLM instances from different providers
Supports: Ollama, OpenAI, Anthropic, Groq
"""
from typing import Any, Dict
from config import LLMConfig, LLMProvider


def create_llm(config: Dict[str, Any]) -> Any:
    """
    Create LLM instance based on provider configuration
    
    Args:
        config: Dictionary with provider, model, and other settings
    
    Returns:
        LLM instance from the appropriate provider
    """
    provider = config.get("provider")
    
    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            base_url=config.get("base_url", "http://localhost:11434"),
            model=config.get("model", "llama3.2:3b"),
            temperature=config.get("temperature", 0.7),
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=config.get("api_key"),
            model=config.get("model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=config.get("api_key"),
            model=config.get("model", "claude-3-5-sonnet-20241022"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=config.get("api_key"),
            model=config.get("model", "llama-3.1-8b-instant"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens"),
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_structured_llm(llm: Any, schema: Any) -> Any:
    """
    Wrap LLM with structured output capability
    
    Args:
        llm: Base LLM instance
        schema: Pydantic schema for structured output
    
    Returns:
        LLM with structured output binding
    """
    try:
        # Try with_structured_output (newer LangChain API)
        return llm.with_structured_output(schema)
    except AttributeError:
        # Fallback to older API
        from langchain.output_parsers import PydanticOutputParser
        from langchain.prompts import PromptTemplate
        
        parser = PydanticOutputParser(pydantic_object=schema)
        return llm, parser
