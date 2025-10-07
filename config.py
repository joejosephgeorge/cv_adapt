"""
Configuration management for CV Analysis
Supports multiple LLM providers: Ollama, OpenAI, Anthropic, etc.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import os


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class LLMConfig(BaseModel):
    """Configuration for LLM provider"""
    provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")
    
    # Groq settings
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama-3.1-8b-instant")
    
    # General LLM settings
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None)
    
    # Hybrid model strategy (per architecture recommendation)
    parser_provider: Optional[LLMProvider] = None  # High reliability for parsing
    parser_model: Optional[str] = None
    
    scoring_provider: Optional[LLMProvider] = None
    scoring_model: Optional[str] = None
    
    rewriter_provider: Optional[LLMProvider] = None  # Used for analysis agent
    rewriter_model: Optional[str] = None
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )
    
    def get_provider_config(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specific agent type or default provider"""
        # If agent-specific configuration exists, use it
        if agent_type:
            provider_attr = f"{agent_type}_provider"
            model_attr = f"{agent_type}_model"
            
            if hasattr(self, provider_attr) and getattr(self, provider_attr):
                provider = getattr(self, provider_attr)
                model = getattr(self, model_attr)
                return self._get_config_for_provider(provider, model)
        
        # Otherwise use default provider
        return self._get_config_for_provider(self.provider)
    
    def _get_config_for_provider(
        self, 
        provider: LLMProvider, 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration dictionary for a specific provider"""
        if provider == LLMProvider.OLLAMA:
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": model or self.ollama_model,
                "temperature": self.temperature,
            }
        
        elif provider == LLMProvider.OPENAI:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": model or self.openai_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        
        elif provider == LLMProvider.ANTHROPIC:
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": model or self.anthropic_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        
        elif provider == LLMProvider.GROQ:
            return {
                "provider": "groq",
                "api_key": self.groq_api_key,
                "model": model or self.groq_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation)"""
    vector_db_path: str = Field(default="./chroma_db")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)
    use_rag_fusion: bool = Field(default=False)  # Multi-query RAG


class WorkflowConfig(BaseModel):
    """Configuration for workflow behavior"""
    min_relevance_score: float = Field(default=50.0)  # Minimum score to proceed with analysis


class Config(BaseModel):
    """Main application configuration"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load complete configuration from environment"""
        return cls(
            llm=LLMConfig.from_env()
        )
