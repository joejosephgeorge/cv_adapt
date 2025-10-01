# CV Adaptor AI

**Intelligent CV Adaptation using Multi-Agent LLM System**

A sophisticated CV tailoring system that uses LangGraph-orchestrated multi-agent architecture with RAG (Retrieval-Augmented Generation) to automatically adapt your CV to match specific job descriptions.

## 🏗️ Architecture

This system implements a state-of-the-art multi-agent architecture as described in academic literature on LLM-based document processing:

### Four Specialized Agents

1. **Parser Agent** - Structured data extraction from unstructured CV and JD text
   - Uses Pydantic schema enforcement with retry mechanism
   - Converts documents into machine-readable JSON formats
   - High-reliability LLM (GPT-4 or equivalent) recommended

2. **Scoring Agent** - RAG-enhanced relevance scoring and gap analysis
   - Compares CV against job requirements using vector similarity
   - Generates explainable relevance scores with citations
   - Identifies skill gaps and target keywords

3. **Rewriter Agent** - Content generation with RAG grounding
   - Rewrites CV sections using Action-Metric-Result framework
   - Ensures factual accuracy through RAG context retrieval
   - Cost-optimized LLM (Llama 3, etc.) suitable

4. **QA Agent** - Quality assurance with self-correction loop
   - Validates keyword integration and factual consistency
   - Triggers cyclical refinement if issues found
   - High-reasoning LLM recommended for critical validation

### LangGraph Orchestration

The workflow uses LangGraph's cyclical state machine to enable:
- **Conditional branching** based on relevance scores
- **Iterative refinement** through QA feedback loops
- **Stateful execution** with progress tracking
- **Error handling** at each stage

### RAG System

ChromaDB vector database provides:
- Dual indexing (CV facts + JD requirements)
- Multi-query retrieval (RAG Fusion)
- Factual grounding to prevent hallucination
- Citation and verification support

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Prerequisites:** Docker and Docker Compose installed

```bash
# Quick start with interactive script
./docker-run.sh

# Or manually
docker-compose up -d

# Access application at http://localhost:8501
```

See [DOCKER.md](DOCKER.md) for complete Docker documentation.

### Option 2: Local Python Installation

**Prerequisites:** Python 3.9+, Ollama (for local LLM) OR API keys

```bash
# Clone repository
git clone <repository-url>
cd cv_adapt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running with Ollama (Local)

1. **Install and start Ollama:**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:3b  # or any other model
ollama serve
```

2. **Run Streamlit app:**
```bash
streamlit run app.py
```

3. **Configure in UI:**
   - Select "ollama" as provider
   - Enter Ollama URL: `http://localhost:11434`
   - Enter model name: `llama3.2:3b`

### Running with API Providers

Set environment variables or configure in UI:

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"
export LLM_PROVIDER="openai"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"
export LLM_PROVIDER="anthropic"

# Groq
export GROQ_API_KEY="your-key-here"
export LLM_PROVIDER="groq"
```

## 📖 Usage

### Streamlit Web Interface

1. Upload your CV (PDF, DOCX, or TXT)
2. Provide job description (URL or paste text)
3. Click "Adapt CV to Job"
4. Download adapted CV

### Programmatic Usage

```python
from config import Config
from workflow import CVAdaptationWorkflow

# Initialize
config = Config.from_env()
workflow = CVAdaptationWorkflow(config)

# Load your CV and JD
with open("cv.txt") as f:
    cv_text = f.read()

with open("job_description.txt") as f:
    jd_text = f.read()

# Run adaptation
result = workflow.run(cv_text, jd_text)

# Access results
if result["success"]:
    adapted_cv = result["adapted_cv"]
    print(f"Relevance Score: {adapted_cv.relevance_score}%")
    print(f"QA Passed: {adapted_cv.qa_passed}")
```

## ⚙️ Configuration

### LLM Provider Configuration

Edit `config.py` or use environment variables:

```python
from config import Config, LLMProvider

config = Config()

# Use Ollama
config.llm.provider = LLMProvider.OLLAMA
config.llm.ollama_model = "llama3.2:3b"

# Use OpenAI
config.llm.provider = LLMProvider.OPENAI
config.llm.openai_api_key = "sk-..."
config.llm.openai_model = "gpt-4o-mini"
```

### Hybrid Model Strategy

For optimal performance and cost efficiency:

```python
# High-reliability for parsing
config.llm.parser_provider = LLMProvider.OPENAI
config.llm.parser_model = "gpt-4o"

# Cost-optimized for rewriting
config.llm.rewriter_provider = LLMProvider.OLLAMA
config.llm.rewriter_model = "llama3.1:8b"

# High-reasoning for QA
config.llm.qa_provider = LLMProvider.ANTHROPIC
config.llm.qa_model = "claude-3-5-sonnet-20241022"
```

### Workflow Configuration

```python
# Enable/disable QA self-correction loop
config.workflow.enable_qa_loop = True
config.workflow.max_qa_iterations = 2

# Set minimum relevance score
config.workflow.min_relevance_score = 50.0
```

## 📁 Project Structure

```
cv_adapt/
├── app.py                 # Streamlit web interface
├── config.py              # Configuration management
├── models.py              # Pydantic data models
├── agents.py              # Four specialized agents
├── workflow.py            # LangGraph orchestration
├── rag_system.py          # Vector database RAG
├── llm_factory.py         # Multi-provider LLM factory
├── utils.py               # Document parsing utilities
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🎯 Features

### Document Processing
- ✅ PDF, DOCX, TXT support
- ✅ Web scraping for job postings
- ✅ Layout-aware text extraction

### Intelligent Adaptation
- ✅ Semantic skill matching with RAG
- ✅ Keyword optimization for ATS
- ✅ Quantifiable achievement rewriting
- ✅ Transferable skills highlighting

### Quality Assurance
- ✅ Factual accuracy verification
- ✅ Keyword coverage validation
- ✅ Self-correction loop
- ✅ Explainable scoring

### Multiple LLM Providers
- ✅ Ollama (local, free)
- ✅ OpenAI (GPT-4, GPT-4o-mini)
- ✅ Anthropic (Claude)
- ✅ Groq (fast inference)

## 🔧 Advanced Usage

### Custom Prompts

Modify agent prompts in `agents.py` to customize behavior.

### RAG Configuration

Adjust vector database settings in `config.py`:
```python
config.rag.chunk_size = 500
config.rag.chunk_overlap = 50
config.rag.top_k = 5
config.rag.use_rag_fusion = True  # Multi-query RAG
```

### Workflow Customization

Modify the LangGraph workflow in `workflow.py` to add custom nodes or change conditional logic.

## 📊 Performance

### Recommended Models

Based on architecture analysis:

| Agent | Task | Recommended Model | Rationale |
|-------|------|-------------------|-----------|
| Parser | Structured extraction | GPT-4, Claude Sonnet | High reliability for JSON |
| Scoring | Relevance analysis | Llama 3.1 8B, GPT-4o-mini | Balanced performance |
| Rewriter | Content generation | Llama 3.2 3B, GPT-4o-mini | High throughput, cost-effective |
| QA | Validation | GPT-4, Claude Sonnet | Critical reasoning |

### Cost Optimization

- Use local Ollama for all agents (free but slower)
- Use hybrid strategy (expensive models only for Parser/QA)
- Adjust `max_qa_iterations` to control refinement loops

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Memory Issues
```bash
# Reduce RAG chunk size
config.rag.chunk_size = 250
config.rag.top_k = 3
```

### API Rate Limits
- Add delays between API calls
- Use local Ollama as fallback
- Implement retry logic with exponential backoff

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📚 References

This implementation follows best practices from:
- LangChain documentation
- LangGraph multi-agent patterns
- RAG architecture papers
- CV optimization research

## 🆘 Support

For issues or questions:
- Open a GitHub issue
- Check documentation in `/docs`
- Review architecture in code comments