# CV Adaptor AI - System Architecture

## Overview

This system implements a sophisticated multi-agent architecture for CV adaptation using LangGraph orchestration and RAG (Retrieval-Augmented Generation) grounding.

## Architecture Components

### 1. Four Specialized Agents (`agents.py`)

#### Parser Agent
- **Purpose**: Structured data extraction from unstructured documents
- **Input**: Raw CV/JD text
- **Output**: Pydantic-validated JSON structures
- **Key Features**:
  - Retry mechanism for failed Pydantic validation
  - Fallback parsing for error handling
  - Extracts contact info, experience, education, skills, etc.

#### Scoring Agent
- **Purpose**: RAG-enhanced relevance scoring and gap analysis
- **Input**: Structured CV + JD data
- **Output**: MatchGapReportSchema with score, gaps, keywords
- **Key Features**:
  - Few-shot learning for consistent scoring
  - RAG Fusion (multi-query retrieval)
  - Explainable AI with reasoning and citations
  - Conditional branching based on scores

#### Rewriter Agent
- **Purpose**: Content generation with factual grounding
- **Input**: CV, JD, match report, optional QA feedback
- **Output**: RewrittenSectionSchema with tailored content
- **Key Features**:
  - Action-Metric-Result framework for achievements
  - RAG grounding to prevent hallucination
  - Style matching from original CV
  - Iterative refinement support

#### QA Agent
- **Purpose**: Quality assurance with self-correction
- **Input**: Rewritten sections, original facts, required keywords
- **Output**: QAReportSchema with pass/fail and issues
- **Key Features**:
  - Factual consistency verification via RAG
  - Keyword coverage validation
  - Severity-based issue reporting
  - Feedback generation for refinement loop

### 2. RAG System (`rag_system.py`)

**Vector Database**: ChromaDB (persistent storage)

**Dual Index Strategy**:
- **CV Facts Index**: Experience, achievements, metrics, skills
- **JD Requirements Index**: Requirements, skills, responsibilities

**Key Functions**:
- `index_cv_facts()`: Index candidate profile into vectors
- `index_job_requirements()`: Index job description into vectors
- `retrieve_cv_facts()`: Semantic search for CV content
- `retrieve_job_requirements()`: Semantic search for JD content
- `retrieve_multi_query()`: RAG Fusion for better retrieval

### 3. LangGraph Workflow (`workflow.py`)

**State Machine**: Cyclical graph with conditional edges

**Node Flow**:
```
parse_documents → score_match → [decision] → rewrite_cv → qa_validate → [loop/finish] → finalize
                                    ↓
                              (low score) → finalize
```

**Conditional Branching**:
- **After Scoring**:
  - Score ≥ 95: Skip rewriting (high_score → finalize)
  - Score 70-94: Proceed with rewriting (proceed)
  - Score 50-69: Optimize approach (optimize)
  - Score < 50: Don't waste resources (fail → finalize)

- **After QA**:
  - QA passed: Finalize output
  - QA failed + iterations < max: Loop back to rewriter
  - Max iterations reached: Finalize anyway

**Cyclical Refinement Loop**:
```
rewrite_cv → qa_validate → [if failed] → rewrite_cv (with feedback)
```

### 4. Multi-Provider LLM Support (`llm_factory.py`, `config.py`)

**Supported Providers**:
- **Ollama**: Local, free, privacy-preserving
- **OpenAI**: GPT-4, GPT-4o-mini (reliable, expensive)
- **Anthropic**: Claude Sonnet (high reasoning)
- **Groq**: Fast inference, cost-effective

**Hybrid Model Strategy**:
```python
# Example configuration for optimal cost/performance
Parser Agent: GPT-4 (high reliability for JSON)
Scoring Agent: Llama 3.1 8B (balanced)
Rewriter Agent: Llama 3.2 3B (high throughput, cost-optimized)
QA Agent: Claude Sonnet (critical reasoning)
```

### 5. Pydantic Data Models (`models.py`)

**Data Contracts** (ensures type safety and validation):

**Phase I (Parsing)**:
- `CandidateProfileSchema`: Structured CV data
- `JobRequirementSchema`: Structured JD data

**Phase II (Scoring)**:
- `MatchGapReportSchema`: Score, gaps, keywords, recommendation

**Phase III (Rewriting)**:
- `RewrittenSectionSchema`: Tailored CV sections

**Phase IV (QA)**:
- `QAReportSchema`: Validation results, issues, feedback

**Final Output**:
- `AdaptedCVSchema`: Complete adapted CV with metadata

**Workflow State**:
- `WorkflowState`: Maintains state across LangGraph nodes

### 6. Streamlit Application (`app.py`)

**UI Structure**:
1. **Configuration Sidebar**: LLM provider selection, advanced settings
2. **Input Section**: CV upload + JD input (URL or paste)
3. **Process Section**: Single button with progress tracking
4. **Output Section**: Metrics, adapted CV, detailed reports

**Progress Tracking**:
- Phase 1/4: Parsing documents
- Phase 2/4: Analyzing match and scoring
- Phase 3/4: Rewriting CV sections
- Phase 4/4: Quality assurance validation

## Data Flow

```
┌─────────────┐
│ CV (PDF/    │
│ DOCX/TXT)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐         ┌──────────────┐
│  Parser Agent       │────────▶│ RAG System   │
│  (Structured        │         │ (Index CV    │
│   Extraction)       │         │  Facts)      │
└──────┬──────────────┘         └──────────────┘
       │
       │  CandidateProfileSchema
       ▼
┌─────────────────────┐
│  Scoring Agent      │◀────── RAG Retrieval
│  (RAG-Enhanced      │
│   Matching)         │
└──────┬──────────────┘
       │
       │  MatchGapReportSchema
       │  (Score, Gaps, Keywords)
       ▼
    [Decision]
       │
       ├─ Score < 50: Stop
       ├─ Score ≥ 95: Skip Rewrite
       └─ Score 50-94: Continue
           │
           ▼
    ┌─────────────────────┐
    │  Rewriter Agent     │◀────── RAG Grounding
    │  (Content Gen)      │
    └──────┬──────────────┘
           │
           │  RewrittenSectionSchema
           ▼
    ┌─────────────────────┐
    │  QA Agent           │◀────── RAG Verification
    │  (Validation)       │
    └──────┬──────────────┘
           │
           │  QAReportSchema
           ▼
        [Check]
           │
           ├─ Passed: Finalize
           └─ Failed: Loop back to Rewriter
                      (max 2 iterations)
```

## Key Design Principles

### 1. Structured Data Contracts
All agents communicate via Pydantic schemas, ensuring type safety and validation.

### 2. RAG Grounding
All generation is grounded in retrieved facts from vector database, preventing hallucination.

### 3. Cyclical Refinement
QA agent can trigger rewriting if quality issues detected, ensuring high output quality.

### 4. Hybrid LLM Strategy
Different agents can use different models optimized for their specific task.

### 5. Explainable AI
Scoring and QA agents provide reasoning and citations for transparency.

### 6. Error Handling
Every stage has fallback mechanisms and error handlers to ensure robustness.

## Configuration

### Environment Variables

See `.env.example` for all configuration options.

### Runtime Configuration

All settings can be configured via:
1. Environment variables
2. Streamlit UI sidebar
3. Programmatic config in code

## Performance Considerations

### Recommended Models by Stage

| Stage | Task Complexity | Recommended Model | Rationale |
|-------|----------------|-------------------|-----------|
| Parsing | High (JSON accuracy critical) | GPT-4, Claude | Best structured output |
| Scoring | Medium | Llama 3.1 8B, GPT-4o-mini | Balanced performance |
| Rewriting | Medium (high volume) | Llama 3.2 3B | Cost-effective, good quality |
| QA | High (critical validation) | GPT-4, Claude | Best reasoning ability |

### Cost Optimization

1. **Use Ollama for all agents** (free, local)
2. **Hybrid approach** (expensive models only for Parser/QA)
3. **Adjust max_qa_iterations** to control refinement loops
4. **Set min_relevance_score** to skip low-quality matches early

### Latency Optimization

1. Use faster models (3B vs 70B parameters)
2. Disable QA loop for faster processing
3. Use Groq for fastest cloud inference
4. Reduce RAG top_k for fewer retrievals

## Extension Points

### Adding New Agents

1. Create agent class in `agents.py`
2. Add node to workflow in `workflow.py`
3. Update `WorkflowState` in `models.py`

### Adding New LLM Providers

1. Add provider enum to `config.py`
2. Implement in `llm_factory.py`
3. Add UI controls in `app.py`

### Custom Prompts

All prompts are in the agent classes and can be customized for domain-specific requirements.

### Custom Scoring Logic

Modify the `ScoringAgent.score_match()` method to implement custom scoring algorithms.

## Testing

See `example.py` for a complete working example with sample CV and JD.

```bash
python example.py
```

## Monitoring

For production deployment, consider integrating:
- **LangSmith**: LangChain tracing and debugging
- **Weights & Biases**: ML experiment tracking
- **Custom logging**: Add logging throughout workflow nodes

## References

- LangChain Documentation: https://python.langchain.com
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- ChromaDB Documentation: https://docs.trychroma.com/
- Pydantic Documentation: https://docs.pydantic.dev/
