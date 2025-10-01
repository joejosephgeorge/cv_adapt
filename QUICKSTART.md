# Quick Start Guide

Get started with CV Adaptor AI in 5 minutes!

## Option 1: Using Ollama (Local, Free)

### Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

### Step 2: Pull a Model

```bash
# Small, fast model (recommended for testing)
ollama pull llama3.2:3b

# Larger, more capable model
ollama pull llama3.1:8b

# Alternative: coding-focused model
ollama pull qwen2.5-coder:7b
```

### Step 3: Start Ollama

```bash
ollama serve
```

### Step 4: Install Dependencies

```bash
cd cv_adapt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 5: Run the App

```bash
streamlit run app.py
```

### Step 6: Configure in UI

1. Select **ollama** as provider
2. Enter URL: `http://localhost:11434`
3. Enter model: `llama3.2:3b` (or your chosen model)
4. Upload your CV
5. Provide job description
6. Click "Adapt CV to Job"

---

## Option 2: Using OpenAI (Cloud, Paid)

### Step 1: Get API Key

1. Go to https://platform.openai.com
2. Create an account or log in
3. Navigate to API Keys
4. Create a new API key

### Step 2: Install Dependencies

```bash
cd cv_adapt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Set Environment Variable

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Step 4: Run the App

```bash
streamlit run app.py
```

### Step 5: Configure in UI

1. Select **openai** as provider
2. Enter your API key
3. Choose model: `gpt-4o-mini` (cost-effective) or `gpt-4o` (most capable)

---

## Option 3: Using Anthropic Claude (Cloud, Paid)

### Step 1: Get API Key

1. Go to https://console.anthropic.com
2. Create account and get API key

### Step 2: Configure

```bash
export ANTHROPIC_API_KEY="your-key-here"
streamlit run app.py
```

Select **anthropic** as provider in the UI.

---

## Testing with Example Script

Run the included example:

```bash
# Make sure Ollama is running first
ollama serve

# In another terminal
python example.py
```

This will process a sample CV and job description, showing you how the system works.

---

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check the URL is correct: `http://localhost:11434`

### "Model not found" error
- Pull the model first: `ollama pull llama3.2:3b`
- Check available models: `ollama list`

### Out of memory errors
- Use a smaller model: `llama3.2:3b` instead of `llama3.1:70b`
- Reduce RAG settings in Advanced Settings

### Slow performance
- Use a faster model (smaller parameter count)
- Consider using cloud APIs (OpenAI, Groq)
- Disable QA self-correction loop for faster processing

---

## Next Steps

1. **Read the full README.md** for detailed architecture information
2. **Customize prompts** in `agents.py` for your specific needs
3. **Configure hybrid model strategy** for optimal cost/performance
4. **Integrate into your workflow** using the Python API

---

## Need Help?

- Check the README.md for full documentation
- Review example.py for usage patterns
- Open an issue on GitHub for bugs or questions