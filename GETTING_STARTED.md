# Getting Started with CV Adaptor AI

Choose your preferred method to run CV Adaptor AI.

---

## 🐳 Method 1: Docker (Recommended - Easiest)

**Best for:** Anyone who wants to get started quickly without installing Python dependencies.

### Quick Start (30 seconds)

```bash
./docker-run.sh
```

The interactive script will:
1. Check if Docker is installed
2. Ask which LLM provider you want to use
3. Build and start the application
4. Open it at http://localhost:8501

### Manual Docker Start

```bash
# With Ollama (local, free)
ollama serve                    # In one terminal
docker-compose up -d            # In another terminal

# Access at http://localhost:8501
```

**See:** [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for 2-minute guide
**Full docs:** [DOCKER.md](DOCKER.md) for complete Docker documentation

---

## 🐍 Method 2: Local Python (For Developers)

**Best for:** Developers who want to modify the code or understand the internals.

### Prerequisites

- Python 3.9 or higher
- pip and venv

### Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

### With Ollama (Local LLM)

```bash
# Terminal 1: Start Ollama
ollama pull llama3.2:3b
ollama serve

# Terminal 2: Run app
streamlit run app.py

# Configure in the UI to use Ollama at http://localhost:11434
```

### With Cloud APIs

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
# OR
export GROQ_API_KEY="your-key-here"

# Run app
streamlit run app.py
```

**See:** [QUICKSTART.md](QUICKSTART.md) for detailed setup

---

## 🧪 Method 3: Test with Example Script

**Best for:** Testing the system without the UI.

```bash
# Activate venv if not already
source venv/bin/activate

# Edit example.py to configure your LLM
# Then run:
python example.py
```

This processes a sample CV and job description, showing you how the system works.

---

## ⚙️ Configuration

### Docker Configuration

Edit `docker-compose.yml`:

```yaml
environment:
  - LLM_PROVIDER=ollama              # or openai, anthropic, groq
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=llama3.2:3b
```

Or use environment variables:
```bash
export OPENAI_API_KEY="your-key"
docker-compose up -d
```

### Local Python Configuration

Set environment variables:
```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
```

Or configure in the Streamlit UI sidebar.

---

## 📖 Documentation Guide

- **README.md** - Project overview and architecture
- **GETTING_STARTED.md** (this file) - How to run the application
- **QUICKSTART.md** - 5-minute local Python setup
- **DOCKER_QUICKSTART.md** - 2-minute Docker setup
- **DOCKER.md** - Complete Docker documentation
- **ARCHITECTURE.md** - Technical architecture details

---

## 🎯 Usage Flow

1. **Start the application** (Docker or local)
2. **Open browser:** http://localhost:8501
3. **Configure LLM** in the sidebar
4. **Upload your CV** (PDF, DOCX, or TXT)
5. **Provide job description** (URL or paste text)
6. **Click "Adapt CV to Job"**
7. **Download** your tailored CV

---

## 🔧 Troubleshooting

### Docker Issues

**Can't connect to Ollama?**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running:
ollama serve
```

**Port 8501 already in use?**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"
```

### Local Python Issues

**Module not found?**
```bash
# Make sure venv is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Ollama connection refused?**
```bash
# Start Ollama
ollama serve

# Check it's running
ollama list
```

---

## 🚀 Next Steps

1. ✅ Get the application running (you're here!)
2. 📚 Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
3. 🎨 Customize prompts in `agents.py` for your needs
4. ⚡ Try different LLM models for performance/quality trade-offs
5. 🔄 Enable QA self-correction loop for higher quality output

---

## 💡 Tips

- **Start with Ollama** (free, local) to test the system
- **Use GPT-4 for Parser/QA agents** for best accuracy (hybrid approach)
- **Enable QA loop** for critical applications
- **Try different models** to find your cost/quality sweet spot

---

## 🆘 Need Help?

1. Check the troubleshooting section above
2. Review documentation:
   - Docker issues → [DOCKER.md](DOCKER.md)
   - Setup issues → [QUICKSTART.md](QUICKSTART.md)
   - Architecture questions → [ARCHITECTURE.md](ARCHITECTURE.md)
3. Check logs:
   ```bash
   # Docker
   docker-compose logs -f
   
   # Local
   # Check terminal output
   ```

---

## 📊 System Requirements

### Docker
- Docker Desktop (Mac/Windows) or Docker Engine (Linux)
- 4GB RAM minimum, 8GB recommended
- 2GB free disk space

### Local Python
- Python 3.9+
- 2GB RAM minimum
- Virtual environment recommended

### For Ollama
- 8GB RAM for 3B models
- 16GB RAM for 7-8B models
- 32GB+ RAM for larger models

---

## ✨ Features Included

✅ Multi-agent architecture (Parser, Scoring, Rewriter, QA)
✅ RAG system with ChromaDB for factual grounding
✅ Support for 4+ LLM providers
✅ Cyclical self-correction loop
✅ Web scraping for job descriptions
✅ PDF/DOCX document parsing
✅ Streamlit web interface
✅ Docker containerization
✅ Complete documentation

---

Happy CV adapting! 🎉
