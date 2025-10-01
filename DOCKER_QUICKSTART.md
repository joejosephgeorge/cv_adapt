# Docker Quick Start

Get CV Adaptor AI running in Docker in 2 minutes!

## Method 1: Interactive Script (Easiest)

```bash
./docker-run.sh
```

Follow the prompts to choose your LLM provider and the script will handle everything.

## Method 2: Docker Compose (Manual)

### With Ollama (Local, Free)

```bash
# 1. Start Ollama on your host
ollama serve
ollama pull llama3.2:3b

# 2. Run CV Adaptor
docker-compose up -d

# 3. Access at http://localhost:8501
```

### With OpenAI

```bash
# 1. Edit docker-compose.yml - uncomment OpenAI section and add your API key
# OR set environment variable:
export OPENAI_API_KEY="your-key-here"

# 2. Run
docker-compose up -d
```

### With Anthropic Claude

```bash
# 1. Edit docker-compose.yml - uncomment Anthropic section
export ANTHROPIC_API_KEY="your-key-here"

# 2. Run
docker-compose up -d
```

## Common Commands

```bash
# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

## Access the Application

Open your browser: **http://localhost:8501**

## Troubleshooting

**Can't connect to Ollama?**
- Make sure Ollama is running: `ollama serve`
- Check the URL in docker-compose.yml uses `host.docker.internal:11434`

**Port 8501 already in use?**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

**See full documentation:** [DOCKER.md](DOCKER.md)
