#!/bin/bash
# Quick start script for CV Adaptor AI with Docker

set -e

echo "========================================="
echo "  CV Adaptor AI - Docker Quick Start"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file - please edit it with your API keys if needed"
    echo ""
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p chroma_db uploads
echo "✅ Directories created"
echo ""

# Detect LLM provider choice
echo "Which LLM provider do you want to use?"
echo "  1) Ollama (local, free) - requires Ollama running on host"
echo "  2) OpenAI (cloud, paid)"
echo "  3) Anthropic Claude (cloud, paid)"
echo "  4) Groq (cloud, fast)"
echo ""
read -p "Enter choice [1-4] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "🔍 Checking if Ollama is running..."
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is running on localhost:11434"
        else
            echo "⚠️  Warning: Ollama does not appear to be running"
            echo "Please start Ollama with: ollama serve"
            echo "And pull a model with: ollama pull llama3.2:3b"
            echo ""
            read -p "Continue anyway? [y/N]: " continue
            if [[ ! $continue =~ ^[Yy]$ ]]; then
                echo "Exiting. Please start Ollama and try again."
                exit 1
            fi
        fi
        PROVIDER="ollama"
        ;;
    2)
        PROVIDER="openai"
        echo ""
        read -p "Enter your OpenAI API key: " api_key
        sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
        sed -i.bak "s/LLM_PROVIDER=.*/LLM_PROVIDER=openai/" .env
        rm .env.bak 2>/dev/null || true
        ;;
    3)
        PROVIDER="anthropic"
        echo ""
        read -p "Enter your Anthropic API key: " api_key
        sed -i.bak "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$api_key/" .env
        sed -i.bak "s/LLM_PROVIDER=.*/LLM_PROVIDER=anthropic/" .env
        rm .env.bak 2>/dev/null || true
        ;;
    4)
        PROVIDER="groq"
        echo ""
        read -p "Enter your Groq API key: " api_key
        sed -i.bak "s/GROQ_API_KEY=.*/GROQ_API_KEY=$api_key/" .env
        sed -i.bak "s/LLM_PROVIDER=.*/LLM_PROVIDER=groq/" .env
        rm .env.bak 2>/dev/null || true
        ;;
    *)
        echo "Invalid choice. Defaulting to Ollama."
        PROVIDER="ollama"
        ;;
esac

echo ""
echo "🏗️  Building Docker image..."
docker-compose build

echo ""
echo "🚀 Starting CV Adaptor AI..."
docker-compose up -d

echo ""
echo "⏳ Waiting for application to start..."
sleep 5

# Check if container is running
if docker ps | grep -q cv-adaptor-ai; then
    echo "✅ Application is running!"
    echo ""
    echo "========================================="
    echo "  Access the application at:"
    echo "  🌐 http://localhost:8501"
    echo "========================================="
    echo ""
    echo "Useful commands:"
    echo "  • View logs:      docker-compose logs -f"
    echo "  • Stop app:       docker-compose down"
    echo "  • Restart app:    docker-compose restart"
    echo ""
else
    echo "❌ Error: Container failed to start"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
