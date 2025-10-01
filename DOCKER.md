# Docker Deployment Guide

Complete guide for running CV Adaptor AI in Docker containers.

## Quick Start

### Option 1: Using Local Ollama (Recommended)

**Prerequisites:**
- Docker and Docker Compose installed
- Ollama running on your host machine

```bash
# 1. Start Ollama on your host (outside Docker)
ollama pull llama3.2:3b
ollama serve

# 2. Build and run CV Adaptor
docker-compose up -d

# 3. Access the application
open http://localhost:8501
```

### Option 2: Using Cloud API (OpenAI/Anthropic/Groq)

```bash
# 1. Copy and configure environment file
cp .env.example .env
# Edit .env and add your API key

# 2. Update docker-compose.yml
# Uncomment the provider section you want to use

# 3. Run
docker-compose up -d
```

### Option 3: Ollama Inside Docker

```bash
# 1. Edit docker-compose.yml
# Uncomment the 'ollama' service section

# 2. Start both services
docker-compose up -d

# 3. Pull model into Ollama container
docker exec -it ollama-service ollama pull llama3.2:3b

# 4. Access application
open http://localhost:8501
```

---

## Detailed Setup

### Building the Image

```bash
# Build the Docker image
docker build -t cv-adaptor-ai .

# Or use docker-compose
docker-compose build
```

### Running the Container

**Basic run:**
```bash
docker run -p 8501:8501 cv-adaptor-ai
```

**With environment variables:**
```bash
docker run -p 8501:8501 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=llama3.2:3b \
  cv-adaptor-ai
```

**With persistent storage:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/uploads:/app/uploads \
  cv-adaptor-ai
```

### Using Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f cv-adaptor

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Configuration

### Environment Variables

Set these in `docker-compose.yml` or `.env` file:

**LLM Provider:**
```yaml
environment:
  - LLM_PROVIDER=ollama  # or openai, anthropic, groq
```

**Ollama:**
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=llama3.2:3b
```

**OpenAI:**
```yaml
environment:
  - LLM_PROVIDER=openai
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - OPENAI_MODEL=gpt-4o-mini
```

**Anthropic:**
```yaml
environment:
  - LLM_PROVIDER=anthropic
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  - ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Groq:**
```yaml
environment:
  - LLM_PROVIDER=groq
  - GROQ_API_KEY=${GROQ_API_KEY}
  - GROQ_MODEL=llama-3.1-8b-instant
```

### Volumes

**Persistent data:**
```yaml
volumes:
  - ./chroma_db:/app/chroma_db      # Vector database
  - ./uploads:/app/uploads          # Uploaded files
```

### Ports

- **8501**: Streamlit web interface
- **11434**: Ollama API (if using Ollama in Docker)

---

## Accessing Ollama from Docker

### Method 1: Ollama on Host (Recommended)

The container uses `host.docker.internal` to access services on your host machine:

```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**Works on:**
- ✅ Docker Desktop (Mac/Windows)
- ✅ Linux (with `--add-host` flag)

**Linux setup:**
```bash
docker run --add-host=host.docker.internal:host-gateway ...
```

### Method 2: Ollama in Docker

Run Ollama as a separate container:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

Then update CV Adaptor to use `http://ollama:11434`.

### Method 3: Network Mode Host (Linux Only)

```bash
docker run --network host cv-adaptor-ai
```

---

## GPU Support for Ollama

If running Ollama in Docker with NVIDIA GPU:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Prerequisites:**
- NVIDIA GPU
- nvidia-docker2 installed
- NVIDIA Container Toolkit

---

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml cv-adaptor
```

### Using Kubernetes

Create deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cv-adaptor
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: cv-adaptor
        image: cv-adaptor-ai:latest
        ports:
        - containerPort: 8501
        env:
        - name: LLM_PROVIDER
          value: "openai"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: cv-adaptor-secrets
              key: openai-api-key
```

### Behind Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name cv-adaptor.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Troubleshooting

### Cannot connect to Ollama

**Error:** `Connection refused to http://localhost:11434`

**Solutions:**
1. Make sure Ollama is running on host: `ollama serve`
2. Use `host.docker.internal` instead of `localhost`
3. Check firewall settings

**Test connection:**
```bash
docker exec cv-adaptor-ai curl http://host.docker.internal:11434/api/tags
```

### Container exits immediately

**Check logs:**
```bash
docker-compose logs cv-adaptor
```

**Common issues:**
- Missing environment variables
- Port 8501 already in use
- Invalid LLM configuration

### Permission denied errors

**Fix volume permissions:**
```bash
sudo chown -R 1000:1000 chroma_db uploads
```

### Out of memory

**Increase Docker memory:**
- Docker Desktop: Settings → Resources → Memory
- Linux: Edit `/etc/docker/daemon.json`

```json
{
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Soft": -1
    }
  }
}
```

---

## Maintenance

### View logs

```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs cv-adaptor
```

### Update application

```bash
# Rebuild and restart
docker-compose up -d --build

# Pull latest changes
git pull
docker-compose build --no-cache
docker-compose up -d
```

### Backup data

```bash
# Backup vector database
tar -czf chroma_db_backup.tar.gz chroma_db/

# Backup uploads
tar -czf uploads_backup.tar.gz uploads/
```

### Clean up

```bash
# Remove stopped containers
docker-compose down

# Remove with volumes
docker-compose down -v

# Remove images
docker rmi cv-adaptor-ai

# Clean everything
docker system prune -a
```

---

## Security Best Practices

1. **Use environment files for secrets:**
   ```bash
   # Don't commit .env files
   echo ".env" >> .gitignore
   ```

2. **Run as non-root user:**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

3. **Use Docker secrets (Swarm/Kubernetes):**
   ```bash
   echo "your-api-key" | docker secret create openai_key -
   ```

4. **Limit resources:**
   ```yaml
   services:
     cv-adaptor:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
   ```

5. **Use HTTPS in production:**
   - Set up SSL/TLS with Let's Encrypt
   - Use Nginx or Traefik as reverse proxy

---

## Multi-Architecture Builds

Build for multiple platforms:

```bash
# Enable buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t cv-adaptor-ai:latest .

# Build and push to registry
docker buildx build --platform linux/amd64,linux/arm64 \
  -t your-registry/cv-adaptor-ai:latest \
  --push .
```

---

## Resources

- **Docker Documentation**: https://docs.docker.com
- **Ollama Docker**: https://hub.docker.com/r/ollama/ollama
- **Streamlit in Docker**: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Review troubleshooting section
3. Open an issue on GitHub
4. Check main README.md for application-specific help
