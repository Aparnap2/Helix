# Helix AI Gateway - Docker Deployment

Production-ready Docker deployment configuration for the Helix AI Gateway with comprehensive monitoring, security, and scalability features.

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM, 2 CPU cores
- 20GB disk space

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-org/helix.git
cd helix

# Copy environment template
cp .env.helix .env

# Edit .env with your configuration
nano .env
```

### 2. Start the Services

```bash
# Start all services (recommended for production)
docker compose -f docker/docker-compose.helix.yml up -d

# Or start with specific profiles
docker compose -f docker/docker-compose.helix.yml --profile monitoring up -d
docker compose -f docker/docker-compose.helix.yml --profile dev up -d
```

### 3. Verify Deployment

```bash
# Check service health
docker compose -f docker/docker-compose.helix.yml ps

# View logs
docker compose -f docker/docker-compose.helix.yml logs -f

# Test API
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello, world!"}]}'
```

## 📊 Services & Ports

| Service | Port | Description | Access |
|---------|------|-------------|--------|
| **Helix API** | 4000 | Main LLM Gateway API | http://localhost:4000 |
| **Streamlit Dashboard** | 8501 | Real-time monitoring dashboard | http://localhost:8501 |
| **Redis** | 6379 | Vector cache & semantic search | redis://localhost:6379 |
| **Redis Commander** | 8081 | Redis GUI (dev profile) | http://localhost:8081 |
| **PostgreSQL** | 5432 | Primary database | postgresql://localhost:5432 |
| **Prometheus** | 9090 | Metrics collection | http://localhost:9090 |
| **Grafana** | 3000 | Visualization & dashboards | http://localhost:3000 |
| **Nginx** | 80/443 | Reverse proxy & load balancer | http://localhost:80 |
| **Loki** | 3100 | Log aggregation (monitoring profile) | http://localhost:3100 |

## 🔧 Configuration

### Environment Variables

Key environment variables to configure in `.env`:

```bash
# Core Configuration
HELIX_ENVIRONMENT=production
HELIX_LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://helix:your-password@postgres:5432/helix
REDIS_URL=redis://:your-password@redis:6379/0

# LLM Providers
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-key
GROQ_API_KEY=gsk-your-key

# Security
MASTER_KEY=your-secure-master-key
HELIX_JWT_SECRET=your-jwt-secret

# Monitoring
PROMETHEUS_ENABLED=true
LANGFUSE_ENABLED=false  # Set to true for detailed LLM observability
```

### Service Profiles

Use Docker Compose profiles to selectively start services:

```bash
# Default services (API, Cache, DB, Dashboard)
docker compose -f docker/docker-compose.helix.yml up -d

# Include monitoring stack (Prometheus, Grafana, Loki)
docker compose -f docker/docker-compose.helix.yml --profile monitoring up -d

# Include development tools (Redis Commander)
docker compose -f docker/docker-compose.helix.yml --profile dev up -d

# Full production deployment with all features
docker compose -f docker/docker-compose.helix.yml --profile monitoring --profile dev up -d
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Client      │    │   Applications  │    │   Monitoring    │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Browser   │ │    │ │   Mobile    │ │    │ │  Grafana    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   CLI Tool  │ │    │ │   Backend   │ │    │ │ Prometheus  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Nginx       │
                    │ Reverse Proxy   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Helix API     │
                    │   (LiteLLM)     │
                    └─────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │   Redis     │ │ PostgreSQL  │ │  Streamlit  │
        │ Vector Cache│ │ Database    │ │ Dashboard   │
        └─────────────┘ └─────────────┘ └─────────────┘
                │                │
                │                │
        ┌─────────────┐ ┌─────────────┐
        │   LLM       │ │ Monitoring  │
        │  Providers  │ │   Stack     │
        └─────────────┘ └─────────────┘
```

## 🔒 Security Features

- **Authentication**: JWT-based API authentication
- **Rate Limiting**: Configurable per-IP and per-key limits
- **PII Protection**: Automated detection and redaction
- **SSL/TLS**: Encrypted communication (configurable)
- **Security Headers**: HSTS, CSP, XSS protection
- **Network Isolation**: Docker networks with proper segmentation

## 📈 Monitoring & Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Custom Metrics**: Request rates, costs, cache performance
- **Health Checks**: Comprehensive health monitoring

### Visualization
- **Grafana Dashboards**: Pre-built dashboards for all metrics
- **Real-time Monitoring**: Live updates and alerting
- **Alert Rules**: Automated alerts for critical issues

### Logging
- **Structured Logging**: JSON-formatted logs
- **Log Aggregation**: Loki integration (optional)
- **Centralized Logs**: All services logging to centralized location

## 🚀 Performance & Scaling

### Optimization Features
- **Vector Search**: Redis with RediSearch for semantic caching
- **Connection Pooling**: Efficient database connection management
- **Compression**: Gzip compression for API responses
- **Caching**: Multi-layer caching strategy

### Scaling Options
- **Horizontal Scaling**: Multiple API instances behind load balancer
- **Database Sharding**: PostgreSQL partitioning support
- **Redis Clustering**: Multi-node Redis setup
- **Container Orchestration**: Kubernetes-ready configuration

## 🔧 Development

### Local Development Setup

```bash
# Development environment with debugging tools
docker compose -f docker/docker-compose.helix.yml --profile dev --profile monitoring up -d

# View logs in real-time
docker compose -f docker/docker-compose.helix.yml logs -f litellm

# Access development tools
open http://localhost:8081  # Redis Commander
open http://localhost:3000  # Grafana
open http://localhost:8501  # Helix Dashboard
```

### Code Integration

```python
# One-line change to integrate with Helix
import openai

# Before: Direct OpenAI API
# response = openai.ChatCompletion.create(...)

# After: Through Helix
openai.api_base = "http://localhost:4000/v1"
openai.api_key = "any-key"  # Helix handles authentication
response = openai.ChatCompletion.create(...)
```

## 🚨 Troubleshooting

### Common Issues

**Service Won't Start**
```bash
# Check logs
docker compose -f docker/docker-compose.helix.yml logs [service-name]

# Check resource usage
docker stats

# Check port conflicts
netstat -tulpn | grep :4000
```

**High Memory Usage**
```bash
# Check Redis memory
docker exec helix-redis redis-cli info memory

# Clean up unused Docker resources
docker system prune -f
```

**Database Connection Issues**
```bash
# Test database connectivity
docker exec helix-postgres psql -U helix -d helix -c "SELECT 1;"

# Check database logs
docker compose -f docker/docker-compose.helix.yml logs postgres
```

### Health Checks

```bash
# API Health Check
curl http://localhost:4000/health

# Database Health Check
docker exec helix-postgres pg_isready -U helix -d helix

# Redis Health Check
docker exec helix-redis redis-cli ping

# Overall Service Status
docker compose -f docker/docker-compose.helix.yml ps
```

## 📚 Additional Configuration

### Advanced Nginx Configuration

Edit `config/nginx/nginx.conf` for:
- Custom rate limiting rules
- SSL/TLS configuration
- Advanced load balancing
- Custom security headers

### Prometheus Scaling

Edit `config/prometheus/prometheus.yml` for:
- Remote write configuration
- Custom scrape intervals
- Additional monitoring targets
- Alert routing

### Database Optimization

Run database migrations:
```bash
docker exec helix-postgres /app/scripts/migrate-db.sh
```

## 🔄 Updates & Maintenance

### Updating Services

```bash
# Pull latest images
docker compose -f docker/docker-compose.helix.yml pull

# Restart services with new images
docker compose -f docker/docker-compose.helix.yml up -d --force-recreate
```

### Backup Strategy

```bash
# Database backup
docker exec helix-postgres pg_dump -U helix helix > backup.sql

# Redis backup
docker exec helix-redis redis-cli BGSAVE
docker cp helix-redis:/data/dump.rdb ./redis-backup.rdb
```

### Log Rotation

Logs are automatically rotated, but you can manually rotate:
```bash
# Rotate nginx logs
docker exec helix-nginx nginx -s reopen

# Clean up old logs
find ./logs -name "*.log" -mtime +30 -delete
```

## 🆘 Support

- **Documentation**: [docs.helix.ai](https://docs.helix.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/helix/issues)
- **Community**: [Discord Community](https://discord.gg/helix)
- **Enterprise Support**: enterprise@helix.ai

## 📄 License

Apache License 2.0 - see [LICENSE](../LICENSE) file for details.