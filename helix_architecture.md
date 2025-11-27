# Helix - AI Gateway Architecture Design

## Overview

Helix is an enterprise-grade AI Gateway built on top of LiteLLM that provides:
- **Semantic + Exact Caching**: Redis Vector Search with HNSW indexes
- **PII Redaction**: Microsoft Presidio integration with custom recognizers
- **Cost Optimization**: Real-time spend tracking and budget controls
- **Performance Monitoring**: Cache hit rates, latency metrics, and cost savings

## Architecture Principles

1. **Minimal Code Changes**: Leverage existing LiteLLM hook systems and middleware
2. **Production-Ready**: Use existing authentication, database, and monitoring systems
3. **Scalable**: Build on top of LiteLLM's distributed architecture
4. **Compliance-First**: Implement enterprise-grade security and audit trails

## Directory Structure

```
helix/
├── helix/                                    # Main Helix package
│   ├── __init__.py
│   ├── core/                                 # Core Helix functionality
│   │   ├── __init__.py
│   │   ├── gateway.py                        # Main Gateway orchestrator
│   │   ├── config.py                         # Helix configuration management
│   │   └── exceptions.py                     # Helix-specific exceptions
│   ├── caching/                             # Enhanced caching layer
│   │   ├── __init__.py
│   │   ├── semantic_cache.py                # Redis vector search implementation
│   │   ├── hybrid_cache.py                 # Semantic + exact cache combination
│   │   ├── cache_metrics.py                # Cache performance tracking
│   │   └── cache_strategies.py              # Advanced caching strategies
│   ├── pii/                                # PII detection and redaction
│   │   ├── __init__.py
│   │   ├── presidio_integration.py          # Microsoft Presidio integration
│   │   ├── custom_recognizers.py            # Custom PII recognizers
│   │   ├── redaction_strategies.py          # Configurable redaction methods
│   │   └── pii_audit.py                     # PII detection logging
│   ├── cost/                               # Cost optimization and tracking
│   │   ├── __init__.py
│   │   ├── budget_manager.py               # Advanced budget management
│   │   ├── cost_tracker.py                 # Real-time cost calculation
│   │   ├── spend_optimizer.py              # Spend optimization strategies
│   │   └── cost_analytics.py               # Cost analytics and reporting
│   ├── monitoring/                         # Monitoring and analytics
│   │   ├── __init__.py
│   │   ├── metrics_collector.py            # Real-time metrics collection
│   │   ├── performance_analyzer.py         # Performance analysis
│   │   └── alerting.py                     # Alert management
│   ├── integrations/                       # Helix-specific integrations
│   │   ├── __init__.py
│   │   └── litellm_hooks.py                # LiteLLM hook implementations
│   └── dashboard/                         # Streamlit dashboard
│       ├── __init__.py
│       ├── app.py                         # Main dashboard application
│       ├── pages/                         # Dashboard pages
│       │   ├── __init__.py
│       │   ├── overview.py                # Overview dashboard
│       │   ├── caching.py                 # Cache analytics
│       │   ├── costs.py                   # Cost analysis
│       │   ├── pii.py                     # PII monitoring
│       │   └── settings.py                # Configuration management
│       └── components/                    # Reusable dashboard components
│           ├── __init__.py
│           ├── charts.py                   # Chart components
│           ├── metrics.py                  # Metrics display
│           └── tables.py                   # Data tables
├── config/                                 # Configuration files
│   ├── helix.example.yaml                 # Example configuration
│   ├── development.yaml                    # Development config
│   └── production.yaml                    # Production config
├── docker/                               # Docker deployment
│   ├── Dockerfile.helix                   # Helix service Dockerfile
│   ├── docker-compose.yml                # Multi-service deployment
│   ├── redis/                            # Redis configuration
│   │   └── redis.conf                    # Redis with vector search config
│   └── monitoring/                       # Monitoring stack
│       ├── prometheus.yml
│       └── grafana/
│           └── dashboards/
├── migrations/                           # Database migrations
│   └── helix_schema_extensions.prisma     # Helix database extensions
├── scripts/                             # Utility scripts
│   ├── setup_helix.py                   # Installation script
│   ├── migrate_helix.py                 # Database migration script
│   └── benchmark_helix.py                # Performance benchmarking
└── tests/                               # Test suite
    ├── unit/
    ├── integration/
    └── performance/
```

## Integration Points with LiteLLM

### 1. Hook System Integration

Helix will leverage LiteLLM's existing hook system:

```python
# litellm/integrations/helix/helix_gateway.py
class HelixGateway:
    """
    Main Helix Gateway class that integrates with LiteLLM hooks
    """

    async def async_pre_call_hook(self, *args, **kwargs):
        """
        Pre-call hook for PII detection and cache lookup
        """
        pass

    async def async_post_call_hook(self, *args, **kwargs):
        """
        Post-call hook for cache storage and cost tracking
        """
        pass

    async def async_log_success_event_hook(self, *args, **kwargs):
        """
        Success logging hook for metrics collection
        """
        pass
```

### 2. Configuration Integration

Helix configuration will be loaded through LiteLLM's YAML configuration system:

```yaml
# litellm/proxy/config.yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "openai/gpt-4"
      api_base: "https://api.openai.com/v1"

# Helix-specific configuration
helix:
  enabled: true

  # Caching configuration
  caching:
    type: "semantic_hybrid"
    redis:
      host: "redis"
      port: 6379
      vector_search:
        enabled: true
        index_name: "helix_semantic_cache"
        similarity_threshold: 0.85
        embedding_model: "text-embedding-3-small"

  # PII configuration
  pii:
    enabled: true
    presidio:
      anonymize: true
      recognizers:
        - "EMAIL_ADDRESS"
        - "PHONE_NUMBER"
        - "CREDIT_CARD"
        - "US_SSN"
      custom_recognizers:
        - name: "API_KEY"
          pattern: "[A-Za-z0-9]{32,}"

  # Cost optimization
  cost:
    budget_tracking: true
    real_time_monitoring: true
    optimization:
      cache_first_strategy: true
      model_fallback: true
      spend_limits:
        daily: 1000.0
        monthly: 30000.0

  # Monitoring
  monitoring:
    metrics_collection: true
    dashboard_port: 8501
    alerting:
      slack_webhook: "${SLACK_WEBHOOK_URL}"
      email_smtp: "${SMTP_CONFIG}"
```

### 3. Database Schema Extensions

Extend LiteLLM's Prisma schema with Helix-specific tables:

```prisma
// Helix Cache Metrics
model HelixCacheMetrics {
  id          String   @id @default(uuid())
  cache_key   String   @unique
  cache_type  String   // "semantic", "exact", "hybrid"
  hit_count   BigInt   @default(0)
  miss_count  BigInt   @default(0)
  hit_rate    Float    @default(0.0)
  last_access DateTime @default(now())
  created_at  DateTime @default(now())
  updated_at  DateTime @updatedAt
}

// Helix PII Detection Logs
model HelixPIIDetectionLog {
  id          String   @id @default(uuid())
  request_id  String
  detected_pii Json     // PII entities detected
  redacted    Boolean  @default(false)
  confidence_score Float @default(0.0)
  processing_time_ms Int @default(0)
  created_at  DateTime @default(now())

  // Relations
  request HelixRequestLog? @relation(fields: [request_id], references: [id])
}

// Helix Cost Tracking
model HelixCostTracking {
  id              String   @id @default(uuid())
  request_id      String   @unique
  model_name      String
  input_tokens    BigInt?
  output_tokens   BigInt?
  cache_hit       Boolean  @default(false)
  cache_savings   Float    @default(0.0)
  original_cost   Float
  actual_cost     Float
  savings_rate    Float    @default(0.0)
  created_at      DateTime @default(now())

  // Relations
  request HelixRequestLog? @relation(fields: [request_id], references: [id])
}

// Helix Request Log (Central request tracking)
model HelixRequestLog {
  id              String   @id @default(uuid())
  user_id         String?
  team_id         String?
  organization_id String?
  api_key         String?
  model_name      String
  endpoint        String
  request_hash    String   @unique
  latency_ms      Int
  success         Boolean
  error_message   String?
  created_at      DateTime @default(now())

  // Relations
  pii_detections  HelixPIIDetectionLog[]
  cost_tracking   HelixCostTracking?
}
```

## Core Components

### 1. Semantic + Hybrid Caching

```python
# helix/caching/hybrid_cache.py
class HelixHybridCache:
    """
    Hybrid caching system combining semantic and exact matching
    """

    def __init__(self, redis_client, embedding_model):
        self.redis_client = redis_client
        self.embedding_model = embedding_model
        self.exact_cache = RedisCache(redis_client)
        self.semantic_cache = RedisSemanticCache(redis_client, embedding_model)

    async def get(self, key: str, query: str = None) -> Optional[dict]:
        """
        Try exact match first, then semantic match
        """
        # Try exact cache first
        result = await self.exact_cache.get(key)
        if result:
            await self._update_metrics("exact_hit")
            return result

        # Try semantic cache if query provided
        if query:
            result = await self.semantic_cache.get(query)
            if result:
                await self._update_metrics("semantic_hit")
                return result

        await self._update_metrics("miss")
        return None

    async def set(self, key: str, value: dict, query: str = None, ttl: int = 3600):
        """
        Store in both exact and semantic caches
        """
        await self.exact_cache.set(key, value, ttl)

        if query:
            await self.semantic_cache.set(query, value, ttl)
```

### 2. PII Detection and Redaction

```python
# helix/pii/presidio_integration.py
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class HelixPIIProcessor:
    """
    PII detection and redaction using Microsoft Presidio
    """

    def __init__(self, config: dict):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.config = config
        self._load_custom_recognizers()

    async def detect_and_redact(self, text: str) -> tuple[str, list]:
        """
        Detect PII and return redacted text with detected entities
        """
        # Analyze text for PII
        results = self.analyzer.analyze(
            text=text,
            entities=self.config.get("recognizers", []),
            language='en'
        )

        # Redact if enabled
        if self.config.get("anonymize", True):
            redacted_text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
        else:
            redacted_text = text

        return redacted_text, results

    def _load_custom_recognizers(self):
        """
        Load custom PII recognizers from configuration
        """
        custom_recognizers = self.config.get("custom_recognizers", [])
        for recognizer_config in custom_recognizers:
            # Implement custom recognizer loading
            pass
```

### 3. Cost Optimization

```python
# helix/cost/spend_optimizer.py
class HelixSpendOptimizer:
    """
    Advanced spend optimization strategies
    """

    def __init__(self, budget_manager: HelixBudgetManager):
        self.budget_manager = budget_manager

    async def optimize_request(self, request_data: dict) -> dict:
        """
        Optimize request for cost efficiency
        """
        # Check cache first
        cached_result = await self._check_cache(request_data)
        if cached_result:
            return cached_result

        # Check for cheaper model alternatives
        optimized_model = await self._find_cheaper_model(request_data)
        if optimized_model:
            request_data["model"] = optimized_model

        # Apply cost-saving strategies
        request_data = await self._apply_optimization_strategies(request_data)

        return request_data

    async def _find_cheaper_model(self, request_data: dict) -> Optional[str]:
        """
        Find cheaper alternative model for the request
        """
        # Implement model selection logic based on:
        # - Request complexity
        # - Required capabilities
        # - Cost constraints
        pass
```

## Deployment Architecture

### Docker Compose Setup

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  helix-gateway:
    build:
      context: ..
      dockerfile: docker/Dockerfile.helix
    ports:
      - "8000:8000"
      - "8501:8501"  # Streamlit dashboard
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/helix
      - REDIS_URL=redis://redis:6379
      - HELIX_CONFIG_PATH=/app/config/production.yaml
    depends_on:
      - postgres
      - redis
    volumes:
      - ../config:/app/config

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=helix
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis/redis-stack-server:latest
    ports:
      - "6379:6379"
    volumes:
      - ../docker/redis/redis.conf:/opt/redis-stack/etc/redis/redis.conf
      - redis_data:/data
    command: redis-server /opt/redis-stack/etc/redis/redis.conf

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ../docker/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ../docker/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Helix Dockerfile

```dockerfile
# docker/Dockerfile.helix
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-helix.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-helix.txt

# Copy application code
COPY . .

# Copy configuration
COPY config/ /app/config/

# Set environment variables
ENV PYTHONPATH=/app
ENV HELIX_ENABLED=true

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "helix.dashboard.app", "--host", "0.0.0.0", "--port", "8501"]
```

## Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Set up Helix package structure and dependencies
2. Implement LiteLLM hook integration
3. Create database schema extensions
4. Set up basic configuration system

### Phase 2: Caching Implementation (Weeks 3-4)
1. Implement Redis vector search integration
2. Build hybrid caching system
3. Add cache metrics collection
4. Create cache management APIs

### Phase 3: PII Detection (Weeks 5-6)
1. Integrate Microsoft Presidio
2. Implement custom recognizers
3. Add redaction strategies
4. Create PII audit logging

### Phase 4: Cost Optimization (Weeks 7-8)
1. Implement real-time cost tracking
2. Build budget management system
3. Add optimization strategies
4. Create cost analytics

### Phase 5: Monitoring & Dashboard (Weeks 9-10)
1. Build Streamlit dashboard
2. Implement metrics collection
3. Add alerting system
4. Create reporting functionality

### Phase 6: Deployment & Testing (Weeks 11-12)
1. Containerize all services
2. Set up Docker Compose deployment
3. Implement monitoring stack
4. Performance testing and optimization

## Performance Targets

### Caching Performance
- **Cache Hit Rate**: >80% for similar queries
- **Semantic Search Latency**: <50ms
- **Cache Storage**: Support for 1M+ cached responses
- **Memory Usage**: <4GB for Redis with vector search

### PII Detection Performance
- **Detection Latency**: <100ms per request
- **Accuracy**: >95% for standard PII types
- **False Positive Rate**: <1%
- **Throughput**: Support for 1000+ requests/second

### Cost Optimization
- **Savings Rate**: 20-40% reduction in LLM costs
- **Budget Control Accuracy**: Real-time monitoring within 1%
- **Alert Latency**: <1 minute for budget alerts
- **Reporting Latency**: <5 seconds for cost reports

## Security & Compliance

### Data Protection
- **Encryption**: AES-256 for sensitive data at rest
- **PII Compliance**: GDPR, CCPA, HIPAA support
- **Audit Trails**: Complete request/response logging
- **Access Control**: Role-based permissions

### Enterprise Security
- **Authentication**: SSO, OAuth 2.0, JWT support
- **Authorization**: Fine-grained permissions
- **Network Security**: TLS 1.3, mTLS support
- **Compliance**: SOC 2 Type II, ISO 27001

This architecture provides a comprehensive, production-ready AI Gateway that extends LiteLLM's capabilities while maintaining compatibility with existing deployments.