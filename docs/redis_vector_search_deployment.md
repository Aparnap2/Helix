# Redis Vector Search Deployment Guide

This guide provides comprehensive instructions for deploying and configuring Redis Vector Search with Helix for production-ready semantic caching.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Redis Installation](#redis-installation)
3. [Vector Search Configuration](#vector-search-configuration)
4. [Helix Integration](#helix-integration)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Benchmarking](#benchmarking)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Memory**: Minimum 8GB RAM, Recommended 16GB+
- **Storage**: SSD with at least 50GB free space
- **CPU**: 4+ cores recommended
- **Python**: 3.9+ with pip
- **Redis**: 7.2+ with RediSearch module

### Network Requirements

- **Port**: 6379 (default Redis port)
- **Bandwidth**: 1Gbps+ recommended for production
- **Latency**: < 1ms for Redis-client connection
- **Security**: TLS/SSL for production deployments

### Software Dependencies

```bash
# Required Python packages
pip install redis[hiredis] sentence-transformers torch torchvision
pip install psutil pandas matplotlib seaborn
pip install helix-python  # If using package
```

## Redis Installation

### Method 1: Docker Deployment (Recommended)

```bash
# Docker Compose for Redis with Vector Search
version: '3.8'
services:
  redis:
    image: redis/redis-stack-server:7.2.0
    container_name: helix-redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis-data:/data
      - ./redis/helix_init.redis:/docker-entrypoint-initdb.d/helix_init.redis
    command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: helix-redis-commander
    ports:
      - "8081:8081"
    environment:
      - REDIS_HOSTS=local:redis:6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    depends_on:
      - redis
```

### Method 2: Source Installation

```bash
# Install Redis 7.2+ with RediSearch
# Ubuntu/Debian
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install redis-stack-server

# CentOS/RHEL
sudo yum install -y https://download.redis.io/redis-stack/redis-stack-server-7.2.0.rhel7.x86_64.rpm

# Enable and start Redis
sudo systemctl enable redis-stack-server
sudo systemctl start redis-stack-server
```

### Method 3: Redis Enterprise

For production deployments requiring high availability:

```bash
# Redis Enterprise deployment
# Download Redis Enterprise Software
wget https://download.redis.io/redis-enterprise-software/redis-enterprise-7.2.0.tgz
tar xvf redis-enterprise-7.2.0.tgz
cd redis-enterprise-7.2.0
sudo ./install.sh
```

## Vector Search Configuration

### Redis Configuration Optimization

Edit `/etc/redis/redis.conf` or use `CONFIG SET` commands:

```bash
# Memory Management
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Performance Optimization
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# Hash Optimization
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

# List Optimization
list-max-ziplist-size -2

# Set Optimization
set-max-intset-entries 512

# Sorted Set Optimization
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Persistence (if needed)
save 900 1
save 300 10
save 60 10000
```

### Initialize Helix Redis Schema

```bash
# Execute initialization script
redis-cli < redis/helix_init.redis

# Or execute individual commands
redis-cli --pipe < redis/helix_init.redis
```

### Vector Index Parameters

The HNSW (Hierarchical Navigable Small World) index parameters are optimized for semantic caching:

```redis
# Optimized HNSW Parameters
M = 64                    # Number of bi-directional links
EF_CONSTRUCTION = 400     # Search accuracy during index build
EF_RUNTIME = 200          # Search accuracy during queries
DIM = 1536               # Embedding dimensions (OpenAI)
DISTANCE_METRIC = COSINE # Cosine similarity for text embeddings
```

## Helix Integration

### Configuration Setup

Create `config/helix.yaml`:

```yaml
helix:
  enabled: true
  general:
    debug: false
    request_timeout: 300

  caching:
    enabled: true
    cache_type: "hybrid"
    default_ttl: 3600
    max_ttl: 86400
    max_cache_size: 1000000
    max_memory_usage: 4294967296  # 4GB
    compression: true
    batch_size: 100
    batch_timeout: 5

    redis:
      host: "localhost"
      port: 6379
      password: null  # Use REDIS_PASSWORD env var
      database: 0
      max_connections: 100
      connection_timeout: 10
      socket_timeout: 10
      cluster_enabled: false

    vector_search:
      enabled: true
      embedding_model: "text-embedding-3-small"
      embedding_provider: "openai"
      index_name: "helix_semantic_cache"
      similarity_threshold: 0.85
      distance_metric: "cosine"
      dimensions: 1536

      hnsw:
        m: 64
        ef_construction: 400
        ef_runtime: 200

  pii:
    enabled: false
    mode: "detect_only"

  cost:
    enabled: true
    real_time_tracking: true

  monitoring:
    enabled: true
    metrics:
      collection_interval: 60
      performance_metrics: true

    dashboard:
      enabled: true
      streamlit:
        enabled: true
        port: 8501
```

### Environment Variables

```bash
# Redis Configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=your_secure_password
export REDIS_DB=0

# OpenAI Configuration (if using)
export OPENAI_API_KEY=your_openai_api_key

# Helix Configuration
export HELIX_CONFIG_PATH=/path/to/config/helix.yaml
export HELIX_DEBUG=false
```

### Application Integration

```python
from helix.core.semantic_cache import get_semantic_cache
from helix.core.cache_manager import get_cache_manager
import asyncio

async def example_usage():
    # Initialize semantic cache
    cache = await get_semantic_cache()

    # Cache lookup
    prompt = "What is machine learning?"
    model = "gpt-4"

    # Try to get from cache
    cached_response = await cache.get(prompt, model)

    if cached_response:
        print(f"Cache hit: {cached_response}")
    else:
        # Generate response (simulated)
        response = "Machine learning is a subset of AI..."

        # Cache the response
        await cache.set(prompt, model, response, cost=0.01, latency=200.0)
        print(f"Cached new response")

# Multi-level cache usage
async def multi_level_cache_example():
    cache_manager = await get_cache_manager()

    # L1-L3 cache lookup
    cache_key = f"gpt-4:what_is_ai"
    result = await cache_manager.get(cache_key)

    if result:
        return result

    # Generate and cache response
    new_response = "AI is computer systems that can perform tasks..."
    await cache_manager.set(cache_key, new_response)

    return new_response
```

## Performance Optimization

### Connection Pooling

```python
# Optimize connection pool size
redis_pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    max_connections=min(100, cpu_count() * 4),  # 4x CPU cores
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
    max_idle_time=60,  # Close idle connections
    idle_check_interval=30
)
```

### Batch Processing Optimization

```python
# Optimize batch sizes based on workload
from helix.core.config import get_config

config = get_config()

# High-throughput configuration
config.caching.batch_size = 50  # Larger batches for throughput
config.caching.vector_search.ef_runtime = 100  # Faster search, less accurate

# High-accuracy configuration
config.caching.batch_size = 10  # Smaller batches for latency
config.caching.vector_search.ef_runtime = 200  # More accurate search
```

### Memory Optimization

```bash
# Redis memory management
redis-cli CONFIG SET maxmemory 8gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Python memory management
import gc
import psutil

# Periodic garbage collection
async def periodic_gc():
    process = psutil.Process()
    if process.memory_percent() > 70:
        gc.collect()
```

### Index Optimization

```redis
# Monitor index statistics
FT.INFO helix:semantic:index

# Optimize index for specific workload
# For high-write workloads:
FT.CONFIG SET MAXDOCTABLESIZE 10000000

# For high-read workloads:
FT.CONFIG SET DEFAULT_QUERY_TIMEOUT 5000
```

### Adaptive Thresholds

```python
# Enable adaptive similarity thresholds
from helix.core.semantic_cache import AdaptiveThreshold

adaptive_threshold = AdaptiveThreshold(
    base_threshold=0.85,
    min_threshold=0.70,
    max_threshold=0.95,
    performance_window=300,  # 5 minutes
    hit_rate_target=80.0,
    adjustment_factor=0.05
)
```

## Monitoring and Maintenance

### Health Checks

```python
async def health_check():
    """Comprehensive health check"""
    try:
        # Redis connectivity
        redis_conn = redis.Redis(connection_pool=redis_pool)
        await redis_conn.ping()

        # Index health
        info = await redis_conn.ft("helix:semantic:index").info()

        # Memory usage
        memory_info = await redis_conn.info("memory")

        return {
            "redis_connected": True,
            "index_documents": info.get("num_docs", 0),
            "memory_usage": memory_info.get("used_memory", 0),
            "health_status": "healthy"
        }

    except Exception as e:
        return {
            "redis_connected": False,
            "error": str(e),
            "health_status": "unhealthy"
        }
```

### Performance Metrics

```python
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    cache_manager = await get_cache_manager()

    metrics = await cache_manager.get_statistics()

    return {
        "cache_performance": {
            "overall_hit_rate": metrics["overall_hit_rate"],
            "total_requests": metrics["global"]["total_requests"],
            "l1_hits": metrics["global"]["l1_hits"],
            "l2_hits": metrics["global"]["l2_hits"],
            "l3_hits": metrics["global"]["l3_hits"]
        },
        "redis_performance": {
            "connected_clients": metrics.get("redis_connected_clients", 0),
            "memory_usage": metrics.get("redis_memory", 0),
            "cpu_usage": metrics.get("system_cpu", 0)
        }
    }
```

### Automated Maintenance

```python
async def maintenance_tasks():
    """Automated maintenance tasks"""
    while True:
        try:
            # Cleanup expired entries
            cache_manager = await get_cache_manager()
            await cache_manager.clear(pattern="expired:*", levels=[CacheLevel.REDIS])

            # Optimize index
            redis_conn = redis.Redis(connection_pool=redis_pool)
            await redis_conn.ft("helix:semantic:index").optimize()

            # Update statistics
            await cache_manager.get_statistics()

            # Sleep for 6 hours
            await asyncio.sleep(21600)

        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")
            await asyncio.sleep(3600)  # Retry sooner on error
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

**Symptoms**: Redis OOM, slow performance

**Solutions**:
```bash
# Check memory usage
redis-cli INFO memory

# Reduce maxmemory if needed
redis-cli CONFIG SET maxmemory 2gb

# Adjust eviction policy
redis-cli CONFIG SET maxmemory-policy volatile-lru

# Monitor memory breakdown
redis-cli MEMORY USAGE helix:vector:*
```

#### 2. Poor Search Performance

**Symptoms**: Slow vector search, high latency

**Solutions**:
```redis
# Check HNSW parameters
FT.INFO helix:semantic:index

# Adjust EF_RUNTIME for speed/accuracy tradeoff
CONFIG SET DEFAULT_QUERY_TIMEOUT 1000

# Monitor search performance
FT.PROFILE helix:semantic:index SEARCH QUERY "*=>[KNN 10 @vector $vector]"
```

#### 3. Connection Issues

**Symptoms**: Connection timeouts, connection pool exhaustion

**Solutions**:
```python
# Adjust connection pool settings
redis_pool = redis.ConnectionPool(
    max_connections=50,  # Reduce if needed
    socket_timeout=10,   # Increase timeout
    retry_on_timeout=True,
    health_check_interval=30
)

# Monitor connection health
async def check_connections():
    pool = get_redis_pool()
    print(f"Active connections: {len(pool._created_connections)}")
    print(f"Available connections: {len(pool._available_connections)}")
```

#### 4. Index Corruption

**Symptoms**: Search returns no results, Redis errors

**Solutions**:
```bash
# Backup current data
redis-cli BGSAVE

# Recreate index
redis-cli FT.DROPINDEX helix:semantic:index
redis-cli < redis/helix_init.redis

# Restore from backup if needed
redis-cli DEBUG POPULATE
```

### Debug Tools

```python
# Debug vector storage
async def debug_vectors():
    """Debug vector storage and retrieval"""
    redis_conn = redis.Redis(connection_pool=redis_pool)

    # Check index info
    info = await redis_conn.ft("helix:semantic:index").info()
    print(f"Index info: {info}")

    # Check vector keys
    cursor = 0
    while True:
        cursor, keys = await redis_conn.scan(cursor, match="helix:vector:*", count=10)
        for key in keys:
            vector_data = await redis_conn.hgetall(key)
            print(f"Vector {key}: {vector_data}")
        if cursor == 0:
            break

# Debug cache performance
async def debug_cache_performance():
    """Debug cache performance bottlenecks"""
    cache = await get_semantic_cache()

    # Test with known data
    test_prompt = "Debug test prompt"
    test_model = "gpt-3.5-turbo"

    # Generate embedding
    embedding = await cache.embedding_processor.encode_single(test_prompt)

    # Search vectors
    results = await cache.vector_manager.search_vectors(
        query_embedding=embedding,
        similarity_threshold=0.5,  # Lower threshold for debugging
        limit=10
    )

    print(f"Found {len(results)} similar vectors")
    for result in results:
        print(f"Score: {result.similarity_score}, Prompt: {result.entry.prompt[:50]}...")
```

## Benchmarking

### Running Benchmarks

```bash
# Run comprehensive benchmark suite
python benchmarks/redis_vector_search_benchmark.py \
    --entries 1000 \
    --searches 100 \
    --concurrent 10 \
    --output benchmark_results

# Run specific benchmark tests
python -m pytest tests/test_redis_vector_search.py::TestVectorIndexManager -v

# Performance testing with custom parameters
python benchmarks/redis_vector_search_benchmark.py \
    --entries 5000 \
    --batch-sizes 10 20 50 \
    --searches 500 \
    --concurrent 20
```

### Benchmark Analysis

The benchmark suite provides detailed performance metrics:

- **Embedding Generation**: Tokens/second, memory usage
- **Vector Insertion**: Vectors/second, batch performance
- **Vector Search**: Queries/second, latency distribution
- **Semantic Cache**: Hit rates, response times
- **Concurrent Access**: Scalability with multiple workers
- **Memory Usage**: Storage efficiency, scaling patterns

### Performance Targets

**Production Performance Targets**:

- **Vector Search**: < 50ms average latency, > 100 queries/second
- **Cache Hit Rate**: > 80% overall hit rate
- **Memory Efficiency**: < 2KB per cached response
- **Concurrent Performance**: Linear scaling up to 50 workers
- **Availability**: 99.9% uptime with < 1s recovery time

### Optimization Recommendations

Based on benchmark results:

1. **Low Search Performance**:
   - Increase `EF_RUNTIME` for better accuracy
   - Optimize HNSW parameters (`M`, `EF_CONSTRUCTION`)
   - Use connection pooling

2. **High Memory Usage**:
   - Implement data compression
   - Adjust TTL policies
   - Use memory-efficient data structures

3. **Poor Cache Hit Rate**:
   - Adjust similarity thresholds
   - Implement cache warming
   - Use adaptive thresholds

4. **Scalability Issues**:
   - Implement Redis clustering
   - Use sharding strategies
   - Optimize network configuration

## Production Deployment Checklist

### Pre-Deployment Checklist

- [ ] Redis 7.2+ with RediSearch installed
- [ ] Vector search index created with HNSW parameters
- [ ] Connection pooling configured
- [ ] Memory limits and eviction policies set
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Performance benchmarks completed
- [ ] Security hardening completed

### Post-Deployment Monitoring

- [ ] Performance metrics collection
- [ ] Health checks every 30 seconds
- [ ] Memory usage monitoring
- [ ] Error rate tracking
- [ ] Cache hit rate analysis
- [ ] Search latency monitoring

### Maintenance Schedule

- **Daily**: Health checks, metric review
- **Weekly**: Performance analysis, threshold tuning
- **Monthly**: Index optimization, backup verification
- **Quarterly**: Capacity planning, scaling review
- **Annually**: Redis version upgrade, architecture review

This comprehensive deployment guide ensures production-ready Redis Vector Search setup with Helix for optimal semantic caching performance.