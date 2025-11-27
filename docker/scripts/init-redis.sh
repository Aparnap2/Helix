#!/bin/bash
# Helix Redis Initialization Script
# Sets up Redis with vector search capabilities and required indexes

set -euo pipefail

# Environment variables
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Redis command wrapper
redis_cmd() {
    if [ -n "$REDIS_PASSWORD" ]; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" "$@"
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" "$@"
    fi
}

# Wait for Redis to be ready
wait_for_redis() {
    local max_attempts=30
    local attempt=1

    log "Waiting for Redis to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if redis_cmd ping >/dev/null 2>&1; then
            log "Redis is ready (attempt $attempt/$max_attempts)"
            return 0
        else
            log "Waiting for Redis... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done

    error "Redis is not available after $max_attempts attempts"
    return 1
}

# Check if Redis modules are loaded
check_redis_modules() {
    log "Checking Redis modules..."

    # Check for RedisSearch module
    if redis_cmd module list | grep -q "search"; then
        log "RedisSearch module is loaded"
    else
        error "RedisSearch module is not available. Please use redis-stack or load RedisSearch module."
        return 1
    fi

    # Check for RedisJSON module
    if redis_cmd module list | grep -q "ReJSON"; then
        log "RedisJSON module is loaded"
    else
        error "RedisJSON module is not available. Please use redis-stack or load RedisJSON module."
        return 1
    fi
}

# Initialize Helix-specific Redis configuration
setup_helix_redis() {
    log "Setting up Helix Redis configuration..."

    # Configure Redis for optimal performance with Helix
    redis_cmd config set "maxmemory" "1gb"
    redis_cmd config set "maxmemory-policy" "allkeys-lru"
    redis_cmd config set "save" "900 1 300 10 60 10000"
    redis_cmd config set "appendonly" "yes"
    redis_cmd config set "appendfsync" "everysec"
    redis_cmd config set "slowlog-log-slower-than" "10000"
    redis_cmd config set "slowlog-max-len" "1000"

    log "Redis configuration updated for Helix"
}

# Create Helix vector search index
create_vector_index() {
    log "Creating Helix vector search index..."

    # Drop existing index if it exists (to handle updates)
    if redis_cmd ft.info idx:semantic >/dev/null 2>&1; then
        log "Dropping existing semantic search index..."
        redis_cmd ft.dropidx idx:semantic
    fi

    # Create semantic vector search index for embeddings
    redis_cmd ft.create idx:semantic \
        ON HASH \
        PREFIX 1 "helix:vector:" \
        SCHEMA \
            prompt TEXT \
            model TEXT \
            response_json TEXT \
            vector VECTOR HNSW 12 \
                TYPE FLOAT32 \
                DIM 384 \
                DISTANCE_METRIC COSINE \
                INITIAL_CAP 10000 \
                M 16 \
                EF_CONSTRUCTION 200 \
                EF_RUNTIME 200 \
            created_at NUMERIC \
            cost_usd NUMERIC \
            token_count NUMERIC \
            user_id TEXT \
            cache_type TAG

    log "Semantic vector search index created successfully"
}

# Create additional indexes for performance
create_supporting_indexes() {
    log "Creating supporting indexes..."

    # Index for user-specific spend tracking
    redis_cmd ft.create idx:user_spend \
        ON HASH \
        PREFIX 1 "helix:user:" \
        SCHEMA \
            user_id TAG \
            total_spend NUMERIC \
            request_count NUMERIC \
            cache_hits NUMERIC \
            last_request_date NUMERIC

    # Index for cost tracking
    redis_cmd ft.create idx:cost_tracking \
        ON HASH \
        PREFIX 1 "helix:cost:" \
        SCHEMA \
            model TAG \
            cost_usd NUMERIC \
            tokens NUMERIC \
            created_at NUMERIC \
            user_id TAG

    # Index for PII incidents
    redis_cmd ft.create idx:pii_incidents \
        ON HASH \
        PREFIX 1 "helix:pii:" \
        SCHEMA \
            user_id TAG \
            entity_type TAG \
            created_at NUMERIC \
            severity TAG

    log "Supporting indexes created successfully"
}

# Set up time series data for metrics
setup_time_series() {
    log "Setting up time series for metrics..."

    # Create time series for request metrics
    redis_cmd ts.create helix:metrics:requests \
        retention 7776000 \
        LABELS "metric_type" "requests" "service" "helix" || log "Time series already exists"

    redis_cmd ts.create helix:metrics:cache_hits \
        retention 7776000 \
        LABELS "metric_type" "cache_hits" "service" "helix" || log "Time series already exists"

    redis_cmd ts.create helix:metrics:response_time \
        retention 7776000 \
        LABELS "metric_type" "response_time" "service" "helix" || log "Time series already exists"

    redis_cmd ts.create helix:metrics:costs \
        retention 7776000 \
        LABELS "metric_type" "costs" "service" "helix" || log "Time series already exists"

    redis_cmd ts.create helix:metrics:pii_incidents \
        retention 7776000 \
        LABELS "metric_type" "pii_incidents" "service" "helix" || log "Time series already exists"

    log "Time series for metrics created"
}

# Initialize sample data (optional)
init_sample_data() {
    if [ "${INIT_SAMPLE_DATA:-false}" = "true" ]; then
        log "Initializing sample data..."

        # Sample vector entry for testing
        local sample_uuid=$(uuidgen || echo "sample-uuid")
        local sample_vector="0.1,0.2,0.3,0.4"  # Truncated for example

        redis_cmd hset "helix:vector:$sample_uuid" \
            prompt "How do I reset my password?" \
            model "gpt-3.5-turbo" \
            response_json '{"response":"To reset your password, click the forgot password link..."}' \
            vector "$(python -c "import numpy as np; print(','.join(map(str, np.random.rand(384).astype(np.float32)))))")" \
            created_at "$(date +%s)" \
            cost_usd "0.0012" \
            token_count "150" \
            user_id "demo-user" \
            cache_type "exact"

        log "Sample data initialized"
    fi
}

# Main execution
main() {
    log "Starting Helix Redis initialization..."

    # Wait for Redis to be available
    wait_for_redis

    # Check required modules
    check_redis_modules

    # Setup Redis configuration
    setup_helix_redis

    # Create search indexes
    create_vector_index
    create_supporting_indexes

    # Setup time series for metrics
    setup_time_series

    # Initialize sample data if requested
    init_sample_data

    # Get Redis info for verification
    local redis_version=$(redis_cmd info server | grep "redis_version" | cut -d: -f2 | tr -d '\r')
    local redis_memory=$(redis_cmd info memory | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')

    log "Redis initialization completed successfully!"
    log "Redis version: $redis_version"
    log "Memory usage: $redis_memory"
    log "Vector search index: idx:semantic"
    log "Ready for Helix AI Gateway"
}

# Execute main function
main "$@"