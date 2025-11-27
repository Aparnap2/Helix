#!/bin/bash
# Helix AI Gateway Startup Script
# Production-ready startup with proper error handling and monitoring

set -euo pipefail

# Environment variables
export PYTHONPATH="/app"
export HELIX_LOG_LEVEL="${HELIX_LOG_LEVEL:-INFO}"
export HELIX_ENVIRONMENT="${HELIX_ENVIRONMENT:-production}"
export GUNICORN_WORKERS="${GUNICORN_WORKERS:-$(nproc)}"
export GUNICORN_BIND="${GUNICORN_BIND:-0.0.0.0:8000}"
export GUNICORN_WORKER_CLASS="${GUNICORN_WORKER_CLASS:-uvicorn.workers.UvicornWorker}"
export GUNICORN_WORKER_CONNECTIONS="${GUNICORN_WORKER_CONNECTIONS:-1000}"
export GUNICORN_MAX_REQUESTS="${GUNICORN_MAX_REQUESTS:-1000}"
export GUNICORN_MAX_REQUESTS_JITTER="${GUNICORN_MAX_REQUESTS_JITTER:-100}"
export GUNICORN_PRELOAD_APP="${GUNICORN_PRELOAD_APP:-true}"
export GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-300}"
export GUNICORN_KEEPALIVE="${GUNICORN_KEEPALIVE:-5}"

# Dashboard settings
export DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"

# Health check settings
export HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
export HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1

    log "Starting health checks..."

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:8000/health" > /dev/null; then
            log "Health check passed (attempt $attempt/$max_attempts)"
            return 0
        else
            log "Health check failed (attempt $attempt/$max_attempts), retrying in ${HEALTH_CHECK_INTERVAL}s..."
            sleep $HEALTH_CHECK_INTERVAL
            ((attempt++))
        fi
    done

    error "Health check failed after $max_attempts attempts"
    return 1
}

# Database connectivity check
check_database() {
    log "Checking database connectivity..."

    # This would be implemented based on your database setup
    # For now, we'll just check if DATABASE_URL is set
    if [ -n "${DATABASE_URL:-}" ]; then
        log "Database URL configured, proceeding with startup..."
        return 0
    else
        log "Warning: No DATABASE_URL configured, some features may not work"
        return 0
    fi
}

# Redis connectivity check
check_redis() {
    log "Checking Redis connectivity..."

    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli -u "${REDIS_URL:-redis://localhost:6379/0}" ping >/dev/null 2>&1; then
            log "Redis connectivity confirmed"
            return 0
        else
            log "Warning: Redis connectivity failed, caching will be disabled"
            return 1
        fi
    else
        log "Warning: redis-cli not available, cannot check Redis connectivity"
        return 1
    fi
}

# Graceful shutdown handler
graceful_shutdown() {
    log "Received shutdown signal, initiating graceful shutdown..."

    # Kill the dashboard process first
    if [ -n "${DASHBOARD_PID:-}" ]; then
        log "Stopping dashboard (PID: $DASHBOARD_PID)..."
        kill -TERM "$DASHBOARD_PID" 2>/dev/null || true
        wait "$DASHBOARD_PID" 2>/dev/null || true
    fi

    # Kill the main process
    if [ -n "${MAIN_PID:-}" ]; then
        log "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi

    log "Graceful shutdown completed"
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT SIGQUIT

# Pre-startup checks
log "Starting Helix AI Gateway..."
log "Environment: $HELIX_ENVIRONMENT"
log "Log Level: $HELIX_LOG_LEVEL"
log "Gunicorn Workers: $GUNICORN_WORKERS"

# Check dependencies
check_database
check_redis

# Create necessary directories
mkdir -p /app/logs /app/data

# Set up Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Start the main application
log "Starting main application on $GUNICORN_BIND..."

# Use gunicorn with uvicorn workers for production
gunicorn \
    --bind "$GUNICORN_BIND" \
    --workers "$GUNICORN_WORKERS" \
    --worker-class "$GUNICORN_WORKER_CLASS" \
    --worker-connections "$GUNICORN_WORKER_CONNECTIONS" \
    --max-requests "$GUNICORN_MAX_REQUESTS" \
    --max-requests-jitter "$GUNICORN_MAX_REQUESTS_JITTER" \
    --preload "$GUNICORN_PRELOAD_APP" \
    --timeout "$GUNICORN_TIMEOUT" \
    --keep-alive "$GUNICORN_KEEPALIVE" \
    --access-logfile - \
    --error-logfile - \
    --log-level "$HELIX_LOG_LEVEL" \
    --access-logformat '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s' \
    --capture-output \
    --enable-stdio-inheritance \
    litellm.proxy.proxy_server:app &

MAIN_PID=$!

log "Main application started (PID: $MAIN_PID)"

# Start the dashboard
if [ "${HELIX_DASHBOARD_ENABLED:-true}" = "true" ]; then
    log "Starting dashboard on $DASHBOARD_HOST:$DASHBOARD_PORT..."

    # Start Streamlit dashboard in background
    streamlit run helix/dashboard/app.py \
        --server.address "$DASHBOARD_HOST" \
        --server.port "$DASHBOARD_PORT" \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --browser.gatherUsageStats false \
        --logger.level "$HELIX_LOG_LEVEL" &

    DASHBOARD_PID=$!
    log "Dashboard started (PID: $DASHBOARD_PID)"
else
    log "Dashboard disabled, skipping dashboard startup"
    DASHBOARD_PID=""
fi

# Wait for health check
sleep 5  # Give the application a moment to start
health_check

log "Helix AI Gateway is running!"
log "Main API: http://$GUNICORN_BIND"
if [ -n "$DASHBOARD_PID" ]; then
    log "Dashboard: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
fi
log "Health endpoint: http://localhost:8000/health"
log "Metrics endpoint: http://localhost:8000/metrics"

# Wait for the main process
wait $MAIN_PID