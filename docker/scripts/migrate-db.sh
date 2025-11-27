#!/bin/bash
# Helix Database Migration Script
# Handles PostgreSQL database initialization and migrations

set -euo pipefail

# Environment variables
DATABASE_URL="${DATABASE_URL:-}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-helix}"
POSTGRES_USER="${POSTGRES_USER:-helix}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"

# Extract database connection details
if [ -n "$DATABASE_URL" ]; then
    # Parse DATABASE_URL: postgresql://user:password@host:port/db
    POSTGRES_USER=$(echo "$DATABASE_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    POSTGRES_PASSWORD=$(echo "$DATABASE_URL" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    POSTGRES_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    POSTGRES_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    POSTGRES_DB=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
fi

# Default values
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-helix}"
POSTGRES_USER="${POSTGRES_USER:-helix}"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# PostgreSQL command wrapper
psql_cmd() {
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -v ON_ERROR_STOP=1 \
        "$@"
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    local max_attempts=30
    local attempt=1

    log "Waiting for PostgreSQL to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD="$POSTGRES_PASSWORD" psql \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d postgres \
            -c "SELECT 1;" >/dev/null 2>&1; then
            log "PostgreSQL is ready (attempt $attempt/$max_attempts)"
            return 0
        else
            log "Waiting for PostgreSQL... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done

    error "PostgreSQL is not available after $max_attempts attempts"
    return 1
}

# Create database if it doesn't exist
create_database() {
    log "Creating database $POSTGRES_DB if it doesn't exist..."

    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d postgres \
        -c "CREATE DATABASE $POSTGRES_DB;" \
        -c "CREATE DATABASE ${POSTGRES_DB}_test;" 2>/dev/null || log "Database already exists"

    log "Database setup completed"
}

# Run migrations
run_migrations() {
    log "Running Helix database migrations..."

    # Check if migrations table exists
    if ! psql_cmd -c "\dt" | grep -q "helix_migrations"; then
        log "Creating migrations table..."
        psql_cmd << 'EOF'
CREATE TABLE helix_migrations (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64) NOT NULL
);

CREATE INDEX idx_helix_migrations_filename ON helix_migrations(filename);
CREATE INDEX idx_helix_migrations_executed_at ON helix_migrations(executed_at);
EOF
    fi

    # Migration files in order of execution
    local migrations=(
        "001_create_users_table.sql"
        "002_create_api_keys_table.sql"
        "003_create_requests_table.sql"
        "004_create_cache_table.sql"
        "005_create_usage_stats_table.sql"
        "006_create_pii_incidents_table.sql"
        "007_create_cost_tracking_table.sql"
        "008_create_sessions_table.sql"
        "009_create_models_table.sql"
        "010_create_indexes.sql"
        "011_create_triggers.sql"
        "012_create_views.sql"
    )

    # Execute migrations
    for migration in "${migrations[@]}"; do
        local migration_file="/docker-entrypoint-initdb.d/migrations/$migration"

        if [ -f "$migration_file" ]; then
            local checksum=$(sha256sum "$migration_file" | cut -d' ' -f1)

            # Check if migration already executed
            local executed=$(psql_cmd -t -c "SELECT filename FROM helix_migrations WHERE filename = '$migration';" | tr -d ' ')

            if [ -z "$executed" ]; then
                log "Executing migration: $migration"

                if psql_cmd -f "$migration_file"; then
                    psql_cmd -c "INSERT INTO helix_migrations (filename, checksum) VALUES ('$migration', '$checksum');"
                    log "Migration $migration completed successfully"
                else
                    error "Migration $migration failed"
                    return 1
                fi
            else
                log "Migration $migration already executed, skipping"
            fi
        else
            log "Migration file $migration_file not found, skipping"
        fi
    done

    log "All migrations completed successfully"
}

# Create the Helix schema (initial setup)
create_helix_schema() {
    log "Creating Helix database schema..."

    psql_cmd << 'EOF'
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    hashed_password VARCHAR(255),
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- Requests table for logging all LLM requests
CREATE TABLE IF NOT EXISTS requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    api_key_id UUID REFERENCES api_keys(id),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    model VARCHAR(100) NOT NULL,
    provider VARCHAR(50),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd DECIMAL(10,6),
    response_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT false,
    pii_detected BOOLEAN DEFAULT false,
    pii_redacted BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cache table for exact matches
CREATE TABLE IF NOT EXISTS cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key VARCHAR(512) UNIQUE NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt_hash VARCHAR(64) NOT NULL,
    response_data JSONB NOT NULL,
    response_metadata JSONB DEFAULT '{}',
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage statistics table
CREATE TABLE IF NOT EXISTS usage_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    date DATE NOT NULL,
    model VARCHAR(100) NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(12,6) DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date, model)
);

-- PII incidents tracking
CREATE TABLE IF NOT EXISTS pii_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    request_id UUID REFERENCES requests(id),
    entity_type VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2),
    start_position INTEGER,
    end_position INTEGER,
    original_text TEXT,
    redacted_text TEXT,
    severity VARCHAR(20) DEFAULT 'medium',
    auto_resolved BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cost tracking with detailed breakdown
CREATE TABLE IF NOT EXISTS cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    request_id UUID REFERENCES requests(id),
    model VARCHAR(100) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    input_cost DECIMAL(10,6) DEFAULT 0,
    output_cost DECIMAL(10,6) DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions for conversation context
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Models configuration
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    provider VARCHAR(50) NOT NULL,
    display_name VARCHAR(255),
    description TEXT,
    max_tokens INTEGER,
    context_window INTEGER,
    input_cost_per_1k DECIMAL(8,6),
    output_cost_per_1k DECIMAL(8,6),
    is_available BOOLEAN DEFAULT true,
    features JSONB DEFAULT '[]',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

    log "Core schema created"
}

# Create indexes for performance
create_indexes() {
    log "Creating performance indexes..."

    psql_cmd << 'EOF'
-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- API Keys indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

-- Requests table indexes
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_api_key_id ON requests(api_key_id);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_cache_hit ON requests(cache_hit);
CREATE INDEX IF NOT EXISTS idx_requests_pii_detected ON requests(pii_detected);
CREATE INDEX IF NOT EXISTS idx_requests_cost_usd ON requests(cost_usd);
CREATE INDEX IF NOT EXISTS idx_requests_response_time ON requests(response_time_ms);

-- Cache table indexes
CREATE INDEX IF NOT EXISTS idx_cache_cache_key ON cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model);
CREATE INDEX IF NOT EXISTS idx_cache_prompt_hash ON cache(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache(last_accessed);

-- Usage stats indexes
CREATE INDEX IF NOT EXISTS idx_usage_stats_user_id ON usage_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_stats_date ON usage_stats(date);
CREATE INDEX IF NOT EXISTS idx_usage_stats_model ON usage_stats(model);
CREATE INDEX IF NOT EXISTS idx_usage_stats_user_date_model ON usage_stats(user_id, date, model);

-- PII incidents indexes
CREATE INDEX IF NOT EXISTS idx_pii_incidents_user_id ON pii_incidents(user_id);
CREATE INDEX IF NOT EXISTS idx_pii_incidents_request_id ON pii_incidents(request_id);
CREATE INDEX IF NOT EXISTS idx_pii_incidents_entity_type ON pii_incidents(entity_type);
CREATE INDEX IF NOT EXISTS idx_pii_incidents_severity ON pii_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_pii_incidents_created_at ON pii_incidents(created_at);

-- Cost tracking indexes
CREATE INDEX IF NOT EXISTS idx_cost_tracking_user_id ON cost_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_request_id ON cost_tracking(request_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_model ON cost_tracking(model);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_created_at ON cost_tracking(created_at);

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);

-- Models indexes
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_models_is_available ON models(is_available);
EOF

    log "Performance indexes created"
}

# Create triggers for data consistency
create_triggers() {
    log "Creating database triggers..."

    psql_cmd << 'EOF'
-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_usage_stats_updated_at ON usage_stats;
CREATE TRIGGER update_usage_stats_updated_at BEFORE UPDATE ON usage_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_models_updated_at ON models;
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update API key last used timestamp
CREATE OR REPLACE FUNCTION update_api_key_last_used()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = NEW.api_key_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_api_key_last_used_trigger ON requests;
CREATE TRIGGER update_api_key_last_used_trigger AFTER INSERT ON requests
    FOR EACH ROW EXECUTE FUNCTION update_api_key_last_used();

-- Function to aggregate usage statistics
CREATE OR REPLACE FUNCTION aggregate_usage_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO usage_stats (user_id, date, model, total_requests, total_tokens, total_cost, cache_hits, cache_misses, avg_response_time_ms)
    VALUES (NEW.user_id, CURRENT_DATE, NEW.model, 1, NEW.total_tokens, NEW.cost_usd,
            CASE WHEN NEW.cache_hit THEN 1 ELSE 0 END,
            CASE WHEN NEW.cache_hit THEN 0 ELSE 1 END,
            NEW.response_time_ms)
    ON CONFLICT (user_id, date, model) DO UPDATE SET
        total_requests = usage_stats.total_requests + 1,
        total_tokens = usage_stats.total_tokens + NEW.total_tokens,
        total_cost = usage_stats.total_cost + NEW.cost_usd,
        cache_hits = usage_stats.cache_hits + CASE WHEN NEW.cache_hit THEN 1 ELSE 0 END,
        cache_misses = usage_stats.cache_misses + CASE WHEN NEW.cache_hit THEN 0 ELSE 1 END,
        avg_response_time_ms = (usage_stats.avg_response_time_ms * usage_stats.total_requests + NEW.response_time_ms) / (usage_stats.total_requests + 1),
        updated_at = CURRENT_TIMESTAMP;

    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS aggregate_usage_stats_trigger ON requests;
CREATE TRIGGER aggregate_usage_stats_trigger AFTER INSERT ON requests
    FOR EACH ROW EXECUTE FUNCTION aggregate_usage_stats();
EOF

    log "Database triggers created"
}

# Create views for reporting
create_views() {
    log "Creating reporting views..."

    psql_cmd << 'EOF'
-- User activity summary view
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT
    u.id as user_id,
    u.email,
    u.username,
    COALESCE(stats.total_requests, 0) as total_requests,
    COALESCE(stats.total_tokens, 0) as total_tokens,
    COALESCE(stats.total_cost, 0) as total_cost,
    COALESCE(stats.cache_hit_rate, 0) as cache_hit_rate,
    u.created_at as user_created_at,
    MAX(r.created_at) as last_request_date
FROM users u
LEFT JOIN (
    SELECT
        user_id,
        SUM(total_requests) as total_requests,
        SUM(total_tokens) as total_tokens,
        SUM(total_cost) as total_cost,
        CASE
            WHEN SUM(cache_hits + cache_misses) > 0
            THEN ROUND(SUM(cache_hits)::decimal / SUM(cache_hits + cache_misses) * 100, 2)
            ELSE 0
        END as cache_hit_rate
    FROM usage_stats
    GROUP BY user_id
) stats ON u.id = stats.user_id
LEFT JOIN requests r ON u.id = r.user_id
GROUP BY u.id, u.email, u.username, stats.total_requests, stats.total_tokens, stats.total_cost, stats.cache_hit_rate, u.created_at;

-- Model performance summary view
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    m.name,
    m.provider,
    m.display_name,
    COUNT(r.id) as total_requests,
    COALESCE(SUM(r.total_tokens), 0) as total_tokens,
    COALESCE(SUM(r.cost_usd), 0) as total_cost,
    COALESCE(AVG(r.response_time_ms), 0) as avg_response_time_ms,
    COUNT(CASE WHEN r.cache_hit THEN 1 END) as cache_hits,
    COUNT(CASE WHEN NOT r.cache_hit THEN 1 END) as cache_misses,
    CASE
        WHEN COUNT(r.id) > 0
        THEN ROUND(COUNT(CASE WHEN r.cache_hit THEN 1 END)::decimal / COUNT(r.id) * 100, 2)
        ELSE 0
    END as cache_hit_rate,
    COUNT(CASE WHEN r.pii_detected THEN 1 END) as pii_incidents
FROM models m
LEFT JOIN requests r ON m.name = r.model
GROUP BY m.id, m.name, m.provider, m.display_name
ORDER BY total_requests DESC;

-- Daily cost tracking view
CREATE OR REPLACE VIEW daily_cost_summary AS
SELECT
    DATE(r.created_at) as date,
    COUNT(r.id) as total_requests,
    COALESCE(SUM(r.cost_usd), 0) as total_cost,
    COUNT(DISTINCT r.user_id) as active_users,
    COUNT(DISTINCT r.model) as models_used,
    COUNT(CASE WHEN r.cache_hit THEN 1 END) as cache_hits,
    COUNT(CASE WHEN NOT r.cache_hit THEN 1 END) as cache_misses,
    COALESCE(AVG(r.response_time_ms), 0) as avg_response_time_ms
FROM requests r
GROUP BY DATE(r.created_at)
ORDER BY date DESC;

-- PII incidents summary view
CREATE OR REPLACE VIEW pii_incidents_summary AS
SELECT
    DATE(pi.created_at) as date,
    pi.entity_type,
    COUNT(pi.id) as incident_count,
    COUNT(DISTINCT pi.user_id) as affected_users,
    AVG(pi.confidence_score) as avg_confidence,
    COUNT(CASE WHEN pi.severity = 'high' THEN 1 END) as high_severity,
    COUNT(CASE WHEN pi.severity = 'medium' THEN 1 END) as medium_severity,
    COUNT(CASE WHEN pi.severity = 'low' THEN 1 END) as low_severity
FROM pii_incidents pi
GROUP BY DATE(pi.created_at), pi.entity_type
ORDER BY date DESC, incident_count DESC;
EOF

    log "Reporting views created"
}

# Main execution
main() {
    log "Starting Helix database migration..."

    # Wait for PostgreSQL to be ready
    wait_for_postgres

    # Create database if it doesn't exist
    create_database

    # Create schema
    create_helix_schema

    # Run migrations
    run_migrations

    # Create performance indexes
    create_indexes

    # Create triggers
    create_triggers

    # Create views
    create_views

    # Get database info for verification
    local db_info=$(psql_cmd -t -c "SELECT version();" | tr -d '\n' | sed 's/^[[:space:]]*//')
    local table_count=$(psql_cmd -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d '\n' | sed 's/^[[:space:]]*//')
    local index_count=$(psql_cmd -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';" | tr -d '\n' | sed 's/^[[:space:]]*//')

    log "Database migration completed successfully!"
    log "PostgreSQL version: ${db_info%%,*}"
    log "Tables created: $table_count"
    log "Indexes created: $index_count"
    log "Ready for Helix AI Gateway"
}

# Execute main function
main "$@"