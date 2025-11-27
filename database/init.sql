-- Helix AI Gateway Database Initialization
-- Extends LiteLLM PostgreSQL schema with Helix-specific features

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schema for Helix if it doesn't exist
CREATE SCHEMA IF NOT EXISTS helix;

-- Helix Cache Performance Metrics
CREATE TABLE IF NOT EXISTS helix.cache_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    cache_type VARCHAR(50) NOT NULL, -- "semantic", "exact", "hybrid"
    hit_count BIGINT DEFAULT 0,
    miss_count BIGINT DEFAULT 0,
    hit_rate FLOAT DEFAULT 0.0,
    avg_lookup_time FLOAT DEFAULT 0.0, -- milliseconds
    total_tokens_saved BIGINT DEFAULT 0,
    cost_savings FLOAT DEFAULT 0.0,
    last_access TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadata for cache entries
    model_name VARCHAR(100),
    request_hash VARCHAR(255),
    similarity_score FLOAT, -- For semantic cache hits
    ttl_seconds INTEGER
);

-- Create indexes for cache_metrics
CREATE INDEX IF NOT EXISTS idx_cache_metrics_cache_type ON helix.cache_metrics(cache_type);
CREATE INDEX IF NOT EXISTS idx_cache_metrics_hit_rate ON helix.cache_metrics(hit_rate);
CREATE INDEX IF NOT EXISTS idx_cache_metrics_last_access ON helix.cache_metrics(last_access);

-- Helix PII Detection and Redaction Logs
CREATE TABLE IF NOT EXISTS helix.pii_detection_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    original_text_hash VARCHAR(255) UNIQUE NOT NULL, -- Hash of original text for audit

    -- PII Detection Results
    pii_entities JSONB, -- Detected PII entities with types and positions
    pii_count INTEGER DEFAULT 0,
    pii_types TEXT[], -- Array of PII types detected
    avg_confidence FLOAT DEFAULT 0.0,

    -- Redaction Information
    redaction_method VARCHAR(50), -- "replace", "mask", "hash", "encrypt"
    redaction_applied BOOLEAN DEFAULT FALSE,
    redaction_time_ms INTEGER DEFAULT 0,

    -- Processing Metrics
    total_processing_time_ms INTEGER DEFAULT 0,
    text_length INTEGER DEFAULT 0,
    pii_density FLOAT DEFAULT 0.0, -- PII characters / total characters

    -- Model Information
    analyzer_version VARCHAR(50),
    recognizers_used TEXT[],
    custom_recognizers TEXT[],

    -- Audit Trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Create indexes for pii_detection_logs
CREATE INDEX IF NOT EXISTS idx_pii_logs_request_id ON helix.pii_detection_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_pii_logs_redaction_applied ON helix.pii_detection_logs(redaction_applied);
CREATE INDEX IF NOT EXISTS idx_pii_logs_pii_count ON helix.pii_detection_logs(pii_count);
CREATE INDEX IF NOT EXISTS idx_pii_logs_created_at ON helix.pii_detection_logs(created_at);

-- Helix Cost Tracking and Optimization
CREATE TABLE IF NOT EXISTS helix.cost_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) UNIQUE NOT NULL,

    -- Request Information
    model_name VARCHAR(100) NOT NULL,
    provider VARCHAR(50), -- "openai", "anthropic", etc.
    endpoint VARCHAR(255),

    -- Token Usage
    input_tokens BIGINT,
    output_tokens BIGINT,
    total_tokens BIGINT,
    cached_tokens BIGINT DEFAULT 0,

    -- Cost Calculation
    original_cost FLOAT NOT NULL, -- Cost without Helix optimizations
    actual_cost FLOAT NOT NULL, -- Actual cost with optimizations
    cache_savings FLOAT DEFAULT 0.0,
    optimization_savings FLOAT DEFAULT 0.0,
    total_savings FLOAT DEFAULT 0.0,
    savings_rate FLOAT DEFAULT 0.0, -- Percentage saved

    -- Cost Breakdown
    input_cost FLOAT DEFAULT 0.0,
    output_cost FLOAT DEFAULT 0.0,
    cache_cost FLOAT DEFAULT 0.0,

    -- Pricing Information
    price_per_mil_input FLOAT,
    price_per_mil_output FLOAT,
    currency VARCHAR(3) DEFAULT 'USD',

    -- Budget Information
    budget_id UUID REFERENCES helix.budgets(id),
    organization_id VARCHAR(255),
    team_id VARCHAR(255),
    user_id VARCHAR(255),

    -- Performance Metrics
    latency_ms INTEGER,
    cache_lookup_time_ms INTEGER,
    processing_overhead_ms INTEGER,

    -- Optimization Details
    optimization_strategy VARCHAR(50), -- "cache_hit", "model_swap", "prompt_optimization"
    alternative_model VARCHAR(100), -- If model was swapped for optimization

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for cost_tracking
CREATE INDEX IF NOT EXISTS idx_cost_tracking_request_id ON helix.cost_tracking(request_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_model_name ON helix.cost_tracking(model_name);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_organization_id ON helix.cost_tracking(organization_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_team_id ON helix.cost_tracking(team_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_user_id ON helix.cost_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_created_at ON helix.cost_tracking(created_at);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_savings_rate ON helix.cost_tracking(savings_rate);

-- Helix Enhanced Budget Management
CREATE TABLE IF NOT EXISTS helix.budgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Basic Budget Info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL, -- "monthly", "quarterly", "yearly", "per_project"

    -- Budget Limits
    max_budget FLOAT NOT NULL,
    soft_budget FLOAT, -- Warning threshold
    daily_limit FLOAT, -- Daily spending limit
    per_request_limit FLOAT, -- Maximum per request

    -- Budget Tracking
    current_spend FLOAT DEFAULT 0.0,
    forecasted_spend FLOAT DEFAULT 0.0,
    cache_savings FLOAT DEFAULT 0.0,
    optimization_savings FLOAT DEFAULT 0.0,

    -- Budget Period
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    reset_date TIMESTAMP WITH TIME ZONE,

    -- Cost Breakdown
    model_spend JSONB DEFAULT '{}', -- Spend by model
    provider_spend JSONB DEFAULT '{}', -- Spend by provider
    daily_spend JSONB DEFAULT '{}', -- Daily spend breakdown

    -- Controls
    auto_block_on_limit BOOLEAN DEFAULT FALSE,
    alert_thresholds JSONB DEFAULT '{}', -- Percentage-based alerts

    -- Relations
    organization_id VARCHAR(255) UNIQUE,
    team_id VARCHAR(255) UNIQUE,
    user_id VARCHAR(255) UNIQUE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- Create indexes for budgets
CREATE INDEX IF NOT EXISTS idx_budgets_organization_id ON helix.budgets(organization_id);
CREATE INDEX IF NOT EXISTS idx_budgets_team_id ON helix.budgets(team_id);
CREATE INDEX IF NOT EXISTS idx_budgets_user_id ON helix.budgets(user_id);
CREATE INDEX IF NOT EXISTS idx_budgets_start_date ON helix.budgets(start_date);
CREATE INDEX IF NOT EXISTS idx_budgets_end_date ON helix.budgets(end_date);

-- Helix Budget Alerting
CREATE TABLE IF NOT EXISTS helix.budget_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    budget_id UUID NOT NULL REFERENCES helix.budgets(id) ON DELETE CASCADE,

    -- Alert Configuration
    alert_type VARCHAR(50) NOT NULL, -- "percentage", "absolute", "forecast"
    threshold_value FLOAT NOT NULL,
    trigger_condition VARCHAR(20) NOT NULL, -- "greater_than", "equal_to", "less_than"

    -- Alert Status
    is_active BOOLEAN DEFAULT TRUE,
    last_triggered TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0,

    -- Alert Methods
    email_enabled BOOLEAN DEFAULT FALSE,
    email_recipients TEXT[],
    slack_enabled BOOLEAN DEFAULT FALSE,
    slack_webhook VARCHAR(255),
    webhook_enabled BOOLEAN DEFAULT FALSE,
    webhook_url VARCHAR(255),

    -- Alert Message Templates
    email_template TEXT,
    slack_template TEXT,
    webhook_template TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- Create indexes for budget_alerts
CREATE INDEX IF NOT EXISTS idx_budget_alerts_budget_id ON helix.budget_alerts(budget_id);
CREATE INDEX IF NOT EXISTS idx_budget_alerts_is_active ON helix.budget_alerts(is_active);

-- Helix Central Request Log (Enhanced with Helix features)
CREATE TABLE IF NOT EXISTS helix.request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Request Identification
    request_hash VARCHAR(255) UNIQUE NOT NULL,
    trace_id VARCHAR(255), -- For distributed tracing
    correlation_id VARCHAR(255), -- For related requests

    -- Authentication & Authorization
    api_key VARCHAR(255),
    user_id VARCHAR(255),
    team_id VARCHAR(255),
    organization_id VARCHAR(255),

    -- Request Details
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    provider VARCHAR(50),

    -- Request Content (Sanitized)
    request_summary TEXT, -- Hashed/redacted summary
    input_length INTEGER,
    output_length INTEGER,

    -- Performance Metrics
    latency_ms INTEGER NOT NULL,
    cache_lookup_time_ms INTEGER,
    pii_processing_time_ms INTEGER,
    total_processing_time_ms INTEGER,

    -- Cache Information
    cache_hit BOOLEAN DEFAULT FALSE,
    cache_type VARCHAR(50), -- "exact", "semantic", "hybrid"
    cache_similarity FLOAT, -- For semantic cache hits

    -- PII Information
    pii_detected BOOLEAN DEFAULT FALSE,
    pii_count INTEGER DEFAULT 0,
    pii_redaction_applied BOOLEAN DEFAULT FALSE,

    -- Request Outcome
    success BOOLEAN NOT NULL,
    error_code VARCHAR(50),
    error_message TEXT,
    response_code INTEGER,

    -- Quality Metrics
    response_quality_score FLOAT,
    user_satisfaction FLOAT, -- If tracked

    -- Flags and Labels
    flags TEXT[] DEFAULT ARRAY[]::TEXT[],
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Environment
    environment VARCHAR(50), -- "production", "staging", "development"
    region VARCHAR(50),

    -- Relations
    cache_metric_id UUID REFERENCES helix.cache_metrics(id),
    pii_detection_log_id UUID REFERENCES helix.pii_detection_logs(id),
    cost_tracking_id UUID REFERENCES helix.cost_tracking(id),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for request_logs
CREATE INDEX IF NOT EXISTS idx_request_logs_request_hash ON helix.request_logs(request_hash);
CREATE INDEX IF NOT EXISTS idx_request_logs_api_key ON helix.request_logs(api_key);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON helix.request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_team_id ON helix.request_logs(team_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_organization_id ON helix.request_logs(organization_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_model_name ON helix.request_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_request_logs_success ON helix.request_logs(success);
CREATE INDEX IF NOT EXISTS idx_request_logs_cache_hit ON helix.request_logs(cache_hit);
CREATE INDEX IF NOT EXISTS idx_request_logs_pii_detected ON helix.request_logs(pii_detected);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON helix.request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_trace_id ON helix.request_logs(trace_id);

-- Helix Performance Monitoring (Time-series data)
CREATE TABLE IF NOT EXISTS helix.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Time Window
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    time_window VARCHAR(20) NOT NULL, -- "1m", "5m", "15m", "1h", "1d"

    -- Request Metrics
    total_requests BIGINT DEFAULT 0,
    successful_requests BIGINT DEFAULT 0,
    failed_requests BIGINT DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,

    -- Cache Metrics
    cache_hits BIGINT DEFAULT 0,
    cache_misses BIGINT DEFAULT 0,
    cache_hit_rate FLOAT DEFAULT 0.0,
    avg_cache_lookup_ms FLOAT DEFAULT 0.0,

    -- PII Metrics
    pii_detection_count BIGINT DEFAULT 0,
    pii_redaction_count BIGINT DEFAULT 0,
    avg_pii_processing_ms FLOAT DEFAULT 0.0,

    -- Cost Metrics
    total_cost FLOAT DEFAULT 0.0,
    cache_savings FLOAT DEFAULT 0.0,
    optimization_savings FLOAT DEFAULT 0.0,
    avg_cost_per_request FLOAT DEFAULT 0.0,

    -- Performance Metrics
    avg_latency_ms FLOAT DEFAULT 0.0,
    p50_latency_ms FLOAT DEFAULT 0.0,
    p95_latency_ms FLOAT DEFAULT 0.0,
    p99_latency_ms FLOAT DEFAULT 0.0,

    -- Model Breakdown
    model_metrics JSONB DEFAULT '{}', -- Metrics by model
    organization_metrics JSONB DEFAULT '{}', -- Metrics by organization

    -- Indexes for efficient querying
    CONSTRAINT unique_performance_window UNIQUE(timestamp, time_window)
);

-- Create indexes for performance_metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON helix.performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_time_window ON helix.performance_metrics(time_window);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_success_rate ON helix.performance_metrics(success_rate);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_cache_hit_rate ON helix.performance_metrics(cache_hit_rate);

-- Helix System Configuration
CREATE TABLE IF NOT EXISTS helix.system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Configuration Key-Value pairs
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) NOT NULL, -- "string", "number", "boolean", "object", "array"

    -- Configuration Metadata
    description TEXT,
    category VARCHAR(100), -- "caching", "pii", "cost", "monitoring"
    is_sensitive BOOLEAN DEFAULT FALSE,

    -- Environment-specific values
    environment_overrides JSONB DEFAULT '{}', -- {"production": "...", "staging": "..."}

    -- Validation Rules
    validation_rules JSONB, -- JSON schema for validation
    allowed_values TEXT[],

    -- Access Control
    read_roles TEXT[] DEFAULT ARRAY[]::TEXT[],
    write_roles TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Audit Trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- Create indexes for system_config
CREATE INDEX IF NOT EXISTS idx_system_config_config_key ON helix.system_config(config_key);
CREATE INDEX IF NOT EXISTS idx_system_config_category ON helix.system_config(category);

-- Helix Audit Log for Compliance
CREATE TABLE IF NOT EXISTS helix.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Event Information
    event_type VARCHAR(100) NOT NULL, -- "config_change", "user_action", "system_event"
    event_category VARCHAR(50) NOT NULL, -- "security", "performance", "compliance", "cost"
    action VARCHAR(50) NOT NULL, -- "create", "update", "delete", "view", "execute"

    -- Resource Information
    resource_type VARCHAR(100), -- "budget", "user", "config", "recognizer"
    resource_id VARCHAR(255),
    resource_name VARCHAR(255),

    -- User Information
    user_id VARCHAR(255),
    user_email VARCHAR(255),
    user_role VARCHAR(100),

    -- Request Context
    request_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,

    -- Change Details
    old_value JSONB,
    new_value JSONB,
    change_summary TEXT,

    -- Compliance Information
    compliance_reason VARCHAR(100), -- "data_protection", "financial", "security"
    retention_required BOOLEAN DEFAULT FALSE,
    retention_until TIMESTAMP WITH TIME ZONE,

    -- System Information
    service_name VARCHAR(100),
    environment VARCHAR(50),
    region VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON helix.audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_category ON helix.audit_logs(event_category);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON helix.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type ON helix.audit_logs(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON helix.audit_logs(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION helix.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_cache_metrics_updated_at BEFORE UPDATE ON helix.cache_metrics
    FOR EACH ROW EXECUTE FUNCTION helix.update_updated_at_column();

CREATE TRIGGER update_budgets_updated_at BEFORE UPDATE ON helix.budgets
    FOR EACH ROW EXECUTE FUNCTION helix.update_updated_at_column();

CREATE TRIGGER update_budget_alerts_updated_at BEFORE UPDATE ON helix.budget_alerts
    FOR EACH ROW EXECUTE FUNCTION helix.update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON helix.system_config
    FOR EACH ROW EXECUTE FUNCTION helix.update_updated_at_column();

-- Insert default system configuration
INSERT INTO helix.system_config (config_key, config_value, config_type, description, category) VALUES
('helix_version', '"1.0.0"', 'string', 'Helix AI Gateway version', 'system'),
('cache_default_ttl', '3600', 'number', 'Default cache TTL in seconds', 'caching'),
('cache_max_memory_usage', '4294967296', 'number', 'Maximum cache memory usage in bytes', 'caching'),
('pii_confidence_threshold', '0.8', 'number', 'PII detection confidence threshold', 'pii'),
('cost_optimization_enabled', 'true', 'boolean', 'Enable cost optimization features', 'cost'),
('monitoring_metrics_interval', '60', 'number', 'Metrics collection interval in seconds', 'monitoring')
ON CONFLICT (config_key) DO NOTHING;

-- Create view for active budgets with current usage
CREATE OR REPLACE VIEW helix.active_budgets AS
SELECT
    b.*,
    (b.current_spend / b.max_budget * 100) as usage_percentage,
    CASE
        WHEN b.current_spend > b.max_budget THEN 'over_limit'
        WHEN b.current_spend > (b.max_budget * 0.9) THEN 'critical'
        WHEN b.current_spend > (b.max_budget * 0.7) THEN 'warning'
        ELSE 'healthy'
    END as status,
    CASE
        WHEN b.end_date < NOW() THEN 'expired'
        WHEN b.end_date < (NOW() + INTERVAL '7 days') THEN 'expiring_soon'
        ELSE 'active'
    END as time_status
FROM helix.budgets b
WHERE b.start_date <= NOW()
  AND (b.end_date IS NULL OR b.end_date >= NOW())
  AND b.max_budget > 0;

-- Grant permissions (adjust as needed for your setup)
-- GRANT USAGE ON SCHEMA helix TO litellm_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA helix TO litellm_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA helix TO litellm_user;

COMMIT;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Helix AI Gateway database initialized successfully';
    RAISE NOTICE 'Schema: helix';
    RAISE NOTICE 'Tables: cache_metrics, pii_detection_logs, cost_tracking, budgets, budget_alerts, request_logs, performance_metrics, system_config, audit_logs';
    RAISE NOTICE 'Views: active_budgets';
END $$;