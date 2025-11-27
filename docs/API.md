# Helix API Documentation
**Complete API Reference for Helix AI Gateway**

## Overview

Helix provides a 100% OpenAI-compatible API that adds enterprise-grade features including intelligent caching, PII protection, and cost optimization. Simply change your OpenAI API endpoint to Helix and gain immediate benefits.

### Base URL
```
Production: https://your-helix-domain.com/v1
Staging: https://staging-helix-domain.com/v1
Local: http://localhost:4000/v1
```

### Authentication
```bash
# All endpoints accept OpenAI-style API keys
Authorization: Bearer your-api-key

# Helix-specific headers (optional)
X-Helix-Cache-Strategy: semantic  # exact | semantic | hybrid
X-Helix-PII-Protection: true     # true | false
X-Helix-Cost-Optimization: true  # true | false
```

## Core Endpoints

### 1. Chat Completions

**Endpoint**: `POST /v1/chat/completions`

OpenAI-compatible chat completions with Helix enhancements.

#### Request Parameters

```json
{
  "model": "gpt-4o",                    // Required: Model name
  "messages": [                          // Required: Message array
    {
      "role": "system",                 // system | user | assistant
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "max_tokens": 1000,                   // Optional: Max completion tokens
  "temperature": 0.7,                    // Optional: 0.0-2.0
  "top_p": 1,                           // Optional: 0.0-1.0
  "stream": false,                      // Optional: Enable streaming
  "stop": ["\n"],                       // Optional: Stop sequences
  "presence_penalty": 0,                 // Optional: -2.0 to 2.0
  "frequency_penalty": 0,                // Optional: -2.0 to 2.0

  // Helix-specific parameters
  "helix": {
    "cache_strategy": "semantic",       // exact | semantic | hybrid | none
    "pii_protection": true,             // true | false
    "cost_optimization": true,          // true | false
    "user_id": "user_123",              // Optional: User identifier
    "cache_ttl": 2592000,               // Optional: Cache TTL in seconds
    "similarity_threshold": 0.88        // Optional: Semantic similarity
  }
}
```

#### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing harnesses..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  },

  // Helix-specific response metadata
  "helix_metadata": {
    "cache_hit": true,                   // Whether response came from cache
    "cache_type": "semantic",            // exact | semantic | none
    "pii_processed": true,                // Whether PII was detected/processed
    "pii_incidents": [                    // PII incidents detected
      {
        "entity_type": "EMAIL_ADDRESS",
        "confidence": 0.95,
        "position_start": 10,
        "position_end": 25
      }
    ],
    "original_cost_usd": 0.00234,        // Cost without caching
    "actual_cost_usd": 0.00000,          // Actual cost after caching
    "savings_usd": 0.00234,             // Money saved
    "latency_ms": 45,                    // Total processing time
    "provider_used": "openai",           // Actual provider used
    "model_swapped": false,              // Whether model was swapped for cost
    "similarity_score": 0.92             // For semantic cache hits
  }
}
```

#### Examples

**Basic Request**:
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

**With Helix Features**:
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "My email is john@example.com"}],
    "helix": {
      "cache_strategy": "semantic",
      "pii_protection": true,
      "cost_optimization": true
    }
  }'
```

**Streaming Request**:
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### 2. Completions (Legacy)

**Endpoint**: `POST /v1/completions`

Legacy completions endpoint for backward compatibility.

#### Request Parameters

```json
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1,
  "n": 1,
  "stream": false,
  "logprobs": null,
  "stop": ["\n"],

  // Helix-specific parameters
  "helix": {
    "cache_strategy": "hybrid",
    "pii_protection": true,
    "cost_optimization": true
  }
}
```

#### Response Format

```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-instruct",
  "choices": [
    {
      "text": " incredibly exciting and full of possibilities.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 9,
    "total_tokens": 17
  },

  "helix_metadata": {
    "cache_hit": false,
    "cache_type": "none",
    "savings_usd": 0.00000,
    "latency_ms": 1200
  }
}
```

### 3. Embeddings

**Endpoint**: `POST /v1/embeddings`

Generate embeddings with intelligent caching and optimization.

#### Request Parameters

```json
{
  "model": "text-embedding-ada-002",
  "input": [
    "Hello world",
    "This is a test"
  ],
  "encoding_format": "float",             // float | base64
  "dimensions": 1536,                     // Optional: For models that support it
  "user": "user_123",                     // Optional: User identifier

  // Helix-specific parameters
  "helix": {
    "cache_strategy": "exact",             // Semantic caching for embeddings
    "batch_optimization": true,           // Optimize multiple inputs
    "vector_compression": true            // Compress stored embeddings
  }
}
```

#### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0017, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.0045, -0.0032, ...],
      "index": 1
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  },

  "helix_metadata": {
    "cache_hits": 1,                      // Number of cached embeddings
    "compression_ratio": 0.65,            // Vector compression efficiency
    "savings_usd": 0.00045
  }
}
```

## Helix Management Endpoints

### 4. Cache Management

#### Get Cache Status

**Endpoint**: `GET /v1/helix/cache/status`

Get comprehensive cache statistics and performance metrics.

```bash
curl -X GET http://localhost:4000/v1/helix/cache/status \
  -H "Authorization: Bearer your-api-key"
```

**Response**:
```json
{
  "exact_cache": {
    "total_entries": 125000,
    "memory_usage_mb": 2048,
    "hit_rate": 0.73,
    "avg_lookup_time_ms": 2.1,
    "ttl_distribution": {
      "1_hour": 15000,
      "1_day": 45000,
      "30_days": 65000
    }
  },
  "semantic_cache": {
    "total_entries": 87500,
    "memory_usage_mb": 4096,
    "hit_rate": 0.64,
    "avg_search_time_ms": 12.3,
    "similarity_threshold": 0.88,
    "avg_similarity_score": 0.91
  },
  "combined_metrics": {
    "overall_hit_rate": 0.71,
    "total_savings_usd": 1254.32,
    "cache_efficiency_score": 0.89
  }
}
```

#### Clear Cache

**Endpoint**: `DELETE /v1/helix/cache`

Clear cache entries with various filters.

```bash
# Clear all cache
curl -X DELETE http://localhost:4000/v1/helix/cache \
  -H "Authorization: Bearer your-api-key"

# Clear specific model cache
curl -X DELETE http://localhost:4000/v1/helix/cache?model=gpt-4o \
  -H "Authorization: Bearer your-api-key"

# Clear expired entries only
curl -X DELETE http://localhost:4000/v1/helix/cache?expired_only=true \
  -H "Authorization: Bearer your-api-key"

# Clear entries older than specified time
curl -X DELETE http://localhost:4000/v1/helix/cache?older_than=7d \
  -H "Authorization: Bearer your-api-key"
```

#### Warm Cache

**Endpoint**: `POST /v1/helix/cache/warm`

Pre-populate cache with common queries.

```bash
curl -X POST http://localhost:4000/v1/helix/cache/warm \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"prompt": "What is machine learning?", "model": "gpt-4o"},
      {"prompt": "Explain neural networks", "model": "gpt-3.5-turbo"}
    ],
    "priority": "high"  # high | medium | low
  }'
```

### 5. PII Management

#### Get PII Statistics

**Endpoint**: `GET /v1/helix/pii/stats`

Get PII detection statistics and incident reports.

```bash
curl -X GET http://localhost:4000/v1/helix/pii/stats \
  -H "Authorization: Bearer your-api-key"
```

**Response**:
```json
{
  "detection_statistics": {
    "total_requests_processed": 50000,
    "requests_with_pii": 1250,
    "pii_detection_rate": 0.025,
    "false_positive_rate": 0.003,
    "avg_processing_time_ms": 4.2
  },
  "entity_types_detected": {
    "EMAIL_ADDRESS": 450,
    "PHONE_NUMBER": 320,
    "CREDIT_CARD": 180,
    "SSN": 95,
    "API_KEY": 125,
    "PASSWORD": 80
  },
  "recent_incidents": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "user_id": "user_123",
      "entity_type": "EMAIL_ADDRESS",
      "confidence": 0.94,
      "action_taken": "redacted",
      "request_id": "req_456"
    }
  ],
  "compliance_status": {
    "gdpr_compliant": true,
    "hipaa_compliant": true,
    "soc2_compliant": true,
    "last_audit_date": "2024-01-01T00:00:00Z"
  }
}
```

#### Update PII Configuration

**Endpoint**: `PUT /v1/helix/pii/config`

Update PII detection and protection settings.

```bash
curl -X PUT http://localhost:4000/v1/helix/pii/config \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled_entities": [
      "EMAIL_ADDRESS",
      "PHONE_NUMBER",
      "CREDIT_CARD",
      "SSN",
      "API_KEY",
      "PASSWORD",
      "BANK_ACCOUNT",
      "PASSPORT_NUMBER"
    ],
    "detection_threshold": 0.85,
    "action": "redact",           // redact | block | log_only
    "strict_mode": true,          // Block requests with high-confidence PII
    "custom_patterns": [
      {
        "name": "INTERNAL_ID",
        "pattern": "INT-\\d{8}",
        "confidence": 0.9
      }
    ]
  }'
```

### 6. Cost Management

#### Get Cost Analytics

**Endpoint**: `GET /v1/helix/cost/analytics`

Get comprehensive cost analysis and optimization insights.

```bash
# Get cost analytics for date range
curl -X GET "http://localhost:4000/v1/helix/cost/analytics?start_date=2024-01-01&end_date=2024-01-31" \
  -H "Authorization: Bearer your-api-key"

# Get cost analytics for specific user
curl -X GET "http://localhost:4000/v1/helix/cost/analytics?user_id=user_123&period=30d" \
  -H "Authorization: Bearer your-api-key"
```

**Response**:
```json
{
  "summary": {
    "period_start": "2024-01-01T00:00:00Z",
    "period_end": "2024-01-31T23:59:59Z",
    "total_requests": 25000,
    "total_cost_usd": 1250.75,
    "savings_from_caching_usd": 487.30,
    "savings_from_model_swapping_usd": 234.15,
    "net_spend_usd": 529.30,
    "savings_percentage": 57.7
  },
  "breakdown_by_model": {
    "gpt-4o": {
      "requests": 15000,
      "cost_usd": 875.50,
      "savings_usd": 342.10,
      "avg_cost_per_request": 0.058
    },
    "gpt-3.5-turbo": {
      "requests": 10000,
      "cost_usd": 375.25,
      "savings_usd": 145.20,
      "avg_cost_per_request": 0.038
    }
  },
  "optimization_recommendations": [
    {
      "type": "model_swapping",
      "description": "Switch 30% of simple queries from gpt-4o to gpt-3.5-turbo",
      "potential_savings_usd": 125.50,
      "confidence": 0.89
    },
    {
      "type": "cache_optimization",
      "description": "Increase similarity threshold for semantic cache",
      "potential_savings_usd": 45.25,
      "confidence": 0.76
    }
  ]
}
```

#### Set Budget Alerts

**Endpoint**: `POST /v1/helix/cost/budget-alerts`

Configure budget monitoring and alerting.

```bash
curl -X POST http://localhost:4000/v1/helix/cost/budget-alerts \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "daily_limit_usd": 10.00,
    "monthly_limit_usd": 250.00,
    "alert_thresholds": [
      {
        "percentage": 50,
        "action": "email",
        "recipients": ["admin@company.com"]
      },
      {
        "percentage": 80,
        "action": "webhook",
        "webhook_url": "https://api.company.com/alerts"
      },
      {
        "percentage": 100,
        "action": "block_requests",
        "message": "Budget limit exceeded"
      }
    ]
  }'
```

### 7. Performance Monitoring

#### Get Performance Metrics

**Endpoint**: `GET /v1/helix/performance/metrics`

Get detailed performance metrics and system health.

```bash
curl -X GET http://localhost:4000/v1/helix/performance/metrics \
  -H "Authorization: Bearer your-api-key"
```

**Response**:
```json
{
  "system_metrics": {
    "uptime_seconds": 2592000,
    "requests_per_second": 125.5,
    "active_connections": 45,
    "memory_usage_mb": 4096,
    "cpu_utilization_percent": 67.3
  },
  "performance_metrics": {
    "avg_latency_ms": 245.6,
    "p50_latency_ms": 180.2,
    "p95_latency_ms": 580.9,
    "p99_latency_ms": 1245.3,
    "cache_hit_latency_ms": 15.7,
    "pii_processing_latency_ms": 8.2
  },
  "error_metrics": {
    "error_rate_percent": 0.3,
    "timeout_rate_percent": 0.1,
    "rate_limit_rate_percent": 0.05,
    "provider_errors_by_type": {
      "openai_timeout": 15,
      "anthropic_rate_limit": 8,
      "groq_overloaded": 3
    }
  },
  "optimization_metrics": {
    "cache_hit_rate_percent": 71.2,
    "semantic_cache_hit_rate_percent": 64.8,
    "model_swap_rate_percent": 23.5,
    "cost_savings_rate_percent": 57.7
  }
}
```

#### Health Check

**Endpoint**: `GET /v1/helix/health`

Comprehensive health check of all system components.

```bash
curl -X GET http://localhost:4000/v1/helix/health \
  -H "Authorization: Bearer your-api-key"
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "api_gateway": {
      "status": "healthy",
      "response_time_ms": 2,
      "last_check": "2024-01-15T10:30:00Z"
    },
    "redis_cache": {
      "status": "healthy",
      "response_time_ms": 5,
      "memory_usage_percent": 65.2,
      "last_check": "2024-01-15T10:30:00Z"
    },
    "semantic_search": {
      "status": "healthy",
      "index_health": 0.98,
      "response_time_ms": 12,
      "last_check": "2024-01-15T10:30:00Z"
    },
    "pii_detector": {
      "status": "healthy",
      "models_loaded": true,
      "response_time_ms": 8,
      "last_check": "2024-01-15T10:30:00Z"
    },
    "database": {
      "status": "healthy",
      "connection_pool": "healthy",
      "response_time_ms": 15,
      "last_check": "2024-01-15T10:30:00Z"
    },
    "external_providers": {
      "openai": {
        "status": "healthy",
        "response_time_ms": 450,
        "last_check": "2024-01-15T10:30:00Z"
      },
      "anthropic": {
        "status": "healthy",
        "response_time_ms": 520,
        "last_check": "2024-01-15T10:30:00Z"
      }
    }
  }
}
```

## WebSocket Endpoints

### Real-time Monitoring

**Endpoint**: `WS /v1/helix/monitoring/stream`

Real-time streaming of metrics and events.

```javascript
const ws = new WebSocket('ws://localhost:4000/v1/helix/monitoring/stream');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  api_key: 'your-api-key'
}));

// Subscribe to metrics
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: [
    'metrics.requests_per_second',
    'metrics.latency',
    'metrics.cache_hit_rate',
    'metrics.cost_savings',
    'events.pii_incidents',
    'alerts.budget_warnings'
  ]
}));

// Receive updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

**Message Format**:
```json
{
  "type": "metric_update",
  "channel": "metrics.requests_per_second",
  "timestamp": "2024-01-15T10:30:00Z",
  "value": 125.5,
  "metadata": {
    "model": "gpt-4o",
    "cache_hit": true
  }
}
```

## SDKs and Client Libraries

### Python SDK

```python
from helix_sdk import HelixClient

client = HelixClient(
    api_key="your-api-key",
    base_url="http://localhost:4000/v1"
)

# Basic chat completion
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    helix_options={
        "cache_strategy": "semantic",
        "pii_protection": True,
        "cost_optimization": True
    }
)

print(response.choices[0].message.content)
print(response.helix_metadata.savings_usd)

# Cache management
cache_status = await client.cache.get_status()
await client.cache.clear(model="gpt-4o")

# PII configuration
await client.pii.update_config({
    "enabled_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
    "action": "redact"
})

# Cost analytics
analytics = await client.cost.get_analytics(
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

### JavaScript/TypeScript SDK

```typescript
import { HelixClient } from '@helix-ai/gateway-sdk';

const client = new HelixClient({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:4000/v1'
});

// Basic chat completion
const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
  helixOptions: {
    cacheStrategy: 'semantic',
    piiProtection: true,
    costOptimization: true
  }
});

console.log(response.choices[0].message.content);
console.log(response.helixMetadata.savingsUsd);

// Streaming completion
const stream = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Tell me a story' }],
  stream: true
});

for await (const chunk of stream) {
  console.log(chunk.choices[0]?.delta?.content);
}

// Real-time monitoring
const monitoring = await client.monitoring.stream({
  channels: ['metrics.requests_per_second', 'events.pii_incidents']
});

monitoring.on('update', (data) => {
  console.log('Real-time update:', data);
});
```

### cURL Examples

```bash
# Basic chat completion
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'

# With all Helix features
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "My email is john@example.com and phone is 555-0123"}],
    "helix": {
      "cache_strategy": "semantic",
      "pii_protection": true,
      "cost_optimization": true,
      "user_id": "user_123"
    }
  }'

# Get cost analytics
curl -X GET "http://localhost:4000/v1/helix/cost/analytics?period=7d" \
  -H "Authorization: Bearer your-api-key"

# Get cache status
curl -X GET http://localhost:4000/v1/helix/cache/status \
  -H "Authorization: Bearer your-api-key"

# Health check
curl -X GET http://localhost:4000/v1/helix/health \
  -H "Authorization: Bearer your-api-key"
```

## Error Handling

### HTTP Status Codes

| Status | Code | Description |
|--------|------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Invalid or missing API key |
| 403 | Forbidden | Insufficient permissions |
| 429 | Rate Limited | Too many requests |
| 500 | Internal Error | Server error |
| 502 | Bad Gateway | Provider unavailable |
| 503 | Service Unavailable | Temporarily overloaded |

### Error Response Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid model specified",
    "param": "model",
    "code": "invalid_model",

    // Helix-specific error details
    "helix_context": {
      "component": "model_validation",
      "suggested_models": ["gpt-4o", "gpt-3.5-turbo"],
      "documentation_url": "https://docs.helix-ai.com/models"
    }
  }
}
```

### Common Error Scenarios

```bash
# Invalid API key
{
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key provided"
  }
}

# Rate limit exceeded
{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}

# PII detected in strict mode
{
  "error": {
    "type": "pii_violation_error",
    "message": "Request contains PII and is blocked by policy",
    "detected_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]
  }
}

# Budget exceeded
{
  "error": {
    "type": "budget_exceeded_error",
    "message": "Budget limit exceeded for user_123",
    "budget_limit": 10.00,
    "current_spend": 10.01
  }
}
```

## Configuration API

### Global Configuration

**Endpoint**: `GET/PUT /v1/helix/config`

Get or update global Helix configuration.

```bash
# Get current configuration
curl -X GET http://localhost:4000/v1/helix/config \
  -H "Authorization: Bearer admin-key"

# Update configuration
curl -X PUT http://localhost:4000/v1/helix/config \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "cache": {
      "semantic_similarity_threshold": 0.88,
      "max_cache_size_mb": 8192,
      "default_ttl_seconds": 2592000
    },
    "pii": {
      "default_action": "redact",
      "strict_mode": true,
      "enabled_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]
    },
    "cost_optimization": {
      "model_swapping_enabled": true,
      "budget_alerts_enabled": true
    }
  }'
```

### Model Configuration

**Endpoint**: `GET/PUT /v1/helix/config/models`

Configure model routing and optimization settings.

```bash
curl -X PUT http://localhost:4000/v1/helix/config/models \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "models": {
      "gpt-4o": {
        "primary_provider": "openai",
        "fallback_providers": ["anthropic", "groq"],
        "cost_threshold_usd": 0.01,
        "cache_strategy": "semantic",
        "preferred_for_complex_tasks": true
      },
      "gpt-3.5-turbo": {
        "primary_provider": "openai",
        "fallback_providers": ["groq"],
        "cost_threshold_usd": 0.002,
        "cache_strategy": "exact",
        "auto_swap_threshold": 0.95
      }
    }
  }'
```

## OpenAPI Specification

You can access the full OpenAPI 3.1 specification at:
```
GET /v1/openapi.json
```

Or explore the interactive documentation at:
```
GET /docs
```

This provides complete API documentation with:
- Interactive API explorer
- Request/response examples
- Authentication methods
- Error handling
- SDK generation

## Rate Limits

| Endpoint | Rate Limit | Burst Limit |
|----------|------------|-------------|
| `/v1/chat/completions` | 1000 req/min | 100 req |
| `/v1/completions` | 1000 req/min | 100 req |
| `/v1/embeddings` | 2000 req/min | 200 req |
| `/v1/helix/cache/*` | 100 req/min | 20 req |
| `/v1/helix/cost/*` | 50 req/min | 10 req |
| `/v1/helix/pii/*` | 50 req/min | 10 req |

Custom rate limits can be configured per user or organization.

## SDK Download and Installation

### Python
```bash
pip install helix-ai-gateway
```

### JavaScript/TypeScript
```bash
npm install @helix-ai/gateway-sdk
# or
yarn add @helix-ai/gateway-sdk
```

### Go
```bash
go get github.com/helix-ai/gateway-go-sdk
```

### Java
```xml
<dependency>
    <groupId>com.helix-ai</groupId>
    <artifactId>gateway-java-sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

For more detailed examples and integration guides, visit our [documentation portal](https://docs.helix-ai.com).