# Helix Development Guide
**Complete Guide for Contributing to and Extending Helix AI Gateway**

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Testing Strategies](#testing-strategies)
5. [Code Style and Standards](#code-style-and-standards)
6. [Architecture Guidelines](#architecture-guidelines)
7. [Performance Guidelines](#performance-guidelines)
8. [Security Guidelines](#security-guidelines)
9. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
10. [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

### Prerequisites

- **Python 3.11+** with pip
- **Git** for version control
- **Docker & Docker Compose** for containerized development
- **Redis Stack** (Redis with RediSearch)
- **Node.js 16+** (for dashboard development)
- **Make** (optional, for convenience commands)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourname/helix-gateway.git
cd helix-gateway

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev,proxy,test]"
pip install -r requirements-dev.txt

# 4. Set up Redis (using Docker)
docker run -d --name redis-dev -p 6379:6379 redis/redis-stack:latest

# 5. Initialize Redis for Helix
redis-cli < scripts/init_redis.redis

# 6. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and settings

# 7. Run tests to verify setup
make test-quick
```

### Detailed Setup

#### 1. Python Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install LiteLLM in development mode
pip install -e ".[dev,proxy,test]"

# Install additional development dependencies
pip install -r requirements-dev.txt
```

#### 2. Dependencies and Requirements

**requirements-dev.txt**:
```
# Core development tools
black>=23.0.0
ruff>=0.1.0
mypy>=1.5.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Pre-commit hooks
pre-commit>=3.4.0

# Documentation
sphinx>=7.1.0
sphinx-autodoc-typehints>=1.24.0

# Performance and profiling
memory-profiler>=0.61.0
py-spy>=0.3.14

# Security scanning
bandit>=1.7.5
semgrep>=1.45.0

# Linting and formatting
isort>=5.12.0
autoflake>=2.2.0

# Testing utilities
factory-boy>=3.3.0
faker>=19.0.0
responses>=0.23.0

# Development database
psycopg2-binary>=2.9.7

# Redis development
redis[hiredis]>=5.0.0

# ML/AI development
sentence-transformers>=2.7.0
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# API development
httpx>=0.25.0
aiohttp>=3.8.0
```

#### 3. Docker Development Setup

**docker-compose.dev.yml**:
```yaml
version: "3.9"
services:
  redis-dev:
    image: redis/redis-stack:latest
    ports: ["6379:6379"]
    volumes:
      - redis_dev_data:/data
      - ./redis/redis-dev.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf

  postgres-dev:
    image: postgres:15
    environment:
      - POSTGRES_DB=helix_dev
      - POSTGRES_USER=helix
      - POSTGRES_PASSWORD=dev_password
    ports: ["5432:5432"]
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init_dev_db.sql:/docker-entrypoint-initdb.d/init.sql

  minio-dev:
    image: minio/minio:latest
    ports: ["9000:9000", "9001:9001"]
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_dev_data:/data
    command: server /data --console-address ":9001"

volumes:
  redis_dev_data:
  postgres_dev_data:
  minio_dev_data:
```

#### 4. IDE Configuration

**VS Code Settings** (.vscode/settings.json):
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.testing.pytestPath": "pytest",
  "python.testing.unittestEnabled": false,
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true
  }
}
```

**VS Code Extensions**:
- Python (Microsoft)
- Pylance (Microsoft)
- Python Docstring Generator (Nils Werner)
- GitLens (GitKraken)
- Docker (Microsoft)
- Thunder Client (Rangav)

#### 5. Environment Configuration

**.env.example**:
```env
# API Keys (Required)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GROQ_API_KEY=gsk-your-groq-key

# Development Database
DATABASE_URL=postgresql://helix:dev_password@localhost:5432/helix_dev

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Helix Configuration
HELIX_ENABLED=true
HELIX_CACHE_STRATEGY=semantic
HELIX_PII_PROTECTION=true
HELIX_COST_OPTIMIZATION=true

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
PROFILING_ENABLED=false

# Testing Settings
TEST_DATABASE_URL=postgresql://helix:dev_password@localhost:5432/helix_test
TEST_REDIS_URL=redis://localhost:6379/15

# External Services
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Security
SECRET_KEY=your-development-secret-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:8501

# Performance
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
REQUEST_TIMEOUT=30
```

## Project Structure

```
helix-gateway/
├── .github/                     # GitHub configuration
│   ├── workflows/              # CI/CD workflows
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md # PR template
├── docs/                       # Documentation
│   ├── API.md                  # API documentation
│   ├── DEVELOPMENT.md          # Development guide
│   ├── DEPLOYMENT.md           # Deployment guide
│   └── TROUBLESHOOTING.md       # Troubleshooting guide
├── scripts/                    # Utility scripts
│   ├── init_redis.redis        # Redis initialization
│   ├── init_dev_db.sql         # Database initialization
│   ├── run_tests.sh            # Test runner
│   ├── setup_dev.sh            # Development setup
│   └── performance_test.py     # Performance testing
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   ├── fixtures/               # Test fixtures
│   └── conftest.py            # Pytest configuration
├── helix/                      # Main Helix package
│   ├── __init__.py
│   ├── core/                   # Core functionality
│   ├── caching/                # Caching layer
│   ├── pii/                    # PII protection
│   ├── cost/                   # Cost optimization
│   ├── monitoring/             # Monitoring and analytics
│   ├── security/               # Security features
│   └── config/                 # Configuration management
├── litellm/                    # LiteLLM source (modified)
├── dashboard/                  # Streamlit dashboard
│   ├── dashboard.py            # Main dashboard
│   ├── components/             # UI components
│   └── utils/                  # Dashboard utilities
├── docker/                     # Docker configurations
├── deployment/                 # Deployment configurations
├── monitoring/                 # Monitoring configurations
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Pre-commit hooks
├── pyproject.toml              # Project configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── docker-compose.yml          # Production compose
├── docker-compose.dev.yml      # Development compose
├── Makefile                    # Convenience commands
└── README.md                   # Project README
```

### Core Package Structure

**helix/core/**:
```
core/
├── __init__.py
├── gateway.py              # Main gateway orchestrator
├── config.py               # Configuration management
├── exceptions.py           # Helix-specific exceptions
├── middleware.py           # Request/response middleware
├── router.py               # Request routing logic
└── utils.py                # Utility functions
```

**helix/caching/**:
```
caching/
├── __init__.py
├── base.py                 # Abstract cache interface
├── exact_cache.py          # Exact match caching
├── semantic_cache.py       # Semantic similarity caching
├── hybrid_cache.py         # Combined caching strategy
├── cache_manager.py        # Cache orchestration
├── strategies.py           # Caching strategies
└── metrics.py              # Cache performance metrics
```

**helix/pii/**:
```
pii/
├── __init__.py
├── detector.py             # PII detection interface
├── presidio_detector.py   # Presidio integration
├── custom_detector.py      # Custom PII detection
├── redactor.py             # PII redaction
├── auditor.py              # PII incident auditing
└── recognizers/            # Custom recognizers
    ├── api_key_recognizer.py
    ├── internal_id_recognizer.py
    └── custom_pattern_recognizer.py
```

## Development Workflow

### 1. Git Workflow

We use a feature branch workflow with GitFlow conventions:

```bash
# 1. Create feature branch
git checkout -b feature/semantic-cache-optimization

# 2. Work on your feature
git add .
git commit -m "feat: implement semantic cache optimization algorithm"

# 3. Run tests and linting
make test
make lint

# 4. Push to feature branch
git push origin feature/semantic-cache-optimization

# 5. Create Pull Request
# Use GitHub UI or gh CLI:
gh pr create --title "Semantic Cache Optimization" --body "Detailed description..."

# 6. Address review comments
git commit -m "fix: address review feedback on cache optimization"

# 7. Merge after approval
# (Use squash merge for clean history)
```

### 2. Pre-commit Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Install specific hooks
pre-commit install --hook-type commit-msg
```

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-xml
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=88]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-ll, -f json, -o bandit-report.json]

  - repo: https://github.com/returntocorp/semgrep
    rev: v1.45.0
    hooks:
      - id: semgrep
        args: [--config=auto, --severity=INFO]
```

### 3. Makefile Commands

**Makefile**:
```makefile
.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-performance - Run performance tests"
	@echo "  lint         - Run all linting tools"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start development Docker services"
	@echo "  docker-down  - Stop development Docker services"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,proxy,test]"
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=helix --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v --cov=helix --cov-report=html

test-integration:
	pytest tests/integration/ -v --maxfail=1

test-performance:
	pytest tests/performance/ -v

lint:
	ruff check .
	mypy helix/
	bandit -r helix/ -f json -o bandit-report.json
	semgrep --config=auto helix/

format:
	black helix/ tests/ --line-length=88
	isort helix/ tests/ --profile black

format-check:
	black --check helix/ tests/
	isort --check-only helix/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ bandit-report.json semgrep-report.json

docker-build:
	docker build -t helix-gateway:dev .

docker-up:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker-compose.dev.yml down

run-dev:
	uvicorn litellm.proxy.proxy_server:app --host 0.0.0.0 --port 4000 --reload

run-dashboard:
	cd dashboard && streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Commit Message Format

We use conventional commits with semantic meaning:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Maintenance tasks

**Examples**:
```
feat(caching): implement semantic cache vector optimization

Add HNSW indexing for improved semantic search performance.
Includes similarity threshold tuning and cache warming strategies.

Closes #123
```

```
fix(pii): resolve false positives in email detection

Update email regex pattern to avoid matching common
non-email formats like "name@domain". Confirmed through
unit tests with edge cases.

Closes #456
```

```
perf(gateway): reduce request latency by 30%

Optimize request processing pipeline through parallel
execution and connection pooling. Measurements show
latency reduction from 250ms to 175ms on average.
```

## Testing Strategies

### 1. Test Structure

**tests/conftest.py**:
```python
import pytest
import asyncio
import redis
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from helix.core.config import get_settings
from helix.caching.semantic_cache import SemanticCache
from helix.pii.detector import PIIDetector


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing."""
    client = redis.Redis.from_url("redis://localhost:6379/15", decode_responses=True)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
async def semantic_cache(redis_client: redis.Redis) -> SemanticCache:
    """Create a semantic cache instance for testing."""
    cache = SemanticCache(redis_client=redis_client)
    await cache._ensure_index()
    return cache


@pytest.fixture
async def pii_detector() -> PIIDetector:
    """Create a PII detector for testing."""
    return PIIDetector()


@pytest.fixture
def sample_request_data() -> dict:
    """Sample request data for testing."""
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }


@pytest.fixture
def sample_response_data() -> dict:
    """Sample response data for testing."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 9,
            "total_tokens": 28
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
```

### 2. Unit Testing

**tests/unit/caching/test_semantic_cache.py**:
```python
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from helix.caching.semantic_cache import SemanticCache


class TestSemanticCache:
    """Test cases for semantic cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_storage_and_retrieval(
        self, semantic_cache: SemanticCache, sample_request_data: dict, sample_response_data: dict
    ):
        """Test storing and retrieving from semantic cache."""
        prompt = sample_request_data["messages"][-1]["content"]
        model = sample_request_data["model"]

        # Store in cache
        await semantic_cache.store(
            prompt=prompt,
            model=model,
            response=sample_response_data,
            metadata={"test": True}
        )

        # Retrieve from cache
        results = await semantic_cache.search(
            prompt=prompt,
            model=model,
            threshold=0.9
        )

        assert len(results) > 0
        assert results[0]["response"]["choices"][0]["message"]["content"] == \
               sample_response_data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_semantic_similarity_search(
        self, semantic_cache: SemanticCache
    ):
        """Test semantic similarity search with similar but not identical prompts."""
        original_prompt = "How do I reset my password?"
        similar_prompt = "I need to change my password"
        model = "gpt-4o"

        response = {
            "choices": [
                {
                    "message": {
                        "content": "To reset your password, go to settings..."
                    }
                }
            ]
        }

        # Store original prompt
        await semantic_cache.store(
            prompt=original_prompt,
            model=model,
            response=response
        )

        # Search with similar prompt
        results = await semantic_cache.search(
            prompt=similar_prompt,
            model=model,
            threshold=0.8  # Lower threshold for semantic matches
        )

        assert len(results) > 0
        assert float(results[0]["similarity"]) > 0.8

    @pytest.mark.asyncio
    async def test_cache_miss_scenario(
        self, semantic_cache: SemanticCache
    ):
        """Test cache miss scenario."""
        prompt = "What is the capital of France?"
        model = "gpt-4o"

        # Search for non-existent cache entry
        results = await semantic_cache.search(
            prompt=prompt,
            model=model,
            threshold=0.8
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_index_creation(
        self, semantic_cache: SemanticCache
    ):
        """Test vector index creation."""
        # Index should be created during initialization
        index_info = await semantic_cache.redis_client.ft("helix:semantic:index").info()

        assert "vector" in index_info["attributes"]
        assert index_info["num_docs"] >= 0

    @pytest.mark.asyncio
    async def test_cache_with_different_models(
        self, semantic_cache: SemanticCache
    ):
        """Test that cache is properly separated by model."""
        prompt = "Hello, world!"
        model1 = "gpt-4o"
        model2 = "gpt-3.5-turbo"

        response1 = {"choices": [{"message": {"content": "Response 1"}}]}
        response2 = {"choices": [{"message": {"content": "Response 2"}}]}

        # Store for model1
        await semantic_cache.store(
            prompt=prompt,
            model=model1,
            response=response1
        )

        # Store for model2
        await semantic_cache.store(
            prompt=prompt,
            model=model2,
            response=response2
        )

        # Search for model1
        results1 = await semantic_cache.search(
            prompt=prompt,
            model=model1
        )

        # Search for model2
        results2 = await semantic_cache.search(
            prompt=prompt,
            model=model2
        )

        assert len(results1) > 0
        assert len(results2) > 0
        assert results1[0]["response"]["choices"][0]["message"]["content"] == "Response 1"
        assert results2[0]["response"]["choices"][0]["message"]["content"] == "Response 2"
```

### 3. Integration Testing

**tests/integration/test_gateway_flow.py**:
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from helix.core.gateway import HelixGateway
from helix.caching.hybrid_cache import HybridCache
from helix.pii.detector import PIIDetector
from helix.cost.optimizer import CostOptimizer


@pytest.mark.integration
class TestGatewayFlow:
    """Integration tests for complete gateway request flow."""

    @pytest.mark.asyncio
    async def test_complete_request_flow_with_cache_hit(
        self, sample_request_data: dict, sample_response_data: dict, redis_client
    ):
        """Test complete request flow with cache hit scenario."""
        # Initialize gateway components
        cache = HybridCache(redis_client=redis_client)
        pii_detector = PIIDetector()
        cost_optimizer = CostOptimizer()

        gateway = HelixGateway(
            cache=cache,
            pii_detector=pii_detector,
            cost_optimizer=cost_optimizer
        )

        # First request (cache miss)
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = sample_response_data

            response1 = await gateway.process_request(
                request_data=sample_request_data,
                user_id="test_user"
            )

            assert mock_completion.called_once

        # Second identical request (cache hit)
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = sample_response_data

            response2 = await gateway.process_request(
                request_data=sample_request_data,
                user_id="test_user"
            )

            assert not mock_completion.called  # Should not call LLM

        # Verify responses are identical
        assert response1["choices"][0]["message"]["content"] == \
               response2["choices"][0]["message"]["content"]

        # Verify cache metadata
        assert "cache_hit" in response2["helix_metadata"]
        assert response2["helix_metadata"]["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_pii_detection_and_redaction_flow(
        self, redis_client
    ):
        """Test request flow with PII detection and redaction."""
        request_with_pii = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "My email is john@example.com and phone is 555-0123"}
            ],
            "max_tokens": 100
        }

        cache = HybridCache(redis_client=redis_client)
        pii_detector = PIIDetector()
        cost_optimizer = CostOptimizer()

        gateway = HelixGateway(
            cache=cache,
            pii_detector=pii_detector,
            cost_optimizer=cost_optimizer
        )

        # Process request with PII
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = {
                "choices": [{"message": {"content": "I understand you have contact information."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            }

            response = await gateway.process_request(
                request_data=request_with_pii,
                user_id="test_user"
            )

            # Verify PII was processed
            assert mock_completion.called_once

            # The LLM call should have redacted content
            call_args = mock_completion.call_args[1]
            processed_content = call_args["messages"][-1]["content"]

            # Check that PII is redacted (contains redaction markers)
            assert "<EMAIL_ADDRESS>" in processed_content or "john@example.com" not in processed_content
            assert "<PHONE_NUMBER>" in processed_content or "555-0123" not in processed_content

        # Verify PII metadata in response
        assert "pii_processed" in response["helix_metadata"]
        assert response["helix_metadata"]["pii_processed"] is True
        assert "pii_incidents" in response["helix_metadata"]
        assert len(response["helix_metadata"]["pii_incidents"]) > 0

    @pytest.mark.asyncio
    async def test_cost_optimization_flow(
        self, redis_client
    ):
        """Test request flow with cost optimization."""
        request_data = {
            "model": "gpt-4o",  # Expensive model
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 50
        }

        cache = HybridCache(redis_client=redis_client)
        pii_detector = PIIDetector()
        cost_optimizer = CostOptimizer()

        gateway = HelixGateway(
            cache=cache,
            pii_detector=pii_detector,
            cost_optimizer=cost_optimizer
        )

        # Configure cost optimizer to swap to cheaper model
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = {
                "choices": [{"message": {"content": "2+2 = 4"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "gpt-3.5-turbo"  # Cheaper model was used
            }

            response = await gateway.process_request(
                request_data=request_data,
                user_id="test_user"
            )

            # Verify cost optimization occurred
            call_args = mock_completion.call_args[1]
            assert call_args["model"] == "gpt-3.5-turbo"

        # Verify cost metadata
        assert "model_swapped" in response["helix_metadata"]
        assert response["helix_metadata"]["model_swapped"] is True
        assert "original_model" in response["helix_metadata"]
        assert response["helix_metadata"]["original_model"] == "gpt-4o"
        assert "actual_model" in response["helix_metadata"]
        assert response["helix_metadata"]["actual_model"] == "gpt-3.5-turbo"
```

### 4. Performance Testing

**tests/performance/test_cache_performance.py**:
```python
import pytest
import asyncio
import time
import statistics
from typing import List

from helix.caching.semantic_cache import SemanticCache


@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for caching components."""

    @pytest.mark.asyncio
    async def test_cache_search_performance(
        self, semantic_cache: SemanticCache
    ):
        """Test semantic cache search performance under load."""
        # Pre-populate cache with test data
        test_prompts = [
            f"This is test prompt number {i}" for i in range(1000)
        ]

        responses = [
            {"choices": [{"message": {"content": f"Response to prompt {i}"}}]}
            for i in range(1000)
        ]

        # Store test data
        for prompt, response in zip(test_prompts, responses):
            await semantic_cache.store(
                prompt=prompt,
                model="gpt-4o",
                response=response
            )

        # Test search performance
        search_times: List[float] = []
        test_queries = [
            "This is test prompt number 500",  # Exact match
            "This is test prompt number 501",  # Semantic match
            "What is the 600th test prompt?"   # Semantic variation
        ]

        for _ in range(100):  # 100 iterations per query
            for query in test_queries:
                start_time = time.time()
                results = await semantic_cache.search(
                    prompt=query,
                    model="gpt-4o",
                    threshold=0.8
                )
                end_time = time.time()

                search_times.append(end_time - start_time)
                assert len(results) > 0  # Should find matches

        # Analyze performance
        avg_search_time = statistics.mean(search_times)
        p95_search_time = statistics.quantiles(search_times, n=20)[18]  # 95th percentile
        p99_search_time = statistics.quantiles(search_times, n=100)[98]  # 99th percentile

        # Performance assertions
        assert avg_search_time < 0.050, f"Average search time too high: {avg_search_time:.3f}s"
        assert p95_search_time < 0.100, f"P95 search time too high: {p95_search_time:.3f}s"
        assert p99_search_time < 0.200, f"P99 search time too high: {p99_search_time:.3f}s"

        print(f"Cache search performance:")
        print(f"  Average: {avg_search_time:.3f}s")
        print(f"  P95: {p95_search_time:.3f}s")
        print(f"  P99: {p99_search_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(
        self, semantic_cache: SemanticCache
    ):
        """Test cache performance under concurrent load."""
        async def worker_task(worker_id: int, operations: int):
            """Worker task for concurrent operations."""
            times = []
            for i in range(operations):
                prompt = f"Worker {worker_id} prompt {i}"

                # Store operation
                start_time = time.time()
                await semantic_cache.store(
                    prompt=prompt,
                    model="gpt-4o",
                    response={"choices": [{"message": {"content": f"Response {i}"}}]}
                )
                store_time = time.time() - start_time
                times.append(("store", store_time))

                # Search operation
                start_time = time.time()
                results = await semantic_cache.search(
                    prompt=prompt,
                    model="gpt-4o"
                )
                search_time = time.time() - start_time
                times.append(("search", search_time))

                assert len(results) > 0

            return times

        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 50

        start_time = time.time()

        tasks = [
            worker_task(i, operations_per_worker)
            for i in range(num_workers)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        total_operations = num_workers * operations_per_worker * 2  # store + search

        # Analyze results
        store_times = []
        search_times = []

        for worker_result in results:
            for op_type, op_time in worker_result:
                if op_type == "store":
                    store_times.append(op_time)
                else:
                    search_times.append(op_time)

        avg_store_time = statistics.mean(store_times)
        avg_search_time = statistics.mean(search_times)
        ops_per_second = total_operations / total_time

        # Performance assertions
        assert ops_per_second > 100, f"Operations per second too low: {ops_per_second}"
        assert avg_store_time < 0.010, f"Average store time too high: {avg_store_time:.3f}s"
        assert avg_search_time < 0.020, f"Average search time too high: {avg_search_time:.3f}s"

        print(f"Concurrent cache operations performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Operations per second: {ops_per_second:.1f}")
        print(f"  Average store time: {avg_store_time:.3f}s")
        print(f"  Average search time: {avg_search_time:.3f}s")
```

### 5. Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run specific test file
pytest tests/unit/caching/test_semantic_cache.py -v

# Run with coverage
pytest tests/ --cov=helix --cov-report=html

# Run with specific markers
pytest -m "not performance"  # Skip performance tests
pytest -m "integration"      # Only integration tests

# Run with verbose output and show slowest tests
pytest --verbose --durations=10

# Run tests with coverage and generate HTML report
pytest --cov=helix --cov-report=html --cov-report=term-missing

# Run performance tests with profiler
pytest tests/performance/ --profile-svg
```

## Code Style and Standards

### 1. Python Code Style

We follow PEP 8 with additional standards:

```python
# Import organization
import os
import sys
from typing import Dict, List, Optional, Tuple

import fastapi
import redis
import numpy as np

from helix.core.config import settings
from helix.caching.base import CacheInterface


# Class definition
class SemanticCache(CacheInterface):
    """
    High-performance semantic cache using vector similarity search.

    Args:
        redis_client: Redis client for storage and search
        embedding_model: Model for generating text embeddings
        similarity_threshold: Minimum similarity score for cache hits

    Attributes:
        redis_client: Redis client instance
        embedder: Sentence transformer model
        threshold: Similarity threshold for matching

    Example:
        >>> cache = SemanticCache(redis_client=redis.Redis())
        >>> results = await cache.search("Hello world", model="gpt-4o")
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.88,
        max_cache_size: int = 1000000
    ) -> None:
        self.redis_client = redis_client
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size

        # Initialize embedding model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedding_model)

    async def search(
        self,
        prompt: str,
        model: str,
        threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Search for semantically similar cached responses.

        Args:
            prompt: Search query
            model: Model filter
            threshold: Override default similarity threshold

        Returns:
            List of cache hits with similarity scores

        Raises:
            RedisConnectionError: If Redis is unavailable
            ValueError: If prompt is empty
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        search_threshold = threshold or self.threshold

        try:
            # Generate embedding for search
            embedding = self.embedder.encode(prompt, normalize_embeddings=True)
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

            # Execute vector search
            query = (
                f"(@model:{model})=>[KNN 10 @vector $blob AS score]"
                f" FILTER __score SCORE {search_threshold}"
            )

            search_result = self.redis_client.ft("helix:semantic:index").search(
                query,
                query_params={"blob": embedding_blob}
            )

            return self._process_search_results(search_result)

        except redis.ConnectionError as exc:
            raise RedisConnectionError(f"Redis connection failed: {exc}")
        except Exception as exc:
            logger.error(f"Semantic search failed: {exc}")
            return []

    async def store(
        self,
        prompt: str,
        model: str,
        response: Dict[str, any],
        metadata: Optional[Dict[str, any]] = None
    ) -> str:
        """Store response in semantic cache with vector index."""
        cache_key = f"helix:vector:{uuid.uuid4()}"

        # Generate embedding
        embedding = self.embedder.encode(prompt, normalize_embeddings=True)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        # Prepare cache entry
        cache_entry = {
            "prompt": prompt,
            "model": model,
            "response_json": json.dumps(response),
            "vector": embedding_blob,
            "created_at": int(time.time()),
            "metadata": json.dumps(metadata or {})
        }

        # Store in Redis
        await self.redis_client.hset(cache_key, mapping=cache_entry)

        return cache_key
```

### 2. Documentation Standards

**Docstring Format**: Google-style docstrings

```python
def calculate_cache_efficiency(
    cache_hits: int,
    cache_misses: int,
    time_window: Optional[timedelta] = None
) -> Dict[str, float]:
    """
    Calculate cache efficiency metrics.

    This function computes various efficiency metrics including hit rate,
    miss rate, and performance improvements. It can calculate metrics
    for a specific time window if provided.

    Args:
        cache_hits: Number of successful cache hits
        cache_misses: Number of cache misses
        time_window: Optional time window for rate calculations

    Returns:
        Dictionary containing efficiency metrics:
        - hit_rate: Percentage of cache hits (0.0-1.0)
        - miss_rate: Percentage of cache misses (0.0-1.0)
        - total_requests: Total number of requests
        - efficiency_score: Overall efficiency score (0.0-1.0)

    Raises:
        ValueError: If both cache_hits and cache_misses are zero

    Example:
        >>> metrics = calculate_cache_efficiency(80, 20)
        >>> print(metrics['hit_rate'])
        0.8
    """
    if cache_hits == 0 and cache_misses == 0:
        raise ValueError("Both cache_hits and cache_misses cannot be zero")

    total_requests = cache_hits + cache_misses
    hit_rate = cache_hits / total_requests
    miss_rate = cache_misses / total_requests

    # Calculate efficiency score (weighted by cache performance)
    efficiency_score = hit_rate * 0.7 + (1 - miss_rate) * 0.3

    return {
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
        "total_requests": total_requests,
        "efficiency_score": efficiency_score
    }
```

### 3. Type Hints

```python
from typing import (
    Dict, List, Optional, Union, Callable, Any,
    AsyncGenerator, Tuple, TypeVar, Generic
)
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    """Generic cache entry with metadata."""
    key: str
    value: T
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    last_accessed: Optional[float] = None

class RequestMetadata(BaseModel):
    """Request metadata model."""
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(default_factory=time.time)
    client_ip: Optional[str] = Field(None, description="Client IP address")

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError("Timestamp cannot be negative")
        return v

class CacheConfig(BaseModel):
    """Cache configuration model."""
    enabled: bool = Field(default=True, description="Whether caching is enabled")
    max_size: int = Field(default=1000000, ge=1, description="Maximum cache size")
    ttl_seconds: int = Field(default=3600, ge=1, description="Default TTL in seconds")
    strategy: str = Field(default="lru", regex="^(lru|lfu|fifo)$", description="Eviction strategy")

class AsyncCache(Generic[T]):
    """Asynchronous cache interface."""

    def __init__(
        self,
        config: CacheConfig,
        serializer: Optional[Callable[[T], bytes]] = None,
        deserializer: Optional[Callable[[bytes], T]] = None
    ) -> None:
        self.config = config
        self._serializer = serializer or self._default_serializer
        self._deserializer = deserializer or self._default_deserializer

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        raise NotImplementedError

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        raise NotImplementedError

    def _default_serializer(self, value: T) -> bytes:
        """Default serializer implementation."""
        return pickle.dumps(value)

    def _default_deserializer(self, data: bytes) -> T:
        """Default deserializer implementation."""
        return pickle.loads(data)
```

### 4. Error Handling

```python
import logging
from typing import Optional

from helix.core.exceptions import (
    HelixError,
    CacheError,
    PIIError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

class CacheMissError(CacheError):
    """Raised when cache miss occurs."""

    def __init__(self, key: str, model: str) -> None:
        self.key = key
        self.model = model
        super().__init__(f"Cache miss for key '{key}' and model '{model}'")

class CacheFullError(CacheError):
    """Raised when cache is full and cannot accommodate new entries."""

    def __init__(self, cache_size: int, attempted_size: int) -> None:
        self.cache_size = cache_size
        self.attempted_size = attempted_size
        super().__init__(
            f"Cache full (size: {cache_size}, attempted: {attempted_size})"
        )

async def safe_cache_operation(
    operation: Callable[[], Awaitable[T]],
    fallback_value: Optional[T] = None,
    max_retries: int = 3,
    retry_delay: float = 0.1
) -> Optional[T]:
    """
    Safely execute cache operation with error handling and retries.

    Args:
        operation: Async callable to execute
        fallback_value: Value to return on failure
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Operation result or fallback_value on failure
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except redis.ConnectionError as exc:
            last_exception = exc
            logger.warning(f"Cache connection error (attempt {attempt + 1}): {exc}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        except CacheError as exc:
            last_exception = exc
            logger.error(f"Cache operation failed: {exc}")
            break  # Don't retry cache errors
        except Exception as exc:
            last_exception = exc
            logger.error(f"Unexpected error in cache operation: {exc}")
            break

    if fallback_value is not None:
        logger.info(f"Using fallback value due to cache errors")
        return fallback_value

    # Re-raise the last exception if no fallback provided
    raise last_exception

def handle_pii_error(error: PIIError, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PII-related errors with appropriate logging and user messaging.

    Args:
        error: The PII error that occurred
        context: Request context for logging

    Returns:
        Error response appropriate for the client
    """
    error_id = uuid.uuid4().hex
    logger.error(
        f"PII error occurred (ID: {error_id}): {error}",
        extra={
            "error_id": error_id,
            "error_type": type(error).__name__,
            "context": context
        }
    )

    if isinstance(error, PIIConfigurationError):
        return {
            "error": {
                "type": "pii_configuration_error",
                "message": "PII protection is misconfigured",
                "error_id": error_id
            }
        }
    elif isinstance(error, PIIProcessingError):
        return {
            "error": {
                "type": "pii_processing_error",
                "message": "PII processing failed",
                "error_id": error_id
            }
        }
    else:
        return {
            "error": {
                "type": "pii_error",
                "message": "PII protection error occurred",
                "error_id": error_id
            }
        }
```

## Contributing Guidelines

### 1. Before You Start

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
3. **Read this development guide** thoroughly
4. **Set up your development environment** as described above

### 2. Making Changes

#### Code Changes

1. **Follow the coding standards** outlined in this guide
2. **Write tests** for your changes
3. **Update documentation** if applicable
4. **Add type hints** to all new functions and classes
5. **Include docstrings** following the Google style

#### Testing Requirements

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: For performance-critical changes
- **Test coverage**: Maintain >80% coverage

#### Documentation Updates

- **API docs**: Update API.md if adding new endpoints
- **Code comments**: Add comments for complex logic
- **README updates**: Update installation/usage if needed
- **Examples**: Provide usage examples for new features

### 3. Pull Request Process

#### Before Submitting

```bash
# 1. Run all tests
make test

# 2. Run linting and formatting
make lint
make format

# 3. Check code coverage
pytest --cov=helix --cov-report=html

# 4. Run performance tests (if applicable)
make test-performance

# 5. Update documentation if needed
# 6. Commit your changes with conventional commit messages
```

#### Pull Request Template

```markdown
## Description
Brief description of your changes and the problem they solve.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactor
- [ ] Other

## How Has This Been Tested?
Describe how you tested your changes. Include details of:
- Unit tests written/modified
- Integration tests performed
- Manual testing steps
- Performance benchmarks (if applicable)

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
- [ ] I have checked that the code passes the security scanning tools

## Performance Impact
If this change affects performance, please describe:
- Performance improvements or regressions
- Benchmarks performed
- Resource usage implications

## Security Considerations
- [ ] My changes have been reviewed for security implications
- [ ] No sensitive data is exposed or logged inappropriately
- [ ] PII handling follows security best practices
```

### 4. Code Review Process

#### Review Guidelines

1. **Functionality**: Does the code work as intended?
2. **Performance**: Is the code efficient and performant?
3. **Security**: Are there any security vulnerabilities?
4. **Maintainability**: Is the code clean and well-documented?
5. **Testing**: Are the tests adequate and well-written?
6. **Style**: Does the code follow project standards?

#### Responding to Reviews

- **Acknowledge feedback** and thank reviewers
- **Address all concerns** raised in the review
- **Provide explanations** for design decisions
- **Make requested changes** or discuss alternatives
- **Keep PR updated** with latest changes

### 5. Release Process

#### Versioning

We follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

#### Release Checklist

```bash
# 1. Update version numbers
# Update pyproject.toml, package.json, etc.

# 2. Update CHANGELOG.md
# Document all changes since last release

# 3. Tag the release
git tag -a v1.2.3 -m "Release version 1.2.3"

# 4. Push tag to trigger CI/CD
git push origin v1.2.3

# 5. Create GitHub release
# Include changelog and installation instructions
```

### 6. Community Guidelines

#### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions
- Report issues politely

#### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Discord/Slack**: For real-time chat (if available)
- **Documentation**: Check docs/ first for existing answers

#### Contributing Types

- **Code contributions**: New features, bug fixes, performance improvements
- **Documentation**: Improving docs, examples, tutorials
- **Testing**: Writing tests, improving test coverage
- **Design**: Architecture reviews, UI/UX improvements
- **Community**: Answering questions, supporting other users

---

Thank you for contributing to Helix! Your contributions help make AI Gateway more powerful, secure, and efficient for everyone.