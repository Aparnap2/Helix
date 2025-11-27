"""
Helix Configuration Management
Handles loading, validation, and management of Helix configuration
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration for Helix"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    max_connections: int = 100
    connection_timeout: int = 10
    socket_timeout: int = 10
    cluster_enabled: bool = False
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VectorSearchConfig:
    """Vector search configuration for semantic caching"""
    enabled: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    index_name: str = "helix_semantic_cache"
    similarity_threshold: float = 0.85
    distance_metric: str = "cosine"
    dimensions: int = 1536


@dataclass
class HNSWConfig:
    """HNSW index parameters"""
    m: int = 16
    ef_construction: int = 200
    ef_runtime: int = 50


@dataclass
class CachingConfig:
    """Caching configuration for Helix"""
    enabled: bool = False
    cache_type: str = "hybrid"  # "exact", "semantic", "hybrid"
    redis: RedisConfig = field(default_factory=RedisConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    default_ttl: int = 3600
    max_ttl: int = 86400
    max_cache_size: int = 1000000
    max_memory_usage: int = 4294967296  # 4GB
    cacheable_models: List[str] = field(default_factory=list)
    excluded_cache_models: List[str] = field(default_factory=list)
    compression: bool = True
    compression_algorithm: str = "gzip"
    batch_size: int = 100
    batch_timeout: int = 5
    warm_cache: bool = False
    warm_cache_queries: List[str] = field(default_factory=list)


@dataclass
class RedactionConfig:
    """PII redaction configuration"""
    default_method: str = "replace"
    replacement_text: str = "[REDACTED]"
    mask_char: str = "*"
    mask_keep_chars: int = 4


@dataclass
class CustomRecognizer:
    """Custom PII recognizer configuration"""
    name: str
    pattern: str
    entity_type: str
    confidence_level: float = 0.9
    context_keywords: List[str] = field(default_factory=list)


@dataclass
class PresidioConfig:
    """Microsoft Presidio configuration"""
    recognizers: List[str] = field(default_factory=lambda: [
        "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"
    ])
    confidence_threshold: float = 0.8
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    max_processing_time_ms: int = 5000
    max_text_length: int = 100000
    languages: List[str] = field(default_factory=lambda: ["en"])
    custom_recognizers: List[CustomRecognizer] = field(default_factory=list)


@dataclass
class PIIDetectionConfig:
    """PII detection and redaction configuration"""
    enabled: bool = False
    mode: str = "detect_only"  # "detect_only", "redact", "audit_only"
    presidio: PresidioConfig = field(default_factory=PresidioConfig)
    stream_processing: bool = True
    batch_processing: bool = True
    batch_size: int = 10
    adaptive_confidence: bool = True


@dataclass
class AlertThresholds:
    """Budget alert thresholds"""
    warning: float = 70.0
    critical: float = 90.0
    blocked: float = 100.0


@dataclass
class ModelSwapRule:
    """Model swap rule for cost optimization"""
    original: str
    alternatives: List[str]
    max_token_ratio: float = 1.2


@dataclass
class ModelSwappingConfig:
    """Model swapping configuration"""
    enabled: bool = False
    swap_rules: List[ModelSwapRule] = field(default_factory=list)
    cost_sensitivity: float = 0.5


@dataclass
class SizeOptimizationConfig:
    """Prompt size optimization configuration"""
    enabled: bool = False
    max_reduction_percent: int = 20


@dataclass
class ContextPruningConfig:
    """Context pruning configuration"""
    enabled: bool = False
    similarity_threshold: float = 0.9


@dataclass
class PromptOptimizationConfig:
    """Prompt optimization configuration"""
    enabled: bool = False
    size_optimization: SizeOptimizationConfig = field(default_factory=SizeOptimizationConfig)
    context_pruning: ContextPruningConfig = field(default_factory=ContextPruningConfig)


@dataclass
class CostOptimizationConfig:
    """Cost optimization configuration"""
    enabled: bool = False
    cache_first: bool = True
    model_swapping: ModelSwappingConfig = field(default_factory=ModelSwappingConfig)
    prompt_optimization: PromptOptimizationConfig = field(default_factory=PromptOptimizationConfig)


@dataclass
class BudgetManagementConfig:
    """Budget management configuration"""
    enabled: bool = False
    default_daily_budget: float = 1000.0
    default_monthly_budget: float = 30000.0
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)


@dataclass
class CostConfig:
    """Cost tracking and optimization configuration"""
    enabled: bool = False
    real_time_tracking: bool = True
    budget_management: BudgetManagementConfig = field(default_factory=BudgetManagementConfig)
    optimization: CostOptimizationConfig = field(default_factory=CostOptimizationConfig)
    price_update_interval: int = 3600
    custom_pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Metrics collection configuration"""
    collection_interval: int = 60
    metrics_to_collect: List[str] = field(default_factory=lambda: [
        "request_count", "success_rate", "average_latency",
        "cache_hit_rate", "pii_detection_rate", "cost_per_request", "savings_rate"
    ])
    performance_metrics: bool = True
    percentiles: List[int] = field(default_factory=lambda: [50, 90, 95, 99])
    resource_metrics: bool = True


@dataclass
class PerformanceAlertsConfig:
    """Performance alerting configuration"""
    high_latency_threshold: int = 5000
    low_success_rate_threshold: float = 95.0
    high_error_rate_threshold: float = 5.0


@dataclass
class CacheAlertsConfig:
    """Cache alerting configuration"""
    low_cache_hit_rate_threshold: float = 50.0
    high_cache_memory_usage_threshold: float = 80.0


@dataclass
class CostAlertsConfig:
    """Cost alerting configuration"""
    daily_spend_spike_threshold: float = 2.0
    model_usage_anomaly_threshold: float = 3.0


@dataclass
class AlertingConfig:
    """Alerting configuration"""
    enabled: bool = False
    performance_alerts: PerformanceAlertsConfig = field(default_factory=PerformanceAlertsConfig)
    cache_alerts: CacheAlertsConfig = field(default_factory=CacheAlertsConfig)
    cost_alerts: CostAlertsConfig = field(default_factory=CostAlertsConfig)


@dataclass
class PrometheusConfig:
    """Prometheus integration configuration"""
    enabled: bool = False
    port: int = 9090
    endpoint: str = "/metrics"


@dataclass
class OpenTelemetryConfig:
    """OpenTelemetry integration configuration"""
    enabled: bool = False
    endpoint: str = "http://localhost:4317"


@dataclass
class GrafanaConfig:
    """Grafana integration configuration"""
    enabled: bool = False
    dashboard_url: str = "http://localhost:3000"


@dataclass
class MonitoringIntegrationsConfig:
    """Monitoring integrations configuration"""
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    opentelemetry: OpenTelemetryConfig = field(default_factory=OpenTelemetryConfig)
    grafana: GrafanaConfig = field(default_factory=GrafanaConfig)


@dataclass
class StreamlitDashboardConfig:
    """Streamlit dashboard configuration"""
    enabled: bool = False
    port: int = 8501
    host: str = "0.0.0.0"


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    enabled: bool = False
    streamlit: StreamlitDashboardConfig = field(default_factory=StreamlitDashboardConfig)
    refresh_interval: int = 30
    dashboard_retention_days: int = 30


@dataclass
class HealthChecksConfig:
    """Health checks configuration"""
    enabled: bool = False
    interval: int = 30
    components: Dict[str, bool] = field(default_factory=lambda: {
        "redis": True, "database": True, "pii_processor": True, "cache": True
    })


@dataclass
class MonitoringConfig:
    """Monitoring and performance configuration"""
    enabled: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    integrations: MonitoringIntegrationsConfig = field(default_factory=MonitoringIntegrationsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    health_checks: HealthChecksConfig = field(default_factory=HealthChecksConfig)


@dataclass
class GeneralConfig:
    """General Helix configuration"""
    excluded_models: List[str] = field(default_factory=list)
    request_timeout: int = 300
    max_request_size: int = 10485760  # 10MB
    enable_tracing: bool = True
    debug: bool = False


@dataclass
class DevelopmentConfig:
    """Development configuration"""
    enabled: bool = False
    debug_logging: bool = False
    debug_endpoints: bool = False
    hot_reload: bool = False


@dataclass
class HelixConfig:
    """Main Helix configuration class"""
    enabled: bool = False
    general: GeneralConfig = field(default_factory=GeneralConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    pii: PIIDetectionConfig = field(default_factory=PIIDetectionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


class HelixConfigManager:
    """Helix configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: HelixConfig = HelixConfig()
        self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        # Check environment variable first
        env_config_path = os.getenv("HELIX_CONFIG_PATH")
        if env_config_path and os.path.exists(env_config_path):
            return env_config_path

        # Check common locations
        possible_paths = [
            "config/helix.yaml",
            "config/helix.example.yaml",
            "helix.yaml",
            "helix.example.yaml",
            os.path.expanduser("~/.helix/config.yaml"),
            "/etc/helix/config.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Default to example config
        return "config/helix.example.yaml"

    def _load_config(self):
        """Load configuration from file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return

            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.warning("Empty config file, using defaults")
                return

            # Extract helix configuration
            helix_data = config_data.get('helix', {})
            if not helix_data:
                logger.warning("No 'helix' section found in config, using defaults")
                return

            # Parse configuration
            self.config = self._parse_config(helix_data)
            logger.info(f"Configuration loaded from: {self.config_path}")
            logger.info(f"Helix enabled: {self.config.enabled}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")

    def _parse_config(self, config_data: Dict[str, Any]) -> HelixConfig:
        """Parse configuration data into HelixConfig object"""
        try:
            # Parse general configuration
            general_data = config_data.get('general', {})
            general_config = GeneralConfig(
                excluded_models=general_data.get('excluded_models', []),
                request_timeout=general_data.get('request_timeout', 300),
                max_request_size=general_data.get('max_request_size', 10485760),
                enable_tracing=general_data.get('enable_tracing', True),
                debug=general_data.get('debug', False)
            )

            # Parse caching configuration
            caching_data = config_data.get('caching', {})
            redis_data = caching_data.get('redis', {})
            vector_data = caching_data.get('vector_search', {})

            redis_config = RedisConfig(
                host=self._get_env_var(redis_data, 'host', 'REDIS_HOST', 'localhost'),
                port=int(self._get_env_var(redis_data, 'port', 'REDIS_PORT', 6379)),
                password=self._get_env_var(redis_data, 'password', 'REDIS_PASSWORD', None),
                database=int(self._get_env_var(redis_data, 'database', 'REDIS_DB', 0)),
                max_connections=redis_data.get('max_connections', 100),
                connection_timeout=redis_data.get('connection_timeout', 10),
                socket_timeout=redis_data.get('socket_timeout', 10),
                cluster_enabled=redis_data.get('cluster_enabled', False),
                cluster_nodes=redis_data.get('cluster_nodes', [])
            )

            vector_search_config = VectorSearchConfig(
                enabled=vector_data.get('enabled', False),
                embedding_model=vector_data.get('embedding_model', 'text-embedding-3-small'),
                embedding_provider=vector_data.get('embedding_provider', 'openai'),
                index_name=vector_data.get('index_name', 'helix_semantic_cache'),
                similarity_threshold=vector_data.get('similarity_threshold', 0.85),
                distance_metric=vector_data.get('distance_metric', 'cosine'),
                dimensions=vector_data.get('dimensions', 1536)
            )

            caching_config = CachingConfig(
                enabled=caching_data.get('enabled', False),
                cache_type=caching_data.get('cache_type', 'hybrid'),
                redis=redis_config,
                vector_search=vector_search_config,
                default_ttl=caching_data.get('default_ttl', 3600),
                max_ttl=caching_data.get('max_ttl', 86400),
                max_cache_size=caching_data.get('max_cache_size', 1000000),
                max_memory_usage=caching_data.get('max_memory_usage', 4294967296),
                cacheable_models=caching_data.get('cacheable_models', []),
                excluded_cache_models=caching_data.get('excluded_cache_models', []),
                compression=caching_data.get('compression', True),
                compression_algorithm=caching_data.get('compression_algorithm', 'gzip'),
                batch_size=caching_data.get('batch_size', 100),
                batch_timeout=caching_data.get('batch_timeout', 5),
                warm_cache=caching_data.get('warm_cache', False),
                warm_cache_queries=caching_data.get('warm_cache_queries', [])
            )

            # Parse PII configuration
            pii_data = config_data.get('pii', {})
            presidio_data = pii_data.get('presidio', {})
            redaction_data = presidio_data.get('redaction', {})

            redaction_config = RedactionConfig(
                default_method=redaction_data.get('default_method', 'replace'),
                replacement_text=redaction_data.get('replacement_text', '[REDACTED]'),
                mask_char=redaction_data.get('mask_char', '*'),
                mask_keep_chars=redaction_data.get('mask_keep_chars', 4)
            )

            # Parse custom recognizers
            custom_recognizers = []
            for cr_data in presidio_data.get('custom_recognizers', []):
                custom_recognizers.append(CustomRecognizer(
                    name=cr_data.get('name', ''),
                    pattern=cr_data.get('pattern', ''),
                    entity_type=cr_data.get('entity_type', ''),
                    confidence_level=cr_data.get('confidence_level', 0.9),
                    context_keywords=cr_data.get('context_keywords', [])
                ))

            presidio_config = PresidioConfig(
                recognizers=presidio_data.get('recognizers', ['EMAIL_ADDRESS', 'PHONE_NUMBER']),
                confidence_threshold=presidio_data.get('confidence_threshold', 0.8),
                redaction=redaction_config,
                max_processing_time_ms=presidio_data.get('max_processing_time_ms', 5000),
                max_text_length=presidio_data.get('max_text_length', 100000),
                languages=presidio_data.get('languages', ['en']),
                custom_recognizers=custom_recognizers
            )

            pii_config = PIIDetectionConfig(
                enabled=pii_data.get('enabled', False),
                mode=pii_data.get('mode', 'detect_only'),
                presidio=presidio_config,
                stream_processing=pii_data.get('stream_processing', True),
                batch_processing=pii_data.get('batch_processing', True),
                batch_size=pii_data.get('batch_size', 10),
                adaptive_confidence=pii_data.get('adaptive_confidence', True)
            )

            # Parse cost configuration
            cost_data = config_data.get('cost', {})
            budget_data = cost_data.get('budget_management', {})
            optimization_data = cost_data.get('optimization', {})
            swapping_data = optimization_data.get('model_swapping', {})

            # Parse model swap rules
            swap_rules = []
            for rule_data in swapping_data.get('swap_rules', []):
                swap_rules.append(ModelSwapRule(
                    original=rule_data.get('original', ''),
                    alternatives=rule_data.get('alternatives', []),
                    max_token_ratio=rule_data.get('max_token_ratio', 1.2)
                ))

            model_swapping_config = ModelSwappingConfig(
                enabled=swapping_data.get('enabled', False),
                swap_rules=swap_rules,
                cost_sensitivity=swapping_data.get('cost_sensitivity', 0.5)
            )

            budget_management_config = BudgetManagementConfig(
                enabled=budget_data.get('enabled', False),
                default_daily_budget=budget_data.get('default_daily_budget', 1000.0),
                default_monthly_budget=budget_data.get('default_monthly_budget', 30000.0),
                alert_thresholds=AlertThresholds(
                    warning=budget_data.get('alert_thresholds', {}).get('warning', 70.0),
                    critical=budget_data.get('alert_thresholds', {}).get('critical', 90.0),
                    blocked=budget_data.get('alert_thresholds', {}).get('blocked', 100.0)
                )
            )

            cost_config = CostConfig(
                enabled=cost_data.get('enabled', False),
                real_time_tracking=cost_data.get('real_time_tracking', True),
                budget_management=budget_management_config,
                optimization=CostOptimizationConfig(
                    enabled=optimization_data.get('enabled', False),
                    cache_first=optimization_data.get('cache_first', True),
                    model_swapping=model_swapping_config,
                    prompt_optimization=PromptOptimizationConfig(
                        enabled=optimization_data.get('prompt_optimization', {}).get('enabled', False)
                    )
                ),
                price_update_interval=cost_data.get('price_update_interval', 3600),
                custom_pricing=cost_data.get('custom_pricing', {})
            )

            # Parse monitoring configuration
            monitoring_data = config_data.get('monitoring', {})
            monitoring_config = MonitoringConfig(
                enabled=monitoring_data.get('enabled', False),
                metrics=MetricsConfig(
                    collection_interval=monitoring_data.get('metrics', {}).get('collection_interval', 60),
                    metrics_to_collect=monitoring_data.get('metrics', {}).get('metrics_to_collect', []),
                    performance_metrics=monitoring_data.get('metrics', {}).get('performance_metrics', True),
                    percentiles=monitoring_data.get('metrics', {}).get('percentiles', [50, 90, 95, 99]),
                    resource_metrics=monitoring_data.get('metrics', {}).get('resource_metrics', True)
                ),
                dashboard=DashboardConfig(
                    enabled=monitoring_data.get('dashboard', {}).get('enabled', False),
                    streamlit=StreamlitDashboardConfig(
                        enabled=monitoring_data.get('dashboard', {}).get('streamlit', {}).get('enabled', False),
                        port=monitoring_data.get('dashboard', {}).get('streamlit', {}).get('port', 8501),
                        host=monitoring_data.get('dashboard', {}).get('streamlit', {}).get('host', '0.0.0.0')
                    ),
                    refresh_interval=monitoring_data.get('dashboard', {}).get('refresh_interval', 30)
                )
            )

            # Parse development configuration
            dev_data = config_data.get('development', {})
            development_config = DevelopmentConfig(
                enabled=dev_data.get('enabled', False),
                debug_logging=dev_data.get('debug', {}).get('logging', False),
                debug_endpoints=dev_data.get('debug', {}).get('endpoints', False),
                hot_reload=dev_data.get('debug', {}).get('hot_reload', False)
            )

            return HelixConfig(
                enabled=config_data.get('enabled', False),
                general=general_config,
                caching=caching_config,
                pii=pii_config,
                cost=cost_config,
                monitoring=monitoring_config,
                development=development_config
            )

        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            return HelixConfig()  # Return default config

    def _get_env_var(
        self,
        config_data: Dict[str, Any],
        key: str,
        env_var: str,
        default: Any
    ) -> Any:
        """Get value from config or environment variable"""
        # Check if config value is an environment variable reference
        config_value = config_data.get(key)
        if isinstance(config_value, str) and config_value.startswith("${") and config_value.endswith("}"):
            env_var_name = config_value[2:-1]
            if ":" in env_var_name:
                env_var_name, default_value = env_var_name.split(":", 1)
                return os.getenv(env_var_name, default_value)
            return os.getenv(env_var_name, default)

        # Return config value or default
        return config_value if config_value is not None else os.getenv(env_var, default)

    def is_enabled(self) -> bool:
        """Check if Helix is enabled"""
        return self.config.enabled

    def get_config(self) -> HelixConfig:
        """Get the current configuration"""
        return self.config

    def reload_config(self):
        """Reload configuration from file"""
        logger.info("Reloading Helix configuration")
        self._load_config()

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        try:
            config = self.config

            # Validate caching configuration
            if config.caching.enabled:
                if not config.caching.redis.host:
                    errors.append("Redis host is required when caching is enabled")
                if config.caching.cache_type not in ["exact", "semantic", "hybrid"]:
                    errors.append(f"Invalid cache type: {config.caching.cache_type}")

                if config.caching.vector_search.enabled:
                    if not config.caching.vector_search.embedding_model:
                        errors.append("Embedding model is required for vector search")
                    if not (0.0 <= config.caching.vector_search.similarity_threshold <= 1.0):
                        errors.append("Similarity threshold must be between 0.0 and 1.0")

            # Validate PII configuration
            if config.pii.enabled:
                if config.pii.mode not in ["detect_only", "redact", "audit_only"]:
                    errors.append(f"Invalid PII mode: {config.pii.mode}")

                if config.pii.presidio.confidence_threshold < 0 or config.pii.presidio.confidence_threshold > 1:
                    errors.append("PII confidence threshold must be between 0 and 1")

            # Validate cost configuration
            if config.cost.enabled:
                if config.cost.budget_management.enabled:
                    if config.cost.budget_management.default_daily_budget <= 0:
                        errors.append("Default daily budget must be greater than 0")

                if config.cost.optimization.model_swapping.enabled:
                    for rule in config.cost.optimization.model_swapping.swap_rules:
                        if not rule.original or not rule.alternatives:
                            errors.append(f"Invalid model swap rule: {rule}")

            # Validate monitoring configuration
            if config.monitoring.enabled:
                if config.monitoring.metrics.collection_interval <= 0:
                    errors.append("Metrics collection interval must be greater than 0")

                if config.monitoring.dashboard.enabled and config.monitoring.dashboard.streamlit.enabled:
                    if not (1024 <= config.monitoring.dashboard.streamlit.port <= 65535):
                        errors.append("Streamlit port must be between 1024 and 65535")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            "enabled": self.config.enabled,
            "debug": self.config.general.debug,
            "caching": {
                "enabled": self.config.caching.enabled,
                "cache_type": self.config.caching.cache_type,
                "vector_search": self.config.caching.vector_search.enabled,
                "redis_host": self.config.caching.redis.host
            },
            "pii": {
                "enabled": self.config.pii.enabled,
                "mode": self.config.pii.mode,
                "recognizers_count": len(self.config.pii.presidio.recognizers),
                "custom_recognizers_count": len(self.config.pii.presidio.custom_recognizers)
            },
            "cost": {
                "enabled": self.config.cost.enabled,
                "real_time_tracking": self.config.cost.real_time_tracking,
                "budget_management": self.config.cost.budget_management.enabled,
                "optimization": self.config.cost.optimization.enabled
            },
            "monitoring": {
                "enabled": self.config.monitoring.enabled,
                "dashboard": self.config.monitoring.dashboard.enabled,
                "metrics_collection": self.config.monitoring.metrics.collection_interval
            },
            "development": {
                "enabled": self.config.development.enabled,
                "debug_logging": self.config.development.debug_logging
            }
        }


# Global configuration manager instance
config_manager = HelixConfigManager()

def get_config() -> HelixConfig:
    """Get the global Helix configuration"""
    return config_manager.get_config()

def is_helix_enabled() -> bool:
    """Check if Helix is enabled"""
    return config_manager.is_enabled()

def reload_config():
    """Reload the global configuration"""
    config_manager.reload_config()