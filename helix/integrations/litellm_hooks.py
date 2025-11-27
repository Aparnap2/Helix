"""
Helix Integration Hooks for LiteLLM
This module provides the main integration points between Helix and LiteLLM's hook system.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse

from helix.core.config import HelixConfig
from helix.caching.hybrid_cache import HelixHybridCache
from helix.caching.cache_metrics import HelixCacheMetrics
from helix.caching.semantic_cache import HelixSemanticCache
from helix.pii.presidio_integration import HelixPIIProcessor
from helix.cost.cost_tracker import HelixCostTracker
from helix.cost.spend_optimizer import HelixSpendOptimizer
from helix.monitoring.metrics_collector import HelixMetricsCollector


class HelixGatewayHooks:
    """
    Main Helix integration class that implements LiteLLM hooks
    """

    def __init__(self, config: HelixConfig):
        self.config = config
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all Helix components based on configuration"""
        verbose_proxy_logger.info("Initializing Helix Gateway Components")

        # Initialize caching components
        if self.config.caching.enabled:
            self.cache = HelixHybridCache(
                redis_client=self._get_redis_client(),
                config=self.config.caching
            )
            self.cache_metrics = HelixCacheMetrics(self.config.caching)
            verbose_proxy_logger.info("✅ Helix Hybrid Cache initialized")
        else:
            self.cache = None
            verbose_proxy_logger.info("❌ Helix Caching disabled")

        # Initialize PII processor
        if self.config.pii.enabled:
            self.pii_processor = HelixPIIProcessor(
                config=self.config.pii
            )
            verbose_proxy_logger.info("✅ Helix PII Processor initialized")
        else:
            self.pii_processor = None
            verbose_proxy_logger.info("❌ Helix PII processing disabled")

        # Initialize cost tracker
        if self.config.cost.enabled:
            self.cost_tracker = HelixCostTracker(
                config=self.config.cost,
                db_client=self._get_db_client()
            )
            self.spend_optimizer = HelixSpendOptimizer(
                cost_tracker=self.cost_tracker,
                config=self.config.cost
            )
            verbose_proxy_logger.info("✅ Helix Cost Tracker initialized")
        else:
            self.cost_tracker = None
            self.spend_optimizer = None
            verbose_proxy_logger.info("❌ Helix Cost tracking disabled")

        # Initialize metrics collector
        if self.config.monitoring.enabled:
            self.metrics_collector = HelixMetricsCollector(
                config=self.config.monitoring
            )
            verbose_proxy_logger.info("✅ Helix Metrics Collector initialized")
        else:
            self.metrics_collector = None
            verbose_proxy_logger.info("❌ Helix Monitoring disabled")

    async def async_pre_call_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Dict[str, Any]:
        """
        Pre-call hook executed before making the LLM request
        Handles cache lookup, PII detection, and cost optimization
        """
        if not self._is_helix_enabled_for_request(data, user_api_key_dict):
            return data

        start_time = time.time()
        request_id = str(uuid.uuid4())
        data["helix_request_id"] = request_id

        try:
            verbose_proxy_logger.debug(f"Helix pre_call_hook for request {request_id}")

            # Extract request content for processing
            request_content = self._extract_request_content(data)
            if not request_content:
                return data

            # Step 1: PII Detection and Redaction
            if self.pii_processor:
                data = await self._process_pii_detection(data, request_content, request_id)

            # Step 2: Cache Lookup
            cached_result = None
            if self.cache:
                cached_result = await self._lookup_cache(data, request_id)

                if cached_result:
                    await self._handle_cache_hit(cached_result, request_id, user_api_key_dict)
                    return cached_result

            # Step 3: Cost Optimization
            if self.spend_optimizer:
                data = await self._optimize_request_cost(data, request_id, user_api_key_dict)

            # Store original content for post-processing
            data["helix_original_content"] = request_content
            data["helix_processing_start_time"] = start_time

            verbose_proxy_logger.debug(f"Helix pre-processing completed for {request_id}")
            return data

        except Exception as e:
            verbose_proxy_logger.error(f"Helix pre_call_hook error: {str(e)}")
            # Continue with original request if Helix processing fails
            return data

    async def async_post_call_hook(
        self,
        original_data: Dict[str, Any],
        response_data: Union[ModelResponse, Dict[str, Any]],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    ) -> Union[ModelResponse, Dict[str, Any]]:
        """
        Post-call hook executed after receiving the LLM response
        Handles cache storage, cost tracking, and metrics collection
        """
        request_id = original_data.get("helix_request_id")
        if not request_id:
            return response_data

        try:
            verbose_proxy_logger.debug(f"Helix post_call_hook for request {request_id}")

            # Step 1: Store in cache
            if self.cache and self._should_cache_response(original_data, response_data):
                await self._store_in_cache(original_data, response_data, request_id)

            # Step 2: Track costs
            if self.cost_tracker:
                await self._track_request_costs(original_data, response_data, user_api_key_dict)

            # Step 3: Collect metrics
            if self.metrics_collector:
                await self._collect_post_call_metrics(original_data, response_data, user_api_key_dict)

            verbose_proxy_logger.debug(f"Helix post-processing completed for {request_id}")
            return response_data

        except Exception as e:
            verbose_proxy_logger.error(f"Helix post_call_hook error: {str(e)}")
            return response_data

    async def async_log_success_event_hook(
        self,
        kwargs: Dict[str, Any],
        response_obj: Union[ModelResponse, Dict[str, Any]],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Success logging hook for comprehensive tracking
        """
        request_id = kwargs.get("helix_request_id")
        if not request_id:
            return None

        try:
            verbose_proxy_logger.debug(f"Helix success logging for request {request_id}")

            # Calculate total processing time
            processing_time = 0
            if start_time and end_time:
                processing_time = end_time - start_time

            # Log comprehensive success metrics
            success_log = {
                "request_id": request_id,
                "processing_time_ms": int(processing_time * 1000),
                "model": kwargs.get("model"),
                "endpoint": kwargs.get("endpoint"),
                "success": True,
                "helix_features_used": self._get_used_features(kwargs),
                "cache_hit": kwargs.get("helix_cache_hit", False),
                "pii_detected": kwargs.get("helix_pii_detected", False),
                "cost_optimized": kwargs.get("helix_cost_optimized", False),
            }

            # Update metrics
            if self.metrics_collector:
                await self.metrics_collector.record_success(success_log)

            verbose_proxy_logger.debug(f"Helix success logging completed for {request_id}")
            return success_log

        except Exception as e:
            verbose_proxy_logger.error(f"Helix success logging error: {str(e)}")
            return None

    async def async_log_failure_event_hook(
        self,
        kwargs: Dict[str, Any],
        exception: Exception,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Failure logging hook for error tracking and alerting
        """
        request_id = kwargs.get("helix_request_id")
        if not request_id:
            return None

        try:
            verbose_proxy_logger.debug(f"Helix failure logging for request {request_id}")

            # Calculate total processing time
            processing_time = 0
            if start_time and end_time:
                processing_time = end_time - start_time

            # Log comprehensive failure metrics
            failure_log = {
                "request_id": request_id,
                "processing_time_ms": int(processing_time * 1000),
                "model": kwargs.get("model"),
                "endpoint": kwargs.get("endpoint"),
                "success": False,
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "helix_features_used": self._get_used_features(kwargs),
                "helix_processing_stage": kwargs.get("helix_processing_stage", "unknown"),
            }

            # Update metrics
            if self.metrics_collector:
                await self.metrics_collector.record_failure(failure_log)

            verbose_proxy_logger.debug(f"Helix failure logging completed for {request_id}")
            return failure_log

        except Exception as e:
            verbose_proxy_logger.error(f"Helix failure logging error: {str(e)}")
            return None

    # Private helper methods

    def _is_helix_enabled_for_request(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ) -> bool:
        """Check if Helix should be enabled for this request"""
        # Global Helix enablement
        if not self.config.enabled:
            return False

        # Check user/organization settings
        if user_api_key_dict:
            user_config = user_api_key_dict.helix_config or {}
            if not user_config.get("enabled", True):
                return False

        # Check model-specific settings
        model = data.get("model", "")
        excluded_models = self.config.general.get("excluded_models", [])
        if model in excluded_models:
            return False

        return True

    def _extract_request_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from request for processing"""
        try:
            # Handle different request formats
            if "messages" in data:
                # Chat completion format
                messages = data["messages"]
                content_parts = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        content_parts.append(str(msg["content"]))
                return " ".join(content_parts)

            elif "input" in data:
                # Text completion or embedding format
                if isinstance(data["input"], list):
                    return " ".join(str(item) for item in data["input"])
                else:
                    return str(data["input"])

            elif "prompt" in data:
                # Legacy completion format
                return str(data["prompt"])

            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Error extracting request content: {str(e)}")
            return None

    async def _process_pii_detection(
        self,
        data: Dict[str, Any],
        content: str,
        request_id: str
    ) -> Dict[str, Any]:
        """Process PII detection and redaction"""
        try:
            pii_start_time = time.time()

            # Detect PII
            redacted_content, pii_entities = await self.pii_processor.detect_and_redact(content)

            processing_time_ms = int((time.time() - pii_start_time) * 1000)

            # Update request data with redacted content
            data = self._replace_content_in_request(data, redacted_content)

            # Store PII metadata
            data["helix_pii_detected"] = len(pii_entities) > 0
            data["helix_pii_count"] = len(pii_entities)
            data["helix_pii_processing_time_ms"] = processing_time_ms
            data["helix_pii_entities"] = [
                {
                    "type": entity.entity_type,
                    "confidence": entity.confidence_score,
                    "start": entity.start,
                    "end": entity.end
                }
                for entity in pii_entities
            ]

            # Log PII detection asynchronously
            asyncio.create_task(
                self._log_pii_detection(request_id, content, pii_entities, processing_time_ms)
            )

            verbose_proxy_logger.info(
                f"PII Processing: {len(pii_entities)} entities detected, "
                f"{processing_time_ms}ms processing time"
            )

            return data

        except Exception as e:
            verbose_proxy_logger.error(f"PII processing error: {str(e)}")
            data["helix_pii_detected"] = False
            data["helix_pii_error"] = str(e)
            return data

    async def _lookup_cache(
        self,
        data: Dict[str, Any],
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Lookup in hybrid cache"""
        try:
            cache_start_time = time.time()

            # Generate cache key and query
            cache_key = self._generate_cache_key(data)
            content = self._extract_request_content(data)

            # Lookup in cache
            cached_result = await self.cache.get(key=cache_key, query=content)

            cache_lookup_time_ms = int((time.time() - cache_start_time) * 1000)
            data["helix_cache_lookup_time_ms"] = cache_lookup_time_ms

            if cached_result:
                data["helix_cache_hit"] = True
                data["helix_cache_type"] = cached_result.get("cache_type", "unknown")
                data["helix_cache_similarity"] = cached_result.get("similarity_score")

                # Update cache metrics
                await self.cache_metrics.record_hit(
                    cache_type=data["helix_cache_type"],
                    lookup_time_ms=cache_lookup_time_ms
                )

                verbose_proxy_logger.info(
                    f"Cache Hit: {data['helix_cache_type']} cache, "
                    f"{cache_lookup_time_ms}ms lookup time"
                )

                return cached_result
            else:
                data["helix_cache_hit"] = False
                await self.cache_metrics.record_miss(lookup_time_ms=cache_lookup_time_ms)

                return None

        except Exception as e:
            verbose_proxy_logger.error(f"Cache lookup error: {str(e)}")
            data["helix_cache_hit"] = False
            data["helix_cache_error"] = str(e)
            return None

    async def _optimize_request_cost(
        self,
        data: Dict[str, Any],
        request_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ) -> Dict[str, Any]:
        """Optimize request for cost efficiency"""
        try:
            optimization_start_time = time.time()

            # Apply optimization strategies
            optimized_data = await self.spend_optimizer.optimize_request(data)

            processing_time_ms = int((time.time() - optimization_start_time) * 1000)

            # Track optimization changes
            original_model = data.get("model")
            optimized_model = optimized_data.get("model")

            data["helix_cost_optimized"] = original_model != optimized_model
            data["helix_original_model"] = original_model
            data["helix_optimization_time_ms"] = processing_time_ms

            if data["helix_cost_optimized"]:
                verbose_proxy_logger.info(
                    f"Cost Optimization: {original_model} -> {optimized_model}, "
                    f"{processing_time_ms}ms processing time"
                )

            return optimized_data

        except Exception as e:
            verbose_proxy_logger.error(f"Cost optimization error: {str(e)}")
            data["helix_cost_optimized"] = False
            data["helix_cost_optimization_error"] = str(e)
            return data

    async def _handle_cache_hit(
        self,
        cached_result: Dict[str, Any],
        request_id: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ):
        """Handle successful cache hit"""
        try:
            # Record cache hit metrics
            if self.metrics_collector:
                await self.metrics_collector.record_cache_hit({
                    "request_id": request_id,
                    "cache_type": cached_result.get("cache_type"),
                    "cache_hit_time": cached_result.get("timestamp"),
                    "similarity_score": cached_result.get("similarity_score"),
                    "original_cost": cached_result.get("original_cost"),
                    "cached_cost": 0.0,  # Cache hits are free
                    "savings": cached_result.get("original_cost", 0.0),
                })

            # Log cache hit for audit
            await self._log_cache_hit(request_id, cached_result, user_api_key_dict)

        except Exception as e:
            verbose_proxy_logger.error(f"Cache hit handling error: {str(e)}")

    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for the request"""
        try:
            # Extract key parameters for caching
            cache_params = {
                "model": data.get("model"),
                "temperature": data.get("temperature"),
                "max_tokens": data.get("max_tokens"),
                "top_p": data.get("top_p"),
                "frequency_penalty": data.get("frequency_penalty"),
                "presence_penalty": data.get("presence_penalty"),
                "messages": data.get("messages", []),
                "stream": data.get("stream", False),
            }

            # Create deterministic hash
            import hashlib
            cache_key = hashlib.sha256(
                json.dumps(cache_params, sort_keys=True).encode()
            ).hexdigest()

            return f"helix:{cache_key}"

        except Exception as e:
            verbose_proxy_logger.error(f"Cache key generation error: {str(e)}")
            return f"helix:{uuid.uuid4()}"

    def _replace_content_in_request(
        self,
        data: Dict[str, Any],
        new_content: str
    ) -> Dict[str, Any]:
        """Replace content in request data with new content"""
        try:
            data_copy = data.copy()

            if "messages" in data_copy:
                # Chat completion format
                for msg in data_copy["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        msg["content"] = new_content
                        break  # Replace first message content only

            elif "input" in data_copy:
                # Text completion or embedding format
                data_copy["input"] = new_content

            elif "prompt" in data_copy:
                # Legacy completion format
                data_copy["prompt"] = new_content

            return data_copy

        except Exception as e:
            verbose_proxy_logger.error(f"Content replacement error: {str(e)}")
            return data

    async def _store_in_cache(
        self,
        original_data: Dict[str, Any],
        response_data: Union[ModelResponse, Dict[str, Any]],
        request_id: str
    ):
        """Store response in cache"""
        try:
            cache_key = self._generate_cache_key(original_data)
            content = self._extract_request_content(original_data)

            # Prepare cache entry with metadata
            cache_entry = {
                "response": response_data,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": original_data.get("model"),
                "cache_type": "hybrid",
                "ttl": self.config.caching.default_ttl,
            }

            # Store in cache
            await self.cache.set(
                key=cache_key,
                value=cache_entry,
                query=content,
                ttl=self.config.caching.default_ttl
            )

            verbose_proxy_logger.debug(f"Response stored in cache: {cache_key}")

        except Exception as e:
            verbose_proxy_logger.error(f"Cache storage error: {str(e)}")

    async def _track_request_costs(
        self,
        original_data: Dict[str, Any],
        response_data: Union[ModelResponse, Dict[str, Any]],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ):
        """Track request costs and savings"""
        try:
            if not self.cost_tracker:
                return

            cost_data = await self.cost_tracker.calculate_cost(
                request_data=original_data,
                response_data=response_data,
                user_api_key_dict=user_api_key_dict
            )

            # Store cost tracking data
            original_data["helix_cost_data"] = cost_data

            # Check budget limits
            await self.cost_tracker.check_budget_limits(
                cost_data=cost_data,
                user_api_key_dict=user_api_key_dict
            )

            verbose_proxy_logger.debug(f"Cost tracked: {cost_data.get('total_cost', 0.0)}")

        except Exception as e:
            verbose_proxy_logger.error(f"Cost tracking error: {str(e)}")

    async def _collect_post_call_metrics(
        self,
        original_data: Dict[str, Any],
        response_data: Union[ModelResponse, Dict[str, Any]],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ):
        """Collect post-call metrics"""
        try:
            metrics = {
                "request_id": original_data.get("helix_request_id"),
                "cache_hit": original_data.get("helix_cache_hit", False),
                "cache_type": original_data.get("helix_cache_type"),
                "cache_lookup_time_ms": original_data.get("helix_cache_lookup_time_ms"),
                "pii_detected": original_data.get("helix_pii_detected", False),
                "pii_count": original_data.get("helix_pii_count", 0),
                "pii_processing_time_ms": original_data.get("helix_pii_processing_time_ms"),
                "cost_optimized": original_data.get("helix_cost_optimized", False),
                "optimization_time_ms": original_data.get("helix_optimization_time_ms"),
                "total_cost": original_data.get("helix_cost_data", {}).get("total_cost", 0.0),
                "cache_savings": original_data.get("helix_cost_data", {}).get("cache_savings", 0.0),
                "optimization_savings": original_data.get("helix_cost_data", {}).get("optimization_savings", 0.0),
            }

            await self.metrics_collector.record_request(metrics)

        except Exception as e:
            verbose_proxy_logger.error(f"Post-call metrics collection error: {str(e)}")

    def _should_cache_response(
        self,
        original_data: Dict[str, Any],
        response_data: Union[ModelResponse, Dict[str, Any]]
    ) -> bool:
        """Determine if response should be cached"""
        try:
            # Don't cache streaming responses
            if original_data.get("stream", False):
                return False

            # Don't cache errors
            if isinstance(response_data, dict) and response_data.get("error"):
                return False

            # Check cacheable models
            model = original_data.get("model", "")
            cacheable_models = self.config.caching.get("cacheable_models", [])
            if cacheable_models and model not in cacheable_models:
                return False

            # Check content length limits
            content_length = len(str(response_data))
            max_content_length = self.config.caching.get("max_content_length", 100000)
            if content_length > max_content_length:
                return False

            return True

        except Exception as e:
            verbose_proxy_logger.error(f"Cache eligibility check error: {str(e)}")
            return False

    def _get_used_features(self, data: Dict[str, Any]) -> List[str]:
        """Get list of Helix features used for this request"""
        features = []
        if data.get("helix_cache_hit"):
            features.append("cache")
        if data.get("helix_pii_detected"):
            features.append("pii")
        if data.get("helix_cost_optimized"):
            features.append("cost_optimization")
        return features

    # Async logging methods
    async def _log_pii_detection(
        self,
        request_id: str,
        original_content: str,
        pii_entities: List,
        processing_time_ms: int
    ):
        """Log PII detection asynchronously"""
        try:
            log_data = {
                "request_id": request_id,
                "original_content_hash": hashlib.sha256(original_content.encode()).hexdigest(),
                "pii_entities": [
                    {
                        "type": entity.entity_type,
                        "confidence": entity.confidence_score,
                        "start": entity.start,
                        "end": entity.end
                    }
                    for entity in pii_entities
                ],
                "pii_count": len(pii_entities),
                "processing_time_ms": processing_time_ms,
                "text_length": len(original_content),
            }

            # Store in database
            await self._store_pii_log(log_data)

        except Exception as e:
            verbose_proxy_logger.error(f"PII detection logging error: {str(e)}")

    async def _log_cache_hit(
        self,
        request_id: str,
        cached_result: Dict[str, Any],
        user_api_key_dict: Optional[UserAPIKeyAuth] = None
    ):
        """Log cache hit asynchronously"""
        try:
            log_data = {
                "request_id": request_id,
                "cache_type": cached_result.get("cache_type"),
                "cache_hit_time": cached_result.get("timestamp"),
                "similarity_score": cached_result.get("similarity_score"),
                "original_cost": cached_result.get("original_cost", 0.0),
                "savings": cached_result.get("original_cost", 0.0),
                "user_id": user_api_key_dict.user_id if user_api_key_dict else None,
                "organization_id": getattr(user_api_key_dict, 'organization_id', None) if user_api_key_dict else None,
            }

            # Store in database
            await self._store_cache_hit_log(log_data)

        except Exception as e:
            verbose_proxy_logger.error(f"Cache hit logging error: {str(e)}")

    # Database interaction methods (to be implemented based on your database setup)
    async def _store_pii_log(self, log_data: Dict[str, Any]):
        """Store PII detection log in database"""
        # Implementation depends on your database setup
        pass

    async def _store_cache_hit_log(self, log_data: Dict[str, Any]):
        """Store cache hit log in database"""
        # Implementation depends on your database setup
        pass

    # Infrastructure helper methods
    def _get_redis_client(self):
        """Get Redis client for caching"""
        # Implementation depends on your Redis setup
        import redis
        return redis.Redis(
            host=self.config.caching.redis.host,
            port=self.config.caching.redis.port,
            password=self.config.caching.redis.password,
            decode_responses=True
        )

    def _get_db_client(self):
        """Get database client for cost tracking"""
        # Implementation depends on your database setup
        return None  # Replace with actual database client