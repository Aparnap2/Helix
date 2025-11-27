# Helix AI Gateway - Main Hook Implementation
# Extends LiteLLM's existing hook system to add Helix features

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guarails.guardrail_helpers import get_guardrail_instance
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.utils import get_model_info, get_model_cost_map
import litellm

try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False


class HelixHooks:
    """
    Main Helix gateway functionality implemented as LiteLLM hooks

    Features:
    - Semantic caching with vector search
    - PII detection and redaction
    - Cost optimization and tracking
    - Request analytics
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self._initialized = False

        # Initialize components
        self._init_semantic_cache()
        self._init_pii_protection()
        self._init_cost_optimizer()
        self._init_analytics()

        self._initialized = True
        verbose_proxy_logger.info("Helix hooks initialized successfully")

    def _init_semantic_cache(self):
        """Initialize semantic caching components"""
        cache_config = self.config.get('semantic_cache', {})

        if not cache_config.get('enabled', False):
            self.semantic_cache = None
            return

        if not redis_available:
            verbose_proxy_logger.warning("Redis not available, semantic caching disabled")
            self.semantic_cache = None
            return

        self.semantic_cache = {
            'enabled': True,
            'similarity_threshold': cache_config.get('similarity_threshold', 0.88),
            'embedding_model': cache_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            'redis_index': cache_config.get('redis_index', 'helix:semantic:index'),
            'max_cache_size': cache_config.get('max_cache_size', 1000000),
            'ttl': cache_config.get('ttl', 2592000),  # 30 days
            'supported_call_types': set(cache_config.get('supported_call_types', ['completion', 'acompletion']))
        }

        # Initialize sentence transformer
        try:
            self.embedder = SentenceTransformer(self.semantic_cache['embedding_model'])
            verbose_proxy_logger.info(f"Semantic cache initialized with model: {self.semantic_cache['embedding_model']}")
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to initialize sentence transformer: {e}")
            self.semantic_cache = None

    def _init_pii_protection(self):
        """Initialize PII protection components"""
        pii_config = self.config.get('pii_protection', {})

        if not pii_config.get('enabled', False):
            self.pii_protection = None
            return

        try:
            # Configure Presidio with Helix settings
            pii_entities_config = {}
            for entity in pii_config.get('entities', []):
                # Map string entities to PiiEntityType enums
                if isinstance(entity, str):
                    try:
                        entity_type = PiiEntityType(entity)
                        pii_entities_config[entity_type] = PiiAction(pii_config.get('action', 'redact'))
                    except ValueError:
                        # Entity not in enum, use as string
                        pii_entities_config[entity] = PiiAction(pii_config.get('action', 'redact'))

            self.pii_protection = {
                'enabled': True,
                'strict_mode': pii_config.get('strict_mode', False),
                'entities': pii_entities_config,
                'action': pii_config.get('action', 'redact'),
                'output_parse_pii': pii_config.get('output_parse_pii', False),
                'log_all_incidents': pii_config.get('log_all_incidents', True),
                'log_level': pii_config.get('log_level', 'INFO')
            }

            # Initialize Presidio guardrail if available
            try:
                # Try to get existing Presidio instance or create new one
                self.presidio_guardrail = _OPTIONAL_PresidioPIIMasking(
                    pii_entities_config=pii_entities_config,
                    output_parse_pii=self.pii_protection['output_parse_pii']
                )
                verbose_proxy_logger.info("PII protection initialized with Presidio")
            except Exception as e:
                verbose_proxy_logger.warning(f"Failed to initialize Presidio: {e}")
                self.presidio_guardrail = None

        except Exception as e:
            verbose_proxy_logger.error(f"Failed to initialize PII protection: {e}")
            self.pii_protection = None

    def _init_cost_optimizer(self):
        """Initialize cost optimization components"""
        cost_config = self.config.get('cost_optimization', {})

        if not cost_config.get('enabled', False):
            self.cost_optimizer = None
            return

        self.cost_optimizer = {
            'enabled': True,
            'model_swapping': cost_config.get('model_swapping', False),
            'intelligent_routing': cost_config.get('intelligent_routing', False),
            'budget_alerts': cost_config.get('budget_alerts', False),
            'auto_scale': cost_config.get('auto_scale', False),
            'rules': cost_config.get('rules', [])
        }

        # Load model cost information
        try:
            self.model_costs = get_model_cost_map()
            verbose_proxy_logger.info("Cost optimizer initialized")
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to initialize cost optimizer: {e}")
            self.cost_optimizer = None

    def _init_analytics(self):
        """Initialize analytics components"""
        analytics_config = self.config.get('analytics', {})

        if not analytics_config.get('enabled', False):
            self.analytics = None
            return

        self.analytics = {
            'enabled': True,
            'retention_days': analytics_config.get('retention_days', 90),
            'export_s3': analytics_config.get('export_s3'),
            'aggregation_interval': analytics_config.get('aggregation_interval', 300),
            'metrics': analytics_config.get('metrics', [])
        }

        # Initialize Redis for analytics
        try:
            if redis_available:
                self.redis_client = redis.from_url("redis://localhost:6379")
                verbose_proxy_logger.info("Analytics initialized with Redis backend")
            else:
                self.analytics = None
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to initialize analytics: {e}")
            self.analytics = None

    def _get_cache_key(self, data: dict) -> str:
        """Generate cache key for exact matching"""
        # Create deterministic key from request data
        cache_data = {
            'model': data.get('model'),
            'messages': data.get('messages', []),
            'temperature': data.get('temperature'),
            'max_tokens': data.get('max_tokens'),
            'top_p': data.get('top_p'),
            'frequency_penalty': data.get('frequency_penalty'),
            'presence_penalty': data.get('presence_penalty')
        }

        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"helix:exact:{hashlib.sha256(cache_str.encode()).hexdigest()}"

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding for semantic search"""
        if not self.semantic_cache:
            return None

        try:
            embedding = self.embedder.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to get embedding: {e}")
            return None

    async def _semantic_search(self, prompt: str, model: str, top_k: int = 1) -> List[Dict]:
        """Search for semantically similar cached responses"""
        if not self.semantic_cache or not self.redis_client:
            return []

        try:
            # Get embedding for search query
            embedding = await self._get_embedding(prompt)
            if embedding is None:
                return []

            # Convert to bytes for Redis search
            embedding_bytes = embedding.tobytes()

            # Create search query with model filtering
            query = f"(@model:{model})=>[KNN {top_k} @vector $embedding AS score]"

            # Execute search
            results = self.redis_client.ft(self.semantic_cache['redis_index']).search(
                query,
                query_params={"embedding": embedding_bytes}
            )

            # Filter results by similarity threshold
            threshold = self.semantic_cache['similarity_threshold']
            filtered_results = []

            for doc in results.docs:
                score = float(doc.score)
                similarity = 1 - score  # Convert distance to similarity

                if similarity >= threshold:
                    filtered_results.append({
                        'prompt': doc.prompt,
                        'model': doc.model,
                        'response_json': doc.response_json,
                        'similarity': similarity,
                        'cache_key': doc.id
                    })

            return filtered_results

        except Exception as e:
            verbose_proxy_logger.error(f"Semantic search failed: {e}")
            return []

    async def _check_exact_cache(self, data: dict) -> Optional[Dict]:
        """Check for exact cache match"""
        if not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(data)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                verbose_proxy_logger.info(f"Exact cache hit for key: {cache_key}")
                return json.loads(cached_data)

        except Exception as e:
            verbose_proxy_logger.error(f"Exact cache check failed: {e}")

        return None

    async def _store_in_cache(self, data: dict, response: dict, cost: float, latency: float):
        """Store response in both exact and semantic cache"""
        if not self.redis_client:
            return

        try:
            # Store in exact cache
            cache_key = self._get_cache_key(data)
            self.redis_client.setex(
                cache_key,
                self.semantic_cache.get('ttl', 2592000),
                json.dumps(response)
            )

            # Store in semantic cache if enabled
            if self.semantic_cache:
                await self._store_in_semantic_cache(data, response, cost, latency)

        except Exception as e:
            verbose_proxy_logger.error(f"Cache storage failed: {e}")

    async def _store_in_semantic_cache(self, data: dict, response: dict, cost: float, latency: float):
        """Store response in semantic cache with vector search"""
        try:
            # Extract user prompt for embedding
            messages = data.get('messages', [])
            if not messages:
                return

            user_prompt = messages[-1].get('content', '')
            model = data.get('model', '')

            # Get embedding
            embedding = await self._get_embedding(user_prompt)
            if embedding is None:
                return

            # Create semantic cache entry
            vector_key = f"helix:vector:{uuid.uuid4()}"

            cache_entry = {
                'prompt': user_prompt,
                'model': model,
                'response_json': json.dumps(response),
                'vector': embedding.tobytes(),
                'cost': cost,
                'latency': latency,
                'created_at': datetime.now().isoformat()
            }

            # Store in Redis hash
            self.redis_client.hset(vector_key, mapping=cache_entry)

            # Set TTL
            self.redis_client.expire(vector_key, self.semantic_cache.get('ttl', 2592000))

        except Exception as e:
            verbose_proxy_logger.error(f"Semantic cache storage failed: {e}")

    async def _detect_and_redact_pii(self, data: dict, user_id: str) -> Tuple[dict, List[Dict]]:
        """Detect and redact PII from request data"""
        incidents = []

        if not self.pii_protection or not self.presidio_guardrail:
            return data, incidents

        try:
            # Extract messages for PII analysis
            messages = data.get('messages', [])
            if not messages:
                return data, incidents

            # Process each message for PII
            redacted_messages = []

            for message in messages:
                content = message.get('content', '')
                if not content:
                    redacted_messages.append(message)
                    continue

                # Detect PII using Presidio
                try:
                    # Create a dummy user_api_key_dict for the guardrail
                    dummy_user_dict = UserAPIKeyAuth(
                        user_id=user_id,
                        api_key="dummy",
                        spend=0,
                        max_budget=1000
                    )

                    # Create dummy cache
                    dummy_cache = None

                    # Apply PII redaction
                    redacted_message = await self.presidio_guardrail.async_pre_call_hook(
                        data={'messages': [message]},
                        user_api_key_dict=dummy_user_dict,
                        cache=dummy_cache,
                        call_type='completion'
                    )

                    if redacted_message and redacted_message.get('messages'):
                        redacted_messages.extend(redacted_message['messages'])
                        # Log incident if PII was found
                        if self.pii_protection['log_all_incidents']:
                            incidents.append({
                                'user_id': user_id,
                                'timestamp': datetime.now().isoformat(),
                                'action': 'redacted',
                                'content_preview': content[:100] + '...' if len(content) > 100 else content
                            })
                    else:
                        redacted_messages.append(message)

                except Exception as pii_error:
                    verbose_proxy_logger.error(f"PII processing failed: {pii_error}")
                    redacted_messages.append(message)

            # Update data with redacted messages
            if redacted_messages != messages:
                data['messages'] = redacted_messages

        except Exception as e:
            verbose_proxy_logger.error(f"PII detection failed: {e}")

        return data, incidents

    async def _optimize_cost(self, data: dict, user_api_key_dict: UserAPIKeyAuth) -> dict:
        """Apply cost optimization rules"""
        if not self.cost_optimizer:
            return data

        try:
            original_model = data.get('model')

            # Check if model swapping is enabled
            if self.cost_optimizer.get('model_swapping', False):
                optimized_model = await self._suggest_cheaper_alternative(data)
                if optimized_model and optimized_model != original_model:
                    data['model'] = optimized_model
                    verbose_proxy_logger.info(f"Swapped model {original_model} -> {optimized_model}")

        except Exception as e:
            verbose_proxy_logger.error(f"Cost optimization failed: {e}")

        return data

    async def _suggest_cheaper_alternative(self, data: dict) -> Optional[str]:
        """Suggest cheaper model alternative"""
        try:
            # Get current model
            current_model = data.get('model')
            if not current_model:
                return None

            # Check optimization rules
            for rule in self.cost_optimizer.get('rules', []):
                if await self._matches_rule(rule, data):
                    action = rule.get('action', {})
                    alternative_model = action.get('model')
                    if alternative_model:
                        return alternative_model

        except Exception as e:
            verbose_proxy_logger.error(f"Failed to suggest alternative: {e}")

        return None

    async def _matches_rule(self, rule: dict, data: dict) -> bool:
        """Check if request matches optimization rule"""
        try:
            conditions = rule.get('conditions', {})

            # Check token conditions
            if 'max_tokens' in conditions:
                max_tokens = data.get('max_tokens', 0)
                if max_tokens > conditions['max_tokens']:
                    return False

            # Add more condition checks as needed
            return True

        except Exception as e:
            verbose_proxy_logger.error(f"Rule matching failed: {e}")
            return False

    async def _track_spend(self, user_id: str, cost: float, model: str):
        """Track spending for analytics and budgeting"""
        if not self.analytics or not self.redis_client:
            return

        try:
            today = datetime.now().strftime("%Y-%m-%d")

            # Track total spend
            self.redis_client.zincrby("helix:spend:total", cost, today)

            # Track user spend
            user_key = f"helix:spend:user:{user_id}"
            self.redis_client.zincrby(user_key, cost, today)

            # Track model spend
            model_key = f"helix:spend:model:{model}"
            self.redis_client.zincrby(model_key, cost, today)

        except Exception as e:
            verbose_proxy_logger.error(f"Spend tracking failed: {e}")

    async def _log_pii_incident(self, incident: dict):
        """Log PII incident for analytics"""
        if not self.analytics or not self.redis_client:
            return

        try:
            incident_key = "helix:pii:incidents"
            self.redis_client.lpush(incident_key, json.dumps(incident))

            # Keep only last 1000 incidents
            self.redis_client.ltrim(incident_key, 0, 999)

        except Exception as e:
            verbose_proxy_logger.error(f"PII incident logging failed: {e}")

    async def _update_analytics(self, data: dict, response: dict, cost: float, latency: float, cache_hit: bool):
        """Update analytics metrics"""
        if not self.analytics or not self.redis_client:
            return

        try:
            # Increment request counters
            self.redis_client.incr("helix:requests:total")

            if cache_hit:
                self.redis_client.incr("helix:requests:cache_hits")

            # Update latency metrics
            latency_key = "helix:latency:p99"
            # In production, use more sophisticated percentile calculation
            current_p99 = float(self.redis_client.get(latency_key) or 0)
            if latency > current_p99:
                self.redis_client.set(latency_key, str(latency))

        except Exception as e:
            verbose_proxy_logger.error(f"Analytics update failed: {e}")

    # Main hook methods
    async def pre_call_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        call_type: str
    ) -> dict:
        """
        Pre-call hook implementing:
        - Exact cache check
        - Semantic cache search
        - PII detection and redaction
        - Cost optimization
        """
        if not self._initialized:
            return data

        start_time = time.time()
        user_id = getattr(user_api_key_dict, 'user_id', 'anonymous')

        try:
            # 1. Check exact cache first
            cached_response = await self._check_exact_cache(data)
            if cached_response:
                await self._update_analytics(data, cached_response, 0, time.time() - start_time, True)
                return cached_response

            # 2. Check semantic cache
            if self.semantic_cache and call_type in self.semantic_cache.get('supported_call_types', []):
                messages = data.get('messages', [])
                if messages:
                    user_prompt = messages[-1].get('content', '')
                    model = data.get('model', '')

                    semantic_results = await self._semantic_search(user_prompt, model)
                    if semantic_results:
                        best_match = semantic_results[0]
                        cached_response = json.loads(best_match['response_json'])
                        await self._update_analytics(data, cached_response, 0, time.time() - start_time, True)
                        return cached_response

            # 3. Apply PII detection and redaction
            if self.pii_protection:
                data, pii_incidents = await self._detect_and_redact_pii(data, user_id)

                # Log PII incidents
                for incident in pii_incidents:
                    await self._log_pii_incident(incident)

            # 4. Apply cost optimization
            if self.cost_optimizer:
                data = await self._optimize_cost(data, user_api_key_dict)

            # Store original data for post-processing
            data['_helix_metadata'] = {
                'start_time': start_time,
                'user_id': user_id,
                'original_model': data.get('model')
            }

            return data

        except Exception as e:
            verbose_proxy_logger.error(f"Pre-call hook failed: {e}")
            return data

    async def post_call_hook(
        self,
        response: Any,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str
    ) -> Any:
        """
        Post-call hook implementing:
        - Cache storage (exact + semantic)
        - Spend tracking
        - Analytics updates
        """
        if not self._initialized:
            return response

        try:
            helix_metadata = data.pop('_helix_metadata', {})
            if not helix_metadata:
                return response

            start_time = helix_metadata.get('start_time', time.time())
            user_id = helix_metadata.get('user_id', 'anonymous')

            # Calculate metrics
            latency = time.time() - start_time

            # Extract cost from response or calculate
            cost = 0
            try:
                if hasattr(response, '_hidden_params'):
                    cost = response._hidden_params.get('response_cost', 0)
                else:
                    # Calculate cost based on model and token usage
                    model = data.get('model', '')
                    if hasattr(response, 'usage'):
                        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                        completion_tokens = response.usage.completion_tokens if response.usage else 0

                        # Get cost from model cost map
                        model_costs = get_model_cost_map()
                        if model in model_costs:
                            prompt_cost = model_costs[model]['input_cost_per_token'] * prompt_tokens
                            completion_cost = model_costs[model]['output_cost_per_token'] * completion_tokens
                            cost = prompt_cost + completion_cost
            except Exception as cost_error:
                verbose_proxy_logger.error(f"Cost calculation failed: {cost_error}")

            # 1. Store in cache
            await self._store_in_cache(data, response, cost, latency)

            # 2. Track spend
            await self._track_spend(user_id, cost, data.get('model', ''))

            # 3. Update analytics
            await self._update_analytics(data, response, cost, latency, False)

            return response

        except Exception as e:
            verbose_proxy_logger.error(f"Post-call hook failed: {e}")
            return response

    async def health_check(self) -> Dict[str, Any]:
        """Health check for Helix components"""
        health = {
            'initialized': self._initialized,
            'semantic_cache': self.semantic_cache is not None,
            'pii_protection': self.pii_protection is not None,
            'cost_optimizer': self.cost_optimizer is not None,
            'analytics': self.analytics is not None,
            'redis_connected': bool(self.redis_client) if hasattr(self, 'redis_client') else False
        }

        # Test Redis connection
        if health['redis_connected']:
            try:
                self.redis_client.ping()
                health['redis_status'] = 'connected'
            except Exception as e:
                health['redis_status'] = f'error: {e}'

        # Test sentence transformer
        if health['semantic_cache'] and hasattr(self, 'embedder'):
            try:
                test_embedding = self.embedder.encode("test", normalize_embeddings=True)
                health['embedder_status'] = 'working'
            except Exception as e:
                health['embedder_status'] = f'error: {e}'

        return health