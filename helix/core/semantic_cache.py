"""
Helix Semantic Cache Implementation
Production-ready semantic caching with Redis Vector Search and HNSW optimization
"""

import json
import time
import hashlib
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import aiofiles
import pickle
import gzip
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

from helix.core.config import get_config, is_helix_enabled

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    prompt: str
    model: str
    response: str
    embedding: List[float]
    cost: float
    latency: float
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    cache_level: str = "semantic"
    embedding_model: str = "text-embedding-3-small"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Vector search result"""
    entry: CacheEntry
    similarity_score: float
    distance: float
    search_time: float


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0
    avg_similarity: float = 0.0
    avg_search_time: float = 0.0
    memory_usage: int = 0
    eviction_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate percentage"""
        return (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Semantic cache hit rate percentage"""
        return (self.semantic_hits / (self.semantic_hits + self.semantic_misses) * 100) if (self.semantic_hits + self.semantic_misses) > 0 else 0.0


@dataclass
class AdaptiveThreshold:
    """Adaptive similarity threshold configuration"""
    base_threshold: float = 0.85
    min_threshold: float = 0.70
    max_threshold: float = 0.95
    performance_window: int = 300  # seconds
    hit_rate_target: float = 80.0
    adjustment_factor: float = 0.05
    current_threshold: float = 0.85


class EmbeddingProcessor:
    """Handles text embedding with batching and caching"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embedding_cache = {}
        self.batch_queue = asyncio.Queue()
        self.processing_batch = False
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        """Initialize the embedding model"""
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                lambda: SentenceTransformer(self.model_name)
            )
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    async def encode_single(self, text: str) -> List[float]:
        """Encode a single text string"""
        if not self.model:
            await self.initialize()

        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        try:
            # Process in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                text
            )

            # Convert to list and cache
            embedding_list = embedding.tolist()
            self.embedding_cache[text_hash] = embedding_list

            # Limit cache size
            if len(self.embedding_cache) > 10000:
                # Remove oldest entries
                keys_to_remove = list(self.embedding_cache.keys())[:1000]
                for key in keys_to_remove:
                    del self.embedding_cache[key]

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts in batch for efficiency"""
        if not self.model:
            await self.initialize()

        try:
            # Filter out cached embeddings
            uncached_texts = []
            uncached_indices = []
            embeddings = [None] * len(texts)

            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.embedding_cache:
                    embeddings[i] = self.embedding_cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            if uncached_texts:
                # Process uncached texts in batch
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.executor,
                    self.model.encode,
                    uncached_texts
                )

                # Update cache and results
                for i, (text, embedding) in enumerate(zip(uncached_texts, batch_embeddings)):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    embedding_list = embedding.tolist()
                    self.embedding_cache[text_hash] = embedding_list
                    embeddings[uncached_indices[i]] = embedding_list

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=False)


class VectorIndexManager:
    """Manages Redis Vector Search operations"""

    def __init__(self, redis_pool: ConnectionPool):
        self.redis_pool = redis_pool
        self.index_name = "helix:semantic:index"
        self.vector_prefix = "helix:vector:"

    async def add_vector(self, entry: CacheEntry) -> bool:
        """Add a vector entry to the index"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            # Create document hash
            doc_key = f"{self.vector_prefix}{entry.key}"

            # Prepare document fields
            doc_data = {
                "prompt": entry.prompt,
                "model": entry.model,
                "response_json": json.dumps({"response": entry.response}),
                "vector": np.array(entry.embedding, dtype=np.float32).tobytes(),
                "cost": str(entry.cost),
                "latency": str(entry.latency),
                "hit_count": str(entry.hit_count),
                "last_accessed": str(entry.last_accessed),
                "created_at": str(entry.created_at),
                "expires_at": str(entry.expires_at),
                "cache_level": entry.cache_level,
                "embedding_model": entry.embedding_model
            }

            # Add to hash
            await redis_conn.hset(doc_key, mapping=doc_data)

            # Set expiration
            ttl = int(entry.expires_at - time.time())
            if ttl > 0:
                await redis_conn.expire(doc_key, ttl)

            await redis_conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to add vector to index: {e}")
            return False

    async def search_vectors(
        self,
        query_embedding: List[float],
        model_filter: Optional[str] = None,
        similarity_threshold: float = 0.85,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            # Build search query
            query_params = {
                "vector": np.array(query_embedding, dtype=np.float32).tobytes(),
                "PARAMS": 4,  # EF_RUNTIME parameter
                "RETURN": 6,  # Return 6 fields
                "DIALECT": 2,  # Use DIALECT 2 for vector search
                "LIMIT": limit
            }

            # Add filter if specified
            filter_clause = ""
            if model_filter:
                filter_clause = f"@model:{{{model_filter}}}"

            # Construct FT.SEARCH command
            search_query = f"*=>[KNN {limit} @vector $vector AS distance]"
            if filter_clause:
                search_query = f"({filter_clause})=>[KNN {limit} @vector $vector AS distance]"

            start_time = time.time()

            # Execute search
            result = await redis_conn.ft(self.index_name).search(
                search_query,
                query_params=query_params
            )

            search_time = time.time() - start_time

            # Process results
            search_results = []
            for doc in result.docs:
                try:
                    # Extract vector data
                    vector_bytes = doc.__dict__.get('vector', b'')
                    vector = np.frombuffer(vector_bytes, dtype=np.float32).tolist()

                    # Calculate similarity from distance (cosine distance to similarity)
                    distance = float(doc.__dict__.get('distance', 1.0))
                    similarity = 1.0 - distance

                    if similarity >= similarity_threshold:
                        # Reconstruct cache entry
                        entry = CacheEntry(
                            key=doc.id.replace(self.vector_prefix, ""),
                            prompt=doc.prompt,
                            model=doc.model,
                            response=json.loads(doc.response_json).get("response", ""),
                            embedding=vector,
                            cost=float(doc.cost),
                            latency=float(doc.latency),
                            hit_count=int(doc.hit_count),
                            created_at=float(doc.created_at),
                            last_accessed=float(doc.last_accessed),
                            expires_at=float(doc.expires_at),
                            cache_level=doc.cache_level,
                            embedding_model=doc.embedding_model
                        )

                        search_results.append(SearchResult(
                            entry=entry,
                            similarity_score=similarity,
                            distance=distance,
                            search_time=search_time
                        ))

                except Exception as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue

            await redis_conn.close()

            # Sort by similarity (highest first)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)

            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def update_hit_count(self, key: str, increment: int = 1):
        """Update hit count for a cache entry"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            doc_key = f"{self.vector_prefix}{key}"

            await redis_conn.hincrby(doc_key, "hit_count", increment)
            await redis_conn.hset(doc_key, "last_accessed", str(time.time()))

            await redis_conn.close()

        except Exception as e:
            logger.error(f"Failed to update hit count: {e}")

    async def delete_vector(self, key: str) -> bool:
        """Delete a vector entry"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            doc_key = f"{self.vector_prefix}{key}"

            result = await redis_conn.delete(doc_key)
            await redis_conn.close()

            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            # Scan for vector keys
            cursor = 0
            deleted_count = 0
            current_time = time.time()

            while True:
                cursor, keys = await redis_conn.scan(cursor, match=f"{self.vector_prefix}*", count=100)

                for key in keys:
                    try:
                        # Check expiration
                        expires_at = await redis_conn.hget(key, "expires_at")
                        if expires_at and float(expires_at) < current_time:
                            await redis_conn.delete(key)
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to check expiration for {key}: {e}")
                        continue

                if cursor == 0:
                    break

            await redis_conn.close()
            return deleted_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


class SemanticCache:
    """
    Production-ready semantic cache with Redis Vector Search
    Features:
    - Hybrid exact + semantic caching
    - Adaptive similarity thresholds
    - Batch processing optimization
    - Connection pooling
    - Async processing
    - Memory optimization
    """

    def __init__(self):
        self.config = get_config()
        self.redis_pool = None
        self.embedding_processor = None
        self.vector_manager = None
        self.metrics = CacheMetrics()
        self.adaptive_threshold = AdaptiveThreshold()
        self.exact_cache = {}
        self.cache_lock = asyncio.Lock()
        self.performance_history = deque(maxlen=1000)
        self.cleanup_task = None
        self.initialized = False

    async def initialize(self):
        """Initialize the semantic cache"""
        if self.initialized:
            return

        try:
            # Initialize Redis connection pool
            redis_config = self.config.caching.redis
            self.redis_pool = ConnectionPool(
                host=redis_config.host,
                port=redis_config.port,
                password=redis_config.password,
                db=redis_config.database,
                max_connections=redis_config.max_connections,
                socket_timeout=redis_config.socket_timeout,
                socket_connect_timeout=redis_config.connection_timeout,
                retry_on_timeout=True,
                decode_responses=False
            )

            # Test Redis connection
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            await redis_conn.ping()
            await redis_conn.close()

            # Initialize embedding processor
            self.embedding_processor = EmbeddingProcessor(
                model_name=self.config.caching.vector_search.embedding_model,
                batch_size=self.config.caching.batch_size
            )
            await self.embedding_processor.initialize()

            # Initialize vector index manager
            self.vector_manager = VectorIndexManager(self.redis_pool)

            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_entries())

            self.initialized = True
            logger.info("Semantic cache initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            raise

    async def get(
        self,
        prompt: str,
        model: str,
        use_exact_cache: bool = True,
        use_semantic_cache: bool = True
    ) -> Optional[str]:
        """
        Get cached response using hybrid strategy
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # 1. Try exact cache first (fastest)
            if use_exact_cache:
                exact_result = await self._get_exact_cache(prompt, model)
                if exact_result:
                    self.metrics.cache_hits += 1
                    self.metrics.avg_search_time = (time.time() - start_time)
                    logger.debug(f"Exact cache hit for model: {model}")
                    return exact_result

            # 2. Try semantic cache
            if use_semantic_cache:
                semantic_result = await self._get_semantic_cache(prompt, model)
                if semantic_result:
                    self.metrics.cache_hits += 1
                    self.metrics.semantic_hits += 1
                    search_time = time.time() - start_time
                    self.metrics.avg_search_time = (
                        self.metrics.avg_search_time * 0.9 + search_time * 0.1
                    )
                    logger.debug(f"Semantic cache hit for model: {model}")
                    return semantic_result
                else:
                    self.metrics.semantic_misses += 1

            self.metrics.cache_misses += 1
            return None

        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            self.metrics.cache_misses += 1
            return None

    async def _get_exact_cache(self, prompt: str, model: str) -> Optional[str]:
        """Get response from exact cache"""
        try:
            # Create exact cache key
            cache_key = self._create_exact_key(prompt, model)

            # Check in-memory cache first
            if cache_key in self.exact_cache:
                entry = self.exact_cache[cache_key]
                if entry.expires_at > time.time():
                    entry.hit_count += 1
                    entry.last_accessed = time.time()
                    return entry.response
                else:
                    # Remove expired entry
                    del self.exact_cache[cache_key]

            # Check Redis exact cache
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            cached_data = await redis_conn.hget("helix:exact:cache", cache_key)

            if cached_data:
                try:
                    entry_data = json.loads(cached_data)
                    entry = CacheEntry(**entry_data)

                    if entry.expires_at > time.time():
                        # Update in-memory cache
                        entry.hit_count += 1
                        entry.last_accessed = time.time()
                        self.exact_cache[cache_key] = entry

                        await redis_conn.close()
                        return entry.response
                    else:
                        # Remove expired entry
                        await redis_conn.hdel("helix:exact:cache", cache_key)

                except Exception as e:
                    logger.warning(f"Failed to parse exact cache entry: {e}")

            await redis_conn.close()
            return None

        except Exception as e:
            logger.error(f"Exact cache lookup failed: {e}")
            return None

    async def _get_semantic_cache(self, prompt: str, model: str) -> Optional[str]:
        """Get response from semantic cache"""
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_processor.encode_single(prompt)

            # Get adaptive threshold
            current_threshold = self._get_adaptive_threshold()

            # Search for similar vectors
            search_results = await self.vector_manager.search_vectors(
                query_embedding=query_embedding,
                model_filter=model,
                similarity_threshold=current_threshold,
                limit=5
            )

            if search_results:
                # Use the best match
                best_match = search_results[0]

                # Update metrics
                self.metrics.avg_similarity = (
                    self.metrics.avg_similarity * 0.9 + best_match.similarity_score * 0.1
                )

                # Update hit count
                await self.vector_manager.update_hit_count(best_match.entry.key)

                # Update performance history
                self.performance_history.append({
                    'timestamp': time.time(),
                    'similarity': best_match.similarity_score,
                    'search_time': best_match.search_time,
                    'hit': True
                })

                # Update adaptive threshold
                await self._update_adaptive_threshold()

                return best_match.entry.response

            # No match found
            self.performance_history.append({
                'timestamp': time.time(),
                'similarity': 0.0,
                'search_time': best_match.search_time if search_results else time.time(),
                'hit': False
            })

            await self._update_adaptive_threshold()
            return None

        except Exception as e:
            logger.error(f"Semantic cache lookup failed: {e}")
            return None

    async def set(
        self,
        prompt: str,
        model: str,
        response: str,
        cost: float = 0.0,
        latency: float = 0.0,
        ttl: Optional[int] = None
    ):
        """Store response in cache"""
        if not self.initialized:
            await self.initialize()

        try:
            # Determine TTL
            if ttl is None:
                ttl = self.config.caching.default_ttl

            # Create cache entry
            cache_key = self._create_cache_key(prompt, model)
            expires_at = time.time() + ttl

            # Generate embedding
            embedding = await self.embedding_processor.encode_single(prompt)

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                prompt=prompt,
                model=model,
                response=response,
                embedding=embedding,
                cost=cost,
                latency=latency,
                expires_at=expires_at,
                cache_level="semantic",
                embedding_model=self.config.caching.vector_search.embedding_model
            )

            # Store in exact cache (Redis)
            await self._store_exact_cache(entry)

            # Store in semantic cache (Vector Search)
            if self.config.caching.vector_search.enabled:
                await self.vector_manager.add_vector(entry)

            logger.debug(f"Cached response for model: {model}")

        except Exception as e:
            logger.error(f"Cache set operation failed: {e}")

    async def _store_exact_cache(self, entry: CacheEntry):
        """Store entry in exact cache"""
        try:
            # Update in-memory cache
            async with self.cache_lock:
                self.exact_cache[entry.key] = entry

                # Limit in-memory cache size
                if len(self.exact_cache) > 1000:
                    # Remove LRU entries
                    sorted_entries = sorted(
                        self.exact_cache.items(),
                        key=lambda x: x[1].last_accessed
                    )
                    keys_to_remove = [k for k, v in sorted_entries[:100]]
                    for key in keys_to_remove:
                        del self.exact_cache[key]

            # Store in Redis exact cache
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            entry_data = json.dumps(asdict(entry), default=str)
            await redis_conn.hset("helix:exact:cache", entry.key, entry_data)
            await redis_conn.close()

        except Exception as e:
            logger.error(f"Failed to store exact cache: {e}")

    async def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries"""
        try:
            if pattern:
                # Invalidate matching entries
                async with self.cache_lock:
                    keys_to_remove = [
                        k for k in self.exact_cache.keys()
                        if pattern in k
                    ]
                    for key in keys_to_remove:
                        del self.exact_cache[key]

                # Invalidate Redis entries
                redis_conn = redis.Redis(connection_pool=self.redis_pool)
                exact_keys = await redis_conn.hkeys("helix:exact:cache")
                for key in exact_keys:
                    if pattern in key.decode():
                        await redis_conn.hdel("helix:exact:cache", key)

                await redis_conn.close()
            else:
                # Clear all cache
                async with self.cache_lock:
                    self.exact_cache.clear()

                redis_conn = redis.Redis(connection_pool=self.redis_pool)
                await redis_conn.delete("helix:exact:cache")
                await redis_conn.close()

            logger.info(f"Invalidated cache entries matching: {pattern}")

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": self.metrics.hit_rate,
            "semantic_hits": self.metrics.semantic_hits,
            "semantic_misses": self.metrics.semantic_misses,
            "semantic_hit_rate": self.metrics.semantic_hit_rate,
            "avg_similarity": self.metrics.avg_similarity,
            "avg_search_time": self.metrics.avg_search_time,
            "adaptive_threshold": self.adaptive_threshold.current_threshold,
            "exact_cache_size": len(self.exact_cache),
            "memory_usage": self.metrics.memory_usage,
            "eviction_count": self.metrics.eviction_count
        }

    async def warm_cache(self, queries: List[str], models: List[str]):
        """Warm cache with common queries"""
        try:
            logger.info(f"Starting cache warm-up with {len(queries)} queries")

            # Batch encode queries
            embeddings = await self.embedding_processor.encode_batch(queries)

            # Process in batches
            batch_size = self.config.caching.batch_size
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]

                # Add to warm-up queue
                redis_conn = redis.Redis(connection_pool=self.redis_pool)
                for query, embedding in zip(batch_queries, batch_embeddings):
                    await redis_conn.xadd(
                        "helix:cache:warming:queue",
                        {
                            "query": query,
                            "embedding": json.dumps(embedding),
                            "timestamp": str(time.time())
                        }
                    )

                await redis_conn.close()

                # Small delay to avoid overwhelming Redis
                await asyncio.sleep(0.1)

            logger.info("Cache warm-up completed")

        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}")

    def _create_cache_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model"""
        combined = f"{model}:{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _create_exact_key(self, prompt: str, model: str) -> str:
        """Create exact cache key"""
        return f"{model}:{hashlib.md5(prompt.encode()).hexdigest()}"

    def _get_adaptive_threshold(self) -> float:
        """Get current adaptive similarity threshold"""
        return self.adaptive_threshold.current_threshold

    async def _update_adaptive_threshold(self):
        """Update adaptive threshold based on performance"""
        try:
            if len(self.performance_history) < 10:
                return

            # Calculate recent hit rate
            recent_window = self.adaptive_threshold.performance_window
            current_time = time.time()
            recent_entries = [
                entry for entry in self.performance_history
                if current_time - entry['timestamp'] <= recent_window
            ]

            if len(recent_entries) < 5:
                return

            hit_rate = sum(1 for entry in recent_entries if entry['hit']) / len(recent_entries)

            # Adjust threshold based on hit rate
            target_rate = self.adaptive_threshold.hit_rate_target / 100.0
            adjustment = self.adaptive_threshold.adjustment_factor

            if hit_rate < target_rate:
                # Increase threshold to be more selective
                new_threshold = min(
                    self.adaptive_threshold.current_threshold + adjustment,
                    self.adaptive_threshold.max_threshold
                )
            elif hit_rate > target_rate + 0.1:
                # Decrease threshold to be more permissive
                new_threshold = max(
                    self.adaptive_threshold.current_threshold - adjustment,
                    self.adaptive_threshold.min_threshold
                )
            else:
                new_threshold = self.adaptive_threshold.current_threshold

            if new_threshold != self.adaptive_threshold.current_threshold:
                logger.info(f"Adjusting similarity threshold: {self.adaptive_threshold.current_threshold:.3f} -> {new_threshold:.3f}")
                self.adaptive_threshold.current_threshold = new_threshold

                # Store in Redis for persistence
                redis_conn = redis.Redis(connection_pool=self.redis_pool)
                await redis_conn.hset(
                    "helix:adaptive:thresholds",
                    "current_threshold",
                    str(new_threshold)
                )
                await redis_conn.close()

        except Exception as e:
            logger.error(f"Failed to update adaptive threshold: {e}")

    async def _cleanup_expired_entries(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                # Cleanup exact cache
                current_time = time.time()
                async with self.cache_lock:
                    expired_keys = [
                        k for k, v in self.exact_cache.items()
                        if v.expires_at <= current_time
                    ]
                    for key in expired_keys:
                        del self.exact_cache[key]
                        self.metrics.eviction_count += 1

                # Cleanup vector entries
                if self.vector_manager:
                    deleted_count = await self.vector_manager.cleanup_expired()
                    self.metrics.eviction_count += deleted_count

                # Sleep for 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
                await asyncio.sleep(60)  # Retry sooner on error

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            if self.embedding_processor:
                await self.embedding_processor.cleanup()

            if self.redis_pool:
                await self.redis_pool.disconnect()

            logger.info("Semantic cache cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global semantic cache instance
_semantic_cache = None


async def get_semantic_cache() -> SemanticCache:
    """Get global semantic cache instance"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
        await _semantic_cache.initialize()
    return _semantic_cache


async def is_semantic_cache_enabled() -> bool:
    """Check if semantic cache is enabled"""
    return (is_helix_enabled() and
            get_config().caching.enabled and
            get_config().caching.vector_search.enabled)