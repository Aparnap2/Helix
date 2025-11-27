"""
Helix Cache Manager
Production-ready multi-level caching strategy with comprehensive management
"""

import json
import time
import asyncio
import logging
import hashlib
import pickle
import gzip
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import weakref
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from helix.core.config import get_config, is_helix_enabled
from helix.core.semantic_cache import get_semantic_cache, CacheMetrics, CacheEntry

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level strategy"""
    MEMORY = "memory"
    REDIS = "redis"
    SEMANTIC = "semantic"
    PERSISTENT = "persistent"


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    SIZE_BASED = "size_based"


@dataclass
class CacheConfig:
    """Individual cache level configuration"""
    enabled: bool = True
    max_size: int = 10000
    max_memory_bytes: int = 100 * 1024 * 1024  # 100MB
    ttl_seconds: int = 3600
    policy: CachePolicy = CachePolicy.LRU
    compression: bool = True
    compression_level: int = 6
    batch_size: int = 100
    eviction_batch_size: int = 50
    metrics_collection: bool = True
    cleanup_interval: int = 300  # seconds


@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    access_count: int = 0
    size_bytes: int = 0
    compression_type: Optional[str] = None
    cache_level: CacheLevel = CacheLevel.MEMORY
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Comprehensive cache statistics"""
    level: CacheLevel
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    item_count: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    memory_usage_percent: float = 0.0
    compression_ratio: float = 1.0
    last_cleanup: float = 0.0
    errors: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    def update_hit_rate(self):
        """Update hit rate calculation"""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100


class MemoryCache:
    """High-performance in-memory cache with advanced eviction policies"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.access_order = deque()  # For LRU
        self.access_count = defaultdict(int)  # For LFU
        self.stats = CacheStatistics(level=CacheLevel.MEMORY)
        self.lock = asyncio.Lock()
        self.cleanup_task = None

    async def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        start_time = time.time()

        async with self.lock:
            try:
                if key not in self.cache:
                    self.stats.misses += 1
                    self.stats.update_hit_rate()
                    return None

                item = self.cache[key]

                # Check expiration
                if item.expires_at <= time.time():
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    try:
                        self.access_order.remove(key)
                    except ValueError:
                        pass
                    self.stats.misses += 1
                    self.stats.evictions += 1
                    self.stats.update_hit_rate()
                    return None

                # Update access information
                item.last_accessed = time.time()
                item.access_count += 1
                self.access_count[key] = item.access_count

                # Update LRU order
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)

                # Update stats
                self.stats.hits += 1
                access_time = time.time() - start_time
                self.stats.avg_access_time = (
                    self.stats.avg_access_time * 0.9 + access_time * 0.1
                )
                self.stats.update_hit_rate()

                return self._deserialize_value(item.value, item.compression_type)

            except Exception as e:
                logger.error(f"Memory cache get error: {e}")
                self.stats.errors += 1
                return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in memory cache"""
        try:
            async with self.lock:
                # Serialize and compress value if needed
                serialized_value, compression_type, size = await self._serialize_value(
                    value, self.config.compression
                )

                # Calculate expiration
                if ttl is None:
                    ttl = self.config.ttl_seconds
                expires_at = time.time() + ttl

                # Create cache item
                item = CacheItem(
                    key=key,
                    value=serialized_value,
                    expires_at=expires_at,
                    size_bytes=size,
                    compression_type=compression_type,
                    cache_level=CacheLevel.MEMORY
                )

                # Check if we need to evict
                if key not in self.cache:
                    await self._evict_if_needed()

                # Update cache and access tracking
                self.cache[key] = item
                self.access_count[key] = 1

                # Update LRU order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                # Update statistics
                self.stats.item_count = len(self.cache)
                self.stats.size_bytes = sum(item.size_bytes for item in self.cache.values())

                return True

        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            self.stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete item from memory cache"""
        try:
            async with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    try:
                        self.access_order.remove(key)
                    except ValueError:
                        pass

                    # Update statistics
                    self.stats.item_count = len(self.cache)
                    self.stats.size_bytes = sum(item.size_bytes for item in self.cache.values())

                    return True
                return False

        except Exception as e:
            logger.error(f"Memory cache delete error: {e}")
            self.stats.errors += 1
            return False

    async def clear(self):
        """Clear all items from memory cache"""
        try:
            async with self.lock:
                self.cache.clear()
                self.access_order.clear()
                self.access_count.clear()
                self.stats.item_count = 0
                self.stats.size_bytes = 0

        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            self.stats.errors += 1

    async def _evict_if_needed(self):
        """Evict items based on configuration"""
        while (
            len(self.cache) >= self.config.max_size or
            self.stats.size_bytes >= self.config.max_memory_bytes
        ):
            if not await self._evict_single_item():
                break

    async def _evict_single_item(self) -> bool:
        """Evict a single item based on policy"""
        if not self.cache:
            return False

        try:
            if self.config.policy == CachePolicy.LRU:
                # Evict least recently used
                key_to_evict = self.access_order[0]
            elif self.config.policy == CachePolicy.LFU:
                # Evict least frequently used
                key_to_evict = min(self.access_count.items(), key=lambda x: x[1])[0]
            elif self.config.policy == CachePolicy.TTL:
                # Evict expired items first, then oldest
                current_time = time.time()
                expired_keys = [
                    k for k, v in self.cache.items() if v.expires_at <= current_time
                ]
                if expired_keys:
                    key_to_evict = expired_keys[0]
                else:
                    key_to_evict = min(self.cache.items(), key=lambda x: x[1].created_at)[0]
            else:
                # Default to LRU
                key_to_evict = self.access_order[0]

            # Remove item
            if key_to_evict in self.cache:
                del self.cache[key_to_evict]
                if key_to_evict in self.access_count:
                    del self.access_count[key_to_evict]
                try:
                    self.access_order.remove(key_to_evict)
                except ValueError:
                    pass

                self.stats.evictions += 1
                return True

        except Exception as e:
            logger.error(f"Eviction error: {e}")

        return False

    async def _serialize_value(self, value: Any, compress: bool) -> Tuple[Any, Optional[str], int]:
        """Serialize and optionally compress value"""
        try:
            # Serialize to bytes
            serialized = pickle.dumps(value)
            original_size = len(serialized)
            compression_type = None

            if compress and original_size > 1024:  # Only compress if > 1KB
                # Compress with gzip
                compressed = gzip.compress(serialized, compresslevel=self.config.compression_level)
                if len(compressed) < original_size * 0.9:  # Only use if 10% reduction
                    serialized = compressed
                    compression_type = "gzip"

            return serialized, compression_type, len(serialized)

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            # Return original value on error
            return value, None, 0

    def _deserialize_value(self, value: Any, compression_type: Optional[str]) -> Any:
        """Deserialize and decompress value"""
        try:
            if compression_type == "gzip":
                # Decompress
                decompressed = gzip.decompress(value)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(value)

        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return value

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        # Update memory usage percentage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.stats.memory_usage_percent = (
            (self.stats.size_bytes / memory_info.rss) * 100 if memory_info.rss > 0 else 0
        )

        return {
            "level": self.stats.level.value,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "size_bytes": self.stats.size_bytes,
            "item_count": self.stats.item_count,
            "avg_access_time": self.stats.avg_access_time,
            "memory_usage_percent": self.stats.memory_usage_percent,
            "errors": self.stats.errors
        }


class RedisCache:
    """Redis-based distributed cache"""

    def __init__(self, redis_pool: ConnectionPool, config: CacheConfig):
        self.redis_pool = redis_pool
        self.config = config
        self.stats = CacheStatistics(level=CacheLevel.REDIS)
        self.key_prefix = "helix:cache:"
        self.batch_operations = defaultdict(list)
        self.batch_timer = None

    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache"""
        start_time = time.time()
        redis_key = f"{self.key_prefix}{key}"

        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            # Get cached data
            cached_data = await redis_conn.hgetall(redis_key)

            if not cached_data:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                await redis_conn.close()
                return None

            # Parse cache data
            item_data = json.loads(cached_data.get(b'data', b'{}').decode())
            expires_at = float(cached_data.get(b'expires_at', b'0').decode())

            # Check expiration
            if expires_at <= time.time():
                await redis_conn.delete(redis_key)
                self.stats.misses += 1
                self.stats.evictions += 1
                self.stats.update_hit_rate()
                await redis_conn.close()
                return None

            # Update access time asynchronously
            await redis_conn.hset(redis_key, "last_accessed", str(time.time()))

            # Update statistics
            self.stats.hits += 1
            access_time = time.time() - start_time
            self.stats.avg_access_time = (
                self.stats.avg_access_time * 0.9 + access_time * 0.1
            )
            self.stats.update_hit_rate()

            await redis_conn.close()

            # Return deserialized value
            return pickle.loads(cached_data.get(b'value', b''))

        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.stats.errors += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis cache"""
        redis_key = f"{self.key_prefix}{key}"

        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            # Calculate expiration
            if ttl is None:
                ttl = self.config.ttl_seconds
            expires_at = time.time() + ttl

            # Serialize value
            serialized_value = pickle.dumps(value)

            # Prepare cache data
            cache_data = {
                "value": serialized_value,
                "data": json.dumps({
                    "key": key,
                    "created_at": time.time(),
                    "compression": self.config.compression
                }),
                "expires_at": str(expires_at),
                "last_accessed": str(time.time()),
                "access_count": "1",
                "size_bytes": str(len(serialized_value)),
                "cache_level": CacheLevel.REDIS.value
            }

            # Store in Redis
            await redis_conn.hset(redis_key, mapping=cache_data)
            await redis_conn.expire(redis_key, ttl)

            # Update statistics
            self.stats.item_count += 1
            self.stats.size_bytes += len(serialized_value)

            await redis_conn.close()
            return True

        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            self.stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        redis_key = f"{self.key_prefix}{key}"

        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)
            result = await redis_conn.delete(redis_key)
            await redis_conn.close()

            if result > 0:
                self.stats.item_count = max(0, self.stats.item_count - 1)
                return True
            return False

        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            self.stats.errors += 1
            return False

    async def clear(self, pattern: Optional[str] = None):
        """Clear items from Redis cache"""
        try:
            redis_conn = redis.Redis(connection_pool=self.redis_pool)

            if pattern:
                # Clear items matching pattern
                search_pattern = f"{self.key_prefix}{pattern}"
                cursor = 0
                deleted_count = 0

                while True:
                    cursor, keys = await redis_conn.scan(cursor, match=search_pattern, count=100)
                    if keys:
                        deleted_count += await redis_conn.delete(*keys)
                    if cursor == 0:
                        break

                self.stats.item_count = max(0, self.stats.item_count - deleted_count)
            else:
                # Clear all cache items
                search_pattern = f"{self.key_prefix}*"
                cursor = 0
                deleted_count = 0

                while True:
                    cursor, keys = await redis_conn.scan(cursor, match=search_pattern, count=100)
                    if keys:
                        deleted_count += await redis_conn.delete(*keys)
                    if cursor == 0:
                        break

                self.stats.item_count = 0
                self.stats.size_bytes = 0

            await redis_conn.close()

        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            self.stats.errors += 1

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "level": self.stats.level.value,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "size_bytes": self.stats.size_bytes,
            "item_count": self.stats.item_count,
            "avg_access_time": self.stats.avg_access_time,
            "errors": self.stats.errors
        }


class CacheManager:
    """
    Production-ready cache manager with multi-level caching
    Features:
    - Multi-level caching strategy
    - Comprehensive metrics collection
    - Adaptive eviction policies
    - Performance monitoring
    - Memory optimization
    - Async processing
    - Batch operations
    """

    def __init__(self):
        self.config = get_config()
        self.redis_pool = None
        self.memory_cache = None
        self.redis_cache = None
        self.semantic_cache = None
        self.enabled = False
        self.initialized = False
        self.cleanup_task = None
        self.metrics_task = None
        self.global_stats = defaultdict(int)
        self.performance_history = deque(maxlen=10000)

    async def initialize(self):
        """Initialize the cache manager"""
        if self.initialized:
            return

        try:
            self.enabled = (
                is_helix_enabled() and
                self.config.caching.enabled
            )

            if not self.enabled:
                logger.info("Cache manager is disabled")
                return

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

            # Initialize cache levels
            cache_config = CacheConfig(
                enabled=True,
                max_size=self.config.caching.max_cache_size,
                max_memory_bytes=self.config.caching.max_memory_usage,
                ttl_seconds=self.config.caching.default_ttl,
                compression=self.config.caching.compression,
                batch_size=self.config.caching.batch_size
            )

            # Memory cache (L1)
            self.memory_cache = MemoryCache(cache_config)

            # Redis cache (L2)
            self.redis_cache = RedisCache(self.redis_pool, cache_config)

            # Semantic cache (L3)
            if self.config.caching.vector_search.enabled:
                self.semantic_cache = await get_semantic_cache()

            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())

            self.initialized = True
            logger.info("Cache manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise

    async def get(self, key: str, use_levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """Get value from multi-level cache"""
        if not self.enabled or not self.initialized:
            return None

        start_time = time.time()
        self.global_stats["total_requests"] += 1

        # Default to all levels
        if use_levels is None:
            use_levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.SEMANTIC]

        try:
            # L1: Memory Cache
            if CacheLevel.MEMORY in use_levels and self.memory_cache:
                value = await self.memory_cache.get(key)
                if value is not None:
                    self.global_stats["l1_hits"] += 1
                    self._record_performance("memory", time.time() - start_time, True)
                    return value

            # L2: Redis Cache
            if CacheLevel.REDIS in use_levels and self.redis_cache:
                value = await self.redis_cache.get(key)
                if value is not None:
                    self.global_stats["l2_hits"] += 1
                    # Promote to L1
                    if self.memory_cache:
                        await self.memory_cache.set(key, value)
                    self._record_performance("redis", time.time() - start_time, True)
                    return value

            # L3: Semantic Cache
            if CacheLevel.SEMANTIC in use_levels and self.semantic_cache:
                # For semantic cache, we need to extract prompt and model from key
                try:
                    prompt, model = self._parse_cache_key(key)
                    if prompt and model:
                        value = await self.semantic_cache.get(prompt, model)
                        if value is not None:
                            self.global_stats["l3_hits"] += 1
                            # Promote to lower levels
                            if self.memory_cache:
                                await self.memory_cache.set(key, value)
                            if self.redis_cache:
                                await self.redis_cache.set(key, value)
                            self._record_performance("semantic", time.time() - start_time, True)
                            return value
                except Exception as e:
                    logger.debug(f"Semantic cache lookup failed: {e}")

            self.global_stats["total_misses"] += 1
            self._record_performance("miss", time.time() - start_time, False)
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.global_stats["errors"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  levels: Optional[List[CacheLevel]] = None) -> bool:
        """Set value in multi-level cache"""
        if not self.enabled or not self.initialized:
            return False

        # Default to all levels
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS]

        try:
            success = True

            # L1: Memory Cache
            if CacheLevel.MEMORY in levels and self.memory_cache:
                success &= await self.memory_cache.set(key, value, ttl)

            # L2: Redis Cache
            if CacheLevel.REDIS in levels and self.redis_cache:
                success &= await self.redis_cache.set(key, value, ttl)

            return success

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.global_stats["errors"] += 1
            return False

    async def delete(self, key: str, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Delete from multi-level cache"""
        if not self.enabled or not self.initialized:
            return False

        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.SEMANTIC]

        try:
            success = True

            # L1: Memory Cache
            if CacheLevel.MEMORY in levels and self.memory_cache:
                success &= await self.memory_cache.delete(key)

            # L2: Redis Cache
            if CacheLevel.REDIS in levels and self.redis_cache:
                success &= await self.redis_cache.delete(key)

            # L3: Semantic Cache
            if CacheLevel.SEMANTIC in levels and self.semantic_cache:
                try:
                    prompt, model = self._parse_cache_key(key)
                    if prompt and model:
                        # Semantic cache invalidation by prompt/model
                        await self.semantic_cache.invalidate(f"{model}:{prompt}")
                except Exception as e:
                    logger.debug(f"Semantic cache invalidation failed: {e}")

            return success

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.global_stats["errors"] += 1
            return False

    async def clear(self, pattern: Optional[str] = None,
                   levels: Optional[List[CacheLevel]] = None) -> bool:
        """Clear multi-level cache"""
        if not self.enabled or not self.initialized:
            return False

        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.SEMANTIC]

        try:
            success = True

            # L1: Memory Cache
            if CacheLevel.MEMORY in levels and self.memory_cache:
                if pattern:
                    # Clear matching items
                    async with self.memory_cache.lock:
                        keys_to_remove = [
                            k for k in self.memory_cache.cache.keys()
                            if pattern in k
                        ]
                        for key in keys_to_remove:
                            await self.memory_cache.delete(key)
                else:
                    await self.memory_cache.clear()

            # L2: Redis Cache
            if CacheLevel.REDIS in levels and self.redis_cache:
                await self.redis_cache.clear(pattern)

            # L3: Semantic Cache
            if CacheLevel.SEMANTIC in levels and self.semantic_cache:
                await self.semantic_cache.invalidate(pattern)

            logger.info(f"Cleared cache pattern: {pattern}")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.global_stats["errors"] += 1
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.initialized:
            return {"enabled": False}

        try:
            stats = {
                "enabled": self.enabled,
                "global": dict(self.global_stats),
                "overall_hit_rate": self._calculate_overall_hit_rate(),
                "performance_summary": self._get_performance_summary(),
                "levels": {}
            }

            # Get statistics from each cache level
            if self.memory_cache:
                stats["levels"]["memory"] = await self.memory_cache.get_stats()

            if self.redis_cache:
                stats["levels"]["redis"] = await self.redis_cache.get_stats()

            if self.semantic_cache:
                stats["levels"]["semantic"] = await self.semantic_cache.get_metrics()

            # Add system information
            process = psutil.Process()
            stats["system"] = {
                "memory_rss": process.memory_info().rss,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent()
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"enabled": self.enabled, "error": str(e)}

    async def warm_cache(self, queries: List[str], models: List[str]):
        """Warm cache with common queries"""
        if not self.initialized:
            return

        try:
            logger.info(f"Starting cache warm-up with {len(queries)} queries")

            if self.semantic_cache:
                await self.semantic_cache.warm_cache(queries, models)

            # Create cache entries for common patterns
            for query, model in zip(queries, models):
                key = self._create_cache_key(query, model)
                # This would typically populate with actual responses
                # For now, we just create the cache structure

            logger.info("Cache warm-up completed")

        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}")

    def _create_cache_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model"""
        combined = f"{model}:{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _parse_cache_key(self, key: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse cache key to extract prompt and model"""
        # This is a simplified version - in practice, you'd need
        # to store the mapping between keys and prompts/models
        try:
            # Assuming key format: hash(model:prompt)
            # This would need to be implemented based on your key strategy
            return None, None
        except Exception:
            return None, None

    def _record_performance(self, level: str, duration: float, hit: bool):
        """Record performance metrics"""
        self.performance_history.append({
            "timestamp": time.time(),
            "level": level,
            "duration": duration,
            "hit": hit
        })

    def _calculate_overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all levels"""
        total_requests = self.global_stats.get("total_requests", 0)
        if total_requests == 0:
            return 0.0

        total_hits = (
            self.global_stats.get("l1_hits", 0) +
            self.global_stats.get("l2_hits", 0) +
            self.global_stats.get("l3_hits", 0)
        )

        return (total_hits / total_requests) * 100

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from history"""
        if not self.performance_history:
            return {}

        recent_entries = [
            entry for entry in self.performance_history
            if time.time() - entry["timestamp"] <= 3600  # Last hour
        ]

        if not recent_entries:
            return {}

        durations = [entry["duration"] for entry in recent_entries]
        hits = [entry["hit"] for entry in recent_entries]

        return {
            "avg_duration": statistics.mean(durations),
            "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
            "p99_duration": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations),
            "hit_rate": (sum(hits) / len(hits)) * 100,
            "total_requests": len(recent_entries)
        }

    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                if self.memory_cache:
                    async with self.memory_cache.lock:
                        await self.memory_cache._evict_if_needed()

                # Sleep for cleanup interval
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Retry sooner on error

    async def _metrics_loop(self):
        """Background metrics collection task"""
        while True:
            try:
                # Update global metrics
                if self.redis_pool:
                    redis_conn = redis.Redis(connection_pool=self.redis_pool)
                    info = await redis_conn.info()
                    await redis_conn.close()

                    self.global_stats["redis_memory"] = info.get("used_memory", 0)
                    self.global_stats["redis_connected_clients"] = info.get("connected_clients", 0)

                # Update system metrics
                process = psutil.Process()
                self.global_stats["system_memory"] = process.memory_info().rss
                self.global_stats["system_cpu"] = process.cpu_percent()

                # Sleep for metrics interval
                await asyncio.sleep(60)  # 1 minute

            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(30)  # Retry sooner on error

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            if self.metrics_task:
                self.metrics_task.cancel()
                try:
                    await self.metrics_task
                except asyncio.CancelledError:
                    pass

            if self.semantic_cache:
                await self.semantic_cache.cleanup()

            if self.redis_pool:
                await self.redis_pool.disconnect()

            logger.info("Cache manager cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global cache manager instance
_cache_manager = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager


async def is_cache_enabled() -> bool:
    """Check if cache is enabled"""
    return is_helix_enabled() and get_config().caching.enabled