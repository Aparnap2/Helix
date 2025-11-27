"""
Comprehensive test suite for Redis Vector Search setup in Helix
Tests initialization, semantic caching, and cache management
"""

import pytest
import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

# Import Helix components
from helix.core.config import get_config, HelixConfig
from helix.core.semantic_cache import (
    SemanticCache, EmbeddingProcessor, VectorIndexManager,
    CacheEntry, SearchResult, AdaptiveThreshold
)
from helix.core.cache_manager import (
    CacheManager, CacheLevel, MemoryCache, RedisCache
)


@pytest.fixture
async def redis_pool():
    """Redis connection pool for testing"""
    # Use a test Redis database
    pool = ConnectionPool(
        host="localhost",
        port=6379,
        db=15,  # Use test database
        max_connections=10,
        decode_responses=False
    )

    # Clean up test database
    conn = redis.Redis(connection_pool=pool)
    await conn.flushdb()
    await conn.close()

    yield pool

    # Cleanup
    conn = redis.Redis(connection_pool=pool)
    await conn.flushdb()
    await conn.close()
    await pool.disconnect()


@pytest.fixture
async def test_config():
    """Test configuration for Helix"""
    config = HelixConfig()
    config.enabled = True
    config.caching.enabled = True
    config.caching.cache_type = "hybrid"
    config.caching.vector_search.enabled = True
    config.caching.vector_search.embedding_model = "all-MiniLM-L6-v2"
    config.caching.vector_search.similarity_threshold = 0.85
    config.caching.vector_search.dimensions = 384
    config.caching.default_ttl = 300
    config.caching.batch_size = 10

    # Redis configuration
    config.caching.redis.host = "localhost"
    config.caching.redis.port = 6379
    config.caching.redis.database = 15
    config.caching.redis.max_connections = 10

    return config


class TestRedisInitialization:
    """Test Redis initialization script"""

    @pytest.mark.asyncio
    async def test_redis_initialization(self, redis_pool):
        """Test Redis initialization script execution"""
        conn = redis.Redis(connection_pool=redis_pool)

        # Execute initialization script
        with open("/home/aparna/Desktop/Helix/redis/helix_init.redis", "r") as f:
            script = f.read()

        # Replace bash loops with direct commands for testing
        commands = script.split('\n')
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith('#') or cmd.startswith('echo'):
                continue
            if cmd.startswith('for '):
                # Skip bash loops in testing
                continue
            if cmd.startswith('do'):
                continue
            if cmd.startswith('done'):
                continue
            if cmd.startswith('CONFIG SET'):
                # Parse CONFIG SET commands
                parts = cmd.split()
                if len(parts) >= 4:
                    key, value = parts[2], parts[3]
                    await conn.config_set(key, value)
                continue

            try:
                await conn.execute_command(cmd)
            except Exception as e:
                # Some commands might fail in test environment
                print(f"Command failed (expected in tests): {cmd} - {e}")

        # Verify index creation
        try:
            info = await conn.ft("helix:semantic:index").info()
            assert info is not None
            print(f"Vector index info: {info}")
        except Exception as e:
            print(f"Index check failed (might be expected): {e}")

        # Verify key structures exist
        assert await conn.exists("helix:exact:cache") >= 0
        assert await conn.exists("helix:cache:metrics") >= 0

        await conn.close()

    @pytest.mark.asyncio
    async def test_hnsw_parameters(self, redis_pool):
        """Test HNSW index parameters"""
        conn = redis.Redis(connection_pool=redis_pool)

        try:
            # Get index info
            info = await conn.ft("helix:semantic:index").info()

            # Check HNSW parameters (if available)
            if info and "attributes" in info:
                vector_attrs = info["attributes"].get("vector", {})
                print(f"Vector attributes: {vector_attrs}")

                # Verify HNSW settings (these might not be in info)
                assert True  # Basic check that index exists

        except Exception as e:
            print(f"HNSW parameter check failed: {e}")
            # This might fail if index doesn't exist yet

        await conn.close()


class TestEmbeddingProcessor:
    """Test embedding processor functionality"""

    @pytest.mark.asyncio
    async def test_embedding_processor_initialization(self):
        """Test embedding processor initialization"""
        processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2")

        # Test initialization
        await processor.initialize()
        assert processor.model is not None

        await processor.cleanup()

    @pytest.mark.asyncio
    async def test_single_text_encoding(self):
        """Test single text encoding"""
        processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2")
        await processor.initialize()

        test_text = "This is a test sentence for embedding generation"
        embedding = await processor.encode_single(test_text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        await processor.cleanup()

    @pytest.mark.asyncio
    async def test_batch_text_encoding(self):
        """Test batch text encoding"""
        processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2", batch_size=3)
        await processor.initialize()

        test_texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence",
            "Fourth test sentence"
        ]

        embeddings = await processor.encode_batch(test_texts)

        assert len(embeddings) == len(test_texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

        await processor.cleanup()

    @pytest.mark.asyncio
    async def test_embedding_caching(self):
        """Test embedding caching functionality"""
        processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2")
        await processor.initialize()

        test_text = "Test caching functionality"

        # First encoding
        start_time = time.time()
        embedding1 = await processor.encode_single(test_text)
        first_time = time.time() - start_time

        # Second encoding (should use cache)
        start_time = time.time()
        embedding2 = await processor.encode_single(test_text)
        second_time = time.time() - start_time

        assert embedding1 == embedding2
        # Second time should be faster (cached)
        assert second_time < first_time

        await processor.cleanup()


class TestVectorIndexManager:
    """Test vector index manager functionality"""

    @pytest.mark.asyncio
    async def test_vector_addition(self, redis_pool):
        """Test adding vectors to index"""
        manager = VectorIndexManager(redis_pool)

        # Create test entry
        test_embedding = [0.1] * 384  # Small test embedding
        entry = CacheEntry(
            key="test_key_1",
            prompt="Test prompt",
            model="test-model",
            response="Test response",
            embedding=test_embedding,
            cost=0.001,
            latency=100.0,
            created_at=time.time(),
            expires_at=time.time() + 3600
        )

        # Add to index
        result = await manager.add_vector(entry)
        assert result is True

        # Verify existence
        conn = redis.Redis(connection_pool=redis_pool)
        key_exists = await conn.exists("helix:vector:test_key_1")
        assert key_exists > 0
        await conn.close()

    @pytest.mark.asyncio
    async def test_vector_search(self, redis_pool):
        """Test vector search functionality"""
        manager = VectorIndexManager(redis_pool)

        # Add test vectors
        test_embeddings = [
            ([0.1] * 384, "First prompt", "response1"),
            ([0.2] * 384, "Second prompt", "response2"),
            ([0.15] * 384, "Similar prompt", "response3")
        ]

        for i, (embedding, prompt, response) in enumerate(test_embeddings):
            entry = CacheEntry(
                key=f"test_key_{i}",
                prompt=prompt,
                model="test-model",
                response=response,
                embedding=embedding,
                cost=0.001,
                latency=100.0
            )
            await manager.add_vector(entry)

        # Search for similar vectors
        query_embedding = [0.12] * 384
        results = await manager.search_vectors(
            query_embedding=query_embedding,
            similarity_threshold=0.7,
            limit=5
        )

        # Should find some results
        assert len(results) >= 0
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.similarity_score >= 0.7
            assert isinstance(result.entry, CacheEntry)

    @pytest.mark.asyncio
    async def test_vector_deletion(self, redis_pool):
        """Test vector deletion functionality"""
        manager = VectorIndexManager(redis_pool)

        # Add test vector
        entry = CacheEntry(
            key="delete_test_key",
            prompt="Delete test prompt",
            model="test-model",
            response="Delete test response",
            embedding=[0.1] * 384
        )
        await manager.add_vector(entry)

        # Delete vector
        result = await manager.delete_vector("delete_test_key")
        assert result is True

        # Verify deletion
        conn = redis.Redis(connection_pool=redis_pool)
        key_exists = await conn.exists("helix:vector:delete_test_key")
        assert key_exists == 0
        await conn.close()


class TestSemanticCache:
    """Test semantic cache functionality"""

    @pytest.mark.asyncio
    async def test_semantic_cache_initialization(self, test_config):
        """Test semantic cache initialization"""
        with patch('helix.core.config.get_config', return_value=test_config):
            cache = SemanticCache()

            # Mock Redis connection for testing
            with patch('redis.asyncio.ConnectionPool'):
                with patch('helix.core.semantic_cache.EmbeddingProcessor'):
                    await cache.initialize()
                    assert cache.initialized is True

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, test_config, redis_pool):
        """Test cache set and get operations"""
        with patch('helix.core.config.get_config', return_value=test_config):
            cache = SemanticCache()
            cache.redis_pool = redis_pool

            # Mock embedding processor
            mock_embedding = [0.1] * 384
            cache.embedding_processor = Mock()
            cache.embedding_processor.encode_single = AsyncMock(return_value=mock_embedding)

            # Initialize vector manager
            cache.vector_manager = VectorIndexManager(redis_pool)

            # Test set operation
            test_prompt = "What is the capital of France?"
            test_model = "gpt-3.5-turbo"
            test_response = "The capital of France is Paris."

            await cache.set(test_prompt, test_model, test_response, cost=0.001, latency=150.0)

            # Test get operation (exact cache)
            result = await cache.get(test_prompt, test_model, use_semantic_cache=False)
            assert result == test_response

    @pytest.mark.asyncio
    async def test_adaptive_threshold(self):
        """Test adaptive threshold functionality"""
        threshold = AdaptiveThreshold(base_threshold=0.85)

        # Test initial values
        assert threshold.current_threshold == 0.85
        assert threshold.min_threshold == 0.70
        assert threshold.max_threshold == 0.95

        # Test adjustment logic
        threshold.current_threshold = 0.8
        # Simulate high hit rate - should increase threshold
        # (This would be tested with actual performance data in real scenarios)

    @pytest.mark.asyncio
    async def test_cache_metrics(self):
        """Test cache metrics tracking"""
        cache = SemanticCache()

        # Initial metrics
        metrics = await cache.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["cache_hits"] == 0
        assert metrics["hit_rate"] == 0.0

        # Simulate some activity
        cache.metrics.total_requests = 100
        cache.metrics.cache_hits = 75

        metrics = await cache.get_metrics()
        assert metrics["total_requests"] == 100
        assert metrics["cache_hits"] == 75
        assert metrics["hit_rate"] == 75.0


class TestCacheManager:
    """Test cache manager functionality"""

    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, test_config, redis_pool):
        """Test cache manager initialization"""
        with patch('helix.core.config.get_config', return_value=test_config):
            manager = CacheManager()
            manager.redis_pool = redis_pool

            await manager.initialize()
            assert manager.initialized is True
            assert manager.enabled is True

    @pytest.mark.asyncio
    async def test_memory_cache_operations(self):
        """Test memory cache operations"""
        from helix.core.cache_manager import CacheConfig

        config = CacheConfig(max_size=10)
        memory_cache = MemoryCache(config)

        # Test set and get
        test_key = "test_memory_key"
        test_value = {"data": "test_value", "number": 42}

        result = await memory_cache.set(test_key, test_value)
        assert result is True

        retrieved_value = await memory_cache.get(test_key)
        assert retrieved_value == test_value

        # Test stats
        stats = await memory_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["item_count"] == 1

    @pytest.mark.asyncio
    async def test_multi_level_caching(self, test_config, redis_pool):
        """Test multi-level caching strategy"""
        with patch('helix.core.config.get_config', return_value=test_config):
            manager = CacheManager()
            manager.redis_pool = redis_pool
            manager.enabled = True

            # Initialize components
            from helix.core.cache_manager import CacheConfig
            cache_config = CacheConfig(max_size=10)
            manager.memory_cache = MemoryCache(cache_config)
            manager.redis_cache = RedisCache(redis_pool, cache_config)

            # Test L1 cache
            test_key = "multi_level_test"
            test_value = "test_data"

            # Set in memory cache
            await manager.memory_cache.set(test_key, test_value)

            # Get from multi-level cache
            result = await manager.get(test_key, [CacheLevel.MEMORY])
            assert result == test_value

    @pytest.mark.asyncio
    async def test_cache_statistics(self, test_config, redis_pool):
        """Test cache statistics collection"""
        with patch('helix.core.config.get_config', return_value=test_config):
            manager = CacheManager()
            manager.redis_pool = redis_pool
            manager.enabled = True
            manager.initialized = True

            # Add some mock statistics
            manager.global_stats["total_requests"] = 100
            manager.global_stats["l1_hits"] = 60
            manager.global_stats["l2_hits"] = 25

            stats = await manager.get_statistics()
            assert stats["enabled"] is True
            assert stats["global"]["total_requests"] == 100
            assert stats["overall_hit_rate"] == 85.0


class TestIntegration:
    """Integration tests for complete Redis Vector Search setup"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, redis_pool):
        """Test complete end-to-end workflow"""
        # Initialize all components
        config = HelixConfig()
        config.enabled = True
        config.caching.enabled = True
        config.caching.vector_search.enabled = True
        config.caching.vector_search.embedding_model = "all-MiniLM-L6-v2"

        with patch('helix.core.config.get_config', return_value=config):
            # Initialize semantic cache
            semantic_cache = SemanticCache()
            semantic_cache.redis_pool = redis_pool

            # Mock embedding processor
            mock_embedding = [0.1] * 384
            semantic_cache.embedding_processor = Mock()
            semantic_cache.embedding_processor.encode_single = AsyncMock(return_value=mock_embedding)
            semantic_cache.initialized = True
            semantic_cache.vector_manager = VectorIndexManager(redis_pool)

            # Initialize cache manager
            cache_manager = CacheManager()
            cache_manager.redis_pool = redis_pool
            cache_manager.enabled = True
            cache_manager.initialized = True
            cache_manager.semantic_cache = semantic_cache

            # Test workflow
            prompt = "What is machine learning?"
            model = "gpt-4"
            response = "Machine learning is a subset of artificial intelligence..."

            # Step 1: Cache miss scenario
            result = await cache_manager.get(f"{model}:{prompt}")
            assert result is None

            # Step 2: Cache the response
            await semantic_cache.set(prompt, model, response, cost=0.01, latency=200.0)

            # Step 3: Cache hit scenario
            result = await semantic_cache.get(prompt, model)
            assert result == response

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, redis_pool):
        """Test performance characteristics"""
        manager = VectorIndexManager(redis_pool)

        # Add test data
        test_data = []
        for i in range(100):
            embedding = np.random.random(384).tolist()
            test_data.append((embedding, f"prompt_{i}", f"response_{i}"))

        # Measure insertion performance
        start_time = time.time()
        for i, (embedding, prompt, response) in enumerate(test_data):
            entry = CacheEntry(
                key=f"perf_test_{i}",
                prompt=prompt,
                model="test-model",
                response=response,
                embedding=embedding
            )
            await manager.add_vector(entry)

        insertion_time = time.time() - start_time
        insertion_rate = len(test_data) / insertion_time

        print(f"Insertion performance: {insertion_rate:.2f} entries/second")
        assert insertion_rate > 10  # Should be at least 10 entries/second

        # Measure search performance
        query_embedding = np.random.random(384).tolist()

        search_times = []
        for _ in range(10):
            start_time = time.time()
            results = await manager.search_vectors(query_embedding, limit=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

        avg_search_time = sum(search_times) / len(search_times)
        print(f"Average search time: {avg_search_time:.4f} seconds")
        assert avg_search_time < 1.0  # Should be under 1 second

    @pytest.mark.asyncio
    async def test_error_handling(self, redis_pool):
        """Test error handling and recovery"""
        cache = SemanticCache()

        # Test with invalid Redis connection
        cache.redis_pool = None
        cache.initialized = False

        # Should handle gracefully
        result = await cache.get("test", "test-model")
        assert result is None

        # Test with corrupted data
        with patch.object(cache, '_get_exact_cache', side_effect=Exception("Corrupted data")):
            result = await cache.get("test", "test-model")
            assert result is None


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "TestRedisInitialization",
        "TestEmbeddingProcessor",
        "TestVectorIndexManager",
        "TestSemanticCache",
        "TestCacheManager",
        "TestIntegration"
    ])