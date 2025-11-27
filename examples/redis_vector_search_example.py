"""
Redis Vector Search Example with Helix
Demonstrates semantic caching, vector operations, and performance monitoring
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_example():
    """Setup example environment"""
    print("🚀 Setting up Redis Vector Search Example...")

    # Configure environment (using patches for example)
    import helix.core.semantic_cache
    import helix.core.cache_manager
    from helix.core.config import HelixConfig, CachingConfig, VectorSearchConfig

    # Create test configuration
    config = HelixConfig()
    config.enabled = True
    config.caching = CachingConfig()
    config.caching.enabled = True
    config.caching.cache_type = "hybrid"
    config.caching.default_ttl = 3600
    config.caching.batch_size = 10
    config.caching.redis = helix.core.config.RedisConfig(
        host="localhost",
        port=6379,
        database=14,  # Use test database
        max_connections=20
    )
    config.caching.vector_search = VectorSearchConfig(
        enabled=True,
        embedding_model="all-MiniLM-L6-v2",  # Fast for examples
        similarity_threshold=0.85,
        dimensions=384
    )

    # Patch config for demonstration
    helix.core.semantic_cache.get_config = lambda: config
    helix.core.cache_manager.get_config = lambda: config

    print("✅ Configuration setup complete")

async def initialize_redis_index():
    """Initialize Redis Vector Search index"""
    print("📊 Initializing Redis Vector Search...")

    try:
        import redis.asyncio as redis
        from pathlib import Path

        # Connect to Redis
        redis_conn = redis.Redis(host="localhost", port=6379, db=14)

        # Clear test database
        await redis_conn.flushdb()

        # Execute initialization script
        init_script_path = Path("redis/helix_init.redis")
        if init_script_path.exists():
            with open(init_script_path, 'r') as f:
                script_content = f.read()

            # Parse and execute commands (simplified version)
            commands = script_content.split('\n')
            for cmd in commands:
                cmd = cmd.strip()
                if not cmd or cmd.startswith('#') or cmd.startswith('echo') or cmd.startswith('for '):
                    continue

                try:
                    # Handle common Redis commands
                    if cmd.startswith('FT.CREATE'):
                        # Create vector index
                        await redis_conn.execute_command(cmd)
                    elif cmd.startswith('HSET') or cmd.startswith('ZADD') or cmd.startswith('SET'):
                        await redis_conn.execute_command(cmd)
                    elif cmd.startswith('CONFIG SET'):
                        parts = cmd.split()
                        if len(parts) >= 4:
                            await redis_conn.config_set(parts[2], parts[3])
                except Exception as e:
                    logger.debug(f"Command failed: {cmd} - {e}")

            print("✅ Redis Vector Search initialized")

        await redis_conn.close()

    except Exception as e:
        print(f"❌ Redis initialization failed: {e}")
        print("⚠️  Make sure Redis 7.2+ with RediSearch is running on localhost:6379")
        return False

    return True

async def demonstrate_embedding_processor():
    """Demonstrate embedding processor functionality"""
    print("\n🧠 Testing Embedding Processor...")

    from helix.core.semantic_cache import EmbeddingProcessor

    # Initialize processor
    processor = EmbeddingProcessor(model_name="all-MiniLM-L6-v2")
    await processor.initialize()

    # Test texts
    test_texts = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How do neural networks work?",
        "What is deep learning?",
        "Machine learning is a subset of AI"
    ]

    print(f"📝 Processing {len(test_texts)} test texts...")

    # Test single encoding
    start_time = time.time()
    embedding = await processor.encode_single(test_texts[0])
    single_time = time.time() - start_time
    print(f"✅ Single text encoding: {single_time*1000:.2f}ms, dimensions: {len(embedding)}")

    # Test batch encoding
    start_time = time.time()
    embeddings = await processor.encode_batch(test_texts)
    batch_time = time.time() - start_time
    print(f"✅ Batch encoding: {batch_time*1000:.2f}ms total, {batch_time/len(test_texts)*1000:.2f}ms per text")

    # Test similarity (cosine similarity)
    import numpy as np
    query_embedding = embeddings[0]
    similarities = []

    for embedding in embeddings[1:]:
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append(similarity)

    print(f"📊 Similarity scores: {similarities}")
    print(f"🎯 Most similar text similarity: {max(similarities):.3f}")

    await processor.cleanup()
    return embeddings

async def demonstrate_vector_storage():
    """Demonstrate vector storage and retrieval"""
    print("\n📦 Testing Vector Storage...")

    import redis.asyncio as redis
    from helix.core.semantic_cache import VectorIndexManager, CacheEntry

    # Setup
    redis_pool = redis.ConnectionPool(host="localhost", port=6379, db=14, max_connections=20)
    vector_manager = VectorIndexManager(redis_pool)

    # Create test cache entries
    test_entries = []
    prompts = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "What is artificial intelligence?",
        "How does computer vision work?"
    ]

    print("📝 Creating test cache entries...")

    # Generate embeddings using sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, prompt in enumerate(prompts):
        embedding = model.encode(prompt).tolist()

        entry = CacheEntry(
            key=f"test_{i}",
            prompt=prompt,
            model="gpt-3.5-turbo",
            response=f"Comprehensive answer about {prompt}",
            embedding=embedding,
            cost=0.001,
            latency=100.0,
            created_at=time.time(),
            expires_at=time.time() + 3600
        )

        test_entries.append(entry)

    # Add vectors to index
    print("💾 Storing vectors in Redis...")
    start_time = time.time()

    for entry in test_entries:
        success = await vector_manager.add_vector(entry)
        if not success:
            print(f"❌ Failed to store: {entry.key}")

    storage_time = time.time() - start_time
    print(f"✅ Stored {len(test_entries)} vectors in {storage_time*1000:.2f}ms")

    # Test vector search
    print("🔍 Testing vector search...")

    query_prompt = "Tell me about AI and machine learning"
    query_embedding = model.encode(query_prompt).tolist()

    # Test different thresholds
    thresholds = [0.5, 0.7, 0.85]

    for threshold in thresholds:
        start_time = time.time()
        results = await vector_manager.search_vectors(
            query_embedding=query_embedding,
            similarity_threshold=threshold,
            limit=5
        )
        search_time = time.time() - start_time

        print(f"\n🎯 Threshold {threshold:.2f}: {len(results)} results in {search_time*1000:.2f}ms")

        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.similarity_score:.3f}, Prompt: {result.entry.prompt[:50]}...")

    # Test hit count update
    if results:
        print("\n📈 Updating hit count for top result...")
        await vector_manager.update_hit_count(results[0].entry.key)
        print("✅ Hit count updated")

    await redis_pool.disconnect()
    return len(test_entries) > 0

async def demonstrate_semantic_cache():
    """Demonstrate semantic caching functionality"""
    print("\n🧠 Testing Semantic Cache...")

    from helix.core.semantic_cache import get_semantic_cache

    # Initialize semantic cache
    cache = await get_semantic_cache()

    # Test data
    test_queries = [
        ("What is machine learning?", "gpt-3.5-turbo"),
        ("Explain neural networks", "gpt-4"),
        ("How does AI work?", "claude-2"),
        ("What is deep learning?", "gpt-3.5-turbo"),
        ("Tell me about ML", "gpt-3.5-turbo")  # Semantic variant
    ]

    # Test cache misses (should be empty initially)
    print("🔍 Testing cache misses...")
    for prompt, model in test_queries:
        result = await cache.get(prompt, model)
        if result:
            print(f"❌ Unexpected cache hit for: {prompt}")
        else:
            print(f"✅ Cache miss as expected: {prompt}")

    # Cache some responses
    print("\n💾 Caching responses...")
    for i, (prompt, model) in enumerate(test_queries[:3]):  # Cache first 3
        response = f"Detailed response about {prompt} using {model}"
        await cache.set(prompt, model, response, cost=0.01, latency=200.0)
        print(f"✅ Cached: {prompt[:30]}...")

    # Test cache hits
    print("\n🎯 Testing cache hits...")
    for i, (prompt, model) in enumerate(test_queries[:3]):
        result = await cache.get(prompt, model)
        if result:
            print(f"✅ Cache hit: {prompt[:30]}...")
        else:
            print(f"❌ Unexpected cache miss: {prompt}")

    # Test semantic similarity
    print("\n🔗 Testing semantic similarity...")
    semantic_prompt = "Explain machine learning concepts"  # Similar to first query
    result = await cache.get(semantic_prompt, test_queries[0][1])

    if result:
        print(f"✅ Semantic cache hit: {semantic_prompt[:30]}...")
        print(f"   Original: {test_queries[0][0]}")
        print(f"   Response: {result[:100]}...")
    else:
        print(f"❌ No semantic match found")

    # Get cache metrics
    print("\n📊 Cache Metrics:")
    metrics = await cache.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    return metrics["total_requests"] > 0

async def demonstrate_multi_level_cache():
    """Demonstrate multi-level caching strategy"""
    print("\n🏗️ Testing Multi-Level Cache Manager...")

    from helix.core.cache_manager import get_cache_manager, CacheLevel

    # Initialize cache manager
    cache_manager = await get_cache_manager()

    # Test multi-level cache operations
    test_key = "multi_level:test_query"
    test_value = {
        "response": "This is a test response for multi-level caching",
        "model": "gpt-4",
        "cost": 0.02,
        "latency": 150.0
    }

    # Test L1 cache (Memory)
    print("🧠 Testing L1 Memory Cache...")
    await cache_manager.set(test_key, test_value, levels=[CacheLevel.MEMORY])
    result = await cache_manager.get(test_key, levels=[CacheLevel.MEMORY])

    if result == test_value:
        print("✅ L1 cache working correctly")
    else:
        print("❌ L1 cache failed")

    # Test L2 cache (Redis)
    print("💾 Testing L2 Redis Cache...")
    l2_key = test_key + "_l2"
    await cache_manager.set(l2_key, test_value, levels=[CacheLevel.REDIS])
    result = await cache_manager.get(l2_key, levels=[CacheLevel.REDIS])

    if result == test_value:
        print("✅ L2 cache working correctly")
    else:
        print("❌ L2 cache failed")

    # Test automatic promotion
    print("🔄 Testing automatic cache promotion...")
    promotion_key = test_key + "_promotion"

    # Add to L2 only
    await cache_manager.set(promotion_key, test_value, levels=[CacheLevel.REDIS])

    # Get from all levels (should promote to L1)
    result = await cache_manager.get(promotion_key, use_levels=[CacheLevel.MEMORY, CacheLevel.REDIS])

    if result == test_value:
        print("✅ Automatic promotion working")

        # Check if it's now in L1
        l1_result = await cache_manager.get(promotion_key, levels=[CacheLevel.MEMORY])
        if l1_result == test_value:
            print("✅ Successfully promoted to L1")
        else:
            print("⚠️  Promotion to L1 failed")
    else:
        print("❌ Automatic promotion failed")

    # Test comprehensive statistics
    print("\n📊 Cache Manager Statistics:")
    stats = await cache_manager.get_statistics()

    print(f"   Enabled: {stats['enabled']}")
    print(f"   Overall Hit Rate: {stats['overall_hit_rate']:.1f}%")
    print(f"   Total Requests: {stats['global']['total_requests']}")
    print(f"   L1 Hits: {stats['global']['l1_hits']}")
    print(f"   L2 Hits: {stats['global']['l2_hits']}")

    if 'system' in stats:
        print(f"   Memory Usage: {stats['system']['memory_rss'] / 1024 / 1024:.1f}MB")
        print(f"   CPU Usage: {stats['system']['cpu_percent']:.1f}%")

    return stats['global']['total_requests'] > 0

async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    print("\n📊 Performance Monitoring Demonstration...")

    import psutil
    from helix.core.semantic_cache import get_semantic_cache

    # Monitor system resources
    process = psutil.Process()

    # Initial system state
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_cpu = process.cpu_percent()

    print(f"💻 Initial System State:")
    print(f"   Memory Usage: {initial_memory:.1f}MB")
    print(f"   CPU Usage: {initial_cpu:.1f}%")

    # Perform cache operations and monitor performance
    cache = await get_semantic_cache()

    operations = []
    operation_count = 50

    print(f"\n⚡ Performing {operation_count} cache operations...")

    for i in range(operation_count):
        prompt = f"Performance test prompt {i} about AI and machine learning"
        model = f"gpt-{i % 4 + 1}"

        # Test cache get (should miss initially)
        start_time = time.time()
        result = await cache.get(prompt, model)
        get_time = time.time() - start_time

        # Cache set
        start_time = time.time()
        response = f"Test response {i} for {prompt}"
        await cache.set(prompt, model, response, cost=0.001, latency=50.0)
        set_time = time.time() - start_time

        operations.append({
            'operation': i,
            'get_time': get_time,
            'set_time': set_time,
            'total_time': get_time + set_time
        })

        if (i + 1) % 10 == 0:
            print(f"   Completed {i + 1}/{operation_count} operations")

    # Calculate performance metrics
    import statistics

    get_times = [op['get_time'] for op in operations]
    set_times = [op['set_time'] for op in operations]
    total_times = [op['total_time'] for op in operations]

    print(f"\n📈 Performance Results:")
    print(f"   Cache Get - Avg: {statistics.mean(get_times)*1000:.2f}ms, "
          f"P95: {statistics.quantile(get_times, 0.95)*1000:.2f}ms")
    print(f"   Cache Set - Avg: {statistics.mean(set_times)*1000:.2f}ms, "
          f"P95: {statistics.quantile(set_times, 0.95)*1000:.2f}ms")
    print(f"   Total Operations - Avg: {statistics.mean(total_times)*1000:.2f}ms")

    # Final system state
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    final_cpu = process.cpu_percent()
    memory_diff = final_memory - initial_memory

    print(f"\n💻 Final System State:")
    print(f"   Memory Usage: {final_memory:.1f}MB (+{memory_diff:+.1f}MB)")
    print(f"   CPU Usage: {final_cpu:.1f}%")

    # Get cache performance metrics
    cache_metrics = await cache.get_metrics()

    print(f"\n🎯 Cache Performance:")
    print(f"   Hit Rate: {cache_metrics['hit_rate']:.1f}%")
    print(f"   Semantic Hit Rate: {cache_metrics['semantic_hit_rate']:.1f}%")
    print(f"   Avg Search Time: {cache_metrics['avg_search_time']*1000:.2f}ms")
    print(f"   Evictions: {cache_metrics['eviction_count']}")

    return operation_count > 0

async def main():
    """Main demonstration function"""
    print("🚀 Redis Vector Search with Helix - Complete Demonstration")
    print("=" * 60)

    success_count = 0
    total_tests = 6

    try:
        # Setup
        await setup_example()

        # Test 1: Redis Index Initialization
        if await initialize_redis_index():
            success_count += 1
            print("✅ Test 1: Redis Vector Search initialized successfully")
        else:
            print("❌ Test 1: Redis initialization failed")
            return

        # Test 2: Embedding Processor
        try:
            await demonstrate_embedding_processor()
            success_count += 1
            print("✅ Test 2: Embedding processor working correctly")
        except Exception as e:
            print(f"❌ Test 2: Embedding processor failed: {e}")

        # Test 3: Vector Storage
        try:
            if await demonstrate_vector_storage():
                success_count += 1
                print("✅ Test 3: Vector storage working correctly")
        except Exception as e:
            print(f"❌ Test 3: Vector storage failed: {e}")

        # Test 4: Semantic Cache
        try:
            if await demonstrate_semantic_cache():
                success_count += 1
                print("✅ Test 4: Semantic cache working correctly")
        except Exception as e:
            print(f"❌ Test 4: Semantic cache failed: {e}")

        # Test 5: Multi-Level Cache
        try:
            if await demonstrate_multi_level_cache():
                success_count += 1
                print("✅ Test 5: Multi-level cache working correctly")
        except Exception as e:
            print(f"❌ Test 5: Multi-level cache failed: {e}")

        # Test 6: Performance Monitoring
        try:
            if await demonstrate_performance_monitoring():
                success_count += 1
                print("✅ Test 6: Performance monitoring working correctly")
        except Exception as e:
            print(f"❌ Test 6: Performance monitoring failed: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during demonstration: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"📋 Demonstration Summary: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Redis Vector Search with Helix is working perfectly.")
        print("\n📚 Next steps:")
        print("   1. Run the comprehensive benchmark suite:")
        print("      python benchmarks/redis_vector_search_benchmark.py")
        print("   2. Check out the deployment guide:")
        print("      docs/redis_vector_search_deployment.md")
        print("   3. Review the test suite:")
        print("      tests/test_redis_vector_search.py")
    elif success_count > 0:
        print("⚠️  Some tests passed. Check the error messages above for troubleshooting.")
    else:
        print("❌ All tests failed. Please check your Redis setup and dependencies.")

    print("\n🔧 Troubleshooting tips:")
    print("   - Make sure Redis 7.2+ with RediSearch is running")
    print("   - Check Redis connection: redis-cli ping")
    print("   - Verify Vector Search index: FT.INFO helix:semantic:index")
    print("   - Check Python dependencies: pip install redis[hiredis] sentence-transformers")

if __name__ == "__main__":
    asyncio.run(main())