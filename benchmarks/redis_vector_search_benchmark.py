"""
Redis Vector Search Benchmark for Helix
Comprehensive performance testing for semantic caching and vector operations
"""

import asyncio
import time
import json
import numpy as np
import psutil
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import threading

import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

# Import Helix components
from helix.core.config import get_config, HelixConfig, CachingConfig, VectorSearchConfig
from helix.core.semantic_cache import SemanticCache, EmbeddingProcessor, VectorIndexManager
from helix.core.cache_manager import CacheManager, CacheLevel


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    # Test data parameters
    num_entries: int = 1000
    num_searches: int = 100
    batch_sizes: List[int] = None
    embedding_dimensions: List[int] = None

    # Performance parameters
    similarity_thresholds: List[float] = None
    search_limits: List[int] = None
    concurrent_requests: int = 10
    warmup_requests: int = 50

    # Test datasets
    use_real_data: bool = True
    data_file: str = "test_prompts.json"

    # Output parameters
    output_dir: str = "benchmark_results"
    generate_plots: bool = True
    save_detailed_logs: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 10, 50, 100]
        if self.embedding_dimensions is None:
            self.embedding_dimensions = [384, 768, 1536]
        if self.similarity_thresholds is None:
            self.similarity_thresholds = [0.7, 0.8, 0.9, 0.95]
        if self.search_limits is None:
            self.search_limits = [1, 5, 10, 20, 50]


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_name: str
    operation: str
    total_time: float
    operations_per_second: float
    avg_time_per_operation: float
    p50_time: float
    p95_time: float
    p99_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    success_count: int
    parameters: Dict[str, Any] = None

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)


class VectorSearchBenchmark:
    """Comprehensive Redis Vector Search benchmarking suite"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.redis_pool = None
        self.cache = None
        self.embedding_processor = None
        self.vector_manager = None
        self.cache_manager = None
        self.test_data = []
        self.results = []
        self.system_monitor = None

    async def initialize(self):
        """Initialize benchmark environment"""
        print("Initializing benchmark environment...")

        # Setup Redis connection
        self.redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=14,  # Use dedicated benchmark database
            max_connections=self.config.concurrent_requests + 5,
            decode_responses=False
        )

        # Clear benchmark database
        conn = redis.Redis(connection_pool=self.redis_pool)
        await conn.flushdb()
        await conn.close()

        # Initialize Helix components
        await self._setup_helix_components()

        # Generate or load test data
        await self._prepare_test_data()

        # Start system monitoring
        self.system_monitor = SystemMonitor()
        await self.system_monitor.start()

        print("Benchmark environment initialized successfully!")

    async def _setup_helix_components(self):
        """Setup Helix components for benchmarking"""
        # Create test configuration
        helix_config = HelixConfig()
        helix_config.enabled = True
        helix_config.caching.enabled = True
        helix_config.caching.cache_type = "hybrid"
        helix_config.caching.vector_search.enabled = True
        helix_config.caching.vector_search.embedding_model = "all-MiniLM-L6-v2"
        helix_config.caching.vector_search.similarity_threshold = 0.85
        helix_config.caching.vector_search.dimensions = 384
        helix_config.caching.default_ttl = 3600
        helix_config.caching.batch_size = 50
        helix_config.caching.redis.host = "localhost"
        helix_config.caching.redis.port = 6379
        helix_config.caching.redis.database = 14
        helix_config.caching.redis.max_connections = self.config.concurrent_requests + 5

        # Patch config for testing
        import helix.core.semantic_cache
        import helix.core.cache_manager

        original_get_config = helix.core.semantic_cache.get_config
        original_cache_get_config = helix.core.cache_manager.get_config

        helix.core.semantic_cache.get_config = lambda: helix_config
        helix.core.cache_manager.get_config = lambda: helix_config

        # Initialize components
        self.embedding_processor = EmbeddingProcessor(
            model_name="all-MiniLM-L6-v2",
            batch_size=self.config.batch_sizes[0]
        )
        await self.embedding_processor.initialize()

        self.vector_manager = VectorIndexManager(self.redis_pool)

        self.cache = SemanticCache()
        self.cache.redis_pool = self.redis_pool
        self.cache.embedding_processor = self.embedding_processor
        self.cache.vector_manager = self.vector_manager
        self.cache.initialized = True

        self.cache_manager = CacheManager()
        self.cache_manager.redis_pool = self.redis_pool
        self.cache_manager.enabled = True
        self.cache_manager.initialized = True
        self.cache_manager.memory_cache = None  # Skip memory cache for pure Redis testing
        self.cache_manager.redis_cache = None
        self.cache_manager.semantic_cache = self.cache

    async def _prepare_test_data(self):
        """Prepare test data for benchmarking"""
        if self.config.use_real_data:
            # Load real test data
            try:
                with open(self.config.data_file, 'r') as f:
                    self.test_data = json.load(f)
                print(f"Loaded {len(self.test_data)} test entries from {self.config.data_file}")
            except FileNotFoundError:
                print(f"Data file {self.config.data_file} not found, generating synthetic data...")
                self.test_data = self._generate_synthetic_data()
        else:
            self.test_data = self._generate_synthetic_data()

        # Limit to configured number of entries
        if len(self.test_data) > self.config.num_entries:
            self.test_data = self.test_data[:self.config.num_entries]

    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic test data"""
        # Common prompts for LLM caching scenarios
        base_prompts = [
            "What is machine learning?",
            "Explain artificial intelligence",
            "How does neural network work?",
            "What is deep learning?",
            "Explain natural language processing",
            "What is computer vision?",
            "How does reinforcement learning work?",
            "What is supervised learning?",
            "Explain unsupervised learning",
            "What is transfer learning?",
            "How does GPT work?",
            "What is transformer architecture?",
            "Explain attention mechanism",
            "What is BERT?",
            "How does Word2Vec work?",
            "What is sentiment analysis?",
            "Explain text classification",
            "What is named entity recognition?",
            "How does machine translation work?",
            "What is language modeling?",
            "Explain clustering algorithms",
            "What is dimensionality reduction?",
            "How does PCA work?",
            "What is random forest?",
            "Explain support vector machines",
            "What is gradient boosting?",
            "How does XGBoost work?",
            "What is cross-validation?",
            "Explain hyperparameter tuning",
            "What is feature engineering?",
            "How does data preprocessing work?",
            "What is overfitting?",
            "Explain regularization techniques",
            "What is ensemble learning?",
            "How does bagging work?",
            "What is boosting?",
            "Explain decision trees",
            "What is naive Bayes classifier?",
            "How does k-means clustering work?",
            "What is hierarchical clustering?",
            "Explain DBSCAN algorithm",
            "What is recommendation system?",
            "How does collaborative filtering work?",
            "What is content-based filtering?",
            "Explain matrix factorization",
            "What is deep reinforcement learning?",
            "How does Q-learning work?",
            "What is policy gradient method?",
            "Explain actor-critic algorithms"
        ]

        test_data = []
        models = ["gpt-3.5-turbo", "gpt-4", "claude-2", "gemini-pro"]

        for i, prompt in enumerate(base_prompts):
            # Create variations
            variations = [
                prompt,
                f"Can you {prompt.lower()}?",
                f"Explain in detail: {prompt}",
                f"Simple explanation of {prompt.lower()}",
                f"Advanced {prompt.lower()} concepts"
            ]

            for var in variations:
                if len(test_data) >= self.config.num_entries:
                    break

                model = models[i % len(models)]
                response = f"This is a detailed response about {prompt}. It includes comprehensive information covering the main concepts, applications, and current developments in the field."

                test_data.append({
                    "prompt": var,
                    "model": model,
                    "response": response,
                    "cost": np.random.uniform(0.001, 0.1),
                    "latency": np.random.uniform(50, 500)
                })

        return test_data

    async def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        print("Starting Redis Vector Search benchmark suite...")

        # Run individual benchmark tests
        await self.benchmark_embedding_performance()
        await self.benchmark_vector_insertion()
        await self.benchmark_vector_search()
        await self.benchmark_semantic_cache()
        await self.benchmark_batch_operations()
        await self.benchmark_concurrent_access()
        await self.benchmark_memory_usage()
        await self.benchmark_scalability()

        # Stop system monitoring
        await self.system_monitor.stop()

        # Generate results report
        await self._generate_report()

        print(f"Benchmark completed! Total results: {len(self.results)}")
        return self.results

    async def benchmark_embedding_performance(self):
        """Benchmark embedding generation performance"""
        print("\n🚀 Benchmarking embedding generation performance...")

        for batch_size in self.config.batch_sizes:
            print(f"Testing batch size: {batch_size}")

            # Warmup
            for _ in range(self.config.warmup_requests):
                sample_prompts = [item["prompt"] for item in self.test_data[:batch_size]]
                await self.embedding_processor.encode_batch(sample_prompts)

            # Benchmark
            times = []
            memory_usage = []

            for _ in range(20):  # 20 iterations
                sample_prompts = [item["prompt"] for item in
                                 np.random.choice(self.test_data, batch_size, replace=True)]

                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

                embeddings = await self.embedding_processor.encode_batch(sample_prompts)

                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
                end_time = time.time()

                times.append(end_time - start_time)
                memory_usage.append(memory_after - memory_before)

            # Calculate metrics
            total_time = sum(times)
            result = BenchmarkResult(
                test_name="embedding_generation",
                operation=f"batch_size_{batch_size}",
                total_time=total_time,
                operations_per_second=(batch_size * len(times)) / total_time,
                avg_time_per_operation=sum(times) / len(times) / batch_size,
                p50_time=statistics.median(times) / batch_size,
                p95_time=np.percentile(times, 95) / batch_size,
                p99_time=np.percentile(times, 99) / batch_size,
                memory_usage_mb=statistics.mean(memory_usage),
                cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                error_count=0,
                success_count=len(times) * batch_size,
                parameters={"batch_size": batch_size}
            )

            self.results.append(result)
            print(f"  ✅ {batch_size}: {result.operations_per_second:.2f} ops/sec, "
                  f"{result.avg_time_per_operation*1000:.2f}ms avg per item")

    async def benchmark_vector_insertion(self):
        """Benchmark vector insertion performance"""
        print("\n📝 Benchmarking vector insertion performance...")

        # Prepare embeddings for test data
        print("Preparing embeddings...")
        embeddings = []
        for item in self.test_data:
            embedding = await self.embedding_processor.encode_single(item["prompt"])
            embeddings.append(embedding)

        # Test different insertion strategies
        strategies = ["individual", "batch_small", "batch_medium", "batch_large"]
        batch_sizes = [1, 10, 50, 100]

        for strategy, batch_size in zip(strategies, batch_sizes):
            print(f"Testing insertion strategy: {strategy} (batch_size={batch_size})")

            # Clear Redis for each test
            conn = redis.Redis(connection_pool=self.redis_pool)
            await conn.flushdb()
            await conn.close()

            times = []
            inserted_count = 0

            # Insert in batches
            for i in range(0, len(self.test_data), batch_size):
                batch_end = min(i + batch_size, len(self.test_data))
                batch_items = self.test_data[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]

                start_time = time.time()

                for j, (item, embedding) in enumerate(zip(batch_items, batch_embeddings)):
                    entry = {
                        "key": f"test_{i+j}",
                        "prompt": item["prompt"],
                        "model": item["model"],
                        "response": item["response"],
                        "embedding": embedding,
                        "cost": item["cost"],
                        "latency": item["latency"]
                    }
                    success = await self.vector_manager.add_vector(entry)
                    if success:
                        inserted_count += 1

                end_time = time.time()
                times.append(end_time - start_time)

            # Calculate metrics
            total_time = sum(times)
            result = BenchmarkResult(
                test_name="vector_insertion",
                operation=strategy,
                total_time=total_time,
                operations_per_second=inserted_count / total_time,
                avg_time_per_operation=total_time / inserted_count,
                p50_time=statistics.median(times) / batch_size,
                p95_time=np.percentile(times, 95) / batch_size,
                p99_time=np.percentile(times, 99) / batch_size,
                memory_usage_mb=self.system_monitor.get_current_memory_mb(),
                cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                error_count=len(self.test_data) - inserted_count,
                success_count=inserted_count,
                parameters={"batch_size": batch_size, "total_entries": len(self.test_data)}
            )

            self.results.append(result)
            print(f"  ✅ {strategy}: {result.operations_per_second:.2f} vectors/sec, "
                  f"{result.avg_time_per_operation*1000:.2f}ms avg per vector")

    async def benchmark_vector_search(self):
        """Benchmark vector search performance"""
        print("\n🔍 Benchmarking vector search performance...")

        # Prepare query embeddings
        query_prompts = [item["prompt"] for item in self.test_data[:self.config.num_searches]]
        query_embeddings = []
        for prompt in query_prompts:
            embedding = await self.embedding_processor.encode_single(prompt)
            query_embeddings.append(embedding)

        # Test different search parameters
        for limit in self.config.search_limits:
            for threshold in self.config.similarity_thresholds:
                print(f"Testing search: limit={limit}, threshold={threshold}")

                times = []
                results_counts = []

                for i, query_embedding in enumerate(query_embeddings):
                    start_time = time.time()

                    search_results = await self.vector_manager.search_vectors(
                        query_embedding=query_embedding,
                        similarity_threshold=threshold,
                        limit=limit
                    )

                    end_time = time.time()
                    times.append(end_time - start_time)
                    results_counts.append(len(search_results))

                # Calculate metrics
                total_time = sum(times)
                avg_results = statistics.mean(results_counts)

                result = BenchmarkResult(
                    test_name="vector_search",
                    operation=f"limit_{limit}_threshold_{threshold}",
                    total_time=total_time,
                    operations_per_second=len(times) / total_time,
                    avg_time_per_operation=statistics.mean(times),
                    p50_time=statistics.median(times),
                    p95_time=np.percentile(times, 95),
                    p99_time=np.percentile(times, 99),
                    memory_usage_mb=self.system_monitor.get_current_memory_mb(),
                    cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                    error_count=0,
                    success_count=len(times),
                    parameters={
                        "limit": limit,
                        "threshold": threshold,
                        "avg_results": avg_results
                    }
                )

                self.results.append(result)
                print(f"  ✅ limit={limit}, threshold={threshold}: "
                      f"{result.operations_per_second:.2f} searches/sec, "
                      f"{result.avg_time_per_operation*1000:.2f}ms avg")

    async def benchmark_semantic_cache(self):
        """Benchmark semantic cache performance"""
        print("\n🧠 Benchmarking semantic cache performance...")

        # Cache some test data
        print("Populating semantic cache...")
        for i, item in enumerate(self.test_data[:500]):  # Use first 500 items
            await self.cache.set(
                prompt=item["prompt"],
                model=item["model"],
                response=item["response"],
                cost=item["cost"],
                latency=item["latency"]
            )

        # Test cache hits and misses
        test_operations = [
            ("cache_hit", lambda: self.cache.get(item["prompt"], item["model"]))
            for item in self.test_data[:200]  # These should be in cache
        ]

        test_operations.extend([
            ("cache_miss", lambda: self.cache.get(f"unique_prompt_{i}", "test-model"))
            for i in range(100)  # These should not be in cache
        ])

        # Shuffle operations
        np.random.shuffle(test_operations)

        times = []
        hits = 0
        misses = 0

        for op_type, op_func in test_operations:
            start_time = time.time()
            result = await op_func()
            end_time = time.time()

            times.append(end_time - start_time)

            if op_type == "cache_hit" and result:
                hits += 1
            elif op_type == "cache_miss" and result is None:
                misses += 1

        # Calculate metrics
        total_time = sum(times)
        hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        result = BenchmarkResult(
            test_name="semantic_cache",
            operation="mixed_operations",
            total_time=total_time,
            operations_per_second=len(times) / total_time,
            avg_time_per_operation=statistics.mean(times),
            p50_time=statistics.median(times),
            p95_time=np.percentile(times, 95),
            p99_time=np.percentile(times, 99),
            memory_usage_mb=self.system_monitor.get_current_memory_mb(),
            cpu_usage_percent=self.system_monitor.get_avg_cpu(),
            error_count=0,
            success_count=len(times),
            parameters={
                "total_operations": len(times),
                "cache_hit_rate": hit_rate,
                "cache_size": 500
            }
        )

        self.results.append(result)
        print(f"  ✅ Semantic cache: {result.operations_per_second:.2f} ops/sec, "
              f"hit_rate: {hit_rate:.1f}%")

    async def benchmark_batch_operations(self):
        """Benchmark batch processing operations"""
        print("\n📦 Benchmarking batch operations...")

        # Test different batch sizes for simultaneous operations
        for batch_size in self.config.batch_sizes:
            print(f"Testing batch operations size: {batch_size}")

            # Prepare batch of unique queries
            queries = []
            for i in range(batch_size):
                prompt = f"Batch test prompt {i} about technology and science"
                model = f"test-model-{i % 3}"
                queries.append((prompt, model))

            # Batch encoding benchmark
            prompts = [q[0] for q in queries]

            start_time = time.time()
            embeddings = await self.embedding_processor.encode_batch(prompts)
            batch_encode_time = time.time() - start_time

            # Batch search benchmark
            search_times = []
            for embedding in embeddings:
                start_time = time.time()
                results = await self.vector_manager.search_vectors(
                    query_embedding=embedding,
                    similarity_threshold=0.7,
                    limit=5
                )
                search_times.append(time.time() - start_time)

            total_batch_time = batch_encode_time + sum(search_times)

            result = BenchmarkResult(
                test_name="batch_operations",
                operation=f"batch_size_{batch_size}",
                total_time=total_batch_time,
                operations_per_second=len(queries) / total_batch_time,
                avg_time_per_operation=total_batch_time / len(queries),
                p50_time=statistics.median(search_times),
                p95_time=np.percentile(search_times, 95),
                p99_time=np.percentile(search_times, 99),
                memory_usage_mb=self.system_monitor.get_current_memory_mb(),
                cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                error_count=0,
                success_count=len(queries),
                parameters={
                    "batch_size": batch_size,
                    "encode_time": batch_encode_time,
                    "avg_search_time": statistics.mean(search_times)
                }
            )

            self.results.append(result)
            print(f"  ✅ Batch size {batch_size}: {result.operations_per_second:.2f} ops/sec")

    async def benchmark_concurrent_access(self):
        """Benchmark concurrent access performance"""
        print("\n🔀 Benchmarking concurrent access performance...")

        async def worker_task(worker_id: int, num_operations: int):
            """Worker task for concurrent benchmarking"""
            local_times = []
            local_hits = 0

            for i in range(num_operations):
                prompt = f"Concurrent test prompt {worker_id}_{i} about machine learning"
                model = f"gpt-3.5-turbo"

                start_time = time.time()
                result = await self.cache.get(prompt, model)
                end_time = time.time()

                local_times.append(end_time - start_time)
                if result is not None:
                    local_hits += 1

            return local_times, local_hits

        # Test different levels of concurrency
        concurrency_levels = [1, 5, 10, 20]
        operations_per_worker = 50

        for concurrent_workers in concurrency_levels:
            print(f"Testing {concurrent_workers} concurrent workers...")

            start_time = time.time()

            # Create and run concurrent tasks
            tasks = [
                worker_task(i, operations_per_worker)
                for i in range(concurrent_workers)
            ]

            worker_results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Aggregate results
            all_times = []
            total_hits = 0
            for times, hits in worker_results:
                all_times.extend(times)
                total_hits += hits

            total_operations = concurrent_workers * operations_per_worker

            result = BenchmarkResult(
                test_name="concurrent_access",
                operation=f"workers_{concurrent_workers}",
                total_time=total_time,
                operations_per_second=total_operations / total_time,
                avg_time_per_operation=statistics.mean(all_times),
                p50_time=statistics.median(all_times),
                p95_time=np.percentile(all_times, 95),
                p99_time=np.percentile(all_times, 99),
                memory_usage_mb=self.system_monitor.get_current_memory_mb(),
                cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                error_count=0,
                success_count=len(all_times),
                parameters={
                    "concurrent_workers": concurrent_workers,
                    "operations_per_worker": operations_per_worker,
                    "total_operations": total_operations,
                    "cache_hit_rate": (total_hits / total_operations * 100) if total_operations > 0 else 0
                }
            )

            self.results.append(result)
            print(f"  ✅ {concurrent_workers} workers: {result.operations_per_second:.2f} ops/sec")

    async def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        print("\n💾 Analyzing memory usage patterns...")

        # Measure memory usage at different cache sizes
        cache_sizes = [100, 500, 1000, 2000, 5000]

        for size in cache_sizes:
            print(f"Testing memory usage with {size} cached items...")

            # Clear cache
            conn = redis.Redis(connection_pool=self.redis_pool)
            await conn.flushdb()
            await conn.close()

            # Add items to cache
            memory_samples = []

            for i in range(size):
                if i % 100 == 0:
                    memory_samples.append(psutil.Process().memory_info().rss / (1024 * 1024))

                prompt = f"Memory test prompt {i} for testing storage efficiency"
                model = "test-model"
                response = f"Response {i} with some content to test memory usage patterns in vector search caching."

                await self.cache.set(prompt, model, response)

            # Final memory measurement
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            # Get Redis memory usage
            conn = redis.Redis(connection_pool=self.redis_pool)
            redis_info = await conn.info()
            redis_memory = redis_info.get("used_memory", 0) / (1024 * 1024)
            await conn.close()

            # Calculate memory per item
            avg_memory_per_item = final_memory / size if size > 0 else 0
            redis_memory_per_item = redis_memory / size if size > 0 else 0

            result = BenchmarkResult(
                test_name="memory_usage",
                operation=f"cache_size_{size}",
                total_time=0,  # Not timing this test
                operations_per_second=0,
                avg_time_per_operation=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                memory_usage_mb=final_memory,
                cpu_usage_percent=0,
                error_count=0,
                success_count=size,
                parameters={
                    "cache_size": size,
                    "avg_memory_per_item_kb": avg_memory_per_item * 1024,
                    "redis_memory_mb": redis_memory,
                    "redis_memory_per_item_kb": redis_memory_per_item * 1024,
                    "total_items": size
                }
            )

            self.results.append(result)
            print(f"  ✅ {size} items: {final_memory:.1f}MB total, "
                  f"{avg_memory_per_item*1024:.1f}KB per item")

    async def benchmark_scalability(self):
        """Benchmark system scalability"""
        print("\n📈 Testing system scalability...")

        # Test with increasing dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        search_operations = 50

        for dataset_size in dataset_sizes:
            print(f"Testing scalability with dataset size: {dataset_size}")

            # Setup test data
            conn = redis.Redis(connection_pool=self.redis_pool)
            await conn.flushdb()
            await conn.close()

            # Insert test data
            for i in range(dataset_size):
                if i % 100 == 0:
                    print(f"  Inserting {i}/{dataset_size}...")

                prompt = f"Scalability test prompt {i} with varying content"
                model = f"model-{i % 3}"
                response = f"Scalability test response {i} with comprehensive content"

                await self.cache.set(prompt, model, response)

            # Measure search performance
            search_times = []

            for i in range(search_operations):
                query_prompt = f"Scalability test query {i}"
                start_time = time.time()

                result = await self.cache.get(query_prompt, "model-1")

                end_time = time.time()
                search_times.append(end_time - start_time)

            # Calculate metrics
            avg_search_time = statistics.mean(search_times)
            p95_search_time = np.percentile(search_times, 95)

            result = BenchmarkResult(
                test_name="scalability",
                operation=f"dataset_{dataset_size}",
                total_time=sum(search_times),
                operations_per_second=search_operations / sum(search_times),
                avg_time_per_operation=avg_search_time,
                p50_time=statistics.median(search_times),
                p95_time=p95_search_time,
                p99_time=np.percentile(search_times, 99),
                memory_usage_mb=self.system_monitor.get_current_memory_mb(),
                cpu_usage_percent=self.system_monitor.get_avg_cpu(),
                error_count=0,
                success_count=search_operations,
                parameters={
                    "dataset_size": dataset_size,
                    "search_operations": search_operations,
                    "search_time_linear_scaling": avg_search_time / dataset_size * 1000
                }
            )

            self.results.append(result)
            print(f"  ✅ Dataset {dataset_size}: avg search {avg_search_time*1000:.2f}ms")

    async def _generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n📊 Generating benchmark report...")

        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save detailed results
        if self.config.save_detailed_logs:
            results_file = os.path.join(self.config.output_dir, "benchmark_results.json")
            with open(results_file, 'w') as f:
                results_data = {
                    "timestamp": datetime.now().isoformat(),
                    "config": asdict(self.config),
                    "results": [r.to_dict() for r in self.results]
                }
                json.dump(results_data, f, indent=2)

            print(f"📝 Detailed results saved to: {results_file}")

        # Generate CSV summary
        csv_file = os.path.join(self.config.output_dir, "benchmark_summary.csv")
        with open(csv_file, 'w', newline='') as csvfile:
            if self.results:
                fieldnames = self.results[0].to_dict().keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    writer.writerow(result.to_dict())

        print(f"📈 CSV summary saved to: {csv_file}")

        # Generate plots
        if self.config.generate_plots:
            await self._generate_plots()

        # Print summary
        self._print_summary()

    async def _generate_plots(self):
        """Generate performance plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Redis Vector Search Performance Benchmarks', fontsize=16)

            # Group results by test type
            by_test = {}
            for result in self.results:
                if result.test_name not in by_test:
                    by_test[result.test_name] = []
                by_test[result.test_name].append(result)

            # Plot 1: Throughput comparison
            throughput_data = []
            for test_name, results in by_test.items():
                for r in results:
                    if r.operations_per_second > 0:
                        throughput_data.append({
                            'Test': f"{test_name}_{r.operation}",
                            'Throughput': r.operations_per_second
                        })

            if throughput_data:
                df_throughput = pd.DataFrame(throughput_data)
                top_throughput = df_throughput.nlargest(10, 'Throughput')
                axes[0, 0].barh(range(len(top_throughput)), top_throughput['Throughput'])
                axes[0, 0].set_yticks(range(len(top_throughput)))
                axes[0, 0].set_yticklabels(top_throughput['Test'], fontsize=8)
                axes[0, 0].set_xlabel('Operations per Second')
                axes[0, 0].set_title('Top 10 Throughput Performance')

            # Plot 2: Latency comparison
            latency_data = []
            for result in self.results:
                if result.avg_time_per_operation > 0:
                    latency_data.append({
                        'Test': result.test_name,
                        'Operation': result.operation,
                        'Avg Latency (ms)': result.avg_time_per_operation * 1000,
                        'P95 Latency (ms)': result.p95_time * 1000
                    })

            if latency_data:
                df_latency = pd.DataFrame(latency_data)
                # Group by test and show average
                latency_summary = df_latency.groupby('Test').agg({
                    'Avg Latency (ms)': 'mean',
                    'P95 Latency (ms)': 'mean'
                }).reset_index()

                x = range(len(latency_summary))
                axes[0, 1].bar(x, latency_summary['Avg Latency (ms)'], alpha=0.7, label='Avg')
                axes[0, 1].bar(x, latency_summary['P95 Latency (ms)'], alpha=0.7, label='P95')
                axes[0, 1].set_xlabel('Test Type')
                axes[0, 1].set_ylabel('Latency (ms)')
                axes[0, 1].set_title('Latency Comparison by Test Type')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(latency_summary['Test'], rotation=45, ha='right')
                axes[0, 1].legend()

            # Plot 3: Memory usage
            memory_data = []
            for result in self.results:
                if result.memory_usage_mb > 0:
                    memory_data.append({
                        'Test': f"{result.test_name}_{result.operation}",
                        'Memory (MB)': result.memory_usage_mb
                    })

            if memory_data:
                df_memory = pd.DataFrame(memory_data)
                top_memory = df_memory.nlargest(10, 'Memory (MB)')
                axes[0, 2].bar(range(len(top_memory)), top_memory['Memory (MB)'])
                axes[0, 2].set_xlabel('Test Operation')
                axes[0, 2].set_ylabel('Memory Usage (MB)')
                axes[0, 2].set_title('Top 10 Memory Usage')
                axes[0, 2].set_xticks(range(len(top_memory)))
                axes[0, 2].set_xticklabels(top_memory['Test'], rotation=45, ha='right')

            # Plot 4: Vector insertion performance
            insertion_results = by_test.get('vector_insertion', [])
            if insertion_results:
                x = [r.parameters.get('batch_size', 0) for r in insertion_results]
                y = [r.operations_per_second for r in insertion_results]
                axes[1, 0].plot(x, y, 'o-', markersize=8, linewidth=2)
                axes[1, 0].set_xlabel('Batch Size')
                axes[1, 0].set_ylabel('Insertions per Second')
                axes[1, 0].set_title('Vector Insertion Performance')
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Search performance by parameters
            search_results = by_test.get('vector_search', [])
            if search_results:
                # Group by threshold
                thresholds = sorted(set(r.parameters.get('threshold', 0) for r in search_results))

                for threshold in thresholds:
                    threshold_results = [r for r in search_results
                                      if r.parameters.get('threshold', 0) == threshold]
                    limits = [r.parameters.get('limit', 0) for r in threshold_results]
                    throughputs = [r.operations_per_second for r in threshold_results]

                    axes[1, 1].plot(limits, throughputs, 'o-',
                                   label=f'Threshold {threshold}', markersize=6)

                axes[1, 1].set_xlabel('Search Limit')
                axes[1, 1].set_ylabel('Searches per Second')
                axes[1, 1].set_title('Vector Search Performance by Threshold')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Concurrent access scaling
            concurrent_results = by_test.get('concurrent_access', [])
            if concurrent_results:
                workers = [r.parameters.get('concurrent_workers', 0) for r in concurrent_results]
                throughputs = [r.operations_per_second for r in concurrent_results]

                axes[1, 2].bar(workers, throughputs, alpha=0.7, color='skyblue')
                axes[1, 2].set_xlabel('Number of Concurrent Workers')
                axes[1, 2].set_ylabel('Operations per Second')
                axes[1, 2].set_title('Concurrent Access Performance')
                axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = os.path.join(self.config.output_dir, "benchmark_plots.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Performance plots saved to: {plot_file}")

        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available, skipping plots")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("🏁 BENCHMARK SUMMARY")
        print("="*80)

        # Group results by test type
        by_test = {}
        for result in self.results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)

        for test_name, results in by_test.items():
            print(f"\n📊 {test_name.upper()}:")
            print("-" * 40)

            # Find best performing operation
            best_result = max(results, key=lambda r: r.operations_per_second)

            print(f"  Best Performance: {best_result.operation}")
            print(f"    Throughput: {best_result.operations_per_second:.2f} ops/sec")
            print(f"    Latency: {best_result.avg_time_per_operation*1000:.2f}ms avg, "
                  f"{best_result.p95_time*1000:.2f}ms p95")
            if best_result.memory_usage_mb > 0:
                print(f"    Memory: {best_result.memory_usage_mb:.1f}MB")

        print(f"\n📈 Overall Statistics:")
        print(f"  Total Tests: {len(self.results)}")
        print(f"  Total Operations: {sum(r.success_count for r in self.results):,}")
        print(f"  Total Errors: {sum(r.error_count for r in self.results)}")

        if self.system_monitor:
            print(f"\n💻 System Performance:")
            print(f"  Peak Memory: {self.system_monitor.get_peak_memory():.1f}MB")
            print(f"  Avg CPU Usage: {self.system_monitor.get_avg_cpu():.1f}%")

        print("\n" + "="*80)


class SystemMonitor:
    """System resource monitoring for benchmarking"""

    def __init__(self):
        self.monitoring = False
        self.monitor_task = None
        self.memory_samples = []
        self.cpu_samples = []
        self.start_time = None

    async def start(self):
        """Start system monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Monitoring loop"""
        process = psutil.Process()

        while self.monitoring:
            try:
                # Sample memory usage
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)

                # Sample CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)

                # Sleep for sampling interval
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Monitoring error: {e}")
                break

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        return max(self.memory_samples) if self.memory_samples else 0

    def get_avg_memory(self) -> float:
        """Get average memory usage in MB"""
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def get_avg_cpu(self) -> float:
        """Get average CPU usage percentage"""
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0

    def get_current_cpu(self) -> float:
        """Get current CPU usage percentage"""
        process = psutil.Process()
        return process.cpu_percent()


async def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description="Redis Vector Search Benchmark for Helix")
    parser.add_argument("--entries", type=int, default=1000, help="Number of test entries")
    parser.add_argument("--searches", type=int, default=100, help="Number of test searches")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Create benchmark configuration
    config = BenchmarkConfig(
        num_entries=args.entries,
        num_searches=args.searches,
        concurrent_requests=args.concurrent,
        output_dir=args.output,
        generate_plots=not args.no_plots
    )

    # Run benchmark
    benchmark = VectorSearchBenchmark(config)
    try:
        results = await benchmark.run_full_benchmark()
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise
    finally:
        # Cleanup
        await benchmark.cache_manager.cleanup() if benchmark.cache_manager else None
        await benchmark.cache.cleanup() if benchmark.cache else None


if __name__ == "__main__":
    asyncio.run(main())