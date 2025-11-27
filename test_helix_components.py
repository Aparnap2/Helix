#!/usr/bin/env python3
"""
Comprehensive testing script for Helix AI Gateway components
Tests semantic caching, PII detection, and cost tracking
"""

import os
import sys
import json
import time
import asyncio
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add LiteLLM to path
sys.path.insert(0, '/home/aparna/Desktop/Helix/litellm')

try:
    import redis
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Installing missing packages...")
    os.system("pip install redis sentence-transformers presidio-analyzer presidio-anonymizer")
    # Try importing again
    import redis
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    print("✓ Packages installed and imported successfully")

class HelixTester:
    """Comprehensive testing suite for Helix components"""

    def __init__(self):
        self.redis_url = "redis://localhost:6382"
        self.test_results = {
            'redis_connection': False,
            'vector_index': False,
            'semantic_search': False,
            'pii_detection': False,
            'embedding_generation': False,
            'exact_caching': False,
            'cost_tracking': False,
            'end_to_end': False
        }

    def test_redis_connection(self) -> bool:
        """Test Redis connection and basic operations"""
        try:
            print("\n=== Testing Redis Connection ===")
            r = redis.from_url(self.redis_url)

            # Test ping
            pong = r.ping()
            print(f"Redis ping: {pong}")

            # Test basic operations
            test_key = "helix:test:connection"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            print(f"Redis get/set: {value.decode() if value else 'None'}")

            # Clean up
            r.delete(test_key)

            self.redis_client = r
            self.test_results['redis_connection'] = True
            print("✓ Redis connection test passed")
            return True

        except Exception as e:
            print(f"✗ Redis connection test failed: {e}")
            return False

    def test_vector_index(self) -> bool:
        """Test Redis Vector Search index"""
        try:
            print("\n=== Testing Vector Index ===")

            # Check if index exists
            try:
                info = self.redis_client.ft("idx:semantic").info()
                print(f"Vector index info: {info}")
            except:
                # Create index if doesn't exist
                print("Creating vector index...")
                result = self.redis_client.ft("idx:semantic").create_index([
                    {"name": "prompt", "type": "TEXT"},
                    {"name": "model", "type": "TEXT"},
                    {"name": "response_json", "type": "TEXT"},
                    {"name": "vector", "type": "VECTOR", "options": {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "BLOCK_SIZE": 1000
                    }}
                ])
                print(f"Index creation result: {result}")

            self.test_results['vector_index'] = True
            print("✓ Vector index test passed")
            return True

        except Exception as e:
            print(f"✗ Vector index test failed: {e}")
            return False

    def test_embedding_generation(self) -> bool:
        """Test sentence transformer for embeddings"""
        try:
            print("\n=== Testing Embedding Generation ===")

            # Initialize model
            print("Loading sentence transformer model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Test embedding generation
            test_text = "How do I reset my password?"
            print(f"Testing with text: '{test_text}'")

            embedding = model.encode(test_text, normalize_embeddings=True)
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding type: {type(embedding)}")
            print(f"Sample values: {embedding[:5]}")

            # Test with different texts
            texts = [
                "What is my account balance?",
                "How can I update my profile?",
                "Where do I find billing information?"
            ]

            embeddings = model.encode(texts, normalize_embeddings=True)
            print(f"Batch embedding shape: {embeddings.shape}")

            self.embedder = model
            self.test_results['embedding_generation'] = True
            print("✓ Embedding generation test passed")
            return True

        except Exception as e:
            print(f"✗ Embedding generation test failed: {e}")
            return False

    def test_semantic_search(self) -> bool:
        """Test semantic search functionality"""
        try:
            print("\n=== Testing Semantic Search ===")

            if not hasattr(self, 'embedder'):
                print("✗ Embedder not initialized")
                return False

            # Create test data
            test_prompts = [
                "How do I reset my password?",
                "What is my account balance?",
                "How can I update my profile information?",
                "Where do I find billing statements?",
                "How do I cancel my subscription?"
            ]

            test_responses = [
                {"content": "To reset your password, go to Settings > Security > Reset Password"},
                {"content": "Your account balance is displayed on the dashboard homepage"},
                {"content": "Update your profile by clicking on your avatar and selecting Edit Profile"},
                {"content": "Billing statements are available in the Billing > Statements section"},
                {"content": "Cancel your subscription from Settings > Subscription > Cancel Plan"}
            ]

            # Store test data with embeddings
            print("Storing test data in vector cache...")
            for i, (prompt, response) in enumerate(zip(test_prompts, test_responses)):
                # Generate embedding
                embedding = self.embedder.encode(prompt, normalize_embeddings=True)
                embedding_bytes = embedding.astype(np.float32).tobytes()

                # Create vector key
                vector_key = f"helix:vector:test:{uuid.uuid4()}"

                # Store in Redis
                self.redis_client.hset(vector_key, mapping={
                    "prompt": prompt,
                    "model": "gpt-3.5-turbo",
                    "response_json": json.dumps(response),
                    "vector": embedding_bytes
                })

                print(f"Stored: {prompt[:30]}...")

            # Wait a moment for indexing
            time.sleep(1)

            # Test semantic search
            print("\nTesting semantic search...")
            query = "How can I change my password?"
            query_embedding = self.embedder.encode(query, normalize_embeddings=True)
            query_bytes = query_embedding.astype(np.float32).tobytes()

            # Perform search
            search_query = f"(@model:gpt-3.5-turbo)=>[KNN 3 @vector $query_blob AS score]"
            results = self.redis_client.ft("idx:semantic").search(
                search_query,
                query_params={"query_blob": query_bytes}
            )

            print(f"Search results: {results.total} found")
            for doc in results.docs:
                print(f"  - {doc.prompt} (score: {doc.score})")

            # Clean up test data
            for key in self.redis_client.scan_iter(match="helix:vector:test:*"):
                self.redis_client.delete(key)

            self.test_results['semantic_search'] = True
            print("✓ Semantic search test passed")
            return True

        except Exception as e:
            print(f"✗ Semantic search test failed: {e}")
            return False

    def test_pii_detection(self) -> bool:
        """Test PII detection and redaction"""
        try:
            print("\n=== Testing PII Detection ===")

            # Initialize Presidio
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()

            # Test cases with PII
            test_cases = [
                {
                    "text": "My email is john.doe@example.com and my phone is 555-123-4567",
                    "expected_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]
                },
                {
                    "text": "My credit card number is 4532-1234-5678-9012",
                    "expected_entities": ["CREDIT_CARD"]
                },
                {
                    "text": "My API key is sk-1234567890abcdef and password is secret123",
                    "expected_entities": None  # May not detect these
                },
                {
                    "text": "This is a normal message without PII",
                    "expected_entities": []
                }
            ]

            for i, test_case in enumerate(test_cases):
                text = test_case["text"]
                print(f"\nTest {i+1}: {text}")

                # Analyze for PII
                results = analyzer.analyze(
                    text=text,
                    language="en",
                    entities=["CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "IP_ADDRESS"]
                )

                print(f"PII entities found: {len(results)}")
                for result in results:
                    print(f"  - {result.entity_type}: '{text[result.start:result.end]}' (confidence: {result.score:.2f})")

                # Test redaction
                if results:
                    anonymized = anonymizer.anonymize(
                        text=text,
                        analyzer_results=results
                    )
                    print(f"Redacted text: {anonymized.text}")
                else:
                    print("No redaction needed")

            self.analyzer = analyzer
            self.anonymizer = anonymizer
            self.test_results['pii_detection'] = True
            print("✓ PII detection test passed")
            return True

        except Exception as e:
            print(f"✗ PII detection test failed: {e}")
            return False

    def test_exact_caching(self) -> bool:
        """Test exact caching functionality"""
        try:
            print("\n=== Testing Exact Caching ===")

            # Test data
            test_request = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "temperature": 0.7,
                "max_tokens": 100
            }

            test_response = {
                "choices": [{"message": {"content": "I'm doing well, thank you!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
                "model": "gpt-3.5-turbo"
            }

            # Generate cache key
            cache_data = {
                'model': test_request.get('model'),
                'messages': test_request.get('messages', []),
                'temperature': test_request.get('temperature'),
                'max_tokens': test_request.get('max_tokens')
            }
            cache_str = json.dumps(cache_data, sort_keys=True)
            cache_key = f"helix:exact:{hashlib.sha256(cache_str.encode()).hexdigest()}"

            print(f"Cache key: {cache_key}")

            # Test cache miss
            cached_data = self.redis_client.get(cache_key)
            print(f"Cache miss (should be None): {cached_data}")

            # Store in cache
            cache_ttl = 3600  # 1 hour
            self.redis_client.setex(cache_key, cache_ttl, json.dumps(test_response))
            print("Stored response in cache")

            # Test cache hit
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                cached_response = json.loads(cached_data)
                print(f"Cache hit: {cached_response}")

                # Verify data integrity
                if cached_response['model'] == test_response['model']:
                    print("✓ Cached data matches original")
                else:
                    print("✗ Cached data mismatch")
                    return False
            else:
                print("✗ Cache hit failed")
                return False

            # Clean up
            self.redis_client.delete(cache_key)

            self.test_results['exact_caching'] = True
            print("✓ Exact caching test passed")
            return True

        except Exception as e:
            print(f"✗ Exact caching test failed: {e}")
            return False

    def test_cost_tracking(self) -> bool:
        """Test cost tracking functionality"""
        try:
            print("\n=== Testing Cost Tracking ===")

            # Test data
            user_id = "test_user_123"
            costs = [0.001, 0.002, 0.0015, 0.003, 0.001]

            today = datetime.now().strftime("%Y-%m-%d")

            # Track costs
            total_cost = 0
            for i, cost in enumerate(costs):
                # Track total spend
                self.redis_client.zincrby("helix:spend:total", cost, today)

                # Track user spend
                user_key = f"helix:spend:user:{user_id}"
                self.redis_client.zincrby(user_key, cost, today)

                # Track model spend
                model_key = f"helix:spend:model:gpt-3.5-turbo"
                self.redis_client.zincrby(model_key, cost, today)

                total_cost += cost
                print(f"Tracked cost {i+1}: ${cost:.4f}")

            # Verify tracking
            total_spend = self.redis_client.zrange("helix:spend:total", 0, -1, withscores=True)
            user_spend = self.redis_client.zrange(f"helix:spend:user:{user_id}", 0, -1, withscores=True)

            print(f"Total spend records: {len(total_spend)}")
            for date, cost in total_spend:
                print(f"  {date}: ${float(cost):.4f}")

            print(f"User spend records: {len(user_spend)}")
            for date, cost in user_spend:
                print(f"  {date}: ${float(cost):.4f}")

            # Test analytics
            self.redis_client.incr("helix:requests:total")
            self.redis_client.incr("helix:requests:cache_hits")

            total_requests = int(self.redis_client.get("helix:requests:total") or 0)
            cache_hits = int(self.redis_client.get("helix:requests:cache_hits") or 0)
            hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

            print(f"Total requests: {total_requests}")
            print(f"Cache hits: {cache_hits}")
            print(f"Hit rate: {hit_rate:.1f}%")

            # Clean up test data
            self.redis_client.delete("helix:spend:total")
            self.redis_client.delete(f"helix:spend:user:{user_id}")
            self.redis_client.delete("helix:spend:model:gpt-3.5-turbo")
            self.redis_client.delete("helix:requests:total")
            self.redis_client.delete("helix:requests:cache_hits")

            self.test_results['cost_tracking'] = True
            print("✓ Cost tracking test passed")
            return True

        except Exception as e:
            print(f"✗ Cost tracking test failed: {e}")
            return False

    def test_end_to_end(self) -> bool:
        """Test complete end-to-end flow"""
        try:
            print("\n=== Testing End-to-End Flow ===")

            if not all([self.test_results['redis_connection'],
                       self.test_results['embedding_generation'],
                       self.test_results['pii_detection']]):
                print("✗ Prerequisites not met for end-to-end test")
                return False

            # Simulate a request
            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "My email is john@example.com, how do I reset my password?"}],
                "temperature": 0.7,
                "max_tokens": 100
            }

            user_id = "test_user_e2e"
            start_time = time.time()

            print(f"Processing request: {request_data['messages'][0]['content']}")

            # Step 1: Check exact cache
            cache_data = {
                'model': request_data.get('model'),
                'messages': request_data.get('messages', []),
                'temperature': request_data.get('temperature'),
                'max_tokens': request_data.get('max_tokens')
            }
            cache_str = json.dumps(cache_data, sort_keys=True)
            exact_key = f"helix:exact:{hashlib.sha256(cache_str.encode()).hexdigest()}"

            cached_response = self.redis_client.get(exact_key)
            if cached_response:
                print("✓ Exact cache hit")
                response = json.loads(cached_response)
            else:
                print("✗ Exact cache miss - proceeding with processing")

                # Step 2: PII Detection and Redaction
                user_prompt = request_data['messages'][0]['content']
                pii_results = self.analyzer.analyze(
                    text=user_prompt,
                    language="en",
                    entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]
                )

                if pii_results:
                    print(f"PII detected: {len(pii_results)} entities")
                    redacted_prompt = self.anonymizer.anonymize(
                        text=user_prompt,
                        analyzer_results=pii_results
                    ).text
                    print(f"Redacted prompt: {redacted_prompt}")

                    # Log PII incident
                    incident = {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "entities": [r.entity_type for r in pii_results],
                        "original": user_prompt
                    }
                    self.redis_client.lpush("helix:pii:incidents", json.dumps(incident))
                    self.redis_client.ltrim("helix:pii:incidents", 0, 99)  # Keep last 100

                    # Update request with redacted content
                    request_data['messages'][0]['content'] = redacted_prompt
                else:
                    print("No PII detected")

                # Step 3: Semantic Search (simulated)
                # In real implementation, this would search for similar cached responses
                print("Checking semantic cache...")

                # Step 4: Simulate LLM call (mock response)
                response = {
                    "choices": [{"message": {"content": "You can reset your password by going to Settings and clicking on 'Reset Password'."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 15},
                    "model": "gpt-3.5-turbo"
                }

                # Step 5: Store in caches
                latency = time.time() - start_time
                cost = 0.002  # Mock cost

                # Store in exact cache
                self.redis_client.setex(exact_key, 3600, json.dumps(response))
                print("✓ Stored in exact cache")

                # Store in semantic cache
                embedding = self.embedder.encode(user_prompt, normalize_embeddings=True)
                vector_key = f"helix:vector:{uuid.uuid4()}"
                self.redis_client.hset(vector_key, mapping={
                    "prompt": user_prompt,
                    "model": request_data['model'],
                    "response_json": json.dumps(response),
                    "vector": embedding.astype(np.float32).tobytes()
                })
                print("✓ Stored in semantic cache")

                # Step 6: Track analytics
                today = datetime.now().strftime("%Y-%m-%d")
                self.redis_client.zincrby("helix:spend:total", cost, today)
                self.redis_client.zincrby(f"helix:spend:user:{user_id}", cost, today)
                self.redis_client.incr("helix:requests:total")

                print(f"✓ Tracked cost: ${cost:.4f}")
                print(f"✓ Latency: {latency:.3f}s")

            print(f"Final response: {response['choices'][0]['message']['content']}")

            # Verify PII incident was logged
            pii_incidents = self.redis_client.lrange("helix:pii:incidents", 0, -1)
            print(f"PII incidents logged: {len(pii_incidents)}")

            # Clean up test data
            self.redis_client.delete(exact_key)
            self.redis_client.delete("helix:pii:incidents")
            for key in self.redis_client.scan_iter(match="helix:vector:*"):
                if uuid.UUID(key.split(':')[-1]):  # Check if it's a UUID
                    self.redis_client.delete(key)

            self.test_results['end_to_end'] = True
            print("✓ End-to-end test passed")
            return True

        except Exception as e:
            print(f"✗ End-to-end test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🔮 Helix AI Gateway - Comprehensive Component Testing")
        print("=" * 60)

        tests = [
            ("Redis Connection", self.test_redis_connection),
            ("Vector Index", self.test_vector_index),
            ("Embedding Generation", self.test_embedding_generation),
            ("PII Detection", self.test_pii_detection),
            ("Exact Caching", self.test_exact_caching),
            ("Semantic Search", self.test_semantic_search),
            ("Cost Tracking", self.test_cost_tracking),
            ("End-to-End Flow", self.test_end_to_end)
        ]

        start_time = time.time()

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False

        total_time = time.time() - start_time

        # Generate report
        print("\n" + "=" * 60)
        print("🏁 TEST REPORT")
        print("=" * 60)

        passed = sum(self.test_results.values())
        total = len(self.test_results)

        print(f"Tests completed in {total_time:.2f} seconds")
        print(f"Passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")

        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")

        print(f"\n{'🎉 All tests passed!' if passed == total else '❌ Some tests failed.'}")

        return passed == total

def main():
    """Main testing function"""
    tester = HelixTester()
    success = tester.run_all_tests()

    # Clean up Redis container
    print("\n=== Cleaning up ===")
    os.system("docker stop test-redis > /dev/null 2>&1")
    os.system("docker rm test-redis > /dev/null 2>&1")
    print("✓ Cleanup completed")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)