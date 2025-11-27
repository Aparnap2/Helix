#!/usr/bin/env python3
# Helix Dashboard Test Script
# Verify dashboard setup and dependencies

import sys
import os
import importlib
import traceback
from datetime import datetime

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    try:
        import redis
        client = redis.from_url("redis://localhost:6379")
        client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_dashboard_imports():
    """Test dashboard module imports"""
    print("🔍 Testing dashboard imports...")

    # Core dependencies
    tests = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("plotly.express", "Plotly Express"),
        ("plotly.graph_objects", "Plotly Graph Objects"),
        ("numpy", "NumPy"),
        ("redis", "Redis"),
        ("psutil", "System utilities (psutil)")
    ]

    core_passed = sum(test_import(module, desc) for module, desc in tests)

    print(f"\n📊 Core dependencies: {core_passed}/{len(tests)} passed")

    # Dashboard pages
    print("\n🔍 Testing dashboard page imports...")

    page_tests = [
        ("pages.overview", "Overview page"),
        ("pages.cost_analysis", "Cost Analysis page"),
        ("pages.cache_performance", "Cache Performance page"),
        ("pages.security", "Security page"),
        ("pages.user_management", "User Management page"),
        ("pages.system_health", "System Health page")
    ]

    page_passed = sum(test_import(module, desc) for module, desc in page_tests)

    print(f"\n📄 Page modules: {page_passed}/{len(page_tests)} passed")

    return core_passed == len(tests) and page_passed == len(page_tests)

def test_dashboard_config():
    """Test dashboard configuration"""
    print("\n🔍 Testing dashboard configuration...")

    config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

    if os.path.exists(config_path):
        print("✅ Streamlit configuration found")
        return True
    else:
        print("❌ Streamlit configuration not found")
        return False

def test_file_structure():
    """Test required file structure"""
    print("\n🔍 Testing file structure...")

    required_files = [
        "dashboard.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "pages/__init__.py",
        "pages/overview.py",
        "pages/cost_analysis.py",
        "pages/cache_performance.py",
        "pages/security.py",
        "pages/user_management.py",
        "pages/system_health.py"
    ]

    passed = 0
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
            passed += 1
        else:
            print(f"❌ {file_path} not found")

    print(f"\n📁 File structure: {passed}/{len(required_files)} files found")
    return passed == len(required_files)

def main():
    """Main test function"""
    print("🔮 Helix AI Gateway Dashboard Test Suite")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test file structure
    structure_ok = test_file_structure()

    # Test configuration
    config_ok = test_dashboard_config()

    # Test imports
    imports_ok = test_dashboard_imports()

    # Test Redis (optional)
    print("\n🔍 Testing Redis connection (optional)...")
    redis_ok = test_redis_connection()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"📁 File Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"⚙️  Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"📦 Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"🔴 Redis: {'✅ PASS' if redis_ok else '❌ FAIL'}")

    all_critical_ok = structure_ok and config_ok and imports_ok

    if all_critical_ok:
        print("\n🎉 All critical tests passed! Dashboard is ready to run.")
        print("\n🚀 To start the dashboard:")
        print("   cd helix/dashboard")
        print("   streamlit run dashboard.py")
        print("\n🐳 Or use Docker:")
        print("   docker-compose -f docker-compose.dashboard.yml up")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the dashboard.")
        print("\n🔧 Common fixes:")
        if not structure_ok:
            print("   - Ensure all required files are present")
        if not config_ok:
            print("   - Create .streamlit/config.toml")
        if not imports_ok:
            print("   - Install dependencies: pip install -r requirements.txt")
        if not redis_ok:
            print("   - Start Redis server or set REDIS_URL correctly")

    return all_critical_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)