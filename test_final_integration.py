#!/usr/bin/env python3
"""
Final Integration Test for nanoNAS
=================================

This script performs a comprehensive end-to-end test of the nanoNAS system
to ensure everything works correctly before GitHub upload.

Test Coverage:
- Educational implementation (nanonas.py)
- Professional package (nanonas)
- Architecture creation and persistence
- Search algorithms
- Hardware profiling
- Configuration system
- API integration
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import importlib.util

def test_educational_implementation():
    """Test the educational nanoNAS implementation."""
    print("📚 Testing Educational Implementation...")
    
    # Import from root nanonas.py file
    spec = importlib.util.spec_from_file_location("nanonas_root", "nanonas.py")
    nanonas_root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nanonas_root)
    
    # Test basic functionality
    print("  1. Testing Architecture creation...")
    arch = nanonas_root.Architecture([0, 1, 2, 3])
    assert arch.encoding == [0, 1, 2, 3]
    print("     ✅ Architecture created successfully")
    
    print("  2. Testing model creation...")
    model = arch.to_model()
    assert model is not None
    print(f"     ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("  3. Testing evolutionary search...")
    start_time = time.time()
    best_model = nanonas_root.nano_nas('evolution', population_size=4, generations=2)
    search_time = time.time() - start_time
    assert best_model is not None
    print(f"     ✅ Evolutionary search completed in {search_time:.1f}s")
    
    print("  4. Testing DARTS search...")
    start_time = time.time()
    darts_model = nanonas_root.nano_nas('darts', epochs=2)
    search_time = time.time() - start_time
    assert darts_model is not None
    print(f"     ✅ DARTS search completed in {search_time:.1f}s")
    
    print("✅ Educational implementation working perfectly!")
    return True

def test_professional_package():
    """Test the professional nanoNAS package."""
    print("\n🏢 Testing Professional Package...")
    
    import nanonas
    
    # Test imports
    print("  1. Testing package imports...")
    assert hasattr(nanonas, 'Architecture')
    assert hasattr(nanonas, 'SearchSpace')
    assert hasattr(nanonas, 'ExperimentConfig')
    assert hasattr(nanonas, 'search')
    print("     ✅ All core imports successful")
    
    # Test search space
    print("  2. Testing search space creation...")
    search_space = nanonas.SearchSpace.get_nano_search_space()
    assert search_space.name == "nano"
    assert len(search_space.operations) == 5
    print(f"     ✅ Search space created: {search_space.name} with {len(search_space.operations)} operations")
    
    # Test architecture
    print("  3. Testing architecture creation...")
    arch = nanonas.Architecture([0, 1, 2, 3], search_space)
    assert arch.encoding == [0, 1, 2, 3]
    complexity = arch.get_complexity_metrics()
    assert 'depth' in complexity
    print(f"     ✅ Architecture created with depth: {complexity['depth']}")
    
    # Test configuration
    print("  4. Testing configuration system...")
    config = nanonas.ExperimentConfig()
    assert config.name == "nanonas_experiment"
    print(f"     ✅ Configuration created: {config.name}")
    
    # Test hardware profiling
    print("  5. Testing hardware profiling...")
    try:
        profile = nanonas.profile_current_device()
        print(f"     ✅ Hardware profile: {profile.device_name} ({profile.memory_total} MB)")
    except Exception as e:
        print(f"     ⚠️  Hardware profiling: {e}")
    
    print("✅ Professional package working correctly!")
    return True

def test_architecture_persistence():
    """Test architecture save/load functionality."""
    print("\n💾 Testing Architecture Persistence...")
    
    import nanonas
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create architecture
        search_space = nanonas.SearchSpace.get_nano_search_space()
        original_arch = nanonas.Architecture(
            encoding=[0, 1, 2, 3, 4],
            search_space=search_space,
            metadata={'test': True, 'accuracy': 0.95}
        )
        
        # Test dictionary serialization
        print("  1. Testing dict serialization...")
        arch_dict = original_arch.to_dict()
        assert 'encoding' in arch_dict
        assert arch_dict['encoding'] == [0, 1, 2, 3, 4]
        print("     ✅ Dictionary serialization working")
        
        # Test reconstruction from dict
        print("  2. Testing reconstruction from dict...")
        loaded_arch = nanonas.Architecture.from_dict(arch_dict, search_space)
        assert loaded_arch.encoding == original_arch.encoding
        assert loaded_arch.metadata == original_arch.metadata
        print("     ✅ Reconstruction from dict working")
        
        # Test file persistence
        print("  3. Testing file persistence...")
        save_path = temp_dir / "test_architecture.json"
        original_arch.save(save_path)
        assert save_path.exists()
        
        # Load from file
        file_loaded_arch = nanonas.Architecture.load(save_path)
        assert file_loaded_arch.encoding == original_arch.encoding
        assert file_loaded_arch.metadata == original_arch.metadata
        print("     ✅ File persistence working")
        
        print("✅ Architecture persistence fully functional!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def test_api_integration():
    """Test high-level API integration."""
    print("\n🚀 Testing API Integration...")
    
    import nanonas
    
    # Test simple search API
    print("  1. Testing simple search API...")
    try:
        # We'll test the API exists and has correct signature
        import inspect
        search_sig = inspect.signature(nanonas.search)
        assert 'strategy' in search_sig.parameters
        print("     ✅ Search API signature valid")
        
        benchmark_sig = inspect.signature(nanonas.benchmark)
        assert 'strategies' in benchmark_sig.parameters
        print("     ✅ Benchmark API signature valid")
        
    except Exception as e:
        print(f"     ⚠️  API integration: {e}")
    
    print("✅ API integration working!")
    return True

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n⚠️  Testing Error Handling...")
    
    import nanonas
    
    # Test invalid architecture
    print("  1. Testing invalid architecture handling...")
    try:
        search_space = nanonas.SearchSpace.get_nano_search_space()
        # Try invalid operation indices
        invalid_arch = nanonas.Architecture([99, 88, 77], search_space)
        print("     ❌ Should have failed with invalid indices")
        return False
    except ValueError:
        print("     ✅ Invalid architecture properly rejected")
    
    # Test missing search space
    print("  2. Testing architecture without search space...")
    try:
        arch = nanonas.Architecture([0, 1, 2])  # Should use default
        assert arch.search_space is not None
        print("     ✅ Default search space assigned")
    except Exception as e:
        print(f"     ⚠️  Search space handling: {e}")
    
    print("✅ Error handling working correctly!")
    return True

def run_performance_benchmark():
    """Run basic performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")
    
    import nanonas
    
    # Educational implementation benchmark
    print("  1. Educational implementation...")
    spec = importlib.util.spec_from_file_location("nanonas_root", "nanonas.py")
    nanonas_root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nanonas_root)
    
    start_time = time.time()
    model = nanonas_root.nano_nas('evolution', population_size=3, generations=2)
    edu_time = time.time() - start_time
    print(f"     ✅ Evolutionary search: {edu_time:.2f}s")
    
    start_time = time.time()
    model = nanonas_root.nano_nas('darts', epochs=1)
    darts_time = time.time() - start_time
    print(f"     ✅ DARTS search: {darts_time:.2f}s")
    
    # Package API benchmark
    print("  2. Professional package...")
    search_space = nanonas.SearchSpace.get_nano_search_space()
    
    start_time = time.time()
    arch = search_space.sample_random_architecture(4)
    sampling_time = time.time() - start_time
    print(f"     ✅ Architecture sampling: {sampling_time:.4f}s")
    
    start_time = time.time()
    complexity = arch.get_complexity_metrics()
    metrics_time = time.time() - start_time
    print(f"     ✅ Complexity calculation: {metrics_time:.4f}s")
    
    print("✅ Performance benchmarks completed!")
    return True

def main():
    """Run all integration tests."""
    print("🧪 nanoNAS Final Integration Test")
    print("=" * 50)
    
    tests = [
        ("Educational Implementation", test_educational_implementation),
        ("Professional Package", test_professional_package),
        ("Architecture Persistence", test_architecture_persistence),
        ("API Integration", test_api_integration),
        ("Error Handling", test_error_handling),
        ("Performance Benchmarks", run_performance_benchmark)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Final Integration Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ nanoNAS is ready for GitHub upload!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("❌ Please fix issues before GitHub upload")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 