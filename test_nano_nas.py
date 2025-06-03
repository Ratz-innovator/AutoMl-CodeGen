"""
Simple Tests for nanoNAS
========================

Clean, minimal tests for the educational NAS implementation.
"""

import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from root nanonas.py file 
import importlib.util
spec = importlib.util.spec_from_file_location("nanonas_root", "nanonas.py")
nanonas_root = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nanonas_root)

nano_nas = nanonas_root.nano_nas
Architecture = nanonas_root.Architecture
Operation = nanonas_root.Operation
from nas_insights import ArchitectureDNA


def test_nano_nas_evolution():
    """Test evolutionary search works."""
    print("🧬 Testing evolutionary search...")
    model = nano_nas('evolution', population_size=5, generations=2)
    
    # Basic checks
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Check it can process input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10)  # CIFAR-10 classes
    
    print("✅ Evolutionary search working!")


def test_nano_nas_darts():
    """Test DARTS search works."""
    print("📈 Testing DARTS search...")
    model = nano_nas('darts', epochs=3)
    
    # Basic checks
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Check it can process input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10)
    
    print("✅ DARTS search working!")


def test_architecture_dna():
    """Test architecture DNA encoding/decoding."""
    print("🧬 Testing architecture DNA...")
    
    dna_analyzer = ArchitectureDNA()
    
    # Test encoding
    arch = Architecture([0, 1, 2, 3])
    dna_string = dna_analyzer.encode_architecture(arch)
    assert len(dna_string) == 4
    assert all(base in 'ATGCN' for base in dna_string)
    
    # Test decoding
    decoded_arch = dna_analyzer.decode_dna(dna_string)
    assert decoded_arch.encoding == arch.encoding
    
    print("✅ Architecture DNA working!")


def test_operations():
    """Test individual operations work."""
    print("🔧 Testing operations...")
    
    channels = 16
    x = torch.randn(1, channels, 32, 32)
    
    # Test all operations
    for op_name in Operation.OPS.keys():
        op = Operation(op_name, channels)
        output = op(x)
        assert output.shape[0] == 1  # Batch size preserved
        assert output.shape[1] == channels  # Channels preserved
    
    print("✅ All operations working!")


def test_architecture_mutation():
    """Test architecture mutation."""
    print("🧬 Testing mutation...")
    
    arch = Architecture([0, 1, 2, 3])
    mutated = arch.mutate(mutation_rate=1.0)  # Force mutation
    
    # Should be different (with high probability)
    assert mutated.encoding != arch.encoding
    assert len(mutated.encoding) == len(arch.encoding)
    
    print("✅ Mutation working!")


if __name__ == "__main__":
    print("🧪 Running nanoNAS Tests")
    print("=" * 30)
    
    # Run all tests
    test_operations()
    test_architecture_dna() 
    test_architecture_mutation()
    test_nano_nas_evolution()
    test_nano_nas_darts()
    
    print("\n🎉 All tests passed!")
    print("nanoNAS is working perfectly!") 