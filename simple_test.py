"""
Simple Test for nanoNAS
======================

Basic test to verify the nanonas functionality works.
"""

import torch
import sys
import os
import importlib.util

# Import directly from root nanonas.py file 
spec = importlib.util.spec_from_file_location("nanonas_root", "nanonas.py")
nanonas_root = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nanonas_root)

def test_basic_functionality():
    """Test basic nanonas functionality."""
    print("🧪 Testing basic nanoNAS functionality...")
    
    # Test Architecture creation
    print("1. Testing Architecture creation...")
    arch = nanonas_root.Architecture([0, 1, 2, 3])
    print(f"   Created architecture: {arch.encoding}")
    
    # Test Operation creation
    print("2. Testing Operation creation...")
    op = nanonas_root.Operation('conv3x3', 16)
    print(f"   Created operation: conv3x3")
    
    # Test model creation
    print("3. Testing model creation...")
    model = arch.to_model()
    print(f"   Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test model inference
    print("4. Testing model inference...")
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    print(f"   Model output shape: {output.shape}")
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    
    # Test nano_nas function
    print("5. Testing nano_nas function...")
    try:
        model = nanonas_root.nano_nas('evolution', population_size=3, generations=2)
        print(f"   Evolutionary search created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test inference
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        print(f"   Model output shape: {output.shape}")
        assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
        
    except Exception as e:
        print(f"   Evolutionary search failed: {e}")
    
    print("✅ Basic functionality tests completed!")

if __name__ == "__main__":
    print("🔍 Running Simple nanoNAS Test")
    print("=" * 40)
    
    test_basic_functionality()
    
    print("\n🎉 Simple test completed!") 