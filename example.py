#!/usr/bin/env python3
"""
Simple nanoNAS Usage Example

This example shows the basic usage of nanoNAS for neural architecture search.
Perfect for getting started or demonstrating the capabilities.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from nanonas import nano_nas

def main():
    """Simple usage example"""
    
    print("nanoNAS Simple Example")
    print("=" * 30)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Basic architecture search
    print("\n1. Running architecture search...")
    model = nano_nas('evolution', population_size=5, generations=2)
    
    # 2. Check the discovered model
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Found model with {param_count:,} parameters")
    
    # 3. Test the model with random input
    print("\n2. Testing forward pass...")
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 input size
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output classes: {output.size(1)}")
    
    # 4. Quick CIFAR-10 test (optional)
    print("\n3. Quick CIFAR-10 test...")
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
        
        # Test on a few batches
        correct = 0
        total = 0
        model.to(device)
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= 3:  # Test only 3 batches for speed
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"   Accuracy on test samples: {accuracy:.2f}%")
        print(f"   (This is untrained - expect low accuracy)")
        
    except Exception as e:
        print(f"   Skipping CIFAR-10 test: {e}")
    
    # 5. Show how to use the model
    print("\n4. How to train this model:")
    print("   # The model is a standard PyTorch nn.Module")
    print("   optimizer = torch.optim.SGD(model.parameters(), lr=0.1)")
    print("   criterion = torch.nn.CrossEntropyLoss()")
    print("   # ... standard PyTorch training loop")
    
    print("\n✅ Example completed!")
    print("   The discovered architecture is ready for training.")
    
    return model

if __name__ == "__main__":
    discovered_model = main() 