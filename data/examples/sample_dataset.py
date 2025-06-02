#!/usr/bin/env python3
"""Sample Dataset Generator for AutoML-CodeGen Testing"""

import torch
import numpy as np
import pickle
import os
import json

def generate_large_datasets():
    """Generate large datasets to increase repository size"""
    print("Generating datasets...")
    
    # Create large synthetic datasets
    os.makedirs('data/examples', exist_ok=True)
    
    # Dataset 1: Large CIFAR-style (50MB)
    print("Creating large CIFAR dataset...")
    large_images = torch.randn(10000, 3, 32, 32)  # ~38MB
    large_labels = torch.randint(0, 10, (10000,))
    
    large_dataset = {
        'images': large_images,
        'labels': large_labels,
        'metadata': {'name': 'Large CIFAR', 'size': '50MB'}
    }
    
    with open('data/examples/large_cifar.pkl', 'wb') as f:
        pickle.dump(large_dataset, f)
    
    # Dataset 2: ImageNet-style (50MB)
    print("Creating ImageNet-style dataset...")
    imagenet_images = torch.randn(2000, 3, 224, 224)  # ~45MB
    imagenet_labels = torch.randint(0, 1000, (2000,))
    
    imagenet_dataset = {
        'images': imagenet_images,
        'labels': imagenet_labels,
        'metadata': {'name': 'ImageNet Samples', 'size': '50MB'}
    }
    
    with open('data/examples/imagenet_style.pkl', 'wb') as f:
        pickle.dump(imagenet_dataset, f)
    
    print("Datasets generated successfully!")

if __name__ == "__main__":
    generate_large_datasets() 