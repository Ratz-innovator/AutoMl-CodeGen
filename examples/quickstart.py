#!/usr/bin/env python3
"""
nanoNAS Quickstart Example
==========================

Get started with Neural Architecture Search in 3 lines of code!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanonas import nano_nas
from nas_insights import explore_nas_concepts

def main():
    print("🚀 nanoNAS Quickstart")
    print("=" * 30)
    
    # Option 1: Quick architecture search
    print("\n1️⃣ Quick Evolution Search:")
    model = nano_nas('evolution', population_size=8, generations=3)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Found model with {params:,} parameters")
    
    # Option 2: Gradient-based search  
    print("\n2️⃣ Quick DARTS Search:")
    model = nano_nas('darts', epochs=5)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Found model with {params:,} parameters")
    
    # Option 3: Explore insights (uncomment to see visualizations)
    print("\n3️⃣ Educational Insights:")
    print("   Uncomment the line below to explore NAS concepts!")
    # explore_nas_concepts()  # Uncomment for interactive visualizations
    
    print("\n✨ That's it! You've done Neural Architecture Search!")
    print("   → Your models are ready to train with standard PyTorch")
    print("   → Try different parameters to explore the search space")

if __name__ == "__main__":
    main() 