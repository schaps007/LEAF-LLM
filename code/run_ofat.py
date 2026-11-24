"""
Main entry point for OFAT experiments
Run: python run_ofat.py
"""

import sys
import os
import logging

# Suppress verbose logging
logging.getLogger('codecarbon').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.ofat_runner import main

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌿 LEAF-LLM: OFAT Energy Efficiency Framework")
    print("=" * 70)
    print("One-Factor-At-A-Time experimental design")
    print("Each factor is varied while others are held constant")
    print("=" * 70)
    
    main()