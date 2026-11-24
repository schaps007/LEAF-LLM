"""
Workload loader for standardized prompts
"""

import json
import os

def load_workload(name):
    """
    Load a workload from a JSON file
    
    Args:
        name: Workload name ('short', 'medium', or 'long')
    
    Returns:
        List of prompt strings
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{name}.json")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to default prompts
        print(f"⚠️  Workload file not found: {file_path}")
        print("   Using default prompts...")
        return get_default_workload(name)

def get_default_workload(name):
    """Fallback default workloads"""
    workloads = {
        "short": [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "What is the capital of France?"
        ],
        "medium": [
            "Explain the concept of machine learning and its applications.",
            "Describe how neural networks work in deep learning.",
            "What are the main differences between supervised and unsupervised learning?"
        ],
        "long": [
            "Provide a comprehensive explanation of transformer architecture in natural language processing, including attention mechanisms and their applications.",
            "Discuss the ethical implications of large language models and propose mitigation strategies for bias and misinformation.",
            "Explain reinforcement learning from human feedback (RLHF) and its role in aligning AI systems with human values."
        ]
    }
    return workloads.get(name, workloads["short"])