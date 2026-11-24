"""
Random seed utilities for reproducibility
"""

import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For GPU reproducibility (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False