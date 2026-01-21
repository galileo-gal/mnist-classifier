"""
Reproducibility utilities - set seeds for deterministic training
"""
import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed_from_config(config):
    """Extract seed from config dict, default to 42"""
    return config.get('seed', 42)