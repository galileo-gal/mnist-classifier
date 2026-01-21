"""Device management utilities"""
import torch


def get_device(force_cpu=False):
    """
    Get appropriate device (GPU/CPU)

    Args:
        force_cpu: If True, always return CPU

    Returns:
        torch.device
    """
    if force_cpu:
        return torch.device('cpu')

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(obj, device):
    """
    Move model or tensor to device

    Args:
        obj: Model or tensor
        device: Target device

    Returns:
        Object on target device
    """
    return obj.to(device)

