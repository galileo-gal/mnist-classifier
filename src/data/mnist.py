"""MNIST dataset with proper train/val/test splits"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(config):
    """
    Create MNIST data loaders with train/val/test split

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 2)
    train_split = config['data'].get('train_split', 0.8)

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training set
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    # Split into train/val
    train_size = int(train_split * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    # Test set (never touched during training/validation)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")

    return train_loader, val_loader, test_loader
