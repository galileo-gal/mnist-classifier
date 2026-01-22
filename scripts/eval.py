"""Evaluate trained model on test set"""
import sys
from pathlib import Path
import argparse
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device
from src.data.mnist import get_mnist_loaders
from src.models.fc import FullyConnectedNet


def find_checkpoint_and_config(run_name):
    """Find checkpoint and config files for a run"""
    runs_dir = Path('runs')

    # Find config (in timestamped directory)
    config_file = None
    for d in runs_dir.glob(f'{run_name}_*'):
        if (d / 'config.yaml').exists():
            config_file = d / 'config.yaml'
            break

    # Find checkpoint (in base run_name directory)
    checkpoint_file = runs_dir / run_name / 'checkpoints' / 'best.pth'

    return config_file, checkpoint_file


def evaluate_model(run_name):
    """Evaluate model on test set"""
    config_file, checkpoint_file = find_checkpoint_and_config(run_name)

    if not config_file or not config_file.exists():
        print(f"ERROR: Config not found for {run_name}")
        return

    if not checkpoint_file.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_file}")
        return

    # Load config
    config = load_config(config_file)
    print(f"Loaded config from: {config_file}")
    print(f"Loaded checkpoint from: {checkpoint_file}")

    # Setup
    device = get_device()
    _, _, test_loader = get_mnist_loaders(config)

    # Load model
    model = FullyConnectedNet.from_config(config).to(device)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Correct: {correct:,} / {total:,}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='baseline_fc', help='Run name (without timestamp)')
    args = parser.parse_args()

    evaluate_model(args.run)