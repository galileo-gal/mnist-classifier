"""Main training script - run experiments from config files"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.mnist import get_mnist_loaders
from src.models.fc import FullyConnectedNet
from src.training.trainer import Trainer


def create_run_dir(config):
    """Create unique run directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['name']}_{timestamp}"
    run_dir = Path(config['logging']['checkpoint_dir']) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def main(config_path):
    """Main training pipeline"""
    # Load config
    config = load_config(config_path)
    print(f"Loaded config: {config['name']}")

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Create run directory
    run_dir, run_name = create_run_dir(config)

    # Save config to run directory
    save_config(config, run_dir / 'config.yaml')
    print(f"Run directory: {run_dir}")

    # Get device
    device = get_device()
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_mnist_loaders(config)

    # Create model
    if config['model']['type'] == 'fc':
        model = FullyConnectedNet.from_config(config)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        run_dir=run_dir
    )

    # Train
    trainer.train(num_epochs=config['training']['epochs'])

    print(f"\n✓ Training complete! Results saved to: {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args.config)