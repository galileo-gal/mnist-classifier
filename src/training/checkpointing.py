"""Model checkpointing utilities"""
import torch
from pathlib import Path


class CheckpointManager:
    """Handles saving and loading model checkpoints"""

    def __init__(self, checkpoint_dir, run_name):
        """
        Args:
            checkpoint_dir: Base directory for checkpoints
            run_name: Name of this run
        """
        self.checkpoint_dir = Path(checkpoint_dir) / run_name / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = None

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        # Always save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (epoch {epoch})")

    def load_checkpoint(self, model, optimizer=None, checkpoint_name='best.pth'):
        """
        Load model checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to restore
            checkpoint_name: Which checkpoint to load

        Returns:
            Loaded epoch number and metrics
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return 0, {}

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']

