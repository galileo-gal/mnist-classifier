"""Logging utilities for TensorBoard and file logging"""
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Handles TensorBoard logging and metrics saving"""

    def __init__(self, run_dir):
        """
        Args:
            run_dir: Directory for this experiment run
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        tb_dir = self.run_dir / 'logs' / 'tensorboard'
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def log_scalar(self, tag, value, step):
        """Log scalar to TensorBoard"""
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log training metrics"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)

        # Log to TensorBoard
        self.log_scalar('Loss/train', train_loss, epoch)
        self.log_scalar('Loss/val', val_loss, epoch)
        self.log_scalar('Accuracy/train', train_acc, epoch)
        self.log_scalar('Accuracy/val', val_acc, epoch)

    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = self.run_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
