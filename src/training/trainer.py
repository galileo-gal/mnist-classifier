"""Main training loop with all production features"""
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.logging import ExperimentLogger
from src.training.checkpointing import CheckpointManager
from src.training.early_stopping import EarlyStopping


class Trainer:
    """Production-grade trainer with logging, checkpointing, early stopping"""

    def __init__(self, model, train_loader, val_loader, config, device, run_dir):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: torch.device
            run_dir: Directory for this run's artifacts
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(config)

        # Infrastructure
        self.logger = ExperimentLogger(run_dir)
        self.checkpoint_manager = CheckpointManager(
            config['logging']['checkpoint_dir'],
            config['name']
        )
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode='min'  # For loss
        )

        self.best_val_loss = float('inf')

    def _create_optimizer(self, config):
        """Create optimizer from config"""
        opt_name = config['training']['optimizer'].lower()
        lr = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 0)

        if opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.view(-1, 784).to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.view(-1, 784).to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def train(self, num_epochs):
        """Full training loop"""
        print(f"\nStarting training: {self.config['name']}")
        print("=" * 60)

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Log metrics
            self.logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc},
                is_best=is_best
            )

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print("=" * 60)
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        # Save final metrics
        self.logger.save_metrics()
        self.logger.close()