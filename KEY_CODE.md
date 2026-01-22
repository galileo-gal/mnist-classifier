# KEY_CODE.md - Essential Code Patterns & Commands

> **Note:** This reference includes patterns for features currently in development. Only the baseline FC model, training pipeline, and utility modules are fully functional. CNN, interpretability, and testing code are planned for upcoming phases.

Quick reference for common tasks and code patterns in this project.
---

## 📦 Class & Function Reference

### `src/models/fc.py`
```python
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, dropout=0.2)
    def forward(self, x)
    @classmethod
    def from_config(cls, config)
```

### `src/training/trainer.py`
```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, run_dir)
    def _create_optimizer(self, config)
    def train_epoch(self, epoch)
    def validate(self)
    def train(self, num_epochs)
```

### `src/training/checkpointing.py`
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, run_name)
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False)
    def load_checkpoint(self, model, optimizer=None, checkpoint_name='best.pth')
```

### `src/training/early_stopping.py`
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min')
    def __call__(self, val_metric)  # Returns True if should stop
```

### `src/utils/logging.py`
```python
class ExperimentLogger:
    def __init__(self, run_dir)
    def log_scalar(self, tag, value, step)
    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc)
    def save_metrics(self)
    def close(self)
```

### `src/utils/config.py`
```python
def load_config(config_path)  # Returns dict
def save_config(config, save_path)
def merge_configs(base_config, override_config)
```

### `src/utils/seed.py`
```python
def set_seed(seed=42)
def get_seed_from_config(config)
```

### `src/utils/device.py`
```python
def get_device(force_cpu=False)  # Returns torch.device
def move_to_device(obj, device)
```

### `src/data/mnist.py`
```python
def get_mnist_loaders(config)  # Returns (train_loader, val_loader, test_loader)
```

### `src/legacy/fc_scratch.py` (Reference Only)
```python
class LinearLayer:
    def __init__(self, input_size, output_size)
    def forward(self, x)
    def backward(self, grad_output)
    def update(self, learning_rate)

class ReLU:
    def __init__(self)
    def forward(self, x)
    def backward(self, grad_output)

class Softmax:
    def __init__(self)
    def forward(self, x)

class CrossEntropyLoss:
    def __init__(self)
    def forward(self, predictions, targets)
    def backward(self)

class FCNetworkScratch:
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10)
    def forward(self, x)
    def backward(self, grad)
    def update_weights(self, learning_rate)
    def train_step(self, x, y, learning_rate=0.01)
    def predict(self, x)
```

### `scripts/train.py`
```python
def create_run_dir(config)  # Returns (run_dir, run_name)
def main(config_path)
```

### `scripts/eval.py`
```python
def find_checkpoint_and_config(run_name)  # Returns (config_file, checkpoint_file)
def evaluate_model(run_name)
```

---

---

## 🚀 Quick Commands

### Training
```bash
# Train with a config file
python scripts/train.py --config configs/baseline.yaml

# Train CNN (once implemented)
python scripts/train.py --config configs/cnn.yaml

# Train with ablation config
python scripts/train.py --config configs/ablations/no_dropout.yaml
```

### Evaluation
```bash
# Evaluate on test set
python scripts/eval.py --run baseline_fc

# Evaluate specific run by timestamp
python scripts/eval.py --run baseline_fc_20260122_014543
```

### Monitoring
```bash
# Start TensorBoard
tensorboard --logdir=runs

# Open browser to: http://localhost:6006

# View specific experiment
tensorboard --logdir=runs/baseline_fc_20260122_014543/logs/tensorboard
```

### Environment
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Update requirements after adding packages
pip freeze > requirements.txt
```

### Git Workflow
```bash
# Standard workflow
git add .
git commit -m "Descriptive message"
git push

# Create milestone tag
git tag -a v0.2-cnn -m "CNN implementation complete"
git push origin v0.2-cnn

# View commit history
git log --oneline --graph

# Check what changed
git diff
```

---

## 📝 Config File Templates

### Baseline Config Template
```yaml
# configs/baseline.yaml
name: baseline_fc
seed: 42

model:
  type: fc  # Options: fc, cnn
  input_size: 784
  hidden_sizes: [256, 128]
  num_classes: 10
  dropout: 0.2

data:
  dataset: mnist
  data_dir: ./data/raw
  train_split: 0.8  # 80% train, 20% val from training set
  batch_size: 64
  num_workers: 2

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: adam  # Options: adam, sgd
  weight_decay: 0.0001
  early_stopping:
    patience: 5
    min_delta: 0.001

logging:
  log_interval: 10
  save_checkpoints: true
  checkpoint_dir: ./runs
```

### Ablation Config Example
```yaml
# configs/ablations/no_dropout.yaml
# Inherits from baseline, just change what's different
name: ablation_no_dropout
seed: 42

model:
  type: fc
  input_size: 784
  hidden_sizes: [256, 128]
  num_classes: 10
  dropout: 0.0  # ← Only difference from baseline

data:
  dataset: mnist
  data_dir: ./data/raw
  train_split: 0.8
  batch_size: 64
  num_workers: 2

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0001
  early_stopping:
    patience: 5
    min_delta: 0.001

logging:
  log_interval: 10
  save_checkpoints: true
  checkpoint_dir: ./runs
```

---

### Load Saved Model for Inference

```python
import torch
from src.utils.config import load_config
from src.models.fc import FullyConnectedNet

config = load_config('runs/baseline_fc_20260122_014543/config.yaml')
model = FullyConnectedNet.from_config(config)

checkpoint = torch.load('runs/baseline_fc/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode
```

### Access Training Metrics

### Access Training Metrics

```python
import json

with open('runs/baseline_fc_20260122_014543/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Best val acc: {max(metrics['val_acc']):.2f}%")
print(f"Training history: {len(metrics['train_loss'])} epochs")
```

---

**Note:** For full code examples (confusion matrix, worst predictions, custom transforms, etc.), see the original detailed patterns below. The class reference above + these minimal patterns should be sufficient for handoffs. Full implementations are preserved for reference.

---

## 📊 Original Detailed Code Patterns

<details>
<summary>Click to expand full implementation examples</summary>

### Manual Training Loop

### Manual Training Loop

```python
import torch
from torch.utils.data import DataLoader
from src.data.mnist import get_mnist_loaders
from src.models.fc import FullyConnectedNet
from src.utils.config import load_config
from src.utils.device import get_device

# Setup
config = load_config('configs/baseline.yaml')
device = get_device()
train_loader, val_loader, test_loader = get_mnist_loaders(config)

# Create model
model = FullyConnectedNet.from_config(config).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/10 complete")
```

### Custom Data Transform

```python
### Custom Data Transform

```python
# src/data/transforms.py
from torchvision import transforms

def get_train_transforms():
    """Training transforms with augmentation"""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_test_transforms():
    """Test transforms (no augmentation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
```

### Analyze Training Results

```python
### Analyze Training Results

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path('runs/baseline_fc_20260122_014543')
with open(run_dir / 'metrics.json', 'r') as f:
    metrics = json.load(f)

epochs = range(1, len(metrics['train_loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs, metrics['train_loss'], label='Train')
ax1.plot(epochs, metrics['val_loss'], label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, metrics['train_acc'], label='Train')
ax2.plot(epochs, metrics['val_acc'], label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Sanity Check: Overfit Test

```python
### Sanity Check: Overfit Test

```python
# tests/test_overfit.py - Model should overfit 128 samples quickly
import torch
from torch.utils.data import DataLoader, Subset
from src.data.mnist import get_mnist_loaders
from src.models.fc import FullyConnectedNet
from src.utils.config import load_config
from src.utils.seed import set_seed

set_seed(42)
config = load_config('configs/baseline.yaml')
train_loader, _, _ = get_mnist_loaders(config)

small_dataset = Subset(train_loader.dataset, range(128))
small_loader = DataLoader(small_dataset, batch_size=32)

model = FullyConnectedNet.from_config(config)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    correct = 0
    total = 0

    for images, labels in small_loader:
        images = images.view(-1, 784)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if accuracy > 95:
        print(f"✓ PASS: Reached {accuracy:.2f}% in {epoch+1} epochs")
        break
```

### Confusion Matrix

```python
### Confusion Matrix

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
```

### Find Worst Predictions

```python
### Find Worst Predictions

```python
def find_worst_predictions(model, test_loader, device, n=25):
    """Find top N most confident wrong predictions"""
    model.eval()
    mistakes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(-1, 784).to(device)
            outputs = model(images_flat)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)

            wrong_mask = predicted != labels.to(device)
            if wrong_mask.any():
                wrong_indices = torch.where(wrong_mask)[0]
                for idx in wrong_indices:
                    mistakes.append({
                        'image': images[idx].cpu(),
                        'true_label': labels[idx].item(),
                        'pred_label': predicted[idx].item(),
                        'confidence': confidences[idx].item()
                    })

    mistakes.sort(key=lambda x: x['confidence'], reverse=True)

    # Visualize
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < min(n, len(mistakes)):
            m = mistakes[i]
            ax.imshow(m['image'].squeeze(), cmap='gray')
            ax.set_title(f"T:{m['true_label']} P:{m['pred_label']}\n{m['confidence']:.2%}",
                        color='red', fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

</details>

---

## 🔧 Debugging Tips

### Check Training Progress
```python
# View metrics during training
import json
with open('runs/experiment_name/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Best val acc: {max(metrics['val_acc']):.2f}%")
print(f"Final train acc: {metrics['train_acc'][-1]:.2f}%")
```

### Inspect Checkpoint
```python
checkpoint = torch.load('runs/baseline_fc/checkpoints/best.pth')
print(f"Saved at epoch: {checkpoint['epoch']}")
print(f"Metrics: {checkpoint['metrics']}")
print(f"Keys: {checkpoint.keys()}")
```

### Check Device Assignment
```python
print(f"Model device: {next(model.parameters()).device}")
print(f"Data device: {images.device}")
# Should match! If not, use .to(device)
```

### Monitor GPU Memory (if using CUDA)
```python
if torch.cuda.is_available():
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 📊 Quick Analysis Commands

```bash
# Count total parameters in saved model
python -c "import torch; m = torch.load('runs/baseline_fc/checkpoints/best.pth'); print(sum(p.numel() for p in m['model_state_dict'].values()))"

# View config without loading full project
python -c "import yaml; print(yaml.safe_load(open('configs/baseline.yaml')))"

# Find best checkpoint across all runs
find runs -name "best.pth" -exec ls -lh {} \;

# Count total training runs
ls -d runs/*/ | wc -l
```

---

**Last Updated:** 2026-01-22
**Project:** MNIST Classifier Production Pipeline