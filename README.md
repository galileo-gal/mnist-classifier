# MNIST Handwritten Digit Classifier


A production-grade machine learning pipeline for MNIST digit classification, built to demonstrate proper ML engineering practices from data loading to model evaluation.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Goals

This project is designed to learn and implement:
- **Neural networks from first principles** (backpropagation, gradient descent)
- **PyTorch fundamentals** (nn.Module, DataLoader, optimizers)
- **Production ML pipeline** (config-driven experiments, logging, checkpointing)
- **Proper evaluation methodology** (train/val/test splits, early stopping)
- **Model interpretability** (confusion matrices, failure analysis, filter visualization)

**Current Status:** Phase 2 Complete - Baseline FC Model (97.82% test accuracy)

---

## 📊 Results

| Model | Architecture | Test Accuracy | Parameters | Training Time |
|-------|-------------|---------------|------------|---------------|
| **FC (Production)** | 784→256→128→10 | **97.82%** | 235,146 | ~5 min (CPU) |
| FC (PyTorch - Legacy) | 784→256→128→10 | 98.16% | 235,146 | ~3 min (CPU, subset) |
| FC (From Scratch - Legacy) | 784→128→64→10 | 91.66% | - | ~10 min (Colab GPU) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/galileo-gal/mnist-classifier.git
cd mnist-classifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Train a Model

```bash
# Train with baseline config
python scripts/train.py --config configs/baseline.yaml

# Output: runs/baseline_fc_TIMESTAMP/
#   ├── config.yaml          # Experiment configuration
#   ├── checkpoints/         # best.pth, last.pth
#   ├── logs/tensorboard/    # Training metrics
#   └── metrics.json         # Final results
```

### Evaluate on Test Set

```bash
# Evaluate best model
python scripts/eval.py --run baseline_fc

# Output: Test Accuracy: 97.82%
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=runs

# Open browser: http://localhost:6006
```

---

## 📁 Project Structure

```
mnist_classifier/
├── configs/                    # Experiment configurations
│   ├── baseline.yaml          # ✅ Baseline FC config
│   ├── cnn.yaml              # 📝 TODO: CNN config
│   └── ablations/            # 📝 TODO: Ablation studies
├── src/
│   ├── data/
│   │   └── mnist.py          # ✅ Data loading with train/val/test splits
│   ├── models/
│   │   ├── fc.py             # ✅ Production fully connected model
│   │   └── cnn.py            # 📝 TODO: CNN implementation 
│   ├── training/
│   │   ├── trainer.py        # ✅ Main training loop
│   │   ├── checkpointing.py  # ✅ Model saving/loading
│   │   ├── early_stopping.py # ✅ Early stopping logic
│   │   └── metrics.py        # ⚠️ Basic metrics (needs expansion)
│   ├── utils/
│   │   ├── config.py         # ✅ YAML config management
│   │   ├── logging.py        # ✅ TensorBoard + JSON logging
│   │   ├── seed.py           # ✅ Reproducibility utilities
│   │   └── device.py         # ✅ GPU/CPU handling
│   └── legacy/               # ✅ Learning reference implementations
│       ├── fc_scratch.py     # From-scratch neural network
│       └── fc_pytorch.py     # Basic PyTorch implementation
├── scripts/
│   ├── train.py              # ✅ Main training script
│   ├── eval.py               # ✅ Evaluation script
│   ├── visualize_filters.py  # 📝 TODO: Filter visualization
│   ├── visualize_failures.py # 📝 TODO: Failure analysis
│   └── run_ablations.py      # 📝 TODO: Parallel ablations
├── tests/                    # 📝 TODO: Sanity checks
│   ├── test_overfit.py
│   ├── test_random_labels.py
│   └── test_single_batch.py
├── notebooks/
│   ├── 01_explore_data.ipynb # ✅ Data exploration
│   └── legacy/               # ✅ Learning notebooks
├── runs/                     # Generated experiment artifacts
├── data/raw/                 # MNIST dataset (auto-downloaded)
├── PROJECT_CONTEXT.md        # ✅ Detailed project state
├── KEY_CODE.md              # ✅ Code patterns reference
└── requirements.txt         # ✅ Python dependencies
```

**Legend:** ✅ Complete | ⚠️ Partial | 📝 TODO

---

## 🔧 Configuration System

All experiments are defined via YAML configs in `configs/`. Example:

```yaml
name: baseline_fc
seed: 42

model:
  type: fc
  input_size: 784
  hidden_sizes: [256, 128]
  num_classes: 10
  dropout: 0.2

data:
  dataset: mnist
  train_split: 0.8  # 80% train, 20% val
  batch_size: 64

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: adam
  early_stopping:
    patience: 5
    min_delta: 0.001
```

**Benefits:**
- Reproducible experiments
- Version-controlled hyperparameters
- Easy comparison across runs
- Config saved with each experiment

---

## 🧪 Key Features

### 1. Proper Data Splits
- **Train (80%):** Model training
- **Validation (20%):** Early stopping, checkpoint selection
- **Test (held-out):** Final evaluation only

Prevents the common antipattern of "tuning on test set."

### 2. Production Training Pipeline
- **TensorBoard logging:** Real-time training curves
- **Checkpointing:** Saves best and last models
- **Early stopping:** Prevents overfitting, saves compute
- **Reproducible:** Seed control for deterministic results

### 3. Experiment Tracking
Each training run creates a timestamped directory:
```
runs/baseline_fc_20260122_014543/
├── config.yaml          # Exact config used
├── checkpoints/
│   ├── best.pth        # Best validation loss
│   └── last.pth        # Final epoch
├── logs/tensorboard/   # Training metrics
└── metrics.json        # Summary statistics
```

### 4. From-Scratch Learning Path
`src/legacy/` contains educational implementations:
- **fc_scratch.py:** Manual backpropagation (91.66% accuracy)
- **fc_pytorch.py:** Basic PyTorch (98.16% accuracy)

These serve as correctness references and demonstrate progression to production code.

---

## 📚 Documentation

- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** - Complete project state, design decisions, next tasks
- **[KEY_CODE.md](KEY_CODE.md)** - API reference, code patterns, quick commands
- **[requirements.txt](requirements.txt)** - Python dependencies

---

## 🎓 Learning Outcomes

### Phase 1: Foundations ✅
- [x] Data exploration and visualization
- [x] Understanding MNIST format and normalization
- [x] Class distribution analysis

### Phase 2: Neural Networks ✅
- [x] Forward pass (matrix multiplications)
- [x] Backpropagation (gradient computation)
- [x] Loss functions (cross-entropy)
- [x] Weight initialization (Xavier/Kaiming)
- [x] Activation functions (ReLU, Softmax)

### Phase 3: PyTorch Fundamentals ✅
- [x] nn.Module architecture
- [x] Automatic differentiation
- [x] DataLoader and transforms
- [x] Optimizers (Adam, SGD)
- [x] GPU/CPU device management

### Phase 4: Production ML Engineering ✅
- [x] Config-driven experiments
- [x] Proper train/val/test splits
- [x] TensorBoard integration
- [x] Model checkpointing
- [x] Early stopping
- [x] Reproducibility (seeding)

### Phase 5: Advanced Topics (In Progress)
- [ ] Sanity checks (overfit test, random labels)
- [ ] CNN implementation
- [ ] Data augmentation
- [ ] Systematic ablation studies
- [ ] Model interpretability (confusion matrix, failure analysis)
- [ ] Filter visualization

---

## 🔜 Roadmap

### Note on Repository Structure
This repository includes placeholder files (empty or minimal implementations) for features planned in upcoming phases. The directory structure is complete to maintain clean organization as features are added.

**Currently Implemented:**
- Full training pipeline (config → train → checkpoint → eval)
- Baseline FC model with 97.82% test accuracy
- All infrastructure utilities (logging, seeding, device management)

**Next to Implement (files exist as placeholders):**
- CNN model and training
- Interpretability scripts
- Sanity check tests
- Ablation experiments

### Immediate Next Steps
1. **Sanity Checks** - Validate training pipeline
   - Overfit 128 samples test
   - Random labels test
   - Single batch training test

2. **CNN Implementation** - Target 99%+ accuracy
   - Conv2d layers with pooling
   - Batch normalization
   - Filter visualization

3. **Ablation Studies** - Understand what matters
   - Initialization schemes (Xavier vs Kaiming)
   - Dropout impact
   - Learning rate schedules
   - Batch normalization effect

4. **Interpretability** - Debug model decisions
   - Confusion matrix analysis
   - Top-25 confident mistakes
   - Failure case clustering
   - Activation map visualization

### Future Enhancements
- Transfer learning to CIFAR-10
- ResNet architecture
- Distributed training
- Model quantization
- ONNX export

---

## 🤝 Contributing

This is a learning project. Contributions that improve:
- Code clarity and documentation
- Educational value
- Production best practices
- Test coverage

...are welcome!

---

## 📖 Resources

- **PyTorch Documentation:** https://pytorch.org/docs/
- **MNIST Dataset:** http://yann.lecun.com/exdb/mnist/
- **TensorBoard Guide:** https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Yann LeCun et al. for the MNIST dataset
- PyTorch team for the deep learning framework
- Anthropic Claude for development assistance

---

## 📧 Contact

**Author:** Abdullah Galib  
**GitHub:** [@galileo-gal](https://github.com/galileo-gal)  
**Repo:** [mnist-classifier](https://github.com/galileo-gal/mnist-classifier)

---

**Last Updated:** January 22, 2026  
**Version:** 0.1 (Baseline Complete)
