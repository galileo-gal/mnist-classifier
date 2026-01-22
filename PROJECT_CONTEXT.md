# MNIST Classifier - Production ML Pipeline

## 🎯 Current State
- **Status:** Phase 2 complete - Baseline FC model working
- **Test Accuracy:** 97.82% (validation: 97.75%)
- **Branch:** main
- **Repo:** https://github.com/galileo-gal/mnist-classifier
- **Last Major Milestone:** Production training pipeline with early stopping

## ✅ What's Working

### Training Pipeline
- Full production trainer with TensorBoard logging
- Early stopping (patience=5, triggered at epoch 15)
- Checkpoint management (best.pth + last.pth)
- Proper train/val/test splits (48k/12k/10k)
- YAML config system for experiments
- Reproducible seeding

### Models Implemented
1. **FC from scratch** (legacy) - 91.66% on Colab (educational reference)
2. **FC PyTorch** (legacy) - 98.16% local testing
3. **Production FC** (current) - 97.82% test accuracy
   - Architecture: 784 → 256 → 128 → 10
   - Dropout: 0.2
   - Optimizer: Adam (lr=0.001)
   - Parameters: 235,146

### Infrastructure
- Config system: YAML-based experiment configs
- Data module: Automatic train/val/test splits with proper seeding
- Logging: TensorBoard + JSON metrics
- Checkpointing: Saves best and last models
- Early stopping: Prevents overfitting
- Device management: CPU/GPU handling
- Seed utilities: Full reproducibility

## 📁 Project Structure

```
mnist_classifier/
├── configs/
│   ├── baseline.yaml          # ✅ Working baseline config
│   ├── cnn.yaml              # 📝 TODO: CNN config
│   └── ablations/            # 📝 TODO: Ablation studies
│       ├── init_xavier.yaml
│       ├── init_kaiming.yaml
│       ├── no_dropout.yaml
│       ├── lr_schedule.yaml
│       └── batch_norm.yaml
├── src/
│   ├── data/
│   │   ├── mnist.py          # ✅ Data loading with splits
│   │   └── transforms.py     # 📝 TODO: Augmentation pipeline
│   ├── models/
│   │   ├── fc.py             # ✅ Production FC model
│   │   └── cnn.py            # 📝 TODO: CNN implementation
│   ├── training/
│   │   ├── trainer.py        # ✅ Main training loop
│   │   ├── metrics.py        # ⚠️ Exists but minimal
│   │   ├── checkpointing.py  # ✅ Save/load logic
│   │   └── early_stopping.py # ✅ Early stopping class
│   ├── utils/
│   │   ├── config.py         # ✅ YAML config loading
│   │   ├── seed.py           # ✅ Reproducibility
│   │   ├── device.py         # ✅ GPU/CPU handling
│   │   └── logging.py        # ✅ TensorBoard + JSON
│   └── legacy/               # ✅ Reference implementations
│       ├── fc_pytorch.py     # Old PyTorch FC
│       └── fc_scratch.py     # From-scratch learning code
├── scripts/
│   ├── train.py              # ✅ Main training script
│   ├── eval.py               # ✅ Evaluation script
│   ├── visualize_filters.py  # 📝 TODO: Filter visualization
│   ├── visualize_failures.py # 📝 TODO: Failure analysis
│   └── run_ablations.py      # 📝 TODO: Parallel ablations
├── tests/
│   ├── test_overfit.py       # 📝 TODO: Overfit 128 samples
│   ├── test_random_labels.py # 📝 TODO: Random labels test
│   └── test_single_batch.py  # 📝 TODO: Single batch test
├── notebooks/
│   ├── legacy/               # ✅ Old exploration notebooks
│   ├── 01_explore_data.ipynb # ✅ Data exploration
│   └── 04_cnn_experiments.ipynb # 📝 TODO: CNN notebooks
├── runs/                     # Generated during training
│   └── {experiment_name}_TIMESTAMP/
│       ├── config.yaml       # Copy of config used
│       ├── checkpoints/
│       │   ├── best.pth
│       │   └── last.pth
│       ├── logs/tensorboard/
│       └── metrics.json
└── data/raw/                 # MNIST dataset (gitignored)
```

## 🚀 How to Use

### Train a Model
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Train with config
python scripts/train.py --config configs/baseline.yaml

# Output: runs/baseline_fc_TIMESTAMP/
```

### Evaluate on Test Set
```bash
python scripts/eval.py --run baseline_fc
```

### View Training Progress
```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```

### Create New Experiment
1. Copy `configs/baseline.yaml` to `configs/my_experiment.yaml`
2. Modify hyperparameters
3. Run: `python scripts/train.py --config configs/my_experiment.yaml`

## 📋 Next Tasks (Priority Order)

### Placeholder Files (Created Structure, Implementation Pending)
The following files exist in the repository but are empty/minimal and marked for next phase:
- `src/models/cnn.py` - CNN architecture
- `src/data/transforms.py` - Data augmentation pipeline  
- `src/training/metrics.py` - Extended metrics (confusion matrix, per-class accuracy)
- `scripts/visualize_filters.py` - Conv filter visualization
- `scripts/visualize_failures.py` - Failure case analysis
- `scripts/run_ablations.py` - Parallel ablation experiments
- `tests/test_overfit.py` - Overfit sanity check
- `tests/test_random_labels.py` - Random labels sanity check
- `tests/test_single_batch.py` - Single batch sanity check
- `configs/ablations/*.yaml` - Ablation configuration files

These will be implemented in Phase 3-6 of the roadmap.

### Phase 3: Sanity Checks (CRITICAL - Do First)
**Why:** Catches 80% of bugs before wasting time on experiments

1. **test_overfit.py** - Model should hit ~100% on 128 samples in <50 epochs
2. **test_random_labels.py** - Model should NOT learn (stay at ~10% accuracy)
3. **test_single_batch.py** - Loss should go to near-zero on single batch

**Implementation:** Use existing trainer, just modify data loading

### Phase 4: CNN Implementation (Target: 99%+ accuracy)
1. Create `src/models/cnn.py` with proper architecture
2. Create `configs/cnn.yaml` with CNN hyperparameters
3. Test on full MNIST dataset (should beat 97.82% baseline)
4. Visualize learned filters (`scripts/visualize_filters.py`)

**Architecture suggestion:**
```
Conv(32, 3x3) → ReLU → BatchNorm → MaxPool(2x2)
Conv(64, 3x3) → ReLU → BatchNorm → MaxPool(2x2)
Flatten → FC(128) → Dropout(0.5) → FC(10)
```

### Phase 5: Ablation Studies (Systematic Comparison)
**Goal:** Understand what matters for performance

Run each with 3 seeds (42, 123, 456):
1. Baseline (reproduction check)
2. Xavier vs Kaiming initialization
3. With vs without dropout
4. Learning rate schedules (step decay, cosine)
5. With vs without batch normalization

**Output:** Table with mean ± std for each configuration

### Phase 6: Interpretability
1. **Confusion matrix** - Which digits get confused?
2. **Top-25 failures** - Highest confidence mistakes
3. **Failure clustering** - Group similar errors
4. **Activation maps** - What does the network "see"?

## ⚙️ Configuration System

### Config File Structure
```yaml
name: experiment_name
seed: 42

model:
  type: fc  # or cnn
  input_size: 784
  hidden_sizes: [256, 128]
  num_classes: 10
  dropout: 0.2

data:
  dataset: mnist
  data_dir: ./data/raw
  train_split: 0.8  # 80% train, 20% val
  batch_size: 64
  num_workers: 2

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: adam  # or sgd
  weight_decay: 0.0001
  early_stopping:
    patience: 5
    min_delta: 0.001

logging:
  log_interval: 10
  save_checkpoints: true
  checkpoint_dir: ./runs
```

## 🐛 Known Issues

### Issue 1: Nested Checkpoint Directories
- **Problem:** CheckpointManager creates `runs/baseline_fc/checkpoints/` separate from `runs/baseline_fc_TIMESTAMP/`
- **Impact:** Config and checkpoints in different directories
- **Workaround:** eval.py handles this by searching both locations
- **TODO:** Refactor CheckpointManager to use single run_dir

### Issue 2: pin_memory Warning
- **Message:** `'pin_memory' argument is set as true but no accelerator is found`
- **Impact:** None (harmless warning)
- **Fix:** Set `pin_memory=False` in data loader when device is CPU

## 💻 Development Environment

### Local (PyCharm)
- **Purpose:** Code development, quick testing
- **Hardware:** Dell Inspiron 15 3511, i5 11th gen, 16GB RAM
- **Limitation:** CPU only, use small subsets for testing
- **Workflow:** Develop → commit → train on Colab

### Google Colab (Training)
- **Purpose:** Full training runs with GPU
- **Access:** https://colab.research.google.com/
- **Workflow:** Clone repo → run training → save to Google Drive
- **Note:** For from-scratch model, achieved 91.66% (baseline for comparison)

### GitHub Codespaces (Mobile)
- **Purpose:** Quick edits, code review on phone
- **Limitation:** Can't run notebooks interactively
- **Use case:** Documentation, minor bug fixes

## 🔑 Key Design Decisions

### 1. Train/Val/Test Split Strategy
- **Why 3 splits:** Prevents "tuning on test set" antipattern
- **Ratio:** 48k train / 12k val / 10k test
- **Val purpose:** Early stopping, checkpoint selection
- **Test purpose:** Final reporting only (never seen during training)

### 2. One Run Directory Per Experiment
- **Format:** `runs/{name}_{timestamp}/`
- **Contents:** config.yaml, checkpoints/, logs/, metrics.json
- **Why timestamp:** Multiple runs of same config don't overwrite
- **Why save config:** Exact reproduction of any experiment

### 3. Early Stopping on Validation Loss
- **Why loss not accuracy:** More stable signal
- **Patience=5:** Allows temporary fluctuations
- **Result:** Saved 5 epochs in baseline (stopped at 15/20)

### 4. Separate Legacy Code
- **Purpose:** Keep learning path visible
- **Location:** `src/legacy/`, `notebooks/legacy/`
- **Benefit:** Reference for correctness, regression testing
- **Maturity signal:** Shows progression to reviewers

## 📊 Baseline Results

| Metric | Value |
|--------|-------|
| Architecture | FC: 784→256→128→10 |
| Parameters | 235,146 |
| Train Accuracy | 98.55% (epoch 14) |
| Val Accuracy | 97.75% (epoch 10, best) |
| **Test Accuracy** | **97.82%** |
| Training Time | ~5 min (15 epochs, CPU) |
| Early Stop | Epoch 15 (patience=5) |
| Optimizer | Adam (lr=0.001) |

### Comparison to Earlier Versions
- From scratch (Colab): 91.66%
- PyTorch FC (local, subset): 98.16%
- **Production FC (current): 97.82%** ✅

## 🎓 Learning Objectives Completed

✅ **Phase 1: Data Exploration**
- Understanding MNIST format (28×28 grayscale)
- Class distribution analysis
- Normalization importance

✅ **Phase 2: Neural Networks from Scratch**
- Forward pass (matrix multiplications)
- Backpropagation (gradient computation)
- Weight updates (gradient descent)
- Loss functions (cross-entropy)

✅ **Phase 3: PyTorch Basics**
- nn.Module, nn.Linear, optimizers
- DataLoader, transforms
- Automatic differentiation
- GPU/CPU device management

✅ **Phase 4: Production ML Engineering**
- Config-driven experiments
- Proper data splits
- Logging and monitoring
- Checkpointing strategies
- Early stopping
- Reproducibility (seeding)

## 🔄 Workflow for Future Development

### Adding a New Model Type
1. Create `src/models/your_model.py` with `from_config()` classmethod
2. Create `configs/your_model.yaml`
3. Update `scripts/train.py` model loading logic:
   ```python
   elif config['model']['type'] == 'your_model':
       model = YourModel.from_config(config)
   ```
4. Train and evaluate as usual

### Running Experiments Systematically
1. Define hypothesis (e.g., "Does dropout improve generalization?")
2. Create two configs: `configs/ablations/with_dropout.yaml`, `configs/ablations/no_dropout.yaml`
3. Run both with multiple seeds
4. Compare metrics.json results
5. Document findings in README or notebook

### Debugging Training Issues
1. Check `runs/{experiment}/config.yaml` - was config correct?
2. View TensorBoard - loss curves, overfitting?
3. Load `runs/{experiment}/metrics.json` - exact numbers
4. Inspect `runs/{experiment}/checkpoints/best.pth` - which epoch was best?

## 📚 References & Resources

### Project Documentation
- Original plan: See initial conversation for full project roadmap
- Baseline config: `configs/baseline.yaml`
- Training examples: `notebooks/legacy/` for learning progression

### External Resources
- PyTorch docs: https://pytorch.org/docs/
- MNIST dataset: http://yann.lecun.com/exdb/mnist/
- TensorBoard guide: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

## 🤝 Handoff Instructions

When switching to a new Claude instance:

1. **Share this file** (PROJECT_CONTEXT.md)
2. **Provide repo link:** https://github.com/galileo-gal/mnist-classifier
3. **State current commit/tag:** [Run: `git log -1 --oneline`]
4. **Specify immediate task:** e.g., "Help me implement test_overfit.py"

### Example Handoff Message
```
Continuing MNIST classifier project.

Repo: https://github.com/galileo-gal/mnist-classifier
Status: Baseline FC complete (97.82% test acc)
Current phase: Need to implement sanity checks

Key context in PROJECT_CONTEXT.md (attached).
Immediate task: Help me write tests/test_overfit.py to verify model can overfit 128 samples.

Environment: PyCharm local (CPU), will train full experiments on Colab.
```

---

**Last Updated:** 2026-01-22
**Project Phase:** 2/6 (Baseline Complete)
**Next Milestone:** Sanity checks + CNN implementation
