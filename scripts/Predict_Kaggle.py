"""Generate predictions for Kaggle submission"""
import sys
from pathlib import Path

# FIRST: Add project root to path (BEFORE any src imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NOW import from src
import torch
import pandas as pd
from src.utils.config import load_config
from src.models.fc import FullyConnectedNet

def predict_kaggle(model_path, test_csv, output_csv):
    """Generate Kaggle submission file"""
    print("Loading model...")

    # Build absolute paths
    config_path = project_root / 'configs' / 'baseline.yaml'
    model_full_path = project_root / model_path
    test_full_path = project_root / test_csv
    output_full_path = project_root / output_csv

    config = load_config(str(config_path))
    model = FullyConnectedNet.from_config(config)

    checkpoint = torch.load(str(model_full_path), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")

    print("Loading test data...")
    test_df = pd.read_csv(str(test_full_path))
    test_images = test_df.values / 255.0
    test_tensor = torch.FloatTensor(test_images)
    print(f"✓ Loaded {len(test_tensor)} test images")

    print("Generating predictions...")
    predictions = []
    batch_size = 1000

    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())

            if (i // batch_size) % 5 == 0:
                print(f"  Processed {i}/{len(test_tensor)} images")

    print("Creating submission file...")
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })

    submission.to_csv(str(output_full_path), index=False)
    print(f"✓ Submission saved to {output_full_path}")
    print(f"  Total predictions: {len(predictions)}")

if __name__ == '__main__':
    predict_kaggle(
        model_path='runs/baseline_fc/checkpoints/best.pth',
        test_csv='data/kaggle/test.csv',
        output_csv='submission.csv'
    )