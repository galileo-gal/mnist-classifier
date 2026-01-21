"""Production fully connected network"""
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    """
    Fully connected neural network with configurable architecture

    Can be initialized from config dict for experiments
    """

    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, dropout=0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @classmethod
    def from_config(cls, config):
        """Create model from config dict"""
        model_cfg = config['model']
        return cls(
            input_size=model_cfg.get('input_size', 784),
            hidden_sizes=model_cfg.get('hidden_sizes', [256, 128]),
            num_classes=model_cfg.get('num_classes', 10),
            dropout=model_cfg.get('dropout', 0.2)
        )

