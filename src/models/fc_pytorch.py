"""
Fully connected neural network using PyTorch.
Compare this to fc_scratch.py to see what PyTorch automates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetworkPyTorch(nn.Module):
    """
    Fully connected network using PyTorch

    Architecture: input(784) -> FC(128) -> ReLU -> FC(64) -> ReLU -> FC(10)
    """

    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout=0.2):
        super(FCNetworkPyTorch, self).__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer (no activation - CrossEntropyLoss includes softmax)
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
