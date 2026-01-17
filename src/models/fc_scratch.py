"""
Fully connected neural network implemented from scratch.
This helps understand backpropagation before using PyTorch's autograd.
"""

import numpy as np


class LinearLayer:
    """
    Fully connected layer: y = Wx + b

    Implements forward and backward pass manually.
    """

    def __init__(self, input_size, output_size):
        # Xavier initialization: helps with gradient flow
        # Scale by sqrt(1/input_size) to prevent exploding/vanishing gradients
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1.0 / input_size)
        self.bias = np.zeros((output_size, 1))

        # Store for backward pass
        self.input = None
        self.dweights = None
        self.dbias = None

    def forward(self, x):
        """
        Forward pass: y = Wx + b

        Args:
            x: Input of shape (input_size, batch_size)

        Returns:
            Output of shape (output_size, batch_size)
        """
        self.input = x  # Store for backward pass
        return np.dot(self.weights, x) + self.bias

    def backward(self, grad_output):
        """
        Backward pass: compute gradients using chain rule

        Args:
            grad_output: Gradient from next layer, shape (output_size, batch_size)

        Returns:
            Gradient w.r.t input, shape (input_size, batch_size)
        """
        batch_size = self.input.shape[1]

        # Gradient w.r.t weights: dL/dW = dL/dy * dy/dW = grad_output * x^T
        self.dweights = np.dot(grad_output, self.input.T) / batch_size

        # Gradient w.r.t bias: dL/db = dL/dy * dy/db = grad_output (sum across batch)
        self.dbias = np.sum(grad_output, axis=1, keepdims=True) / batch_size

        # Gradient w.r.t input: dL/dx = dL/dy * dy/dx = W^T * grad_output
        grad_input = np.dot(self.weights.T, grad_output)

        return grad_input

    def update(self, learning_rate):
        """Update weights using gradient descent"""
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias


class ReLU:
    """
    ReLU activation: f(x) = max(0, x)

    Derivative: f'(x) = 1 if x > 0 else 0
    """

    def __init__(self):
        self.input = None

    def forward(self, x):
        """Apply ReLU activation"""
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Backward pass: gradient flows only where input was positive

        Chain rule: dL/dx = dL/dy * dy/dx
        dy/dx = 1 if x > 0, else 0
        """
        grad_input = grad_output * (self.input > 0)
        return grad_input


class Softmax:
    """
    Softmax activation: converts logits to probabilities

    f(x_i) = exp(x_i) / sum(exp(x_j))
    """

    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Numerically stable softmax

        Subtract max for numerical stability (prevents overflow)
        """
        # Shift values for numerical stability
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return self.output

    def backward(self, grad_output):
        """
        Backward pass for softmax

        This is complex, but when combined with cross-entropy,
        it simplifies to: (predictions - targets)
        """
        # For numerical stability, we'll compute this in the loss function
        # So this method isn't typically called separately
        return grad_output


class CrossEntropyLoss:
    """
    Cross-entropy loss for classification

    L = -sum(y_true * log(y_pred))
    """

    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss

        Args:
            predictions: Softmax outputs, shape (num_classes, batch_size)
            targets: One-hot encoded labels, shape (num_classes, batch_size)

        Returns:
            Scalar loss value
        """
        self.predictions = predictions
        self.targets = targets

        batch_size = predictions.shape[1]

        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions, 1e-10, 1 - 1e-10)

        # Cross-entropy: -sum(target * log(pred))
        loss = -np.sum(targets * np.log(predictions_clipped)) / batch_size

        return loss

    def backward(self):
        """
        Backward pass for cross-entropy + softmax

        Magical simplification: gradient = (predictions - targets)
        This is why softmax + cross-entropy are often combined
        """
        batch_size = self.predictions.shape[1]
        grad = (self.predictions - self.targets) / batch_size
        return grad


class FCNetworkScratch:
    """
    Simple fully connected neural network from scratch

    Architecture: input -> FC(128) -> ReLU -> FC(64) -> ReLU -> FC(10) -> Softmax
    """

    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        self.layers = []

        # Build network architecture
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1]))

            # Add ReLU activation (except for last layer)
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

        # Output activation and loss
        self.softmax = Softmax()
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, learning_rate):
        """Update all layer weights"""
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.update(learning_rate)

    def train_step(self, x, y, learning_rate=0.01):
        """
        Single training step

        Args:
            x: Input data, shape (input_size, batch_size)
            y: One-hot targets, shape (num_classes, batch_size)
            learning_rate: Learning rate for gradient descent

        Returns:
            loss: Scalar loss value
        """
        # Forward pass
        logits = self.forward(x)
        predictions = self.softmax.forward(logits)

        # Compute loss
        loss = self.criterion.forward(predictions, y)

        # Backward pass
        grad = self.criterion.backward()
        self.backward(grad)

        # Update weights
        self.update_weights(learning_rate)

        return loss

    def predict(self, x):
        """
        Make predictions

        Args:
            x: Input data, shape (input_size, batch_size)

        Returns:
            Class predictions, shape (batch_size,)
        """
        logits = self.forward(x)
        predictions = self.softmax.forward(logits)
        return np.argmax(predictions, axis=0)
