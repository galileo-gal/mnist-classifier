"""Early stopping to prevent overfitting"""


class EarlyStopping:
    """
    Stop training when validation metric stops improving
    """

    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        """
        Check if training should stop

        Args:
            val_metric: Current validation metric value

        Returns:
            True if training should stop, False otherwise
        """
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
