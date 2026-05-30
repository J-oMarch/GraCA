import torch
import copy


class EarlyStopping:
    """Early stopping with patience and best model tracking."""

    def __init__(self, patience: int = 100, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: validation metric (higher is better)
            model: model to save best state
            epoch: current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def load_best_model(self, model: torch.nn.Module):
        """Restore model to best state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
