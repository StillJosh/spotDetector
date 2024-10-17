# early_stopping.py
# Description: Implements early stopping, which halts the model training if the validation metric does not improve.
# Author: Joshua Stiller
# Date: 16.10.24

from typing import Optional


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.

    Parameters
    ----------
    patience : int
        How long to wait after last time validation metric improved.
    verbose : bool
        If True, prints a message for each validation metric improvement.
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    mode : str
        'min' or 'max' to decide whether to look for decreasing or increasing metric.

    Attributes
    ----------
    counter : int
        Counts how many times validation metric has not improved.
    best_score : Optional[float]
        Best score achieved so far.
    early_stop : bool
        Whether to stop training early.
    """

    def __init__(
            self, patience: int = 7, verbose: bool = False, delta: float = 0.0, mode: str = 'min'
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best_score is None:
            self.best_score = metric
            return

        if self.mode == 'min':
            is_improvement = metric < self.best_score - self.delta
        else:
            is_improvement = metric > self.best_score + self.delta

        if is_improvement:
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'Validation metric improved to {metric:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
