class EarlyStopping:
    """Stop training if val_acc does not improve for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = 0.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
