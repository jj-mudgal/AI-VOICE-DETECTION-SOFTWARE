import csv
import os


class TrainingLogger:
    """Logs epoch metrics to console and a CSV file."""

    def __init__(self, log_path: str = "logs/training_log.csv"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self._write_header()

    def _write_header(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.3f}"
        )
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
