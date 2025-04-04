# %%
# 3. implement training metrics

import torch

# metrics class
class Metrics:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    # mean squared error loss
    def mse_loss(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

    # accuracy
    def accuracy(self, predictions, targets):
        correct = (torch.abs(predictions - targets) < self.tolerance).float()
        return correct.mean()

    # log metrics (epoch, loss, accuracy)
    def log(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        if train_acc is not None and val_acc is not None:
            msg += f" | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        print(msg)

# %%