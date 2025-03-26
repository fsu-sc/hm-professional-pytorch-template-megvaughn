# %%
# 5. tensorboard analysis

import sys
import os
import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.dynamic_model import DenseModel
from data_loader.function_dataset import SimpleFunctionsDataset
from model.metric import Metrics

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load configuration from JSON
config_file = os.path.join(os.path.dirname(__file__), 'configs', 'config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

print("Loaded config keys:", list(config.keys()))  # Optional: inspect config

# Initialize TensorBoard writer
log_dir = '/Users/meganvaughn/Desktop/health/Homework8/ml_torch_templates/runs/experiment_1'
writer = SummaryWriter(log_dir)

# Initialize model
model = DenseModel(
    input_dim=1,
    hidden_layers=config.get("hidden_layers", 2),
    neurons_per_layer=config.get("neurons_per_layer", 64),
    hidden_activation=config.get("activation_hidden", "relu"),
    output_activation=config.get("activation_output", "linear")
)

# Datasets and loaders
train_dataset = SimpleFunctionsDataset(n_samples=1000, function="linear")
val_dataset = SimpleFunctionsDataset(n_samples=200, function="linear")

batch_size = config.get("batch_size", 32)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Log model graph
dummy_input = torch.ones(1, 1)
writer.add_graph(model, dummy_input)

# Hyperparameters
epochs = config.get("num_epochs", 100)
learning_rate = config.get("learning_rate", 0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
metrics = Metrics()

# Training loop
for epoch in range(epochs):
    start_time = time.time()

    # Training phase
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_accuracy += metrics.accuracy(predictions, y).item()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            predictions = model(x)
            y = y.view(-1, 1)  # ensure shape consistency
            loss = loss_fn(predictions, y)
            val_loss += loss.item()
            val_accuracy += metrics.accuracy(predictions, y).item()

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    writer.add_scalar('Time/epoch', time.time() - start_time, epoch)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# Close TensorBoard writer
writer.close()
# %%