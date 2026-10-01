import numpy as np
import torch
import torch.nn as nn

from src.preprocessing import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from src.models.efficientnet_b0_model import EfficientNetB0Model

images, labels = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).permute(0, 3, 1, 2)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).permute(0, 3, 1, 2)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

# Create the model
model = EfficientNetB0Model().to(device)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.00005
)

num_epochs = 10

model.train()

for epoch in range(num_epochs):

    running_loss = 0

    for batch_images, batch_labels in train_loader:

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_images)

        loss = loss_function(outputs, batch_labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}")

# -----------------------
# Testing
# -----------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_images, batch_labels in test_loader:

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_images)

        _, predicted = torch.max(outputs,1)

        total += batch_labels.size(0)

        correct += (predicted==batch_labels).sum().item()

accuracy = correct/total*100

print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(
    model.state_dict(),
    "models/efficientnet_b0_model.pth"
)

print("Model saved successfully!")