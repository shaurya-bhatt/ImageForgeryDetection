import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt

from src.preprocessing import load_dataset
from src.models.vgg16_model import VGG16Model
images, labels = load_dataset()

# Split the dataset (we only need the test set for evaluation)
_, X_test, _, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Convert the test images to PyTorch tensors
X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).permute(0, 3, 1, 2)

# Convert the test labels to PyTorch tensors
y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

# Create the test dataset
test_dataset = TensorDataset(
    X_test,
    y_test
)

# Create the test dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Select the device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

all_labels = []
all_predictions = []
all_probabilities = []

# Create the VGG16 model
model = VGG16Model()

# Load the trained model weights
model.load_state_dict(
    torch.load(
        "outputs/saved_models/vgg16_model.pth",
        map_location=device,
        weights_only=True
    )
)

# Move the model to the selected device
model = model.to(device)

# Switch the model to evaluation mode
model.eval()

with torch.no_grad():

    for batch_images, batch_labels in test_loader:

        # Move the batch to the selected device
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # Get the model predictions
        outputs = model(batch_images)

        # Convert outputs to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # Store the true labels
        all_labels.extend(batch_labels.cpu().numpy())

        # Store the predicted labels
        all_predictions.extend(predicted.cpu().numpy())

        # Store the probability of the Forged class (class 1)
        all_probabilities.extend(probabilities[:, 1].cpu().numpy())

# Calculate Accuracy
accuracy = accuracy_score(all_labels, all_predictions)

# Calculate Precision
precision = precision_score(all_labels, all_predictions)

# Calculate Recall
recall = recall_score(all_labels, all_predictions)

# Calculate F1 Score
f1 = f1_score(all_labels, all_predictions)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Calculate Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], ["Authentic","Forged"])
plt.yticks([0,1], ["Authentic","Forged"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.show()

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)

# Calculate AUC
roc_auc = auc(
    fpr,
    tpr
)

print(f"\nAUC Score : {roc_auc:.4f}")

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()