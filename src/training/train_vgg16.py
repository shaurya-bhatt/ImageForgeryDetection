import numpy as np
import torch
import torch.nn as nn

from src.preprocessing import load_dataset
from src.models.vgg16_model import VGG16Model

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

images, labels = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2, # 20% of the dataset is reserved for testing
    random_state=42, # Ensures shuffling is consistent across every run
    stratify=labels # Ensures that the class distribution is uniform for both training and testing sets
)

print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)

print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

print("\nTraining Set Distribution")
print("Authentic Images:", np.sum(y_train == 0))
print("Forged Images:", np.sum(y_train == 1))

print("\nTesting Set Distribution")
print("Authentic Images:", np.sum(y_test == 0))
print("Forged Images:", np.sum(y_test == 1))

X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) # Convert training images to PyTorch tensors and reorder dimensions to (N, C, H, W)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) # Convert testing images to PyTorch tensors and reorder dimensions to (N, C, H, W)

y_train = torch.tensor(y_train, dtype=torch.long) # Convert training labels to PyTorch tensors
y_test = torch.tensor(y_test, dtype=torch.long) # Convert testing labels to PyTorch tensors 

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Use MPS if available, otherwise use CPU
print("Using device:", device)

# Create the VGG16 model
model = VGG16Model().to(device)

#Define the loss function
loss_function = nn.CrossEntropyLoss() # Calculates how wrong is the model

# Define the optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), #Only update parameters that require gradients (i.e., the last convolution block and the final classification layer)
    lr=0.00005
) 

# Number of times the model will see the entire dataset
num_epochs = 10

# Switch the model to training mode
model.train()

for epoch in range(num_epochs):

    running_loss = 0.0 
    
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device) # Move batch images to the selected device
        batch_labels = batch_labels.to(device) # Move batch labels to the selected device

        # Clear previous batch gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(batch_images)

        # Calculate the loss
        loss = loss_function(outputs, batch_labels)

        # Backpropagation
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss for this batch
        running_loss += loss.item() 

    # Calculate the average loss for this epoch
    average_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {average_loss:.4f}")

# Switch the model to evaluation mode
model.eval()

correct = 0
total = 0

with torch.no_grad(): # Disable gradient calculation during testing to increase speed and reduce memory usage
    for batch_images, batch_labels in test_loader:

        batch_images = batch_images.to(device) # Move batch images to the selected device
        batch_labels = batch_labels.to(device) # Move batch labels to the selected device

        outputs = model(batch_images) # Get model predictions
        _, predicted = torch.max(outputs, 1) # Predict the class (0 = Authentic, 1 = Forged) with the highest score

        total += batch_labels.size(0) # Count the total number of images in this batch
        correct += (predicted == batch_labels).sum().item() # Count the number of correct predictions

# Calculate the accuracy as a percentage
accuracy = (correct / total) * 100 

print(f"Test Accuracy: {accuracy:.2f}%") # Print the accuracy of the model on the test dataset

# Save the trained model
torch.save(model.state_dict(), "outputs/saved_models/vgg16_model.pth")

print("Model saved successfully!")