import matplotlib.pyplot as plt
import numpy as np

from preprocessing import load_dataset
images, labels = load_dataset()

print("Dataset Information")
print("-" * 35)

print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)

print("Total Images: ", len(images))

print("Authentic Images: ", np.sum(labels == 0))
print("Forged Images:", np.sum(labels == 1))

classes = ["Authentic", "Forged"]
counts = [np.sum(labels == 0), np.sum(labels == 1)]

plt.bar(classes, counts)

plt.title("Class Distribution")
plt.xlabel("Image Class")
plt.ylabel("Number of Images")

plt.show()

def show_samples(images, labels): # Function to display sample images from the dataset
    authentic_indices = np.where(labels == 0)[0] # Get indices of authentic images
    forged_indices = np.where(labels == 1)[0] # Get indices of forged images

    sample_indices = np.concatenate((authentic_indices[:3], forged_indices[:3])) # Select first 3 authentic and first 3 forged images

    plt.figure(figsize=(12, 6)) # Matplotlib figure size

    for i, index in enumerate(sample_indices):
        plt.subplot(2, 3, i + 1) # Create a subplot for each sample
        plt.imshow(images[index]) # Display the image
        plt.axis("off") # Turn off axis
        plt.title("Authentic" if labels[index] == 0 else "Forged") # Set the title based on the label

    plt.tight_layout() # Adjust the layout to prevent overlap
    plt.show() # Display the figure

show_samples(images, labels)