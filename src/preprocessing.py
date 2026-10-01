import os 
import cv2
import numpy as np

#Dataset paths
AUTHENTIC_PATH = "dataset/Au"
FORGED_PATH = "dataset/Tp"
#Valid Extensions for the input images
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#Image Size
IMAGE_SIZE = (224, 224)

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image -> {image_path}")
        return None

    return image

def preprocess_image(image):
    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Normalize pixel values to [0,1]
    image = image.astype(np.float32) / 255.0

    return image

def load_dataset():

    images = [] # Stores preprocessed images
    labels = [] # Stores class labels (0 = Authentic, 1 = Forged)

    # Load authentic images
    for filename in os.listdir(AUTHENTIC_PATH):

        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        image_path = os.path.join(AUTHENTIC_PATH, filename)

        image = load_image(image_path)

        if image is not None:

            image = preprocess_image(image)

            images.append(image)

            labels.append(0)

    # Load forged images
    for filename in os.listdir(FORGED_PATH):

        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        image_path = os.path.join(FORGED_PATH, filename)

        image = load_image(image_path)

        if image is not None:

            image = preprocess_image(image)

            images.append(image)

            labels.append(1)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Temporary testing code
if __name__ == "__main__":

    images, labels = load_dataset()

    print("Dataset Loaded Successfully!")

    print("Images Shape:", images.shape)
    print("Labels Shape:", labels.shape)

    print("Number of Authentic Images:", np.sum(labels == 0))
    print("Number of Forged Images:", np.sum(labels == 1))
# Till here