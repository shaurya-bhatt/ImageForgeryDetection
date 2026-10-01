# Image Forgery Detection using Deep Learning

## Overview

This project detects whether an image is **Authentic** or **Forged** using multiple Deep Learning models.

The project compares the performance of a custom CNN with four transfer learning models:

- CNN
- VGG16
- ResNet50
- EfficientNet-B0
- Vision Transformer (ViT)

All models were trained and evaluated on the **CASIA 2.0 Image Forgery Dataset** using PyTorch.

---

## Features

- Image preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Custom CNN implementation
- Transfer Learning using VGG16
- Transfer Learning using ResNet50
- Transfer Learning using EfficientNet-B0
- Transfer Learning using Vision Transformer (ViT)
- Performance comparison of all models
- Confusion Matrix
- ROC Curve
- AUC Score
- Precision, Recall and F1 Score

---

## Dataset

Dataset used:

**CASIA 2.0 Image Tampering Detection Dataset**

Classes:

- Authentic Images
- Forged Images

Dataset statistics:

- Total Images: 12,614
- Authentic: 7,491
- Forged: 5,123

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Scikit-Image
- Matplotlib

---

## Project Structure

ImageForgeryDetection/

├── dataset/
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── saved_models/
│
├── src/
│   ├── models/
│   ├── training/
│   ├── evaluate/
│   ├── preprocessing.py
│   └── eda.py
│
├── README.md
├── requirements.txt
├── .gitignore

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd ImageForgeryDetection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Example:

```bash
python -m src.training.train_vgg16
```

Available training scripts

- train_cnn.py
- train_vgg16.py
- train_resnet50.py
- train_efficientnet_b0.py
- train_vit.py

---

## Evaluation

Example

```bash
python -m src.evaluate.evaluate_vgg16
```

The evaluation scripts generate:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|--------|----------|-----------|---------|----------|-----|
| CNN | 0.6560 | 0.5684 | 0.6361 | 0.6004 | 0.6871 |
| VGG16 | 0.7816 | 0.6798 | 0.8741 | 0.7648 | 0.8422 |
| ResNet50 | 0.7812 | 0.7221 | 0.7502 | 0.7359 | 0.8645 |
| EfficientNet-B0 | 0.7095 | 0.6276 | 0.7005 | 0.6621 | 0.7897 |
| Vision Transformer | 0.6746 | 0.5981 | 0.6068 | 0.6024 | 0.7563 |

---

## Results

Confusion matrices and ROC curves for every model are available in:

```
outputs/
```

---

## Future Improvements

- Hyperparameter tuning
- Data augmentation
- Ensemble learning
- Attention-based CNN models
- Larger forgery datasets
- Real-time forgery detection

---

## Author

**Shaurya Bhatt**

B.Tech Computer Science Engineer