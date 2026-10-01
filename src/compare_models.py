import pandas as pd

# Replace these values with your actual metrics
results = {
    "Model": [
        "CNN",
        "VGG16",
        "ResNet50",
        "EfficientNet-B0",
        "Vision Transformer"
    ],

    "Accuracy": [
        0.6472,
        0.6992,
        0.6948,
        0.6774,
        0.6746
    ],

    "Precision": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ],

    "Recall": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ],

    "F1 Score": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ],

    "AUC": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ]
}

df = pd.DataFrame(results)

print("\nModel Comparison\n")
print(df)

df.to_csv(
    "outputs/model_comparison/model_comparison.csv",
    index=False
)

print("\nCSV saved successfully!")