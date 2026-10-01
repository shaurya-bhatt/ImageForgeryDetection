import torch.nn as nn
from torchvision import models


class EfficientNetB0Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze feature blocks 7 and 8
        for param in self.model.features[7].parameters():
            param.requires_grad = True

        for param in self.model.features[8].parameters():
            param.requires_grad = True

        # Replace classifier
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            2
        )

        for param in self.model.classifier[1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)