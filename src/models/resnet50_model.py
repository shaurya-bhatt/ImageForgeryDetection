import torch.nn as nn
from torchvision import models


class ResNet50Model(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze Layer 2
        for param in self.model.layer2.parameters():
            param.requires_grad = True

        # Unfreeze Layer 3
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        # Unfreeze Layer 4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=2
        )

        # Make the new classifier trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):

        x = self.model(x)

        return x