import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):

    def __init__(self):
        super().__init__()

        # Load the pretrained VGG16 model
        self.model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last convolution block
        for param in self.model.features[24:].parameters():
            param.requires_grad = True

        # Replace the final classification layer
        self.model.classifier[6] = nn.Linear(
            in_features=4096,
            out_features=2
        )

        # Make the new classifier trainable
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):

        x = self.model(x)

        return x