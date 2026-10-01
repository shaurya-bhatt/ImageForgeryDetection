import torch.nn as nn
from torchvision import models


class ViTModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained ViT-B/16
        self.model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last encoder block
        for param in self.model.encoder.layers[-1].parameters():
            param.requires_grad = True

        # Replace classification head
        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features,
            2
        )

        # Make classifier trainable
        for param in self.model.heads.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)