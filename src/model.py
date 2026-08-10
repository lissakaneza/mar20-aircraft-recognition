"""Small CNN trained from scratch - the baseline every later model must beat."""

import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_mobilenet(num_classes: int, freeze_backbone: bool = True, unfreeze_last_n: int = 0) -> nn.Module:
    """MobileNetV2 pretrained on ImageNet, used as a feature extractor.

    The backbone is frozen (its ImageNet features are reused as-is) and only
    the final classification layer is replaced and trained. Set
    unfreeze_last_n > 0 to also let the last N backbone blocks (out of 19 -
    the ones closest to the classifier, so the most task-specific) train too,
    for fine-tuning after the frozen version has already been trained.
    """
    model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        if unfreeze_last_n > 0:
            for layer in list(model.features.children())[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
