import torch
import torch.nn as nn

""" AlexNet model architecture using PyTorch

Paper: ImageNet Classification with Deep Convolutional Neural Networks

Reference: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

"""
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adding an adaptive pooling layer to ensure the input size to the classifier is fixed as 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            # Fully connected layer 1
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # Fully connected layer 2
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # Output layer
            nn.Linear(4096, num_classes),
            # Softmax is ommited, because it's included in the loss function
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
