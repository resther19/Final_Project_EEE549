# %%
import torch.nn as nn
def CNNmodel():
    model = nn.Sequential(
        # First convolution layer
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.3),

        # Second convolution layer
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.3),

        # Flatten
        nn.Flatten(),

        # Fully connected layers
        nn.Linear(128* 4 * 4, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),

        nn.Linear(128, 10)
    )
    return model

