import torch.nn as nn

def CNNmodel():
    model = nn.Sequential(
        # First convolution layer
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Flatten
        nn.Flatten(),

        # Fully connected layers
        nn.Linear(32 * 14 * 14, 128),  
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.15),

        nn.Linear(128, 10)  
    )
    return model

