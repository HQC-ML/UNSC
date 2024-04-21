import torch.nn as nn
import torch
from .base import Base

class AlexNet(Base):
    def __init__(self, n_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, 4, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2, bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256*2*2, 2048, bias=False),
            nn.BatchNorm1d(2048, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes, bias=False)
        )
        self.register_hook()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x