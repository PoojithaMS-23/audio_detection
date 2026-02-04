# app/models/aasist3.py
import torch.nn as nn
import torch

class AASIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x: (B, T, 1024)
        x = x.transpose(1, 2)  # (B, 1024, T)
        x = self.pool(x).squeeze(-1)  # (B, 1024)
        return self.classifier(x)
