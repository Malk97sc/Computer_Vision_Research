import torch 
import torch.nn as nn
import torch.nn.functional as F

class SheetCNN(nn.Module):
    def __init__(self, n_clases):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #150x150

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  #75x75

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   #37x37
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64*37*37, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_clases)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)