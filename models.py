




import torch
import torch.nn as nn
import torch.nn.functional as F


# 3x32x32 -> 32x28x28
# -> 32x14x14 -> 64x10x10
# -> 64x5x5 -> 128x1x1
# -> 128 -> 64 -> 42

class ConvClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 34)
        #self.fc3 = nn.Linear(84, 34)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x




