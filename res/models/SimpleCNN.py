import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10, h = 8, k = 3):
        super(SimpleCNN, self).__init__()
        
        # ordered layers
        self.conv1 = nn.Conv2d(1, h, kernel_size = k)
        self.pool1 = nn.MaxPool2d(kernel_size = k, stride = 2)
        self.conv2 = nn.Conv2d(h, h * 2, kernel_size = k)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(h * 2 * 4 * 4, h * 2 * 4)
        self.fc2 = nn.Linear(h * 2 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Softmax called in the loss function
        return x