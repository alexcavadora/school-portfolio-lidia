import torch.nn as nn
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Flatten, Dropout2d, BatchNorm2d, BatchNorm1d

class CustomLenet(Module):
    def __init__(self, nChanels, nClasses):
        super(CustomLenet, self).__init__()
        self.conv1 = Conv2d(nChanels, 32, kernel_size=5, stride=1, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(6*6*64, 256)
        self.bn3 = BatchNorm1d(256)
        self.dropout_fc1 = Dropout2d(0.5)
        self.fc2 = Linear(256, 128)
        self.dropout_fc2 = Dropout2d(0.5)
        self.bn4 = BatchNorm1d(128)
        self.fc3 = Linear(128, nClasses)
        self.flatten = nn.Flatten()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = self.dropout_fc2(x)
    
        x = self.fc3(x)
        return x