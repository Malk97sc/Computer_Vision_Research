import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_ly1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_ly2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.full_c1 = nn.Linear(32 * 56 * 56, 64)
        self.relu3 = nn.ReLU()
        self.full_c2 = nn.Linear(64, 2) #where 2 is the number of classes, in this case 2

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv_ly1(x)))
        x = self.pool2(self.relu2(self.conv_ly2(x)))
        x = x.view(-1, 32 * 56 * 56) #flattening the output to match the Linear layer input
        x = self.relu3(self.full_c1(x))
        x = self.full_c2(x)
        return x

