# Two layer fully connected neural network

import torch
import torch.nn as nn
import copy

class TwoLayerNN(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, num_classes=10):
        super(TwoLayerNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.num_classes = num_classes


        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        x = x / self.width
        return x
    

if __name__ == '__main__':
    x = torch.randn(5, 1, 32, 32)
    net = TwoLayerNN(input_dim=32*32, width=123)
    print(net(x))
