import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_dims, out_dims, shape, wide=1024, depth=2, act_fn=None):
        if act_fn is None:
            act_fn = nn.ReLU
        assert depth >= 2
        super(CNN, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_dims, wide, kernel_size=3, padding=1))
        layers.append(act_fn())

        for i in range(depth-1):
            layers.append(nn.Conv2d(wide, wide, kernel_size=3, padding=1))
            layers.append(act_fn())

        layers.append(nn.Flatten())
        layers.append(nn.Linear(wide*shape[0]*shape[1], out_dims))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        return x


def cnn8(in_dims, out_dims, shape, wide=1024):
    return CNN(in_dims, out_dims, shape, wide, 8)

def cnn_x(in_dims, out_dims, shape, depth, wide=1024, act_fn=None):
    return CNN(in_dims, out_dims, shape, wide, depth, act_fn)
