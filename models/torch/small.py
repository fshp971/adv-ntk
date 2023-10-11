import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dims, out_dims, wide=1024, depth=2, act_fn=None):
        if act_fn is None:
            act_fn = nn.ReLU
        assert depth >= 2
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(in_dims, wide))
        layers.append(act_fn())

        for i in range(depth-2):
            layers.append(nn.Linear(wide, wide))
            layers.append(act_fn())

        layers.append(nn.Linear(wide, out_dims))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mlp5(in_dims, out_dims, wide=1024):
    return MLP(in_dims, out_dims, wide, 5)

def mlp_x(in_dims, out_dims, depth, wide=1024, act_fn=None):
    return MLP(in_dims, out_dims, wide, depth, act_fn)
