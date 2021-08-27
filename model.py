import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512), nn.ReLU(), nn.Linear(512, output_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        y = self.relu2(h + x)
        return y


DN_FILTERS = 32
RESIDUAL_NUM = 3


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


class ResNet(nn.Module):
    def __init__(self, input_shape, actions_n):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], DN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(DN_FILTERS),
            nn.LeakyReLU(),
        )

        layers = [ResBlock(DN_FILTERS) for _ in range(RESIDUAL_NUM)]
        self.residual_layers = nn.ModuleList(layers)
        body_out_shape = (DN_FILTERS,) + input_shape[1:]

        self.conv_policy = nn.Sequential(
            nn.Conv2d(DN_FILTERS, 2, kernel_size=1), nn.BatchNorm2d(2), nn.LeakyReLU()
        )
        self.avg_pool = GlobalAvgPool2d()
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, actions_n), nn.Softmax(dim=1)
        )

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        o = self.avg_pool(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        h = self.conv_in(x)
        for layer in self.residual_layers:
            h = layer(h)
        h = self.conv_policy(h)
        h = self.avg_pool(h)
        pol = self.policy(h.view(batch_size, -1))
        return pol
