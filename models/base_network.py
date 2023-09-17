import torch
import torch.nn as nn
from pdb import set_trace

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 96x96
                           nn.BatchNorm2d(32),
                           nn.ReLU(inplace=True))
        b2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 48*48
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True))
        b3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24
                           nn.BatchNorm2d(128),
                           nn.ReLU(inplace=True))
        b4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 12x12
                           nn.BatchNorm2d(256),
                           nn.ReLU(inplace=True))
        b5 = nn.Sequential(ResidualBlock(256),
                           ResidualBlock(256),
                           ResidualBlock(256),
                           ResidualBlock(256),
                           ResidualBlock(256),)
        self.net = nn.Sequential(b1, b2, b3, b4, b5)

    def forward(self, x):
        return self.net(x)
    
class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        b1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 24x24
                           nn.BatchNorm2d(128),
                           nn.ReLU(inplace=True))
        b2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 48x48
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True))
        b3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 96x96
                           nn.BatchNorm2d(32),
                           nn.ReLU(inplace=True))
        b4 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 192x192
                           nn.Sigmoid())
        self.net = nn.Sequential(b1, b2, b3, b4)

    def forward(self, x):
        f_list = []
        for layer in self.net:
            x = layer(x)
            f_list.append(x)
        return x, f_list[:-1]

class Segmentor(nn.Module):
    def __init__(self, nclass=5):
        super(Segmentor, self).__init__()
        b1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 24x24
                           nn.BatchNorm2d(128),
                           nn.ReLU(inplace=True))
        b2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),             # 24x24
                           nn.BatchNorm2d(128),
                           nn.ReLU(inplace=True), 
                           nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 48x48
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True))
        b3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),              # 48x48
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True), 
                           nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),      # 96x96
                           nn.BatchNorm2d(32),
                           nn.ReLU(inplace=True))
        b4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),               # 96x96
                           nn.BatchNorm2d(32),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(32, nclass, kernel_size=4, stride=2, padding=1),  # 128x128
                           nn.Sigmoid())
        self.net = nn.Sequential(b1, b2, b3, b4)

    def forward(self, x, f):
        f_index = 0
        for layer_index, layer in enumerate(self.net):
            if layer_index > 0:
                x = torch.concat([x, f[f_index]], dim=1)
                f_index += 1
            x = layer(x)
        return x
