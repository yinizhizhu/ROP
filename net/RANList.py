import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

torch.manual_seed(1)
np.random.seed(1)


class residualBlock(nn.Module):
    def __init__(self, inN, outN):
        super(residualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(inN, outN, 3, 1, 1),
            nn.BatchNorm2d(outN),
            nn.ReLU(inplace=True),
            nn.Conv2d(outN, outN, 3, 1, 1),
            nn.BatchNorm2d(outN),
        )

    def forward(self, x):
        return self.res(x)


class restorator(nn.Module):
    def __init__(self):
        super(restorator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.layer2 = nn.ReLU(inplace=True)
        self.middle = nn.ModuleList()
        for i in xrange(10):
            self.middle.append(residualBlock(64, 64))
        self.layer13 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer14 = nn.BatchNorm2d(64)
        self.layer15 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        tmp = self.layer2(self.layer1(x))
        step = tmp
        for i in xrange(10):
            step = self.middle[i](step)+step
        step = self.layer13(step)
        step = self.layer14(step)+tmp
        return self.layer15(step)


class stridedBlock(nn.Module):
    def __init__(self, inN, x):
        super(stridedBlock, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(inN, x, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(x),
            nn.Conv2d(x, x, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(x),
        )

    def forward(self, x):
        return self.dis(x)


class discirminator(nn.Module):
    def __init__(self):
        super(discirminator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.layer2 = nn.LeakyReLU(inplace=True)
        self.middle = nn.ModuleList()
        self.middle.append(stridedBlock(64, 64))
        self.middle.append(stridedBlock(64, 128))
        self.middle.append(stridedBlock(128, 256))
        self.middle.append(stridedBlock(256, 512))
        self.middle.append(stridedBlock(512, 512))
        self.layer8 = nn.Linear(2048, 512)
        self.layer9 = nn.LeakyReLU(inplace=True)
        self.layer10 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        for i in xrange(5):
            x = self.middle[i](x)
        x = x.view(1, -1, 2048)
        x = self.layer8(x)
        x = self.layer9(x)
        return torch.sigmoid(self.layer10(x))

#
# x = Variable(torch.randn(3, 64, 64)).view(1, -1, 64, 64)
# R = restorator()
# y = R(x)
# print(y)
#
# D = discirminator()
# z = D(y)
# print (z)