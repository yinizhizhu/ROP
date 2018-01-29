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
        self.layer3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer9 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer10 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer13 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer14 = nn.BatchNorm2d(64)
        self.layer15 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        tmp = self.layer2(self.layer1(x))
        step = self.layer3(tmp)+tmp
        step = self.layer4(step)+step
        step = self.layer5(step)+step
        step = self.layer6(step)+step
        step = self.layer7(step)+step
        step = self.layer8(step)+step
        step = self.layer9(step)+step
        step = self.layer10(step)+step
        step = self.layer11(step)+step
        step = self.layer12(step)+step
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
        self.layer3 = stridedBlock(64, 64)
        self.layer4 = stridedBlock(64, 128)
        self.layer5 = stridedBlock(128, 256)
        self.layer6 = stridedBlock(256, 512)
        self.layer7 = stridedBlock(512, 512)
        self.layer8 = nn.Linear(2048, 512)
        self.layer9 = nn.LeakyReLU(inplace=True)
        self.layer10 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
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