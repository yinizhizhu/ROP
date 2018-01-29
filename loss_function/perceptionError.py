from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()


class PERLossBatch(_Loss):
    def __init__(self):
        super(PERLossBatch, self).__init__()
        self.vgg19_bn = models.vgg19_bn(pretrained=True)
        for param in self.vgg19_bn.parameters():
            param.requires_grad = False

    def extractF(self, x):
        lists = []
        counter = 0
        for module in self.vgg19_bn.features._modules.values():
            x = module(x)
            if (counter == 2):
                lists.append(x)
            elif (counter == 7):
                lists.append(x)
            elif (counter == 16):
                lists.append(x)
            elif (counter == 25):
                lists.append(x)
            elif (counter == 34):
                lists.append(x)
                break
            counter += 1
        return lists

    def forward(self, x, y):
        _assert_no_grad(y)
        listx = self.extractF(x)
        listy = self.extractF(y)
        ans = F.mse_loss(listx[0], listy[0])
        for i in xrange(1, 5):
            ans = ans + F.mse_loss(listx[i], listy[i])
        return ans


class PERLoss(_Loss):
    def __init__(self):
        super(PERLoss, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        # print (self.vgg19)
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def extractF(self, x):
        lists = []
        counter = 0
        for module in self.vgg19.features._modules.values():
            x = module(x)
            if (counter == 2):
                lists.append(x)
            elif (counter == 7):
                lists.append(x)
            elif (counter == 16):
                lists.append(x)
            elif (counter == 25):
                lists.append(x)
            elif (counter == 34):
                lists.append(x)
                break
            counter += 1
        return lists

    def forward(self, x, y):
        _assert_no_grad(y)
        listx = self.extractF(x)
        listy = self.extractF(y)
        ans = F.mse_loss(listx[0], listy[0])
        for i in xrange(1, 5):
            ans = ans + F.mse_loss(listx[i], listy[i])
        return ans
