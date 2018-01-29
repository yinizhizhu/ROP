from __future__ import print_function
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from RAN import restorator
from data import get_training_set

from pycrayon import CrayonClient
import time

cc = CrayonClient(hostname="localhost", port=8889)

def removeAll():
    cc.remove_experiment("train_loss")

removeAll()

class trainer:
    def __init__(self, epochs, l):
        self.timeH = time.time()
        self.trainLN = "train_loss"
        self.trainL = cc.create_experiment(self.trainLN)

        self.nEpochs = epochs

        self.cuda = True
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(123)
        if self.cuda:
            print('*******Cuda!*******')
            torch.cuda.manual_seed(123)

        print('===> Loading datasets')
        train_set = get_training_set()
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=5, shuffle=True)

        print('===> Building gblurConv')
        self.model = restorator()
        criterion = nn.MSELoss()

        if self.cuda:
            print('*******Cuda!!!*******')
            self.model = self.model.cuda()
            self.criterion = criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=l)

    def train(self, epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(self.training_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1])
            if self.cuda:
                input = input.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(input), target)
            epoch_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(self.training_data_loader), loss.data[0]))

            self.trainL.add_scalar_value(self.trainLN, loss.data[0], time.time() - self.timeH)

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(self.training_data_loader)))

        torch.save(self.model, '1.pth')
        print("Checkpoint saved to 1.pth")

    def training(self):
        for epoch in range(1, self.nEpochs + 1):
            self.train(epoch)
#
# t = trainer(9, 0.0001)
# t.training()
