from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from RAN import restorator
from RANList import restorator
from data import get_training_set
from perceptionError import PERLoss, PERLossBatch

# from pycrayon import CrayonClient
# import time
#
# cc = CrayonClient(hostname="localhost", port=8889)

# cc.remove_experiment("train_loss")
class trainer:
    def __init__(self, typeDir, epochs, l, load, dir):
        self.load = load
        self.typeDir = typeDir
        # self.timeH = time.time()
        self.dir = dir

        self.trainLN = "train_loss"
        # cc.remove_experiment(self.trainLN)
        # self.trainL = cc.create_experiment(self.trainLN)

        self.nEpochs = epochs

        self.cuda = True
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(123)
        if self.cuda:
            print('*******Cuda!*******')
            torch.cuda.manual_seed(123)

        print('===> Loading datasets')
        train_set = get_training_set(self.typeDir)
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=5, shuffle=True)

        print('===> Building model')
        if (load == 2):
            self.model = torch.load('%s/%s_22.pth' % (self.dir, self.typeDir))
        elif (load == 1):
            self.model = torch.load('%s/%sF.pth' % (self.dir, self.typeDir))
        else:
            self.model = restorator()
#         criterion = nn.MSELoss()
        criterion = PERLoss()
        # criterion = PERLossBatch()

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

            # self.trainL.add_scalar_value(self.trainLN, loss.data[0], time.time() - self.timeH)

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(self.training_data_loader)))
        # if (self.load == 2):
        #     torch.save(self.model, '%s/%sF_ALL_%d.pth' % (self.dir, self.typeDir, epoch))
        #     print("Checkpoint saved to %s/%sF_ALL_%d.pth" % (self.dir, self.typeDir, epoch))
        # elif (self.load == 1):
        #     torch.save(self.model, '%s/%sF_%d.pth' % (self.dir, self.typeDir, epoch))
        #     print("Checkpoint saved to %s/%sF_%d.pth" %  (self.dir, self.typeDir, epoch))
        # else:
        #     torch.save(self.model, '%s/%s_%d.pth' % (self.dir,self.typeDir, epoch))
        #     print("Checkpoint saved to %s/%s_%d.pth" % (self.dir,self.typeDir, epoch))

    def training(self):
        for epoch in range(1, self.nEpochs + 1):
            self.train(epoch)
        # *.pth - Pretraining gblurConv with 1/3
        # *F.pth - Finetuning gblurConv with 1/3 on pretraining gblurConv
        # *F_ALL.pth - Finetuning gblurConv with 100% on pretraining gblurConv
        if (self.load == 2):
            torch.save(self.model, '%s/%sF_ALL.pth' % (self.dir, self.typeDir))
            print("Checkpoint saved to %s/%sF_ALL.pth" % (self.dir, self.typeDir))
        elif (self.load == 1):
            torch.save(self.model, '%s/%sF.pth' % (self.dir, self.typeDir))
            print("Checkpoint saved to %s/%sF.pth" %  (self.dir, self.typeDir))
        else:
            torch.save(self.model, '%s/%s.pth' % (self.dir,self.typeDir))
            print("Checkpoint saved to %s/%s.pth" % (self.dir,self.typeDir))

#                   0         1             2           3             4         5
modelList = ['gblurConv', 'gblurRes', 'm_mseloss', 'm_perloss', 'vgg19_bn', 'vgg19']
select = 3

# pretraining with 100%
# t = trainer('gblur', 22, 0.00001, 0, modelList[select])
# t.training()
#
# t = trainer('wn', 22, 0.00001, 0, modelList[select])
# t.training()
#
# t = trainer('jpeg', 22, 0.00001, 0, modelList[select])
# t.training()
#
# t = trainer('jp2k', 22, 0.00001, 0, modelList[select])
# t.training()

# finetuning on pretraining gblurConv with 100%
t = trainer('gblur', 22, 0.000001, 2, modelList[select])
t.training()

t = trainer('wn', 22, 0.000001, 2, modelList[select])
t.training()

t = trainer('jpeg', 22, 0.000001, 2, modelList[select])
t.training()

t = trainer('jp2k', 22, 0.000001, 2, modelList[select])
t.training()
