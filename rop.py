from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage
from perceptionError import PERLoss
import matplotlib.pyplot as plt

from pycrayon import CrayonClient

cc = CrayonClient(hostname="localhost", port=8889)

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class denosior:
    def __init__(self, size, typeDir, dir, select):
        self.typeDir = typeDir
        self.size = size
        self.dir = dir
        self.sel = select

        print ('******%s******' % dir)
        print ('Loading model for %s...' % self.typeDir)
        model = torch.load('%s/%sF_ALL.pth' % (dir, self.typeDir))
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.cpu()
        print ('Model is ready!')

        if (select == 2):
            self.criterion = PERLoss()
        else:
            self.criterion = nn.MSELoss()

        self.basis = dir + '_' + self.typeDir
        self.src_dis = self.basis + "_src_dis"
        self.src_res = self.basis + "_src_ress"
        self.dis_res = self.basis + "_dis_res"
        print ('Removing the former experiments...')
        cc.remove_experiment(self.basis)
        print ('Creating new experiments...')
        self.experiment = cc.create_experiment(self.basis)


    def draw(self, stock_data):
        fig = plt.figure(self.basis+'_%d' % self.sel)
        xrow = [i for i in xrange(1, 11)]
        # print (xrow)
        # print (stock_data[0])
        # print (stock_data[1])
        # print (stock_data[2])
        plt.plot(xrow, stock_data[0], '-', label='src_dis')
        plt.plot(xrow, stock_data[1], '-', label='src_res')
        plt.plot(xrow, stock_data[2], '-', label='dis_res')
        plt.plot(xrow, stock_data[3], '-', label='dis_res + src_res')

        plt.xlabel('level')
        plt.ylabel('differ')
        plt.legend(loc='upper right', fontsize=12)
        plt.title(self.basis+'_%d' % self.sel)
        # fig.show()
        if (self.sel == 2):
            fig.savefig('curve/PERLoss/%s/%s.png' % (self.dir, self.typeDir))
        else:
            fig.savefig('curve/MSELoss/%s/%s.png' % (self.dir, self.typeDir))

    def denoiseList(self):
        src = Image.open('Database/waterloo/pristine_images/00001.bmp').convert('RGB')
        src = CenterCrop((self.size, self.size))(src)
        src.save('level_distortion/%s/%s/src.bmp' % (self.dir, self.typeDir), 'bmp', quality=100)
        src = Variable(ToTensor()(src)).view(1, -1, src.size[1], src.size[0])
        stock_data = [[], [], [], []]
        for i in xrange(1, 11):
            print ('    checking %dth...' % i)
            dis = Image.open('level_distortion/%s/%s/00001_%d.bmp' % (self.dir, self.typeDir, i)).convert('RGB')
            dis = CenterCrop((self.size, self.size))(dis)
            dis.save('level_distortion/%s/%s/dis_%d.bmp' % (self.dir, self.typeDir, i), 'bmp', quality=100)
            dis = Variable(ToTensor()(dis)).view(1, -1, dis.size[1], dis.size[0])
            dis = dis.cpu()
            # print (input)
            res = self.model(dis)

            tmp = {}
            loss = self.criterion(src, dis)
            tmp[self.src_dis] = loss.data[0]
            stock_data[0].append(loss.data[0])
            # self.experiment.add_scalar_value(self.src_dis, loss.data[0], i)

            loss = self.criterion(res, src)
            tmp[self.src_res] = loss.data[0]
            stock_data[1].append(loss.data[0])
            # self.experiment.add_scalar_value(self.src_res, loss.data[0], i)

            loss = self.criterion(res, dis)
            tmp[self.dis_res] = loss.data[0]
            stock_data[2].append(loss.data[0])
            # self.experiment.add_scalar_value(self.dis_res, loss.data[0], i)

            stock_data[3].append(stock_data[1][i-1]+stock_data[2][i-1])

            self.experiment.add_scalar_dict(tmp)

            res = res.data[0]
            res.clamp_(0.0, 1.0)
            res = ToPILImage()(res)
            res.save('level_distortion/%s/%s/res_%d.bmp' % (self.dir, self.typeDir, i), 'bmp', quality=100)
        self.draw(stock_data)

        # tmp = self.dis_res_.get_scalar_names()
        # print(tmp)
        # tmp = self.dis_res_.get_scalar_values(self.dis_res)
        # print(tmp)
        # print (self.dis_res_.get_histogram_names())
        # print (self.dis_res_.get_histogram_values(self.dis_res))
        print ('Done this work!\n\n')

modelList = ['deconv', 'm_mseloss', 'm_perloss', 'vgg19_bn']
typeList = ['gblur', 'wn', 'jpeg', 'jp2k']

for i in xrange(1,3):
    for model in modelList:
        for lr in typeList:
            t = denosior(224, lr, model, i)
            t.denoiseList()