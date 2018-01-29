from __future__ import print_function
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage

import numpy as np

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class denosior:
    def __init__(self, size, typeDir, load, dir):
        self.typeDir = typeDir
        self.size = size
        self.load = load
        self.dir = dir
        # self.denoise()


        self.distortedPath = 'Database/waterloo/distorted_images/%s/%05d_%d.bmp' % (self.typeDir, 1, 2)
        self.distortedImg = load_img(self.distortedPath)
        # self.distortedImg.show()
        [m, n] = self.distortedImg.size
        self.getMN(m, n)

    def getMN(self, m, n):
        self.row = n/64
        self.col = m/64
        self.x = (m%64)/2
        self.y = (n%64)/2
        self.moveX = 0
        self.moveY = 0
        print (self.row, self.col)

    def getNext(self):
        x = self.x+self.moveX*64
        y = self.y+self.moveY*64
        # print x, '-', y

        self.fake = self.distortedImg.crop((x, y, x+64, y+64))
        self.moveX += 1

        if (self.moveX == self.col):
            self.moveX = 0
            self.moveY += 1

    def clip(self, out_img):
        for i in xrange(out_img.size(0)):
            for j in xrange(out_img.size(1)):
                for k in xrange(out_img.size(2)):
                    if (out_img[i][j][k] > 1):
                        out_img[i][j][k] = 1.0
                    elif (out_img[i][j][k] < 0):
                        out_img[i][j][k] = 0.0
        return out_img

    def denoiseList(self):
        img = Image.open('Database/waterloo/pristine_images/00001.bmp').convert('RGB')
        img = CenterCrop((self.size, self.size))(img)
        img.save('outPic/src.bmp', 'bmp', quality=100)
        img = Image.open('Database/waterloo/distorted_images/%s/00001_%d.bmp' % (self.typeDir, 2)).convert('RGB')
        img = CenterCrop((self.size, self.size))(img)
        img.save('outPic/sorted.bmp', 'bmp', quality=100)
        for i in xrange(1, 23):
            input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
            input = input.cuda()

            if (self.load == 2):
                model = torch.load('%s/%sF_ALL.pth' % (self.dir, self.typeDir))
            elif (self.load == 1):
                model = torch.load('%s/%sF.pth' % (self.dir, self.typeDir))
            else:
                model = torch.load('%s/%s_%d.pth' % (self.dir, self.typeDir, i))
            model = model.cuda()

            out_img = model(input)
            out_img = out_img.cpu()
            out_img = out_img.data[0]

            out_img.clamp_(0.0, 1.0)
            # out_img = self.clip(out_img)
            out_img = ToPILImage()(out_img)
            # out_img.save('outPic/%d.bmp'%i, 'bmp', quality=100)
            out_img.save('outPic/%d.bmp' % i, 'bmp', quality=100)

    def denoiseImg(self):
        if (self.load == 2):
            model = torch.load('%s/%sF_ALL.pth' % (self.dir, self.typeDir))
        elif (self.load == 1):
            model = torch.load('%s/%sF.pth' % (self.dir, self.typeDir))
        else:
            model = torch.load('%s/%s.pth' % (self.dir, self.typeDir))
        model = model.cuda()

        self.getNext()
        img = Variable(ToTensor()(self.fake)).view(1, -1, self.fake.size[1], self.fake.size[0])
        img = img.cuda()
        out_img = model(img)
        out_img = out_img.cpu()
        out_img = out_img.data[0]
        for j in xrange(1, self.col):
            self.getNext()
            img = Variable(ToTensor()(self.fake)).view(1, -1, self.fake.size[1], self.fake.size[0])
            img = img.cuda()
            img = model(img)
            img = img.cpu()
            img = img.data[0]
            out_img = torch.cat((out_img, img), 2)
        ans = out_img

        for i in xrange(1, self.row):
            self.getNext()
            img = Variable(ToTensor()(self.fake)).view(1, -1, self.fake.size[1], self.fake.size[0])
            img = img.cuda()
            out_img = model(img)
            out_img = out_img.cpu()
            out_img = out_img.data[0]
            print (out_img.size(0), out_img.size(1), out_img.size(2))

            for j in xrange(1, self.col):
                self.getNext()
                img = Variable(ToTensor()(self.fake)).view(1, -1, self.fake.size[1], self.fake.size[0])
                img = img.cuda()
                img = model(img)

                img = img.cpu()
                img = img.data[0]
                out_img = torch.cat((out_img, img), 2)
            ans = torch.cat((ans, out_img), 1)

        ans.clamp_(0.0, 1.0)
        # ans = self.clip(ans);
        print (ans.size(0), ans.size(1), ans.size(2))
        ans = ToPILImage()(ans)
        ans.show()

    def denoisePatch(self):
        if (self.load == 2):
            model = torch.load('%s/%sF_ALL.pth' % (self.dir, self.typeDir))
        elif (self.load == 1):
            model = torch.load('%s/%sF.pth' % (self.dir, self.typeDir))
        else:
            model = torch.load('%s/%s.pth' % (self.dir, self.typeDir))
        model = model.cuda()
        # img = Image.open('Database/waterloo/pristine_images/00001.bmp').convert('RGB')
        # img = CenterCrop((self.size, self.size))(img)
        # img.save('outPic/0.bmp', 'bmp', quality=100)
        for i in xrange(1, 5):
            img = Image.open('Database/waterloo/distorted_images/%s/00001_%d.bmp' % (self.typeDir, i)).convert('RGB')
            # img = Image.open('Database/waterloo/pristine_images/00001.bmp').convert('RGB')
            img = CenterCrop((self.size, self.size))(img)
            # img.save('outPic/%d_0.bmp' % i, 'bmp', quality=100)
            img.show()
            # print (img.size)

            img = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
            # print(img)
            input = img.cuda()

            out_img = model(input)

            out_img = out_img.cpu()
            out_img = out_img.data[0]

            out_img.clamp_(0.0, 1.0)
            # out_img = self.clip(out_img)
            out_img = ToPILImage()(out_img)
            # out_img.save('outPic/%d.bmp'%i, 'bmp', quality=100)
            out_img.show()

#                   0         1             2           3             4         5
modelList = ['gblurConv', 'gblurRes', 'othersConv', 'othersRes', 'vgg19_bn', 'vgg19']
select = 3


t = denosior(224, 'gblur', 0, modelList[select])
# t.denoiseImg()
# t.denoisePatch()
t.denoiseList()