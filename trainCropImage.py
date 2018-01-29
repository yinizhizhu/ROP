from __future__ import print_function
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage

import numpy as np

class denosier:
    def __init__(self, dir):
        self.dir = dir
        self.func()

    def func(self):
        # img = Image.open('Database/waterloo/distorted_images/gblurConv/00001_1.bmp').convert('RGB')
        img = Image.open('Database/waterloo/pristine_images/00001.bmp').convert('RGB')
        img = CenterCrop((224, 224))(img)
        img.save('outPic/src_pristine.bmp', 'bmp', quality=100)
        # img.show()
        # print (img.size)

        img = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
        # print(img)

        model = torch.load('%s/gblurConv.pth' %  self.dir)
        model = model.cuda()
        input = img.cuda()

        out_img = model(input)
        out_img = out_img.cpu().data[0]
        out_img.clamp_(0.0, 1.0)
        out_img = ToPILImage()(out_img)
        # out_img.save('outPic/1_1.bmp', 'bmp', quality=100)
        out_img.show()

denosier('othersRes')