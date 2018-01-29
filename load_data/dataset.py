import torch.utils.data as data
from PIL import Image

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, typeDir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.pristinePath = 'Database/waterloo/pristine_images/'
        self.distortedPath = 'Database/waterloo/distorted_images/'
        self.typeDir = typeDir
        self.filenames = []
        self.filenames.append('')

        self.indexs = []
        self.indexs.append([])
        self.indexs[0].append(0)
        self.indexs[0].append(0)
        self.indexs[0].append(0)

        j = 1
        index = 0
        for i in xrange(1, 4745):
            step = '%05d' % i
            self.filenames.append(step)

            img = load_img(self.pristinePath + self.filenames[j] + '.bmp')
            [m, n] = img.size
            self.getMN(m, n)
            index += self.row*self.col*4  #four level distorted images, each image has m * n patches
            self.indexs.append([])
            self.indexs[j].append(index)
            self.indexs[j].append(self.row)
            self.indexs[j].append(self.col)
            j+=1

        self.input_transform = input_transform
        self.target_transform = target_transform

    def getMN(self, m, n):
        self.row = n / 64
        self.col = m / 64

    def getI(self, index):
        index += 1
        for i in xrange(len(self.indexs)):
            # print 'Judege: ', self.indexs[i][0], index, '?'
            if (self.indexs[i][0] >= index):
                return i

    def __getitem__(self, index):
        i = self.getI(index)
        row = self.indexs[i][1]
        col = self.indexs[i][2]
        index -= self.indexs[i-1][0]
        level = '_%d.bmp' % (index/(row*col)+1)
        index = index % (row*col)
        # print index, '-', i, ': ',
        x = (index % col) * 64
        y = (index / col) * 64
        # print x, y, x*64, y*64, ',  index =', index,
        # print level

        input = load_img(self.distortedPath + self.typeDir + '/' + self.filenames[i] + level)
        # input.show()
        input = input.crop((x, y, x + 64, y + 64))
        # input.show()
        if self.input_transform:
            input = self.input_transform(input)
        target = load_img(self.pristinePath + self.filenames[i] + '.bmp')
        # target.show()
        target = target.crop((x, y, x + 64, y + 64))
        # target.show()
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.filenames)

