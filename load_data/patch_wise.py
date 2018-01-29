from PIL import Image
from torchvision.transforms import ToTensor

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class patch:
    def __init__(self):
        self.pristinePath = 'Database/waterloo/pristine_images/'
        self.distortedPath = 'Database/waterloo/distorted_images/'
        self.src = []
        self.iteration = 0
        self.filenames = []
        self.basic = 20
        for i in xrange(1, 4745):
            number = '%05d' % i
            self.src.append(self.pristinePath+number+'.bmp')
            for j in xrange(1, 6):
                step = '_%d.bmp' % j
                self.filenames.append(self.distortedPath + 'gblurConv/' + number + step)
                self.filenames.append(self.distortedPath + 'wn/' + number + step)
                self.filenames.append(self.distortedPath + 'jpeg/' + number + step)
                self.filenames.append(self.distortedPath + 'jp2k/' + number + step)

        self.pristineImg = load_img(self.src[0])
        self.distortedImg = load_img(self.filenames[0])
        # self.distortedImg.show()
        self.moveImg = 1

        [m, n] = self.distortedImg.size
        self.getMN(m, n)

    def getMN(self, m, n):
        self.row = n/64
        self.col = m/64
        self.x = (m%64)/2
        self.y = (n%64)/2
        self.moveX = 0
        self.moveY = 0
        if (self.moveImg == len(self.filenames)):
            self.moveImg = 0
            self.iteration +-1

        print '\nGetting {}th image...'.format(self.moveImg)
        # print m , n
        # print self.row, self.col
        # print self.x, self.y

    def getNext(self):
        if (self.moveY == self.row):
            if ((self.moveImg-1)/self.basic != self.moveImg/self.basic):
                self.pristineImg = load_img(self.src[self.moveImg/self.basic])
            self.distortedImg = load_img(self.filenames[self.moveImg])
            self.moveImg += 1
            [m, n] = self.distortedImg.size
            self.getMN(m, n)

        x = self.x+self.moveX*64
        y = self.y+self.moveY*64
        # print x, '-', y

        self.real = ToTensor()(self.pristineImg.crop((x, y, x+64, y+64)))
        self.fake = ToTensor()(self.distortedImg.crop((x, y, x+64, y+64)))
        self.moveX += 1

        if (self.moveX == self.col):
            self.moveX = 0
            self.moveY += 1

# patches = patch()
# for i in xrange(8):
#     patches.getNext()