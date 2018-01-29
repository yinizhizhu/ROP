from torchvision.transforms import Compose, ToTensor, CenterCrop

from torch.utils.data import DataLoader
from dataset import DatasetFromFolder

def input_transform():
    return Compose([
        ToTensor(),
    ])

def target_transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(typeDir):
    return DatasetFromFolder(
        typeDir,
        input_transform=input_transform(),
        target_transform=target_transform())

# training = get_training_set('gblur')
# data = DataLoader(dataset=training)
#
# print len(data)
#
# for i in xrange(2):
#     data.dataset[i]

# st = '%s' % 'ab'
# print st