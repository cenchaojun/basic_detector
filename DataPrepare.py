import os
import re
from torch.utils import data
from PIL import Image
from torchvision import transforms
import PIL as pil
import json

#这是另外一种方法
from torchvision.datasets import ImageFolder

# label_file_path为空字符串的时候就不返回label了。
class DefaultDataset(data.Dataset):
    # load_index 需要加载的样本下标
    # image_folder: 图片文件夹
    # transform: transforms类型的变量，用transforms.Compose定义
    def __init__(self, index_file_path,
                 transform: transforms=None, load_index=None):
        super(data.Dataset, self).__init__()
        self.load_index = load_index
        self.transform = transform
        self.__has_label = True
        # 加载图像，标签
        if index_file_path:
            self.label_loader = LabelLoader(index_file_path, load_index)
            self.image_loader = ImageLoader(index_file_path, load_index)
            if len(self.image_loader) != len(self.label_loader):
                raise Exception('Number of label images does not match')
        else:
            self.image_loader = ImageLoader(index_file_path, load_index)
            self.__has_label = False

    def __getitem__(self, index):
        img = self.image_loader[index]
        if self.transform:
            img = self.transform(img)
        if self.__has_label:
            label = self.label_loader[index]
            return [img, label]
        else:
            return [self.image_loader.file_names[index], img]

    def __len__(self):
        return len(self.image_loader)

class BasicLoader(object):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise Exception('%s FILE does not Exist'% file_path)
        self.data = None
        self.path = file_path

    def __getitem__(self, index):
        raise NotImplementedError

class LabelLoader(BasicLoader):
    def __init__(self, index_file, load_index=None):
        super(LabelLoader, self).__init__(index_file)
        self.labels = []
        with open(index_file, 'r') as f:
            self.data = json.load(f)
        for [i, info] in enumerate(self.data):
            self.labels.append([info['dets'], info['img_id']])
            if i % int(len(self.data) / 15):
                print('load %d / %d' % (i, len(self.data)))
        print('load label Done!')

        if load_index:
            try:
                self.labels = [self.labels[i] for i in load_index]
            except IndexError:
                for i in load_index:
                    if i > len(self.labels):
                        print('out index!!%d' % i)
                raise Exception('load_index in LabelLoader out of range')

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def __add__(self, other):
        return

# INPUT: 索引文件， 截取下标列表
# load_index 表示的是截取的下标（为了产生训练集与验证集），使用range表示，如range(1000)
class ImageLoader(BasicLoader):
    def __init__(self, index_file, load_index=None):
        super(ImageLoader, self).__init__(index_file)
        self.file_names: list = []
        with open(index_file, 'r') as f:
            self.data = json.load(f)
        for info in self.data:
            self.file_names.append(info['file_path'])

        if load_index:
            try:
                self.file_names = [self.file_names[i] for i in load_index]
            except IndexError:
                raise Exception('load_index in ImageLoader out of range')

    def __getitem__(self, index):
        img_name = self.file_names[index]
        img = pil.Image.open(img_name)
        # img = transforms.ToTensor()(img) # tensor转变
        return img

    def __len__(self):
        return len(self.file_names)


transform_train = transforms.Compose([
    transforms.Resize([240, 260]),
    transforms.RandomCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

if __name__ == '__main__':
    import torch

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = DefaultDataset('D:/DataBackup/VOC2012/VOC2012_index.json', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
    for data in trainloader:
        print(data[1])
        a = 0
    # index_path = './PIE_DATA/train_label.txt'
    # a = ImageLoader(index_path)
    # b = a[0]
    # v = 0

