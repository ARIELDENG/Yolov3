import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader

# 继承pytorch的torch.utils.data.Dataset方法构建数据集迭代器
# 后续使用torch.utils.data.DataLoader迭代数据集
class ImageFolder(Dataset):
    '''
    需要重载__len__和__getitem__方法
    __len__返回数据集大小
    __getitem__支持从0到len(self)的整数索引
    '''
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.image_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(Image.open(img_path))
        h, w, c = img.shape
        dim_diff = np.abs(h - w)
        # h > w, left and rigth padding
        # h < w, upper and lower padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        if h <= w:
            pad = ((pad1, pad2), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad1, pad2), (0, 0))

        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
        input_img = resize(input_img, (*self.image_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        return img_path, input_img


    def __len__(self):
        return len(self.files)

# TODO
# when training
# 每10个batches  模型随机选择一种输入尺寸   都是32倍数
# dim = (rand() % 10 + 10) * 32


# TODO
# when training
# data argument
# 抖动  需要同时修正label位置
# 色彩调整
class DetDataset(Dataset):
    def __init__(self, img_path, img_size=None, num_workers=None, shuffle=False, transform=None, train=False, seen=0):
        with open(img_path, 'r') as f:
            self.imgfiles = f.readlines()
        if shuffle:
            random.shuffle(self.imgfiles)
        self.labelfiles = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.imgfiles]
        self.image_shape = (img_size, img_size)
        self.max_objects = 50
        self.transform = transform
        self.train = train
        self.seen = seen
        self.num_workers = num_workers

        self.jitter = 0.3      # 抖动
        self.hue = 0.1         # 色差偏差最大值
        self.saturation = 1.5  # 色彩饱和度缩放最大值
        self.exposure = 1.5    # 色彩明亮缩放最大值
        #  水平翻转


    def __getitem__(self, index):
        img_path = self.imgfiles[index % len(self.imgfiles)].rstrip()
        img = np.array(Image.open(img_path))  # channel last

        while len(img.shape) != 3:
            index += 1
            img_path = self.imgfiles[index % len(self.imgfiles)].rstrip()
            img = np.array(Image.open(img_path))


        # if self.train and index % 64 == 0:
        #     if self.seen < 4000*64:
        #         width = 13 * 32
        #         self.image_shape = (width, width)
        #     elif self.seen < 8000*64:
        #         width = (random.randint(0,3) + 13) * 32
        #         self.image_shape = (width, width)
        #     elif self.seen < 12000*64:
        #         width = (random.randint(0,5) + 13) * 32
        #         self.image_shape = (width, width)
        #     elif self.seen < 16000*64:
        #         width = (random.randint(0,7) + 13) * 32
        #         self.image_shape = (width, width)
        #     elif self.seen < 20000*64:
        #         width = (random.randint(0,9) + 13) * 32
        #         self.image_shape = (width, width)

        if self.train and self.seen % 640 == 0:
            width = (random.randint(0, 9) + 13) * 32
            self.image_shape = (width, width)

        if self.train:
            # TODO
            # data argument
            h, w, c = img.shape
            dim_diff = np.abs(h - w)
            # h > w, left and rigth padding
            # h < w, upper and lower padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            if h <= w:
                pad = ((pad1, pad2), (0, 0), (0, 0))
            else:
                pad = ((0, 0), (pad1, pad2), (0, 0))
            # add padding
            input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
            padded_h, padded_w, _ = input_img.shape
            # resize
            input_img = resize(input_img, (*self.image_shape, 3), mode='reflect')
            input_img = np.transpose(input_img, (2, 0, 1))

            # ------
            # labels
            # ------

            # label order:class x, y, w, h
            labels = None
            label_path = self.labelfiles[index % len(self.imgfiles)].rstrip()
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path).reshape(-1, 5)

                # origin x1, y1, x2, y2坐标
                x1 = w * (labels[:, 1] - labels[:, 3] / 2)
                y1 = h * (labels[:, 2] - labels[:, 4] / 2)
                x2 = w * (labels[:, 1] + labels[:, 3] / 2)
                y2 = h * (labels[:, 2] + labels[:, 4] / 2)

                # Adjust for padding
                x1 += pad[1][0]
                y1 += pad[0][0]
                x2 += pad[1][0]
                y2 += pad[0][0]

                # final ratios
                labels[:, 1] = ((x1 + x2) / 2) / padded_w
                labels[:, 2] = ((y1 + y2) / 2) / padded_h
                labels[:, 3] *= w / padded_w
                labels[:, 4] *= h / padded_h
            img_labels = np.zeros((self.max_objects, 5))
            if labels is not None:
                img_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        else:
            h, w, c = img.shape
            dim_diff = np.abs(h - w)
            # h > w, left and rigth padding
            # h < w, upper and lower padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            if h <= w:
                pad = ((pad1, pad2), (0, 0), (0, 0))
            else:
                pad = ((0, 0), (pad1, pad2), (0, 0))
            # add padding
            input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
            padded_h, padded_w, _ = input_img.shape
            # resize
            input_img = resize(input_img, (*self.image_shape, 3), mode='reflect')
            input_img = np.transpose(input_img, (2, 0, 1))

            # ------
            # labels
            # ------

            # label order: x, y, w, h
            labels = None
            label_path = self.labelfiles[index % len(self.imgfiles)].rstrip()
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path).reshape(-1, 5)

                # origin x1, y1, x2, y2坐标
                x1 = w * (labels[:, 1] - labels[:, 3]/2)
                y1 = h * (labels[:, 2] - labels[:, 4]/2)
                x2 = w * (labels[:, 1] + labels[:, 3]/2)
                y2 = h * (labels[:, 2] + labels[:, 4]/2)

                # Adjust for padding
                x1 += pad[1][0]
                y1 += pad[0][0]
                x2 += pad[1][0]
                y2 += pad[0][0]

                # final ratios
                labels[:, 1] = ((x1 + x2) / 2) / padded_w
                labels[:, 2] = ((y1 + y2) / 2) / padded_h
                labels[:, 3] *= w / padded_w
                labels[:, 4] *= h / padded_h
            img_labels = np.zeros((self.max_objects, 5))
            if labels is not None:
                img_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

        self.seen += 1
        #print('self.seen:', self.seen)
        return img_path, input_img, img_labels

    def __len__(self):
        return len(self.imgfiles)


#class MultiTrainData(object):
#    def __init__(self, batch_size):
#        self.batch_size = batch_size
#        self.image_size = 416
#        self.dataset = None

    # def build(self):
    #     self.dataset = DetDataset(train_path, self.image_size)
    #     dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    #
    # def next(self):
    #     pass



def prep_image(img, image_size):
    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # h > w, left and rigth padding
    # h < w, upper and lower padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    if h <= w:
        pad = ((pad1, pad2), (0, 0), (0, 0))
    else:
        pad = ((0, 0), (pad1, pad2), (0, 0))

    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
    input_img = resize(input_img, (image_size, image_size, 3), mode='reflect')
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = torch.from_numpy(input_img).float().unsqueeze(0)
    return input_img
