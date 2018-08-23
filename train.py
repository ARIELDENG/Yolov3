import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import cv2
import argparse
import os
import random
import time
import datetime
import pickle as pkl

from utils.util import *
from utils.datasets import *
from models import *

import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLOv3 Train Module')
    parser.add_argument('--image_folder',type=str,default='test_examples',help='path to images')
    parser.add_argument('--model_config_path',type=str,default='cfg/yolov3.cfg',help='path to config path')
    parser.add_argument('--data_config_path',type=str,default='data/coco.data',help='path to config path')
    parser.add_argument('--weights_path',type=str,default='weights/yolov3.weights',help='path tp weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names',help='path to class label file')
    parser.add_argument('--conf_th',type=float,default=0.8,help='object confidence threshold')
    parser.add_argument('--nms_th',type=float,default=0.4,help='iou threshold for non-maximum suppression')
    parser.add_argument('--batch_size',type=int,default=16,help='batch size')
    parser.add_argument('--epochs',type=int,default=300,help='number of epochs')
    parser.add_argument('--n_cpu',type=int,default=8,help='numbers of cpu threads to use during batch generatioin')
    parser.add_argument('--img_size',type=int,default=416,help='size of each image')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='dir to save model weights')
    return parser.parse_args()

def main():
    opt = arg_parse()
    print(opt)

    #    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # load class
    classes = load_classes(opt.class_path)

    # data config
    data_config = parse_data_cfg(opt.data_config_path)
    train_path = data_config['train']

    model = DarkNet(opt.model_config_path)
    model.apply(weight_init_normal)

    hyperparams = model.net_info
    learning_rate = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])

    saturation = float(hyperparams['saturation'])
    exposure = float(hyperparams['exposure'])
    hue = float(hyperparams['hue'])
    # ...

    pmodel = nn.DataParallel(model, device_ids=[0, 1])
    pmodel.to(device)
    pmodel.train()

#    model.to(device)
#    model.train()
    # Get dataloader
    init_imgsize = opt.img_size
    dataloader = torch.utils.data.DataLoader(DetDataset(train_path, img_size=init_imgsize, num_workers=2, shuffle=True,
                                                        transform=transforms.Compose([transforms.ToTensor()]),
                                                        train=True,
                                                        seen=pmodel.module.seen),
                                             batch_size=opt.batch_size, num_workers=0)
    # 这里未使用多线程加载数据(TODO:多线程下每隔一定batch改变输入尺寸)
    # some bug to fix

    optimizer = optim.SGD(pmodel.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    flog = open('log.txt','w')
    for epoch in range(opt.epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            input_imgs = imgs.to(device).type(Tensor)
            targets = targets.to(device).type(Tensor)
            #print('input_images.shape:', input_imgs.shape)
            #print('targets.shape:', targets.shape)

            optimizer.zero_grad()
            # sum for multi-gpu
            loss = pmodel(input_imgs, targets).sum()
            loss.backward()
            optimizer.step()

            if batch_i % 10 == 0:
                print('[Epoch %d/%d, Batch %d/%d] [Losses: total %f]'
                      %(epoch+1, opt.epochs, batch_i+1, len(dataloader), loss.item()))
                flog.write('[Epoch %d/%d, Batch %d/%d] [Losses: total %f]'
                      %(epoch+1, opt.epochs, batch_i+1, len(dataloader), loss.item()))
                flog.write('||')
                # print(
                #    '[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                #    (epoch+1, opt.epochs, batch_i+1, len(dataloader),
                #     pmodel.module.losses['x'], pmodel.module.losses['y'], pmodel.module.losses['w'],
                #     pmodel.module.losses['h'], pmodel.module.losses['conf'], pmodel.module.losses['cls'],
                #     loss.item(), pmodel.module.losses['recall']))
                pmodel.module.seen += imgs.size(0)
                print('pmodel.module.seen:', pmodel.module.seen)
                flog.write('pmodel.module.seen: %d' % pmodel.module.seen)
                flog.write('\n')
#             if batch_i % 10 == 0:
#                print(
#                    '[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
#                    (epoch+1, opt.epochs, batch_i+1, len(dataloader),
#                     model.losses['x'], model.losses['y'], model.losses['w'],
#                     model.losses['h'], model.losses['conf'], model.losses['cls'],
#                     loss.item(), model.losses['recall']))
#                model.seen += imgs.size(0)
#                print('pmodel.module.seen:', model.seen)
        # save weights every epoch
        pmodel.module.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))


        if epoch in [200,250]:
            learning_rate *= 0.1
            optimizer = optim.SGD(pmodel.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    flog.close()

if __name__ == '__main__':
    main()