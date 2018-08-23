import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import cv2
import argparse
import os
import random

from utils.util import *
from utils.datasets import *
from models import *

import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLOv3 Eval')
    parser.add_argument('--image_folder',type=str,default='test_examples',help='path to images')
    parser.add_argument('--model_config_path',type=str,default='cfg/yolov3.cfg',help='path to config path')
    parser.add_argument('--data_config_path',type=str,default='data/coco.data',help='path to config path')
    parser.add_argument('--weights_path',type=str,default='checkpoints/7.weights',help='path tp weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names',help='path to class label file')

    parser.add_argument('--iou_th', type=float, default=0.5, help='iou threshold to qualify as detected')
    parser.add_argument('--conf_th',type=float,default=0.5,help='object confidence threshold')
    parser.add_argument('--nms_th',type=float,default=0.45,help='iou threshold for non-maximum suppression')
    parser.add_argument('--batch_size',type=int,default=16,help='batch size')
    parser.add_argument('--n_cpu',type=int,default=8,help='numbers of cpu threads to use during batch generatioin')
    parser.add_argument('--img_size',type=int,default=416,help='size of each image')
    return parser.parse_args()

def main():
    opt = arg_parse()
    print(opt)

    #    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    # load class
    classes = load_classes(opt.class_path)

    # data config
    data_config = parse_data_cfg(opt.data_config_path)
    valid_path = data_config['valid']

    model = DarkNet(opt.model_config_path)
    model.load_weights(opt.weights_path)


    pmodel = nn.DataParallel(model, device_ids=[0, 1])
    pmodel.to(device)
    pmodel.eval()

#    model.to(device)
#    model.train()
    # Get dataloader
    init_imgsize = opt.img_size
    detdataset = DetDataset(valid_path, img_size=init_imgsize, num_workers=2, shuffle=False,
                                                        transform=transforms.Compose([transforms.ToTensor()]),
                                                        train=False,
                                                        seen=pmodel.module.seen)
    dataloader = torch.utils.data.DataLoader(detdataset ,batch_size=opt.batch_size, num_workers=opt.n_cpu)

    print('Computing mAP......')


    APs = []

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        input_imgs = imgs.to(device).type(Tensor)
        targets = targets.to(device).type(Tensor)    # shape (16, 50, 5)
        print('input_images.shape:', input_imgs.shape)
        print('targets.shape:', targets.shape)

        with torch.no_grad():
            output = pmodel(input_imgs)   # shape: torch.size([16,10647,85])
            output = non_max_suppression(output, 80, conf_th=opt.conf_th, nms_th=opt.nms_th)
            # list,  lenth:16(bs)  ,  each:  tensor shape(n, 7)  (eg: n=13)
            # output 7  order: x1 y1 x2 y2 conf max_class_conf max_class  (真实坐标)
            # target 5  order: class x y w h  （比例）
        #
        for sample_i in range(targets.size(0)):
            correct = []

            annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
            detections = output[sample_i]

            if detections is None:
                if annotations.size(0) != 0:
                    APs.append(0)
                continue
            # 按conf降序排列
            detections = detections[np.argsort(-detections[:, 4])]

            if annotations.size(0) == 0:
                correct.extend([0 for _ in range(len(detections))])
            else:
                target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                target_boxes[:, 0] = annotations[:, 1] - annotations[:, 3]/2
                target_boxes[:, 1] = annotations[:, 2] - annotations[:, 4]/2
                target_boxes[:, 2] = annotations[:, 1] + annotations[:, 3]/2
                target_boxes[:, 3] = annotations[:, 2] + annotations[:, 4]/2
                target_boxes *= opt.img_size

                detected = []
                # 判断每张图片经过nms得到的所有框中多少是正确的
                for *pred_bbox, conf, cls_conf, cls in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    iou = bbox_iou(pred_bbox, target_boxes)
                    best_i = np.argmax(iou)

                    if iou[best_i] > opt.iou_th and cls == annotations[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # 计算mAP
            # PASCAL VOC challenge计算方法
            # 假设这N个样本中有M个正例,那么我们会得到M个recall值,
            # 对于每个recall值r,我们可以计算出对应（r' > r）的最大precision
            # 然后对这M个precision值取平均即得到最后的AP值

            # 对于COCO数据集中设置IOU=0.5时得到AP50或者(AP@0.5)
            # AP75或者(AP@0.75为严格模式)
            # APsmall  APmedium  APlarge对应 ( ,32²)  (32²,96²)  (96², )

            true_positives = np.array(correct)
            false_positives = 1 - true_positives

            true_positives = np.cumsum(true_positives)
            false_positives = np.cumsum(false_positives)

            recall = true_positives/annotations.size(0) if annotations.size(0) else true_positives
            precision = true_positives/np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            AP = computeAP(recall, precision)
            APs.append(AP)

            APMean = sum(APs)/len(APs)
            print('+ Sample [%d/%d] AP: %.4f  (%.4f)' %(len(APs), len(detdataset), AP, APMean))
    APMean = sum(APs) / len(APs)
    print('Mean Average Precision: %.4f' % np.mean(APMean))



if __name__ == '__main__':
    main()