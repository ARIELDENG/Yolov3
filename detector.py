import torch
from torch.utils.data import DataLoader

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
    parser = argparse.ArgumentParser(description='YOLOv3 Detection Module')
    parser.add_argument('--image_folder',type=str,default='test_examples',help='path to images')
    parser.add_argument('--config_path',type=str,default='cfg/yolov3.cfg',help='path to config path')
    parser.add_argument('--weights_path',type=str,default='weights/yolov3.weights',help='path tp weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names',help='path to class label file')
    parser.add_argument('--conf_th',type=float,default=0.8,help='object confidence threshold')
    parser.add_argument('--nms_th',type=float,default=0.4,help='iou threshold for non-maximum suppression')
    parser.add_argument('--batch_size',type=int,default=1,help='batch size')
    parser.add_argument('--n_cpu',type=int,default=8,help='numbers of cpu threads to use during batch generatioin')
    parser.add_argument('--img_size',type=int,default=416,help='size of each image')
    parser.add_argument('--det_folder',type=str,default='det_result',help='path to store detection results')
    return parser.parse_args()

def main():
    opt = arg_parse()
    print(opt)

#    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load network
    print('Loading networks......')
    model = DarkNet(opt.config_path)
    model.load_weights(opt.weights_path)
    print('Network loaded successfully')
    # load class
    classes = load_classes(opt.class_path)

    model.to(device)
    model.eval()


    if not os.path.exists(opt.det_folder):
        os.makedirs(opt.det_folder)

    dataloader = DataLoader(ImageFolder(opt.image_folder, opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    imgs = []
    img_detections = []

    print('\n Detecting......')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.to(device)

        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_th, opt.nms_th)
            # detections.shape:(bs,7),  x1,y1,x2,y2,conf,max_class_conf,max_class_ind
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time-prev_time)
        prev_time = current_time
        print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        imgs.extend(img_paths)
        img_detections.extend(detections)

    print('image_paths:', imgs)

#    # Bounding-box colors
#    cmap = plt.get_cmap('tab20b')
#    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\n Saving images......')

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        img = cv2.imread(path)

#        img = np.array(Image.open(path))
#        plt.figure()
#        fig, ax = plt.subplots()
#        ax.imshow(img)

        # h > w , x pading
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # think: resize then pad
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_pred = len(unique_labels)
            colors = pkl.load(open("utils/colors", "rb"))
            bbox_colors = random.sample(colors, n_cls_pred)
            for x1, y1, x2, y2, conf, cls_conf, cls_ind in detections:
                print('\t+ Label: %s, Conf: %.5f' %(classes[int(cls_ind)], cls_conf))

                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                y2 = y1 + box_h
                x2 = x1 + box_w

                color = bbox_colors[int(np.where(unique_labels == int(cls_ind))[0])]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = classes[int(cls_ind)]
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = x1 + t_size[0] + 3, y1 + t_size[1] + 4
                cv2.rectangle(img, (x1, y1), c2, color, -1)
                cv2.putText(img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        cv2.imwrite(opt.det_folder + '//%d.png' % img_i, img)

'''
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                         edgecolor=color,facecolor='none')
                # add box
                ax.add_patch(bbox)
                # add label
                plt.text(x1, y1, s=classes[int(cls_ind)], color='white', verticalalignment='top',
                         bbox={'color':color, 'pad':0})

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(opt.det_folder + '//%d.png' % img_i, bbox_inches='tight', pad_inches=0.0)
        plt.close()
'''


if __name__ == '__main__':
    main()