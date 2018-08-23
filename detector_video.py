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
    parser = argparse.ArgumentParser(description='YOLOv3 Video Detection Module')
    parser.add_argument('--video_path',type=str,default='test_video/test.mp4',help='path to video')
    parser.add_argument('--config_path',type=str,default='cfg/yolov3.cfg',help='path to config path')
    parser.add_argument('--weights_path',type=str,default='weights/yolov3.weights',help='path tp weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names',help='path to class label file')
    parser.add_argument('--conf_th',type=float,default=0.8,help='object confidence threshold')
    parser.add_argument('--nms_th',type=float,default=0.4,help='iou threshold for non-maximum suppression')
    parser.add_argument('--batch_size',type=int,default=1,help='batch size')
    parser.add_argument('--n_cpu',type=int,default=8,help='numbers of cpu threads to use during batch generatioin')
    parser.add_argument('--img_size',type=int,default=416,help='size of each image')
    parser.add_argument('--video_output',type=str,default='det_result/test_out.mp4',help='path to store detection results')
    return parser.parse_args()

def get_result(img, orimg_dim, detections, img_size, classes):
    orimg_w = orimg_dim[0]
    orimg_h = orimg_dim[1]
    # h > w , x pading
    pad_x = max(orimg_h - orimg_w, 0) * (img_size / max(orimg_dim))
    pad_y = max(orimg_w - orimg_h, 0) * (img_size / max(orimg_dim))
    # think: resize then pad
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_pred = len(unique_labels)
        colors = pkl.load(open("utils/colors", "rb"))
        bbox_colors = random.sample(colors, n_cls_pred)
        for x1, y1, x2, y2, conf, cls_conf, cls_ind in detections:
            #print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_ind)], cls_conf))

            box_h = ((y2 - y1) / unpad_h) * orimg_h
            box_w = ((x2 - x1) / unpad_w) * orimg_w
            y1 = ((y1 - pad_y // 2) / unpad_h) * orimg_h
            x1 = ((x1 - pad_x // 2) / unpad_w) * orimg_w
            y2 = y1 + box_h
            x2 = x1 + box_w

            color = bbox_colors[int(np.where(unique_labels == int(cls_ind))[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = classes[int(cls_ind)]
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = x1 + t_size[0] + 3, y1 + t_size[1] + 4
            cv2.rectangle(img, (x1, y1), c2, color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def main():
    opt = arg_parse()
    print(opt)


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


    vid = cv2.VideoCapture(opt.video_path)
    if not vid.isOpened():
        raise IOError("couldn't open video.")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    print('video_fps:', video_fps)
    image_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    output_path = opt.video_output
    isOutput = True if output_path != '' else False
    if isOutput:
        out_video = cv2.VideoWriter(output_path, video_FourCC, video_fps, image_size)

    print('\n Detecting......')
    prev_time = time.time()
    all_time = 0
    current_fps = 0
    fps = "FPS: ???"
    while True:
        ret, frame = vid.read()
        if frame is None:
            print('end of video')
            break
        image = np.asarray(frame)
        input_img = prep_image(image, opt.img_size).to(device)
        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, 80, opt.conf_th, opt.nms_th)

        result = get_result(image, image_size, detections[0], opt.img_size, classes)
        # every second calculate fps
        current_time = time.time()
        cost_time = current_time - prev_time
        prev_time = current_time
        all_time += cost_time
        current_fps += 1

        if all_time > 1:
            all_time = all_time - 1
            fps = 'FPS: ' + str(current_fps)
            print('fps now: %d' % current_fps)
            current_fps = 0

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        #cv2.imshow('result', result)
        if isOutput:
            out_video.write(result)

if __name__ == '__main__':
    main()