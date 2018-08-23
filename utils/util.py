import torch
import numpy as np
import math


def non_max_suppression(prediction, num_classes, conf_th=0.5, nms_th=0.4):
    # conf<conf_th的置0
    #conf_mask = (prediction[:,:,4]>conf_th).float().unsqueeze(2)
    #prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    # top-left x, top-left y, right-bottom x, right-bottom y
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]

    bs = prediction.shape[0]
    output = [None for _ in range(bs)]
    for ind in range(bs):
        image_pred = prediction[ind]
        conf_mask = (image_pred[:, 4] >= conf_th).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # 每个box的类别最大置信度和最大置信度所属类别
        max_conf, max_conf_ind = torch.max(image_pred[:,5:5+num_classes], 1, keepdim=True)
        # max_conf = max_conf.float().unsqueeze(1)
        # max_conf_ind = max_conf_ind.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:,:5], max_conf.float(), max_conf_ind.float()),1)

        #non_zero_ind = torch.nonzero(image_pred[:,4])
        #image_pred = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        # 获得当前image所有存在的类别
        unique_labels = image_pred[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        # NMS
        for c in unique_labels:
            pred_classc = image_pred[image_pred[:, -1] == c]
            _, conf_sort_index = torch.sort(pred_classc[:,4], descending=True)
            pred_classc = pred_classc[conf_sort_index]
            max_detections = []
            while pred_classc.size(0): # 剩余框>=0
                max_detections.append(pred_classc[0].unsqueeze(0))

                if pred_classc.size(0) == 1:  # 剩余框=1
                    break
                ious = bbox_iou(max_detections[-1], pred_classc[1:])
                # 剩余的框继续nms
                pred_classc = pred_classc[1:][ious < nms_th]

            max_detections = torch.cat(max_detections).data
            output[ind] = max_detections if output[ind] is None else torch.cat((output[ind],max_detections))

    return output



def bbox_iou(box1, box2, x1y1x2y2=True):
    '''
    x1y1x2y2 order
    :param box1:
    :param box2:
    :return:
    '''
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_x1 = torch.max(b1_x1, b2_x1)  # x轴相交左侧的值
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)  # x轴相交右侧的值
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]
    return names



# % VOCap计算方式如下: function ap = VOCap(rec,prec)
# mrec=[0 ; rec ; 1]; % 在召回率列表首尾添加两个值
# mpre=[0 ; prec ; 0];
# for i=numel(mpre)-1:-1:1
#   mpre(i)=max(mpre(i),mpre(i+1)); % 使mpre单调递减
# end
# i=find(mrec(2:end)~=mrec(1:end-1))+1; % 找出召回率产生变化的下标
# ap=sum((mrec(i)-mrec(i-1)).*mpre(i)); % 计算ROC曲线下面积

def computeAP(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size-1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i-1],mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i+1]-mrec[i]) * mpre[i+1])
    return ap



def set_targets(pred_boxes, targets, anchors, dim, num_anchors, num_classes):
    nB = targets.shape[0]   # target.shape:(batch_size, 50, 5)
    nA = num_anchors   # 3
    nC = num_classes   # 80
    ignore_thresh = 0.5

    nGT = 0   # batch中ground truth个数
    nCorrect = 0
    dim = dim # 当前feature map的dim

    mask = torch.zeros(nB, nA, dim, dim)
    conf_mask = torch.ones(nB, nA, dim, dim)
    tx = torch.zeros(nB, nA, dim, dim)
    ty = torch.zeros(nB, nA, dim, dim)
    tw = torch.zeros(nB, nA, dim, dim)
    th = torch.zeros(nB, nA, dim, dim)
    tconf = torch.zeros(nB, nA, dim, dim)
    tcls = torch.zeros(nB, nA, dim, dim, nC)


    for b in range(nB):
        for t in range(targets.shape[1]):
            if targets[b, t].sum() == 0:
                continue
            nGT += 1
            gx = targets[b, t, 1] * dim
            gy = targets[b, t, 2] * dim
            gw = targets[b, t, 3] * dim
            gh = targets[b, t, 4] * dim

            index_X = int(gx)
            index_Y = int(gy)

            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            anchor_box = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # 当前feature map上的3种anchors与gt的iou
            anchor_ious = bbox_iou(gt_box, anchor_box)
            # 任意anchor与gt的iou>0.5但不是最大的都不参与训练
            conf_mask[b, anchor_ious > ignore_thresh] = 0
            # 相当于找出每个gt落在哪个cell
            # 然后找出当前cell与gt的iou最大的anchor
            best_n = np.argmax(anchor_ious)

            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # 注意这里index_Y index_X顺序
            # eg: 坐标(2,0)  矩阵中按行读所以为(0,2)
            pred_box = pred_boxes[b, best_n, index_Y, index_X].unsqueeze(0)

            # 最大iou设置系数1进行坐标和类别预测 即mask
            # 除了不参与训练的其他所有正负样本设置系数为1 即 conf_mask
            mask[b, best_n, index_Y, index_X] = 1
            conf_mask[b, best_n, index_Y, index_X] = 1

            tx[b, best_n, index_Y, index_X] = gx - index_X
            ty[b, best_n, index_Y, index_X] = gy - index_Y
            tw[b, best_n, index_Y, index_X] = math.log(gw/anchors[best_n][0] + 1e-16)
            tw[b, best_n, index_Y, index_X] = math.log(gh/anchors[best_n][1] + 1e-16)

            tcls[b, best_n, index_Y, index_X, int(targets[b, t, 0])] = 1
            tconf[b, best_n, index_Y, index_X] = 1

            iou = bbox_iou(gt_box, pred_box ,x1y1x2y2=False)
            if iou > 0.5:
                nCorrect += 1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls