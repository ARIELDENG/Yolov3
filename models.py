from utils.parse_config import *
from utils.util import set_targets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import defaultdict
import numpy as np
import cv2

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, image_dim):
        super(YoloLayer,self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_dim = image_dim
        self.bbox_attrs = 5 + num_classes
        self.num_classes = num_classes

        self.mse_loss = nn.MSELoss()  # 均方损失函数
        self.bce_loss = nn.BCELoss()  # 二分类交叉熵

        self.lambda_coord = 1

    def forward(self, x, targets=None):
        # x.size = (b,c,w,h)
        bs = x.size(0)     # batch size
        g_dim = x.size(2)   # H , W
        stride = self.image_dim / g_dim  # 缩放倍数
        is_training = targets is not None

#        if x.is_cuda:
#            print('x is cuda')
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs, self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0,1,3,4,2).contiguous()
        # eg: prediction shape: (bs, 3, 13, 13, 85)
        # output
        x = torch.sigmoid(prediction[...,0])  # eg: x.size = (bs,3,13,13,1)
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        conf = torch.sigmoid(prediction[...,4])
        pred_cls = torch.sigmoid(prediction[...,5:])

        # offset
        # att: 这里cy需要t()  cx和cy为距离图像左上角的距离
        cx = torch.linspace(0,g_dim-1,g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors,1,1).view(x.shape).type(FloatTensor)
        cy = torch.linspace(0,g_dim-1,g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors,1,1).view(x.shape).type(FloatTensor)
        scaled_anchors = [(a_w/stride, a_h/stride) for a_w,a_h in self.anchors]
        pw = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        ph = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        pw = pw.repeat(bs, 1).repeat(1,1,g_dim*g_dim).view(w.shape)
        ph = ph.repeat(bs, 1).repeat(1,1,g_dim*g_dim).view(h.shape)

        # pred_bboxes : x,y,w,h  on feature map
        pred_bboxes = FloatTensor(prediction[...,:4].shape)
        pred_bboxes[...,0] = x + cx
        pred_bboxes[...,1] = y + cy
        pred_bboxes[...,2] = torch.exp(w) * pw
        pred_bboxes[...,3] = torch.exp(h) * ph


        # 训练的话直接返回loss
        if is_training:
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = set_targets(pred_bboxes.cpu().data,
                                                                                      targets.cpu().data,
                                                                                      scaled_anchors,
                                                                                      g_dim,
                                                                                      self.num_anchors,
                                                                                      self.num_classes)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            #if x.is_cuda:
             #   self.bce_loss = self.bce_loss.cuda()
              #  self.mse_loss = self.mse_loss.cuda()
            recall = float(nCorrect / nGT) if nGT else 1

            mask = mask.to(device)
            cls_mask = mask.unsqueeze(-1).repeat(1,1,1,1,self.num_classes).type(FloatTensor)
            conf_mask = conf_mask.to(device)

            tx = tx.to(device).requires_grad_(False)
            ty = ty.to(device).requires_grad_(False)
            tw = tw.to(device).requires_grad_(False)
            th = th.to(device).requires_grad_(False)
            tconf = tconf.to(device).requires_grad_(False)
            tcls = tcls.to(device).requires_grad_(False)

            loss_x = self.lambda_coord * self.mse_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.mse_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask)
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            # 这里注意各个具体的类的loss只是为了记录
            # 只会backprop总的loss   所以这里除了总loss外只是使用item()取值
            # 不要用.data   .data返回新的tensor和之前tensor共享内存  .data[0]取值
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall

        else:
            # return x,y,w,h,conf,80   on  input_image_size
            output = torch.cat((pred_bboxes.view(bs,-1,4)*stride, conf.view(bs,-1,1), pred_cls.view(bs,-1,self.num_classes)), -1)
            return output.data



def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()  # just python list

    index = 1
    output_filters = [int(net_info['channels'])]

    for x in blocks[1:]:
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            # use bn without bias
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(output_filters[-1],filters,kernel_size,stride,pad,bias=bias)
            module.add_module('conv_%d' % index, conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('bn_%d' % index, bn)
            if activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_%d' % index, act)

        elif x['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(x['stride']), mode='nearest')
            module.add_module('upsample_%d' % index, upsample)

        elif x['type'] == 'route':
            layers = [int(i) for i in x['layers'].split(',')]
            filters = sum(output_filters[layer_i] for layer_i in layers)
            module.add_module('route_%d' % index, EmptyLayer())

        elif x['type'] == 'shortcut':
            filters = output_filters[int(x['from'])]
            module.add_module('shortcut_%d' % index, EmptyLayer())

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors ]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            num_classes = int(x['classes'])
            img_height = int(net_info['height'])

            yolo_layer = YoloLayer(anchors, num_classes, img_height)
            module.add_module('yolo_%d' % index, yolo_layer)


        index += 1
        module_list.append(module)
        output_filters.append(filters)

    return net_info, module_list

class DarkNet(nn.Module):
    def __init__(self, cfgfile_path):
        super(DarkNet,self).__init__()
        self.blocks = parse_cfg(cfgfile_path)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.seen = 0
        self.header_info = np.array([0,0,0,self.seen,0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']


    def forward(self, x, targets=None):
        is_training = targets is not None
        modules = self.blocks[1:]
        outputs = []
        yolo_output = []
        self.losses = defaultdict(float)


        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type in ['convolutional', 'upsample']:
                x = self.module_list[i](x)
            elif module_type == 'shortcut':
                src = int(module['from'])
                x = outputs[i-1] + outputs[i+src]
            elif module_type == 'route':
                layers = [int(i) for i in module['layers'].split(',')]
                x = torch.cat([outputs[i] for i in layers], 1)  # cat at channel
            elif module_type == 'yolo':
                if is_training:
                    x, *losses = self.module_list[i][0](x, targets=targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:
                    x = self.module_list[i](x)
                yolo_output.append(x)  # loss or prediction
            outputs.append(x)
        self.losses['recall'] /= 3
        return torch.sum(torch.stack(yolo_output,dim=0).squeeze(),dim=0,keepdim=True) if is_training else torch.cat(yolo_output, 1)

    # two types of weights : bn and conv
    def load_weights(self, weights_path):
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        ptr = 0
        for i, module in enumerate(self.blocks[1:]):
            if module['type'] == 'convolutional':
                conv_layer = self.module_list[i][0]
                try:
                    batch_normalize = int(module['batch_normalize'])
                except:
                    batch_normalize = 0
                if batch_normalize == 1:
                    #
                    bn_layer = self.module_list[i][1]
                    num_bn_bias = bn_layer.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_bias]).view_as(bn_layer.bias)
                    ptr += num_bn_bias
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_bias]).view_as(bn_layer.weight)
                    ptr += num_bn_bias
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_bias]).view_as(bn_layer.running_mean)
                    ptr += num_bn_bias
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_bias]).view_as(bn_layer.running_var)
                    ptr += num_bn_bias

                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.data.copy_(bn_running_mean)
                    bn_layer.running_var.data.copy_(bn_running_var)

                else:
                    num_conv_bias = conv_layer.bias.numel()
                    conv_bias = torch.from_numpy(weights[ptr:ptr+num_conv_bias]).view_as(conv_layer.bias)
                    ptr += num_conv_bias
                    conv_layer.bias.data.copy_(conv_bias)

                num_conv_weights = conv_layer.weight.numel()
                conv_weight = torch.from_numpy(weights[ptr:ptr+num_conv_weights]).view_as(conv_layer.weight)
                ptr += num_conv_weights
                conv_layer.weight.data.copy_(conv_weight)

    def save_weights(self, path, cutoff=-1):
        fp = open(path,'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i, module_l in enumerate(self.blocks[1:cutoff]):
            if module_l['type'] == 'convolutional':
                conv_layer = self.module_list[i][0]
                try:
                    batch_normalize = int(module_l['batch_normalize'])
                except:
                    batch_normalize = 0
                if batch_normalize == 1:
                    bn_layer = self.module_list[i][1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



def get_test_input():
    img = cv2.imread('./test_examples/dog-cycle-car.png')
    img = cv2.resize(img,(416,416))   # 读入shape为(h,w,c), c为BGR order
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR转为RGB, 再转为(c,h,w) 即channel first
    img_ = img_[np.newaxis,:,:,:]/255.0    # 添加batch axis
    img_ = torch.from_numpy(img_).float()
#    img_ = Variable(img_)
    return img_


def test_module():
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    model = DarkNet('cfg/yolov3.cfg')
#    model.load_weights('weights/yolov3.weights')
#    t_input = get_test_input()
#    print(t_input)
#    print(t_input.shape)
#    model = model.cuda()
#    t_input = t_input.cuda()
#    pred = model(t_input)
#    print(pred)
#    print(pred.size())  # ([1,10647,85])  10647 = (13*13 + 26*26 + 52*52)*3

    blocks = parse_cfg('cfg/yolov3.cfg')
    net_info, module_list = create_modules(blocks)
    print(blocks)
    print('------------')
    print(module_list)
    pass

if __name__ == '__main__':
    test_module()