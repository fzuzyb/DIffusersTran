#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: infer_model.py
# @time: 2023/2/27 下午2:18

import os
import cv2
import numpy as np
import torch
import math
import torchvision
import time
from ivface.AppBaseModel import BaseModel


__WEIGHTS_FNS__ = {'v1': "fd_yolov5_v1.onnx", 'v2': "fd_yolov5_v2.onnx"}


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


class FaceDetecter(BaseModel):
    def __init__(self, IVFACE_ROOT="./weight",version="v1",device=0):
        __WEIGHTS_PATH__ = os.path.join(IVFACE_ROOT,'detection/face_detection/yolov5_widerface/')
        assert version in list(__WEIGHTS_FNS__.keys()), \
            "version only support {} , your version is {}".format(list(__WEIGHTS_FNS__.keys()), version)
        super(FaceDetecter, self).__init__(version)
        self.load_model(__WEIGHTS_PATH__, __WEIGHTS_FNS__,device)
        # TODO 支持多个版本模型的选择
        self.stride = 32  # models stride
        self.names = ["face"]  # get class names
        self.conf_thres = 0.5
        self.iou_conf = 0.5
        self.__first_forward__()

    def __first_forward__(self):
        print('initialize FaceDetection Model >>> YOLOv5 onnx {}...'.format(self.version))

        _ = torch.from_numpy(
            np.array(self.model.run([self.model.get_outputs()[0].name],
                                    {self.model.get_inputs()[0].name: np.random.randn(1, 3, 640, 640).astype(
                                        np.float32)})))[0]

    def pre_process(self, img):
        """
        :param img:  ndarray BRG [0-255]
        :return: [[cls conf,xmin ymin xmax ymax]]
        """
        img0 = img.copy()
        imgsz = check_img_size([640, 640], self.stride)
        # Padded resize
        img = letterbox(img, imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        return img, img0, imgsz

    def post_process(self, pred, img, img0):

        pred = non_max_suppression(pred, self.conf_thres, iou_thres=self.iou_conf, classes=None, agnostic=False,
                                   max_det=1000)

        out_lists = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det.numpy()):
                    out_txt = [int(cls), float(conf), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    out_lists.append(out_txt)

        return out_lists

    def forward(self, img, **kwargs):
        '''
        :param img: BGR NDArray 0-255
        :return: 检测出来的人脸 List, 变换参数 List [class,conf,x1,y1,x2,y2]
        '''
        self.conf_thres = kwargs.get('conf', 0.4)
        self.iou_conf = kwargs.get('iou_conf', 0.5)
        self.max_detfaces = kwargs.get('max_detfaces', 100)
        img, img0, imgsz = self.pre_process(img)

        pred = torch.from_numpy(
            np.array(self.model.run([self.model.get_outputs()[0].name],
                                    {self.model.get_inputs()[0].name: img.numpy()})))[0]
        out = self.post_process(pred, img, img0)
        out = np.array(out).astype(np.int32)
        face_locs = np.array([x[2:] for x in out])
        # 通过face_locs的框面积进行排序 选择最大面积的max_detfaces个框
        # max_detfaces = max_detfaces
        if len(face_locs) > 0:
            areas = (face_locs[:, 2] - face_locs[:, 0]) * (face_locs[:, 3] - face_locs[:, 1])
            idxes = np.argsort(-areas, axis=0)[: self.max_detfaces]
            out = out[idxes].tolist()

            return out
        return []


def vis_bbox(img, bboxes, labels=None, scores=None):
    '''

    :param img: HWC [0-255] BGR
    :param bboxes: ndarrdy [n,4]
    :param labels: ndarray [n,]
    :param scores: ndarray [n,]
    :return:
    '''
    label_names = ['bg'] + ['face']

    # input valid

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of label must be same as that of bbox')
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of score must be same as that of bbox')

    # to avoid error EXPECTED PTR＜CV::UMAT＞ FOR ARGUMENT ‘img‘
    imgb = img.copy()

    for i, bbox in enumerate(bboxes):

        caption = []

        colorobject = (255, 0, 0)
        colortext = (255, 255, 255)
        colortextrec = (0, 255, 0)

        #  (int(cmap[labels[i]][0]),int(cmap[labels[i]][1]),int((cmap[labels[i]][2]))
        cv2.rectangle(imgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      colorobject, 2)
        if labels is not None:
            label = labels[i]
            caption.append(label_names[label])
        if scores is not None:
            score = scores[i]
            caption.append('{:.2f}'.format(score))

        # int(bbox[1])+10 to make text in bbox
        text_size = cv2.getTextSize(": ".join(caption), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(imgb, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0]) + (text_size[0][0]), int(bbox[1]) + text_size[0][1]),
                      colortextrec, -1)
        cv2.putText(imgb, ": ".join(caption), (int(bbox[0]), int(bbox[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colortext, 1)

    return imgb


if __name__ == '__main__':

    inputdir = "/home/bobo/Temp_new/ZYB/IV_WORKING/dataset/TEST/Pseries/original100_valid_resolution"
    yolo_detecter = FaceDetecter(version="v1")
    for img_path in os.listdir(inputdir):
        img = cv2.imread(
            os.path.join(inputdir, img_path))

        out_list = yolo_detecter.forward(img, conf=0.4, iou_conf=0.5)

        bboxes = [x[2:] for x in out_list]
        lables = [x[0] + 1 for x in out_list]
        imgvis = vis_bbox(img, bboxes, lables)
        cv2.imshow("vis", imgvis)
        cv2.waitKey(0)
