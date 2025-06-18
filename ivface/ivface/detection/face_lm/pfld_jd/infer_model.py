#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: infer_model.py
# @time: 2021/8/26 下午6:00

from ivface.detection.face_detection.yolov5_widerface.infer_model import FaceDetecter
import torch
import numpy as np
import cv2
import os
from ivface.AppBaseModel import BaseModel

__WEIGHTS_FNS__ = {'v1': "flm_pfldmask_v1.onnx", 'v2': "flm_pfldmask_v1.onnx"}

LEFT_EYEBROW_LEFT = [6.825897, 6.760612, 4.402142]
LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
RIGHT_EYEBROW_RIGHT = [-6.825897, 6.760612, 4.402142]
LEFT_EYE_LEFT = [5.311432, 5.485328, 3.987654]
LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
RIGHT_EYE_RIGHT = [-5.311432, 5.485328, 3.987654]
NOSE_LEFT = [2.005628, 1.409845, 6.165652]
NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
MOUTH_RIGHT = [-2.774015, -2.080775, 5.048531]
LOWER_LIP = [0.000000, -3.116408, 6.097667]
CHIN = [0.000000, -7.415691, 4.070434]
landmarks_3D = np.float32([LEFT_EYEBROW_LEFT,
                           LEFT_EYEBROW_RIGHT,
                           RIGHT_EYEBROW_LEFT,
                           RIGHT_EYEBROW_RIGHT,
                           LEFT_EYE_LEFT,
                           LEFT_EYE_RIGHT,
                           RIGHT_EYE_LEFT,
                           RIGHT_EYE_RIGHT,
                           NOSE_LEFT,
                           NOSE_RIGHT,
                           MOUTH_LEFT,
                           MOUTH_RIGHT,
                           LOWER_LIP,
                           CHIN])


class PnpHeadPoseEstimator:
    """ Head pose estimation class which uses the OpenCV PnP algorithm.
        It finds Roll, Pitch and Yaw of the head given a figure as input.
        It uses the PnP algorithm and it requires the dlib library
    """

    def __init__(self, cam_w=128, cam_h=128):
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x

        # Estimated camera matrix values.
        self.camera_matrix = np.float32([[f_x, 0.0, c_x],
                                         [0.0, f_y, c_y],
                                         [0.0, 0.0, 1.0]])
        # Distortion coefficients
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    def get_pitch_yaw_roll(self, landmarks106_2D):
        # TRACKED_POINTS_106 = [33, 37, 42, 46, 66, 70, 75, 79, 57, 63, 84, 90, 93, 16] # 106
        TRACKED_POINTS_106 = [33, 37, 42, 46, 66, 70, 75, 79, 58, 62, 84, 90, 93, 16]  # 106_to_68
        TRACKED_POINTS_106_MOD = [38, 50]
        landmarks_2D = landmarks106_2D[TRACKED_POINTS_106]
        landmarks_2D[1:3] = (landmarks_2D[1:3] + landmarks106_2D[TRACKED_POINTS_106_MOD]) / 2.

        retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                          landmarks_2D,
                                          self.camera_matrix,
                                          self.camera_distortion)
        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        return list(euler_angles)


class LMDetecter(BaseModel):
    def __init__(self, IVFACE_ROOT="./weight",version="v1",device=0):
        '''
            :param version: v1,v2
        '''

        __WEIGHTS_PATH__ = os.path.join(IVFACE_ROOT, 'detection/face_lm/pfld_jd/')
        assert version in list(__WEIGHTS_FNS__.keys()), \
            "version only support {} , your version is {}".format(list(__WEIGHTS_FNS__.keys()), version)
        super(LMDetecter, self).__init__(version)

        self.load_model(__WEIGHTS_PATH__, __WEIGHTS_FNS__,device)
        self.MAX_BATCH_SIZE = 10
        self.filter_est = PnpHeadPoseEstimator()
        self.__first_forward__()

    def __first_forward__(self):
        # 无法直接调用forward()计算最大显存 [随机数输入无法提供人脸框]
        # 退而求其次, 使用[batch, 3, 128, 128]调用self.model.forward()
        print('initialize FaceLM Model >>> PFLD_MASK_onnx {} ...'.format(self.version))

        _ = torch.tensor(self.model.run([self.model.get_outputs()[0].name], {
            self.model.get_inputs()[0].name: np.random.randn(1, 3, 128, 128).astype(np.float32)}))

    def crop_face_and_resize(self, simg, face_locs):
        # xmin ymin xmax ymax
        # TODO check simg range [0-1] or [0-255]

        height, width, _ = simg.shape
        face_crops, mod_face_locs = [], []
        # todo 使用threadpool并行
        for face_loc in face_locs:
            w = face_loc[2] - face_loc[0] + 1
            h = face_loc[3] - face_loc[1] + 1
            cx = face_loc[0] + w // 2
            cy = face_loc[1] + h // 2

            size = int(max([w, h]) * 1.2)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            xx1 = max(0, x1)
            yy1 = max(0, y1)
            xx2 = min(width, x2)
            yy2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = simg[yy1:yy2, xx1:xx2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            img = cv2.resize(cropped, (128, 128))

            face_crops.append(img)
            mod_face_locs.append([x1, y1, size, edy1, edy2, edx1, edx2])

        # 为限制由于人脸数量不确定造成的显存变动, 将预处理置于cpu中操作
        face_crops = np.stack(face_crops, axis=0)  # [n_face, 128, 128, 3]
        # to rgb order then perform normalization
        face_crops = face_crops[:, :, :, (2, 1, 0)]

        return torch.from_numpy(face_crops).float().permute(0, 3, 1, 2) / 255.0, np.array(mod_face_locs)

    def facecrop_resize(self, simg):
        h, w, _ = simg.shape
        face_crop = cv2.resize(simg, dsize=(128, 128)).astype(np.float32)
        # to rgb order then perform normalization
        face_crop = face_crop[:, :, (2, 1, 0)] / 255.

        return torch.from_numpy(face_crop). \
                   permute(2, 0, 1).unsqueeze(0), np.array([0, 0, w, h])

    def forward(self, img, json_data, **kwargs):
        '''

        :param img: BGR 0-255 image
        :param json_data:  坐标字典  ‘face_locs’ [x1,y1,x2,y2]
        :param kwargs:
        :return:
        '''

        angle = kwargs.get('angle', 45)
        posefilter = kwargs.get('posefilter', True)
        maskfilter = kwargs.get('maskfilter', False)

        face_locs = json_data['face_locs']  # [-1, 4]
        num_faces = len(face_locs)
        num_forwards = int(np.ceil(1. * num_faces / self.MAX_BATCH_SIZE))
        bs = int(np.ceil(1. * num_faces / num_forwards))

        face_crops, mod_face_locs = self.crop_face_and_resize(img, face_locs)
        # print(face_crops.size(),mod_face_locs.shape)
        box_rellms = []
        masks = []
        with torch.no_grad():
            for idx in range(num_forwards):
                pred = self.model.run(None, {
                    self.model.get_inputs()[0].name: face_crops[idx * bs: (idx + 1) * bs].numpy()})

                results = torch.tensor(pred[0])
                mask = torch.tensor(pred[1])
                masks.append(mask.argmax(dim=1).squeeze(1))  # n,128,128
                box_rellms.append(results)

        box_rellms = torch.cat(box_rellms, dim=0)

        masks = torch.cat(masks, dim=0).numpy()
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]
        box_rellms = box_rellms.view(num_faces, -1, 2).numpy()

        # angle filter
        ####
        box_rellms_pyr = [abs(self.filter_est.get_pitch_yaw_roll(rellm * 128)[1]) for rellm in box_rellms]
        box_rellms_pyr = np.array(box_rellms_pyr).squeeze(1)
        ####

        mask_index = []
        # mask filter
        filtermasks = masks[box_rellms_pyr < angle]
        for mask in filtermasks:
            left_eye = np.where(mask == 2)
            right_eye = np.where(mask == 3)
            mouth = np.where(mask == 5)

            if len(left_eye[0]) != 0 and len(right_eye[0]) != 0 and len(mouth[0]) != 0:
                mask_index.append(1)
            else:
                mask_index.append(0)
        mask_index = np.array(mask_index)

        img_abslms = np.stack((
            (mod_face_locs[:, 2, None] * box_rellms[:, :, 0] + (mod_face_locs[:, 0, None])),
            (mod_face_locs[:, 2, None] * box_rellms[:, :, 1] + (mod_face_locs[:, 1, None]))), axis=2)

        if posefilter and maskfilter:
            return None, {'img_abslms': img_abslms[box_rellms_pyr < angle][mask_index == 1],
                          "masks": masks[box_rellms_pyr < angle][mask_index == 1],
                          "mod_face_locs": mod_face_locs[box_rellms_pyr < angle][mask_index == 1]}
        elif posefilter and not maskfilter:
            return None, {'img_abslms': img_abslms[box_rellms_pyr < angle],
                          "masks": masks[box_rellms_pyr < angle],
                          "mod_face_locs": mod_face_locs[box_rellms_pyr < angle]}
        else:
            return None, {'img_abslms': img_abslms,
                          "masks": masks,
                          "mod_face_locs": mod_face_locs}


class FaceLMer(object):
    def __init__(self, IVFACE_ROOT="./weight",version="v1",device=0):
        '''

        :param version: v1,v2,v1&v2
        '''
        self.version = version
        assert version in ['v1', 'v2', 'v1&v2'], "version only support v1 v2, your version is {}".format(self.version)
        if self.version != "v1&v2":
            self.det_model = FaceDetecter(IVFACE_ROOT,version,device)
            self.lm_model = LMDetecter(IVFACE_ROOT,version,device)
        else:
            self.det_model = [FaceDetecter(IVFACE_ROOT,"v1",device), FaceDetecter(IVFACE_ROOT,"v2",device)]
            self.lm_model = [LMDetecter(IVFACE_ROOT,"v1",device), LMDetecter(IVFACE_ROOT,"v2",device)]

    def forward(self, img, **kwargs):
        '''

        :param img:  BGR 0-255 ndarray
        :param kwargs:
        :return: 对齐后的人脸
        '''
        if self.version != "v1&v2":
            detection_list = self.det_model.forward(img, **kwargs)
            bboxes = {"face_locs": [x[2:] for x in detection_list]}
            # 有人脸
            if len(bboxes['face_locs']) != 0:
                _, landmarks = self.lm_model.forward(img, bboxes, **kwargs)
                return landmarks
            else:
                return {}
        else:

            detection_list = self.det_model[0].forward(img, **kwargs)
            bboxes = {"face_locs": [x[2:] for x in detection_list]}
            if len(bboxes['face_locs']) != 0:
                # remark 先使用第一个模型检测，检测失败再使用第二个模型检测
                _, landmarks = self.lm_model[0].forward(img, bboxes, **kwargs)

            if len(bboxes['face_locs']) == 0 or len(landmarks['img_abslms']) == 0:

                detection_list = self.det_model[1].forward(img, **kwargs)
                bboxes = {"face_locs": [x[2:] for x in detection_list]}
                if len(bboxes['face_locs']) != 0:
                    _, landmarks = self.lm_model[1].forward(img, bboxes, **kwargs)
                else:
                    landmarks = {}
            return landmarks
