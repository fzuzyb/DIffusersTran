#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: infer_model.py
# @time: 2/27/23 6:09 PM


from ivface.detection.face_lm.pfld_jd.infer_model import FaceLMer
import cv2
from functools import partial
import numpy as np
import os
import PIL.Image as Image
from skimage import transform as trans

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

ffhq_src = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32)


def coordinate_valid_and_pad(quad_genhao_r, quad_r, img):
    '''

    Args:
        # quad cor
        # 0  2
        # 1  3
        quad: coordinator of image [left ; bottom ; left ; right]
        img: image that will be cropped

    Returns:
        img : padded image
        pad_params :

    '''

    h, w = img.shape[0:2]

    top = quad_genhao_r[0][1]
    bottom = quad_genhao_r[1][1]
    left = quad_genhao_r[0][0]
    right = quad_genhao_r[2][0]

    img = img[max(top, 0):min(bottom, h), max(left, 0):min(right, w)]
    pad_left, pad_top, pad_right, pad_bottom = max(-left, 0), max(-top, 0), max(right - w, 0), max(
        bottom - h, 0)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    top = quad_r[0][1]
    bottom = quad_r[1][1]
    left = quad_r[0][0]
    right = quad_r[2][0]
    pad_left, pad_top, pad_right, pad_bottom = max(-left, 0), max(-top, 0), max(right - w, 0), max(
        bottom - h, 0)

    return img, (pad_top, pad_bottom, pad_left, pad_right)


def cal_rotation_angle(eye_to_mouth, eye_to_eye, eye_avg):
    '''

    :param eye_to_mouth:
    :param eye_to_eye:
    :param eye_avg:
    :return:
    '''
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    opp = quad[1][1] - quad[2][1]
    adj = quad[1][0] - quad[2][0]
    angle = np.arctan2(opp, adj) * 180 / np.pi

    return angle - 180


def face_harmonize_circle(bg_crop, face_crop, center, radius, qsize_scale=0.84):
    qsize = np.int32(qsize_scale * radius)
    # qsize = np.int(np.round(np.sqrt(2) * 1 / start_margin * radius))
    # qsize = np.int(0.98 * radius)
    center_x = center[0]
    center_y = center[1]

    length = radius - qsize

    x, y = np.meshgrid(np.arange(2 * radius + 1), np.arange(2 * radius + 1))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)  # [2 * radius, 2 * radius]
    r = np.round(r).astype(np.int32)

    alpha = np.where(r > qsize, (r - qsize) / length, 0.)[:, :, np.newaxis]
    alpha[r > radius] = 1.0

    bg_crop = (bg_crop * alpha + face_crop * (1 - alpha)).astype(np.uint8)

    return bg_crop


def face_jd106_alignment(lms, lr, upscale, **kwargs):
    """

    Args:
        hr: input high resolution image [HWC 0-255] BGR
        lms: high resolution landmarks  [n,106,2[x,y]]

    Returns:

    """

    # resize lr to hr
    radio = kwargs.get('radio', 1.0)

    def _face_alignment(hr, lm):
        """

        Args:
            hr: input high resolution image [HWC 0-255] BGR
            lm: high resolution landmark

        Returns:

        """

        # get hr size
        # get hr size
        H, W, _ = hr.shape
        if len(lm) == 106:
            lm_eye_left = lm[66: 74, :]  # left-clockwise
            lm_eye_right = lm[75: 83, :]  # right-clockwise
            lm_mouth_outer = lm[84: 96, :]  # mouth-clockwise
        else:
            lm_eye_left = lm[36: 41, :]  # left-clockwise
            lm_eye_right = lm[42: 47, :]  # right-clockwise
            lm_mouth_outer = lm[48: 60, :]  # mouth-clockwise
        eye_left_center = np.mean(lm_eye_left, axis=0)  # center of left eye
        eye_right_center = np.mean(lm_eye_right, axis=0)  # center of right eye

        eye_avg = (eye_left_center + eye_right_center) * 0.5  # center between left and right eye
        eye_to_eye = eye_right_center - eye_left_center  # distance between left and right eye

        mouth_left = lm_mouth_outer[0]  # left mouth point
        mouth_right = lm_mouth_outer[6]  # right mouth point

        mouth_avg = (mouth_left + mouth_right) * 0.5  # center of mouth
        eye_to_mouth = mouth_avg - eye_avg  # distance between eye and mouth
        face_center = np.round(eye_avg + eye_to_mouth * 0.1).astype(np.int32)
        radius = max(np.hypot(*eye_to_eye), np.hypot(*eye_to_mouth))
        radius = int(np.round(radius * 1.9) * radio)

        if radius < 16 or radius > 2048:
            return None, None

        else:
            angle = cal_rotation_angle(eye_to_mouth, eye_to_eye, eye_avg)
            first_pad_radius = int(1.4 * radius)
            crop_pad = int(1.4 * radius) - radius
            # get first pad coordinator
            quad_genhao_r = np.stack(
                ([(face_center[0] - first_pad_radius), (face_center[1] - first_pad_radius)],  # top left
                 [(face_center[0] - first_pad_radius), (face_center[1] + first_pad_radius + 1)],  # bottom left
                 [(face_center[0] + first_pad_radius + 1), (face_center[1] - first_pad_radius)],  # top right
                 [(face_center[0] + first_pad_radius + 1), (face_center[1] + first_pad_radius + 1)]))  # bottom right

            quad_r = np.stack(([(face_center[0] - radius), (face_center[1] - radius)],  # top left
                               [(face_center[0] - radius), (face_center[1] + radius + 1)],  # bottom left
                               [(face_center[0] + radius + 1), (face_center[1] - radius)],  # top right
                               [(face_center[0] + radius + 1), (face_center[1] + radius + 1)]))  # bottom right

            # get real coordinator
            tx, ty = face_center[0], face_center[1]
            # get the first pad image and the pad params
            pad1_img, pad_params = coordinate_valid_and_pad(quad_genhao_r, quad_r, hr)
            # pad second to rotation
            rot_mat = cv2.getRotationMatrix2D(center=(first_pad_radius, first_pad_radius), angle=angle, scale=1)

            # rotation image
            rotation_img = cv2.warpAffine(pad1_img, rot_mat, (pad1_img.shape[1], pad1_img.shape[0]),
                                          flags=cv2.INTER_LANCZOS4)  # 旋转图像
            # crop image to enhance face

            rotation_img = rotation_img[crop_pad:-crop_pad,
                           crop_pad:-crop_pad, :]

            assert radius * 2 + 1 == rotation_img.shape[
                0], 'radius shape is not ematch ratation_img shape {}!={}'.format(
                radius * 2 + 1, rotation_img.shape)
        # assert rotation_img.shape[0]%2!=0,  'rotation img size is not odd {}'.format(rotation_img.shape)

        return rotation_img, [angle, (tx, ty), rotation_img.shape[0:2], radius, pad_params]

    lr_scale = cv2.resize(lr, dsize=None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    align_lqs = []
    tfs = []

    for lm in lms:
        # align face and get transformation params
        lq, tf = _face_alignment(lr_scale, lm)

        # resize align face to input resolution so that the SR face model can process
        if lq is not None and lq.shape[0] // 2 > 24:
            align_lqs.append(lq)
            # align_lqs.append(lq)
            tfs.append(tf)
        # break
    return align_lqs, tfs


def trans_points2d_v2(pts, M):
    new_pts = np.dot(np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=np.float32)), axis=1), M.T)
    return new_pts


def fivpoints_reverse_face(hr, hr_faces, tfms, exp=-0.15):
    # remark 采用原尺寸并对tfms进行相应变换
    hr_h, hr_w, _ = hr.shape
    for hr_face, tfm in zip(hr_faces, tfms):
        final_frame_size = hr_face.shape[0]
        IM = cv2.invertAffineTransform(tfm)
        corner_pts = trans_points2d_v2(
            np.array([[0, 0], [0, final_frame_size - 1], [final_frame_size - 1, final_frame_size - 1],
                      [final_frame_size - 1, 0]]), IM)
        lt, rb = corner_pts.min(axis=0), corner_pts.max(axis=0)
        lt_exp = np.round((1 + exp) * lt - exp * rb).astype(np.int32)
        exp_size = np.round((1 + 2 * exp) * (rb - lt)).astype(np.int32)
        exp_size = exp_size + exp_size % 2 + 1
        rb_exp = lt_exp + exp_size
        x1 = max(0, lt_exp[0])
        y1 = max(0, lt_exp[1])
        x2 = min(hr_w, rb_exp[0])
        y2 = min(hr_h, rb_exp[1])

        padx1 = max(0, -lt_exp[0])
        pady1 = max(0, -lt_exp[1])
        padx2 = max(0, rb_exp[0] - hr_w)
        pady2 = max(0, rb_exp[1] - hr_h)

        hr_patch = hr[y1:y2, x1:x2]
        if pady1 != 0 or pady2 != 0 or padx1 != 0 or padx2 != 0:
            hr_patch = cv2.copyMakeBorder(hr_patch, pady1, pady2, padx1, padx2,
                                          borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        IM[:, -1] -= lt_exp
        swap_t = cv2.warpAffine(hr_face, IM, exp_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        harmonize_back_image = face_harmonize_circle(
            hr_patch, swap_t,
            (exp_size[0] // 2, exp_size[1] // 2),
            exp_size[0] // 2)
        # harmonize_back_image = swap_t

        edx1 = max(0, -lt_exp[0])
        edy1 = max(0, -lt_exp[1])
        harmonize_back_image = harmonize_back_image[edy1: edy1 + (y2 - y1), edx1: edx1 + (x2 - x1)]

        hr[y1:y2, x1:x2] = harmonize_back_image  # (mask_t * swap_t + (1 - mask_t) * full_frame[y1:y2, x1:x2])

    return hr.astype(np.uint8)


def ema_reverse_face(hr, hr_faces, tfs, **kwargs):
    H, W = hr.shape[0], hr.shape[1]

    for idx, (hq_img, tform) in enumerate(zip(hr_faces, tfs)):

        # resize img to before rotation

        hq_img = Image.fromarray(hq_img).resize(tform[2], Image.Resampling.LANCZOS)
        hq_img = np.array(hq_img)

        rot_mat = cv2.getRotationMatrix2D(center=(tform[3], tform[3]),
                                          angle=-tform[0], scale=1)
        # # rotate image  W,H
        back_img = cv2.warpAffine(hq_img, rot_mat, (hq_img.shape[1], hq_img.shape[0]), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # make border so that the outlines coordinator can be done easily
        if tform[4][0] != 0 or tform[4][1] != 0 or tform[4][2] != 0 or tform[4][3] != 0:
            hr = cv2.copyMakeBorder(hr, tform[4][0], tform[4][1], tform[4][2], tform[4][3],
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        left_corner = tform[3]
        top = max((tform[1][1] - left_corner), 0)
        left = max((tform[1][0] - left_corner), 0)

        assert hr[top:top + back_img.shape[0], left:left + back_img.shape[1], :].shape[0] == back_img.shape[
            0], 'back img size not match'

        harmonize_back_image = face_harmonize_circle(
            hr[top:top + back_img.shape[0], left:left + back_img.shape[1], :],
            back_img,
            (tform[3], tform[3]),
            tform[3])

        hr[top:top + back_img.shape[0], left:left + back_img.shape[1], :] = harmonize_back_image
        # crop the image to original size of hr image
        top = max(tform[4][0], 0)
        down = tform[4][1] if tform[4][1] != 0 else -(top + H)
        left = max(tform[4][2], 0)
        right = tform[4][3] if tform[4][3] != 0 else -(left + W)
        hr = hr[top:-down, left:-right, :]

    return hr


class FaceAlignment(object):
    def __init__(self, IVFACE_ROOT="./weight",version='v1',device=0):
        self.DetLMer = FaceLMer(IVFACE_ROOT,version,device)
        self.img_abslms = None
        self.img_modlms = None
    def forward(self, img, **kwargs):
        '''
        :param img: BGR NDArray 0-255
        :return: 对齐后的人脸 List, 变换参数 List
        '''
        scale = kwargs.get('scale', 1)
        img_size = kwargs.get('img_size', None)
        alignment_mode = kwargs.get('alignment_mode', 'ema')
        landmarks = self.DetLMer.forward(img, **kwargs)
        # 有人脸关键点
        if landmarks:
            img_abslms = landmarks['img_abslms']
            img_modlms = landmarks['mod_face_locs']
            img_abslms = np.round(img_abslms).astype(np.int32)
            if alignment_mode == "ema":
                img_abslms = img_abslms * scale
                faces, transforms = face_jd106_alignment(img_abslms, img, scale, **kwargs)
            else:
                faces, transforms = face_align_2dx106(img_abslms, img, image_size=img_size, mode=alignment_mode)
            # for return image_abslms
            self.img_abslms = img_abslms
            self.img_modlms = img_modlms
            assert alignment_mode in ['ema', 'arcface',
                                      'ffhq'], "only 'ema','arcface','ffhq' support but current alignment mode is {}".format(
                alignment_mode)
            # if kwargs.get('return_abslms', False):
            #     return faces, transforms,self.img_abslms
            return faces, transforms
        # 无人脸
        else:
            self.img_abslms = []
            self.img_modlms = []
            return [], []

    def get_img_abslms(self):
        return self.img_abslms

    def get_img_modlms(self):
        return self.img_modlms

    def backward(self, img, faces, tfs, **kwargs):

        """

        Args:
            img: img ndarray0-255 BGR
            faces: face list
            tfs: [angle,(tx,ty),rotation_img.shape[0:2],radius,pad_params[top down left right]]/M

        Returns:

        """
        alignment_mode = kwargs.get('alignment_mode', 'ema')
        if alignment_mode == "ema":
            return ema_reverse_face(img, faces, tfs, **kwargs)
        else:
            return fivpoints_reverse_face(img, faces, tfs)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    if mode == 'arcface':
        src = arcface_src * image_size / 112
    else:
        src = ffhq_src * image_size / 512
    tform.estimate(lmk, src)  # 求lmk -> arcface_src的相似变换矩阵M
    M = tform.params[0:2, :]

    return M


def norm_crop(landmark, img, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def flm2dx106_to_flm2dx5(flms2dx106):
    assert flms2dx106.ndim == 3 and flms2dx106.shape[1] == 106, \
        'shape of lms must be [bs, 106, 2], {} not supported'.format(flms2dx106.shape)
    flms2dx5 = np.zeros((flms2dx106.shape[0], 5, 2), dtype=np.float32)

    flms2dx5[:, 0] = np.mean(flms2dx106[:, 66: 74, :], axis=1)
    flms2dx5[:, 1] = np.mean(flms2dx106[:, 75: 83, :], axis=1)
    # remark method1 偏左/偏右+偏下
    # flms2dx5[:, 2] = (np.mean(flms2dx106[:, 57: 64, :], axis=1) + flms2dx106[:, 54, :]) / 2.
    # remark method2 偏左/偏右+偏上
    # flms2dx5[:, 2, 0] = (np.mean(flms2dx106[:, 57: 64, 0], axis=1) + flms2dx106[:, 54, 0]) / 2.
    # flms2dx5[:, 2, 1] = flms2dx106[:, 54, 1]
    # remark method3 偏上
    # flms2dx5[:, 2] = flms2dx106[:, 54]
    # remark method4
    flms2dx5[:, 2, 0] = flms2dx106[:, 54, 0]
    flms2dx5[:, 2, 1] = (np.mean(flms2dx106[:, 57: 64, 1], axis=1) * 0.15 + flms2dx106[:, 54, 1] * 0.85)
    flms2dx5[:, 3] = (flms2dx106[:, 84, :] * 0.8 + flms2dx106[:, 96, :] * 0.2)
    flms2dx5[:, 4] = (flms2dx106[:, 90, :] * 0.8 + flms2dx106[:, 100, :] * 0.2)

    return flms2dx5


def face_align_2dx106_arcface(flm2dx106, img, image_size=112):
    flm2dx5 = flm2dx106_to_flm2dx5(flm2dx106[np.newaxis, ...])[0]  # [5, 2]
    face_crop, M = norm_crop(flm2dx5, img, image_size, mode='arcface')

    return face_crop, M


def face_align_2dx106_ffhq(flm2dx106, img, image_size=512):
    flm2dx5 = flm2dx106_to_flm2dx5(flm2dx106[np.newaxis, ...])[0]  # [5, 2]
    face_crop, M = norm_crop(flm2dx5, img, image_size, mode='ffhq')

    return face_crop, M


def face_align_2dx106(flms2dx106, img, image_size=None, mode='arcface'):
    flms2dx5 = flm2dx106_to_flm2dx5(flms2dx106)
    face_crops = []
    transforms = []
    if image_size is None:
        if mode == "arcface":
            image_size = 112
        else:
            image_size = 512
    norm_crop_func = partial(norm_crop, img=img, image_size=image_size, mode=mode)
    for flm2dx5 in flms2dx5:
        face, tf = norm_crop_func(flm2dx5)
        face_crops.append(face)
        transforms.append(tf)

    return face_crops, transforms


if __name__ == '__main__':
    inputdir = "/home/bobo/Temp_new/ZYB/IV_WORKING/dataset/TEST/Pseries/original100_valid_resolution"
    facedetlm = FaceAlignment(FaceAlignment,version="v1")

    for img_path in os.listdir(inputdir):

        img = cv2.imread(
            os.path.join(inputdir, img_path))
        # alignment
        facelist, tfs = facedetlm.forward(img,
                                          conf=0.4,
                                          iou_conf=0.5,
                                          radio=1.0,
                                          maskfilter=False,
                                          posefilter=False,
                                          alignment_mode="ema")
        # do something
        facelist = [img[:, :, (2, 1, 0)] for img in facelist]
        # reverse
        backimg = facedetlm.backward(img, facelist, tfs, alignment_mode="ema")
        cv2.imshow("backimg", backimg)
        for face, tran in zip(facelist, tfs):
            cv2.imshow("vis", face)
            # print(tran)
            cv2.waitKey(0)
