import cv2
import numpy as np


def pad_and_crop(quad_r, img):
    h, w = img.shape[0:2]
    top, bottom, left, right = quad_r

    img = img[max(top, 0):min(bottom, h), max(left, 0):min(right, w)]
    pad_left, pad_top, pad_right, pad_bottom = \
        max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    return img # (pad_top, pad_bottom, pad_left, pad_right)

def _face_center_crop(img, lm, expand=1.):
    """

    Args:
        img: input high resolution image [HWC 0-255] BGR
        lm: high resolution landmark

    Returns:

    """
    # get hr size
    H , W, _ = img.shape
    if len(lm)==106:
        lm_eye_left = lm[66: 74,:]  # left-clockwise
        lm_eye_right = lm[75: 83,:]  # right-clockwise
        lm_mouth_outer = lm[84: 96,:]  # mouth-clockwise
    else:
        lm_eye_left = lm[36: 41, :]  # left-clockwise
        lm_eye_right = lm[42: 47, :]  # right-clockwise
        lm_mouth_outer = lm[48: 60, :]  # mouth-clockwise
    eye_left_center = np.mean(lm_eye_left, axis=0) # center of left eye
    eye_right_center = np.mean(lm_eye_right, axis=0) # center of right eye

    eye_avg = (eye_left_center + eye_right_center) * 0.5  # center between left and right eye
    eye_to_eye = eye_right_center - eye_left_center  # distance between left and right eye

    mouth_left = lm_mouth_outer[0]   # left mouth point
    mouth_right = lm_mouth_outer[6]  # right mouth point

    mouth_avg = (mouth_left + mouth_right) * 0.5  # center of mouth
    eye_to_mouth = mouth_avg - eye_avg  # distance between eye and mouth

    face_center = np.round(eye_avg + eye_to_mouth * 0.1).astype(np.int32)
    radius = max(np.hypot(*eye_to_eye), np.hypot(*eye_to_mouth))
    radius = int(np.round(radius*1.8) * expand)

    quad_r = [(face_center[1] - radius), (face_center[1] + radius + 1),
              (face_center[0] - radius), (face_center[0] + radius + 1)]

    # get the first pad image and the pad params
    faceimg = pad_and_crop(quad_r, img)

    return faceimg, [quad_r[2], quad_r[0]]


def resize_maxsize(image, size=512):
    h, w, _ = image.shape
    assert h == w # 人脸裁剪图h=w

    scale = 1. * size / h
    if scale < 1.:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    else: scale = 1.

    return image, scale


def stacked_img_formatter(pos_viz, num_cols=15, size=256):
    num_extra = len(pos_viz) % num_cols
    num_rows = int(np.ceil(len(pos_viz) / num_cols))

    if num_extra == 0:
        hstacks = [np.hstack(pos_viz[idx * num_cols: (idx + 1) * num_cols]) for idx in range(num_rows)]
    else:
        empty_content = np.ones((size, size * (num_cols - num_extra), 3), dtype=np.uint8) * 255
        hstacks = [np.hstack(pos_viz[idx * num_cols: (idx + 1) * num_cols]) for idx in range(num_rows - 1)]
        hstacks.append(np.hstack(pos_viz[-num_extra:] + [empty_content]))

    vstacks = np.vstack(hstacks)
    return vstacks
