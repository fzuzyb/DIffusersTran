#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: infer_model.py
#@time: 10/17/23 10:26 AM
import os

import PIL.Image
import cv2
import torch
from sklearn.cluster import KMeans
from ivface.AppBaseModel import BaseModel
from ivface.face_seg.bisenet.model import BiSeNet


cls_bisenet = ['background', 'skin', 'l_brow', 'r_brow',
               'l_eye', 'r_eye', 'eye_g',
               'l_ear', 'r_ear', 'ear_r',
               'nose', 'mouth', 'u_lip', 'l_lip',
               'neck', 'neck_l', 'cloth', 'hair', 'hat']


__WEIGHTS_FNS__ = {'v1': "fs_bisenet_v1.pth"}

class FaceSegModel(BaseModel):
    def __init__(self, IVFACE_ROOT="./weights",version="v1",device=0):
        super(FaceSegModel, self).__init__()

        assert version in list(__WEIGHTS_FNS__.keys()), \
            "version only support {} , your version is {}".format(list(__WEIGHTS_FNS__.keys()), version)


        __WEIGHTS_PATH__ = os.path.join(IVFACE_ROOT,'face_seg/BiSeNet/')
        self.model = BiSeNet(n_classes=19)
        self.load_model(__WEIGHTS_PATH__, __WEIGHTS_FNS__,device=device)
        if device != -1:
            device = "cuda:{}".format(device)
        else:
            device = "cpu"
        self.model = self.model.to(device)
        self.device = device
        self.bisenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).float()
        self.bisenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).float()

        self.ones = torch.ones(1).to(device)
        self.zeros = torch.zeros(1).to(device)
        self.model.eval()
        self.__first_forward__()

    def __first_forward__(self, input_size=[512, 512, 3]):
        # 直接调用forward()计算最大显存
        print('initialize FaceSeg Model >>> BiSeNet pth {}...'.format(self.version))
        input = (np.random.rand(*input_size) * 255).round().astype(np.uint8)
        _ = self.forward(input)

    def resize(self, img, insize=128):
        img = cv2.resize(img, (insize, insize), interpolation=cv2.INTER_LINEAR)
        return img

    def input_preprocess(self, img):
        # resize
        H,W,C = img.shape
        if H==512 and W==512:
            input = img
        else:
            input = self.resize(img, insize=512)
        # bgr2rgb hwc2bchw
        input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0)
        input = input[:, (2, 1, 0), :, :] / 255.
        # norm
        input = (input - self.bisenet_mean) / self.bisenet_std
        return input.float()

    def forward(self, img,**kwargs):
        '''

        :param img:  HWC, BGR, NDArray 0-255 输入只是人脸，最好是输入是512x512,不然会进行resize
        :param kwargs:
        :return: Mask H,W,1
        '''
        h,w,c = img.shape

        input = self.input_preprocess(img)

        with torch.no_grad():
            segment = self.model(input.to(self.device)) #1, C,H,W
            top = torch.argmax(segment, dim=1) # 1,1,H,W
            # remark 嘴部 unet=[10, 12] | bisenet=[11, 13]
            #mask = torch.where(((top >= 1) & (top <= 13)), self.ones, self.zeros)
            if kwargs.get('contain_neck', True):
                mask = torch.where(((top >= 1) & (top <= 14)), self.ones, self.zeros)
            else:
                mask = torch.where(((top >= 1) & (top <= 13)), self.ones, self.zeros)
            hair_mask = torch.where((top==17) | (top==18), self.ones, self.zeros)
            mask = mask + hair_mask
           # mask = F.max_pool2d(mask,kernel_size=3,padding=1).cpu().numpy().transpose(1,2,0)
            mask = mask.cpu().numpy().transpose(1, 2, 0)
        # 输出resize
        mask = cv2.resize(mask,dsize=(w,h),interpolation=cv2.INTER_LINEAR)
        return mask

def classify_face_color(face):
    '''

    :param face:
    :return: 黑，白，黄（0，1，2）
    '''

    lower_black = (1, 120, 0)
    upper_black = (10, 255, 180)

    lower_back_other = (11, 0, 180)
    upper_black_other = (180, 255, 255)

    lower_white = (11, 0, 180)
    upper_white = (40, 60, 255)

    lower_yellow = (1, 60, 170)
    upper_yellow = (180, 255, 255)

    # lower_black = (0, 0, 0)
    # upper_black = (40, 255, 100)
    #
    # lower_white = (0, 0, 150)
    # upper_white = (20, 30, 255)
    #
    # lower_yellow = (20, 30, 100)
    # upper_yellow = (40, 255, 255)

    hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    # 黑色人脸的颜色范围
    mask_black = cv2.inRange(hsv_face, lower_black, upper_black)
    black_pixels = np.sum(mask_black)
    mask_black_other = cv2.inRange(hsv_face, lower_back_other, upper_black_other)
    black_pixels_other = np.sum(mask_black_other)
    if black_pixels > black_pixels_other:
        return 0
    else:

        # 白色人脸的颜色范围
        mask_white = cv2.inRange(hsv_face, lower_white, upper_white)
        white_pixels = np.sum(mask_white)

        # 黄色人脸的颜色范围
        mask_yellow = cv2.inRange(hsv_face, lower_yellow, upper_yellow)
        yellow_pixels = np.sum(mask_yellow)

        # 判断哪种颜色占据的像素最多
        if white_pixels > yellow_pixels:
            return 1
        else:
            return 2


# def classify_face_color(face):
#     '''
#
#     :param face:
#     :return: 黑，白，黄（0，1，2）
#     '''
#
#
#
#     gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#
#     # 定义肤色的灰度阈值范围
#     white_lower = 190
#     white_upper = 255
#     yellow_lower = 150
#     yellow_upper = 190
#     black_lower = 0
#     black_upper = 100
#
#     # 根据阈值提取不同肤色区域
#     white_skin_mask = cv2.inRange(gray_face, white_lower, white_upper)
#     white_pixels = np.sum(white_skin_mask)
#     yellow_skin_mask = cv2.inRange(gray_face, yellow_lower, yellow_upper)
#     yellow_pixels = np.sum(yellow_skin_mask)
#     black_skin_mask = cv2.inRange(gray_face, black_lower, black_upper)
#     black_pixels = np.sum(black_skin_mask)
#
#     # 判断哪种颜色占据的像素最多
#     max_pixels = max(black_pixels, white_pixels, yellow_pixels)
#     if max_pixels == black_pixels:
#         return 0
#     elif max_pixels == white_pixels:
#         return 1
#     else:
#         return 2


# 2. 特征向量聚类
def cluster_faces(face_features):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(face_features)
    return kmeans.cluster_centers_


# 3. 剔除不符合类别的人脸
def filter_faces(face_features, cluster_center, threshold=0.5):
    filter_index = []
    for index,feature in enumerate(face_features):
        distances = np.linalg.norm(np.array(feature) - cluster_center)/210.0
        print('dis:',distances)
        if distances > threshold:
            filter_index.append(index)
    return filter_index


if __name__ == '__main__':
    # 对齐的人脸
    # inputdir = "/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/VAL/FFHQA_Stylegan/fsr_128_512/random_degradation_nomblur_ISP/HR"
    # outputdir = "/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/VAL/FFHQA_Stylegan/fsr_128_512/random_degradation_nomblur_ISP/mask"
    # input_files = os.listdir(inputdir)
    # bisenet_faceseger = FaceSegModel(version="v1")
    # for img_path in input_files:
    #     img = cv2.imread(
    #         os.path.join(inputdir, img_path))
    #
    #     mask = bisenet_faceseger.forward(img)
    #
    #     # blur 边缘
    #     padding = 10
    #     kernel_size = 21
    #     # Add padding to the input mask
    #     mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #     # Define the kernel for average pooling
    #     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #     # Apply average pooling using cv2.filter2D
    #     pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #     # Crop the padded regions to get the final output
    #     pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #     cv2.imwrite(os.path.join(outputdir, img_path), (pooled_mask*255).astype(np.uint8))
    # 非对齐的人脸
    # inputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/人物测试集/man_y"
    # outputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/测试集/"
    #
    # input_files = sorted(os.listdir(inputdir))
    # from applications.detection.face_alignment import FaceAlignment
    #
    # bisenet_faceseger = FaceSegModel(version="v1")
    # jd_facealignmer = FaceAlignment(version='v1')
    # os.makedirs(os.path.join(outputdir, 'mask'),exist_ok=True)
    # os.makedirs(os.path.join(outputdir, 'crop'),exist_ok=True)
    # for img_path in input_files:
    #     img = cv2.imread(
    #         os.path.join(inputdir, img_path))
    #     ouput_mask = np.zeros_like(img, dtype=np.uint8)
    #     faces, transforms = jd_facealignmer.forward(img)
    #     masks = []
    #     for face in faces:
    #         class_face = np.ones_like(face, dtype=np.uint8) * 128
    #         mask = bisenet_faceseger.forward(face)
    #         # blur 边缘
    #         padding = 10
    #         kernel_size = 21
    #         # Add padding to the input mask
    #         mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #         # Define the kernel for average pooling
    #         kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #         # Apply average pooling using cv2.filter2D
    #         pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #         # Crop the padded regions to get the final output
    #         pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #         pooled_mask = pooled_mask[:, :, None].repeat(3, axis=2)
    #         # 用于区分人脸颜色
    #         class_face = class_face * (1 - pooled_mask) + pooled_mask * face
    #         masks.append((pooled_mask * 255).astype(np.uint8))
    #         # 分类人脸颜色,这里仅使用了分割来测试，检测之后也可以进行人脸肤色分类
    #         print(img_path, classify_face_color(cv2.resize((class_face).astype(np.uint8), dsize=(512, 512))))
    #     # reverse
    #     ouput_mask = jd_facealignmer.backward(ouput_mask, masks, transforms)
    #     # 保存mask
    #     cv2.imwrite(os.path.join(outputdir, 'mask',img_path.replace(img_path.split('.')[-1], 'mask.jpg')), ouput_mask)
    #     cv2.imwrite(os.path.join(outputdir, 'crop',img_path), (class_face).astype(np.uint8))

    # 非对齐人脸，分割+人脸聚类
    # inputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/人物测试集/woman_w"
    # outputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/人物测试集/聚类测试"
    # input_files = sorted(os.listdir(inputdir))
    # from applications.detection.face_alignment import FaceAlignment
    # from applications.recognition.face_recognition.common_filter_face import FaceRecogFilter
    # bisenet_faceseger = FaceSegModel(version="v1")
    # jd_facealignmer = FaceAlignment(version='v1')
    # face_recogfilter = FaceRecogFilter(version='v1')
    # face_features = []
    # file_index = {}
    #
    #
    # img_abslmses = []
    # images = []
    # for index,img_path in enumerate(input_files):
    #
    #     img = cv2.imread(
    #         os.path.join(inputdir, img_path))
    #     ouput_mask = np.zeros_like(img, dtype=np.uint8)
    #     faces, transforms = jd_facealignmer.forward(img,max_detfaces=1)
    #     imgabslms = jd_facealignmer.get_img_abslms()  #  landmarks  [1,106,2[x,y]]
    #
    #     if len(imgabslms) != 0:
    #         h,w,c = img.shape
    #         file_index[index] = img_path
    #         images.append(img)
    #         img_abslmses.append(imgabslms)
    #         masks = []
    #         for face in faces:
    #             class_face = np.ones_like(face, dtype=np.uint8) * 128
    #             mask = bisenet_faceseger.forward(face)
    #             # blur 边缘
    #             padding = 10
    #             kernel_size = 21
    #             # Add padding to the input mask
    #             mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #             # Define the kernel for average pooling
    #             kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #             # Apply average pooling using cv2.filter2D
    #             pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #             # Crop the padded regions to get the final output
    #             pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #             pooled_mask = pooled_mask[:, :, None].repeat(3, axis=2)
    #             # 用于区分人脸颜色
    #             class_face = class_face * (1 - pooled_mask) + pooled_mask * face
    #             masks.append((pooled_mask * 255).astype(np.uint8))
    #             # 分类人脸颜色,这里仅使用了分割来测试，检测之后也可以进行人脸肤色分类
    #             print(img_path, classify_face_color(cv2.resize((class_face).astype(np.uint8), dsize=(512, 512))))
    #         # reverse
    #         ouput_mask = jd_facealignmer.backward(ouput_mask, masks, transforms)
    #         # 保存mask
    #         cv2.imwrite(os.path.join(outputdir, 'mask',img_path.replace(img_path.split('.')[-1], 'mask.jpg')), ouput_mask)
    #         cv2.imwrite(os.path.join(outputdir, 'crop',img_path), (class_face).astype(np.uint8))
    #
    # filter_index = face_recogfilter.forward(images,img_abslmses,filter_threshold=0.8)
    #
    # for fin in filter_index:
    #     print('filter:',file_index[fin])
    #
    # # 模拟如果生成了新的图片
    # inputdir = "/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/AI写真/韩式证件照/女/长批发/train/lora/刘亦菲/聚类测试/生成测试"
    # input_files = sorted(os.listdir(inputdir))
    # images = []
    # img_abslmses = []
    # file_index = {}
    # for index, img_path in enumerate(input_files):
    #     img = cv2.imread(
    #         os.path.join(inputdir, img_path))
    #     faces, transforms = jd_facealignmer.forward(img, max_detfaces=1)
    #     imgabslms = jd_facealignmer.get_img_abslms()  # landmarks  [1,106,2[x,y]]
    #
    #     if len(imgabslms) != 0:
    #         h, w, c = img.shape
    #         file_index[index] = img_path
    #         images.append(img)
    #         img_abslmses.append(imgabslms)
    #
    # # remark filter_threshold 这个阈值需要LoRA实验验证
    # filter_index = face_recogfilter.forward(images,img_abslmses,filter_threshold=0.7)
    #
    # for fin in filter_index:
    #     print('filter:',file_index[fin])
    # from applications.detection.face_alignment import FaceAlignment
    # inputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/yearbook"
    # outputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/分割与颜色F"
    # color_map = {0:'B',1:'W',2:'Y'}
    # bisenet_faceseger = FaceSegModel(version="v1")
    # jd_facealignmer = FaceAlignment(version='v1')
    # for rootdir,subdir,files in os.walk(inputdir):
    #     # if 'man' in rootdir:
    #     #     continue
    #     if 'images' not in rootdir:
    #         continue
    #     if 'woman' in rootdir:
    #         if len(files)!=0:
    #
    #             input_files = os.listdir(rootdir)
    #             output_path = rootdir.replace(inputdir,outputdir)
    #             os.makedirs(output_path,exist_ok=True)
    #             for img_path in input_files:
    #                 img = cv2.imread(
    #                     os.path.join(rootdir, img_path))
    #                 ouput_mask = np.zeros_like(img, dtype=np.uint8)
    #                 faces, transforms = jd_facealignmer.forward(img,radio=1.8)
    #                 masks = []
    #                 for face in faces:
    #                     class_face = np.ones_like(face, dtype=np.uint8) * 128
    #                     mask = bisenet_faceseger.forward(face)
    #                     # blur 边缘
    #                     padding = 4
    #                     kernel_size = 9
    #                     # Add padding to the input mask
    #                     mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT,value=0)
    #                     # Define the kernel for average pooling
    #                     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #                     # # Apply average pooling using cv2.filter2D
    #                     pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #
    #
    #
    #
    #                     # Crop the padded regions to get the final output
    #                     pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #                     pooled_mask = pooled_mask[:, :, None].repeat(3, axis=2)
    #                     # 用于区分人脸颜色
    #                     class_face = class_face * (1 - pooled_mask) + pooled_mask * face
    #                     masks.append((pooled_mask * 255).astype(np.uint8))
    #                     # 分类人脸颜色,这里仅使用了分割来测试，检测之后也可以进行人脸肤色分类
    #                     COLOR = color_map[classify_face_color(cv2.resize((class_face).astype(np.uint8), dsize=(512, 512)))]
    #                 # reverse
    #                 ouput_mask = jd_facealignmer.backward(ouput_mask, masks, transforms)
    #                 # 保存mask
    #                 # cv2.imwrite(os.path.join(output_path,  img_path.replace(img_path.split('.')[-1], 'mask_{}.jpg'.format(COLOR))),
    #                 #             ouput_mask)
    #                 cv2.imwrite(
    #                     os.path.join(output_path, img_path),
    #                     ouput_mask)
    #
    #     else:
    #         continue

    # inputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/人物测试集"
    # outputdir = "/home/node-unknow/Temp/VideoClip/写真/yearbook风格模版图片/人物测试集out"
    #
    # input_files = sorted(os.listdir(inputdir))
    # from applications.detection.face_alignment import FaceAlignment
    #
    # bisenet_faceseger = FaceSegModel(version="v1")
    # jd_facealignmer = FaceAlignment(version='v1')
    # os.makedirs(os.path.join(outputdir, 'crop', 'W'), exist_ok=True)
    # os.makedirs(os.path.join(outputdir, 'crop', 'B'), exist_ok=True)
    # os.makedirs(os.path.join(outputdir, 'crop', 'Y'), exist_ok=True)
    # #os.makedirs(os.path.join(outputdir, 'crop'), exist_ok=True)
    # index = 0
    # for rootdir,subdir,files in os.walk(inputdir):
    #     if len(files)==0:
    #         continue
    #     # if 'images' not in rootdir:
    #     #     continue
    #
    #
    #     for img_path in files:
    #         img = cv2.imread(
    #             os.path.join(rootdir, img_path))
    #         ouput_mask = np.zeros_like(img, dtype=np.uint8)
    #         faces, transforms = jd_facealignmer.forward(img)
    #         masks = []
    #         for face in faces:
    #             class_face = np.ones_like(face, dtype=np.uint8) * 128
    #             mask = bisenet_faceseger.forward(face)
    #             # blur 边缘
    #             padding = 10
    #             kernel_size = 21
    #             # Add padding to the input mask
    #             mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #             # Define the kernel for average pooling
    #             kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #             # Apply average pooling using cv2.filter2D
    #             pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #             # Crop the padded regions to get the final output
    #             pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #             pooled_mask = pooled_mask[:, :, None].repeat(3, axis=2)
    #             # 用于区分人脸颜色
    #             class_face = class_face * (1 - pooled_mask) + pooled_mask * face
    #             masks.append((pooled_mask * 255).astype(np.uint8))
    #             # 分类人脸颜色,这里仅使用了分割来测试，检测之后也可以进行人脸肤色分类
    #             print(img_path, classify_face_color(cv2.resize((class_face).astype(np.uint8), dsize=(512, 512))))
            # reverse
            #ouput_mask = jd_facealignmer.backward(ouput_mask, masks, transforms)
            # 保存mask
            #cv2.imwrite(os.path.join(outputdir, 'mask', img_path.replace(img_path.split('.')[-1], 'mask.jpg')), ouput_mask)
            # if 'Y' in img_path:
            #     cv2.imwrite(os.path.join(outputdir, 'crop', 'Y', str(index)+img_path), (class_face).astype(np.uint8))
            # elif 'B' in img_path:
            #     cv2.imwrite(os.path.join(outputdir, 'crop', 'B', str(index)+img_path), (class_face).astype(np.uint8))
            # else:
            # cv2.imwrite(os.path.join(outputdir, 'crop', 'W', str(index)+'.jpg'), (class_face).astype(np.uint8))
            # index = index+1
    # from tqdm import tqdm
    # inputdir = "/home/node-unknow/Temp/DW/Project/Stable-Diffusion/webui_test/"
    # outputdir = "/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/TRAIN/face_color/celeba_hq_256/mask"
    # input_files = os.listdir(inputdir)
    # bisenet_faceseger = FaceSegModel(version="v1")
    # for img_path in tqdm(input_files):
    #
    #     img = cv2.imread(
    #         os.path.join(inputdir, img_path))
    #     class_face = np.ones_like(img, dtype=np.uint8) * 128
    #     mask = bisenet_faceseger.forward(img)
    #
    #     # blur 边缘
    #     padding = 10
    #     kernel_size = 21
    #     # Add padding to the input mask
    #     mask_padded = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    #     # Define the kernel for average pooling
    #     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    #     # Apply average pooling using cv2.filter2D
    #     pooled_mask = cv2.filter2D(mask_padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #     # Crop the padded regions to get the final output
    #     pooled_mask = pooled_mask[padding:-padding, padding:-padding]
    #
    #     pooled_mask = pooled_mask[:, :, None].repeat(3, axis=2)
    #     # 用于区分人脸颜色
    #     class_face = class_face * (1 - pooled_mask) + pooled_mask * img
    #     cv2.imwrite(os.path.join(outputdir, img_path), (class_face).astype(np.uint8))

    import torchvision.transforms.transforms as tvs_trans
    from PIL import Image
    import onnxruntime as ort
    import numpy as np

    classfiy_transform = tvs_trans.Compose([
        tvs_trans.Resize(224),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cls_names = {
        0: 'B',
        1: 'W',
        2: 'Y',

    }

    classifier_model = ort.InferenceSession(
        "/home/node-unknow/Temp/ZYB/IV_WORKING/codes/Classification/mmpretrain/work_dirs/mobileone-s0_8xb32_facecolor_v3/epoch_191_reparmeter.onnx",
        providers=[('CUDAExecutionProvider', {'device_id': 0}),
                   'CPUExecutionProvider'])
    input_file = "/home/node-unknow/Temp/ZYB/IV_WORKING/dataset/VAL/人种测试/亚洲女"

    from applications.detection.face_alignment import FaceAlignment
    jd_facealignmer = FaceAlignment(version='v1')

    for rootdir, subdir, files in os.walk(input_file):
        if len(files) == 0:
            continue


        for img_path in files:
            img = cv2.imread(
                os.path.join(rootdir, img_path))
            faces, transforms = jd_facealignmer.forward(img)
            for face in faces:
                face = PIL.Image.fromarray(face).convert('RGB')
                img = classfiy_transform(face).numpy()[None, ...]

                pred = torch.from_numpy(np.array(classifier_model.run([classifier_model.get_outputs()[0].name],
                                                                      {classifier_model.get_inputs()[0].name: img}))[0])
                pred_cls2 = torch.argmax(pred, dim=1).numpy()  # classifier prediction label
                print(cls_names[int(pred_cls2)])
                # blur 边缘

