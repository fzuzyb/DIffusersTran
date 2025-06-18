import cv2
import numpy as np
from os import path, listdir
from sklearn.cluster import KMeans

from ivface.detection.face_alignment.infer_model import face_align_2dx106
from ivface.recognition.face_recognition.elasticface import ElasticFaceModel
from ivface.recognition.face_recognition.common_utils import _face_center_crop, resize_maxsize




def face_detect_and_prealign(images, img_abslmses, min_size=32):
    """

    :param images: 输入list [m 个 （bgr，image）]
    :param img_abslmses: list 输入关键点 [m个 （n,106,2）]
    :return: 提取出的人脸与关键点 [nm 个(face, (106x2))]
    """
    # 检测[包括预扩充与裁剪], 支持2dx106关键点检测输出
    faceimg_flm2dx106_pair=[]
    for image, img_abslms in zip(images,img_abslmses):

        num_faces = min(1, len(img_abslms))
        # remark 将最大人脸进行裁剪, 大图缩放至长边512
        for face_idx in range(num_faces):
            # centerimg     xy
            faceimg, lt_pos = _face_center_crop(
                image, img_abslms[face_idx], expand=1.25
            )
            # remark 过滤较小的人脸
            h, w, _ = faceimg.shape
            if h >= min_size and w >= min_size:
                # resize image to 512
                faceimg, scale = resize_maxsize(faceimg, size=512)
                # get croped cor
                img_abslm = (
                    img_abslms[face_idx] - np.array(lt_pos)[np.newaxis, ...]
                ) * scale
                # (face_image(h,w,c) kp(106x2))
                faceimg_flm2dx106_pair.append([faceimg, img_abslm])

            else:
                break

    return faceimg_flm2dx106_pair


def get_aligned_facechips_and_feats(
    faceimg_flm2dx106_pairs, facerecog_model
):
    """

    :param faceimg_flm2dx106_pairs: face_detect_and_prealign输出
    :param facerecog_model: 人脸识别模型, 默认为elasticface

    :return: 人脸特征 #(n,512)
    """

    facechips_recog = []
    alignment_mode, alignment_size = (
        facerecog_model.alignment_mode,
        facerecog_model.alignment_size,
    )
    for (face_img, flm2dx106) in faceimg_flm2dx106_pairs:
        # image flm2dx106 （106,2)
        # align face
        facechip_recog = face_align_2dx106(
            flm2dx106[np.newaxis, ...],
            face_img,
            image_size=alignment_size,
            mode=alignment_mode,
        )[0][0]
        facechips_recog.append(facechip_recog)

    feats = facerecog_model.forward(facechips_recog)

    return feats  # (n,512)


def cluster_faces(face_features):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(face_features)
    return kmeans.cluster_centers_

def filter_faces(face_features, cluster_center, threshold=0.5):
    filter_index = []
    for index,feature in enumerate(face_features):
        distances = np.linalg.norm(np.array(feature) - cluster_center)
        if distances > threshold:
            filter_index.append(index)
        print(distances)
    return filter_index

class FaceRecogFilter(object):
    def __init__(
        self,IVFACE_ROOT="./weight",
            version='v1'
    ):
        """

        """
        self.facerecog_model = ElasticFaceModel(IVFACE_ROOT,version=version)
        self.cluster_center = None

    def forward(self, imgs,  img_abslmses, **kwargs):
        """

        :param images_key: 载入的含有待注册人物人脸的图像(列表) [（HWC）,(1,106,2)]
        :param kwargs:
                filter_threshold: 过滤阈值
        :return:
        """
        filter_threshold = kwargs.get('filter_threshold',0.9)
        assert len(imgs) == len(img_abslmses), "input image shape should equal img_abslms"

        # 获取crop image , keypoint
        faceimg_flm2dx106_pairs = face_detect_and_prealign(imgs, img_abslmses)
        # n,512
        feats = get_aligned_facechips_and_feats(
            faceimg_flm2dx106_pairs, self.facerecog_model
        )
        if self.cluster_center is None:
            self.cluster_center = cluster_faces(feats)

        return filter_faces(feats,self.cluster_center,filter_threshold)


class FaceRecogFeatures(object):
    def __init__(
            self, IVFACE_ROOT="./weight",version='v1',
            device=0
    ):
        """

        """
        self.facerecog_model = ElasticFaceModel(IVFACE_ROOT,version=version,device=device)

    def forward(self, imgs, img_abslmses, **kwargs):
        """

        :param img_abslmses:  [(1,106,2)]
        :param kwargs:
                filter_threshold: 过滤阈值
        :return:
        """

        assert len(imgs) == len(img_abslmses), "input image shape should equal img_abslms"
        # 获取crop image , keypoint
        faceimg_flm2dx106_pairs = face_detect_and_prealign(imgs, img_abslmses)
        # n,512
        feats = get_aligned_facechips_and_feats(
            faceimg_flm2dx106_pairs, self.facerecog_model
        )

        return feats
if __name__ == '__main__':
    pass
