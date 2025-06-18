import cv2
import numpy as np
from os import path, listdir

from ivface.detection.face_lm import PFLDJDFaceLMer
from ivface.detection.face_alignment.infer_model import face_align_2dx106

from ivface.recognition.face_recognition.elasticface import ElasticFaceModel
from ivface.recognition.face_recognition.common_utils import _face_center_crop, resize_maxsize, stacked_img_formatter


def get_database_for_match(exist_db_path, snap_size=96):
    """

    :param exist_db_path: 由opt['pretraining_database_path']确定
    :return: db_data (dict)
    """

    if path.exists(exist_db_path) and exist_db_path.split('.')[-1] == 'npz':
        print('载入数据库...')
        db_data = np.load(exist_db_path, allow_pickle=True)
        facechips_all = db_data['facechips'].astype(
            np.uint8
        )  # jpg~[n, ] / png~[n, 256, 256, 3]
        enclens_all = db_data['enclens'].astype(np.int32)  # [n, ]
        feats_key = db_data['feats'].astype(np.float32)  # [n, 512]
        names_key = db_data['names'].astype(str)  # [n, ]

        print('当前数据库ID容量:', len(names_key))
        if facechips_all.ndim == 1:
            facechips_key = np.split(facechips_all, enclens_all.cumsum(), axis=0)[:-1]
            facechips_key = [cv2.imdecode(facechip, 1) for facechip in facechips_key]
        elif facechips_all.ndim == 4:
            facechips_key = np.split(facechips_all, facechips_all.shape[0], axis=0)
            facechips_key = [item.squeeze(0) for item in facechips_key]

        facechips_key_snap = [
            cv2.resize(img, dsize=(snap_size, snap_size), interpolation=cv2.INTER_AREA)
            for img in facechips_key
        ]
    else:
        print('数据库不存在')
        return None

    return {
        'facechips_key_snap': facechips_key_snap,
        'feats_key': feats_key,
        'names_key': names_key,
    }


def face_detect_and_prealign_inference(image, faceregpre_model, min_size=32):
    """

    :param image: 单张输入图像
    :param faceregpre_model: facedet+facelm模型
    :return: 提取出的人脸与关键点 人脸框(yolo格式)
    """

    def convert_xyxy2xywh(size, bbox):
        h, w = size
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=1, a_max=w - 2)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=1, a_max=h - 2)
        norm_bbox = [
            1.0 * (bbox[:, 2] + bbox[:, 0]) / (2 * w),
            1.0 * (bbox[:, 3] + bbox[:, 1]) / (2 * h),
            1.0 * (bbox[:, 2] - bbox[:, 0]) / w,
            1.0 * (bbox[:, 3] - bbox[:, 1]) / h,
        ]
        norm_bbox = np.stack(norm_bbox, axis=1)
        norm_bbox = np.round(norm_bbox, 5)  # [n, 4]
        return norm_bbox

    faceimg_flm2dx106_pair, norm_bboxes = [], []
    landmarks = faceregpre_model.forward(image, posefilter=False)
    img_abslms = landmarks['img_abslms'] if 'img_abslms' in landmarks else []

    face_locs = []
    if len(img_abslms) != 0:
        face_locs = landmarks['mod_face_locs']
        face_locs = np.hstack((face_locs[:, :2], face_locs[:, :2] + face_locs[:, 2:3]))

    num_faces = len(img_abslms)
    # remark 将所有人脸进行裁剪, 支持vam逐人脸操作, 大图缩放至长边640
    for face_idx in range(num_faces):
        faceimg, lt_pos = _face_center_crop(image, img_abslms[face_idx], expand=1.25)
        # remark 过滤较小的人脸
        h, w, _ = faceimg.shape
        if h >= min_size and w >= min_size:
            faceimg, scale = resize_maxsize(faceimg, size=512)
            img_abslm = (
                img_abslms[face_idx] - np.array(lt_pos)[np.newaxis, ...]
            ) * scale
            faceimg_flm2dx106_pair.append([faceimg, img_abslm])
        else:
            break
    if len(faceimg_flm2dx106_pair) > 0:
        norm_bboxes = convert_xyxy2xywh(
            image.shape[:2], face_locs[: len(faceimg_flm2dx106_pair)]
        )

    return faceimg_flm2dx106_pair, norm_bboxes


def get_aligned_facechips_and_feats(
    faceimg_flm2dx106_pairs, facerecog_model, facechip_size=256, save_format='jpg'
):
    """

    :param faceimg_flm2dx106_pairs: face_detect_and_prealign输出
    :param facerecog_model: 人脸识别模型, 默认为elasticface
    :param facechip_size: 存储的人脸尺寸, 默认256
    :param save_format: training 可选png无损存储/jpg压缩存储, not training 固定为png
    :return: 人脸特征 人脸裁剪图像
    """

    facechips_recog, facechips_viz = [], []
    alignment_mode, alignment_size = (
        facerecog_model.alignment_mode,
        facerecog_model.alignment_size,
    )
    for (face_img, flm2dx106) in faceimg_flm2dx106_pairs:
        facechip_recog = face_align_2dx106(
            flm2dx106[np.newaxis, ...],
            face_img,
            image_size=alignment_size,
            mode=alignment_mode,
        )[0][0]
        facechip_viz = face_align_2dx106(
            flm2dx106[np.newaxis, ...], face_img, image_size=facechip_size, mode='ffhq'
        )[0][0]
        facechips_recog.append(facechip_recog)
        if save_format == 'png':
            facechips_viz.append(facechip_viz)
        else:
            facechips_viz.append(
                cv2.imencode('.jpg', facechip_viz, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[
                    1
                ]
            )
    feats = facerecog_model.forward(facechips_recog)

    return feats, facechips_viz


def face_matching(
    feats_query,
    facechips_ffhq_query,
    norm_boxes,
    db_data,
    pos_thr=(0.4 + 1) / 2,
    snap_size=96,
    num_cols=10,
):
    """

    :param feats_query: 待匹配特征
    :param db_data: dict
    :param pos_thr: 是否为同一ID人脸的分水岭阈值
    :return:
    """

    def verify_MN(emb_a, emb_b):
        # todo 若MN非常大, 需要进行拆分处理
        """

        :param emb_a: 待匹配样例 | shape [M, 512] | ndarray
        :param emb_b: 已知身份样例 | shape [N, 512] | ndarray
        :return: 提供M:N比对-最相似+阈值二分的匹配-返回相似度与序列号
        """
        if isinstance(emb_a, np.ndarray) and isinstance(emb_b, np.ndarray):
            sim_ab = np.dot(emb_a, emb_b.T)  # [M, N]
            # remark 获取最相似匹配[O(n)]+阈值二分[O(1)]
            idxes_max = np.argmax(sim_ab, axis=1)[:, np.newaxis]  # [M, 1]
            sim_ab_max = np.take_along_axis(sim_ab, idxes_max, axis=1)  # [M, 1]
        else:
            raise ValueError('特征类型异常, 请确保操作无误')

        return sim_ab_max.squeeze(1), idxes_max.squeeze(1)

    facechips_key_snap, feats_key, names_key = (
        db_data['facechips_key_snap'],
        db_data['feats_key'],
        db_data['names_key'],
    )

    sim_ab_max, idxes_max = verify_MN(feats_query, feats_key)
    # remark 将[-1, 1]相似度线性变换到[0, 1]
    sim_ab_max = (sim_ab_max + 1) / 2
    pos_masks = sim_ab_max >= pos_thr
    pos_facechips_key = [facechips_key_snap[idx] for idx in idxes_max]
    pos_names_key = [names_key[idx] for idx in idxes_max]

    print('共检出待匹配人脸:', len(facechips_ffhq_query))
    print('匹配成功人脸:', pos_masks.sum())

    output = {}
    viz_matches = []
    for (sim, idx, facechip_query, pos_facechip_key, pos_name_key) in zip(
        sim_ab_max, idxes_max, facechips_ffhq_query, pos_facechips_key, pos_names_key
    ):
        viz_graph = facechip_query.copy()
        if sim >= pos_thr:
            viz_graph[-snap_size:, -snap_size:] = pos_facechip_key
            viz_graph = cv2.putText(
                viz_graph,
                'ID{:03d} {}: {}%'.format(idx + 1, pos_name_key, round(sim * 100, 1)),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        viz_matches.append(viz_graph)

    # 匹配结果可视化图像
    output['labels_snapshot'] = stacked_img_formatter(
        viz_matches, num_cols=min(num_cols, len(viz_matches))
    )

    # remark 查询图像名 | (匹配成功)查询图像人脸序号-匹配数据库ID-数据库人名-概率
    output['labels'] = []
    for (sim, norm_box, pos_name_key) in zip(sim_ab_max, norm_boxes, pos_names_key):
        output['labels'] += [
            norm_box.tolist()
            + [[round(sim, 3), pos_name_key], [0.0, '未知人脸']][float(sim) < pos_thr]
        ]

    return output


class FaceRecogInference(object):
    def __init__(
        self,
        flm_model='pfldjd',
        flm_version='v1',
        recog_model='elasticface',
        recog_version='v1',
    ):
        """

        :param flm_model: pfldjd
        :param flm_version: v1,v2,v1&v2
        :param recog_model: elasticface
        :param recog_version: v1
        """

        assert flm_model in [
            'pfldjd'
        ], 'flm_model only support pfldjd, ' 'your flm_model is {}'.format(flm_model)
        assert recog_model in [
            'elasticface'
        ], 'recog_model only support elasticface, ' 'your recog_model is {}'.format(
            recog_model
        )

        if flm_model == 'pfldjd':
            self.faceregpres_model = PFLDJDFaceLMer(version=flm_version)

        if recog_model == 'elasticface':
            self.facerecog_model = ElasticFaceModel(version=recog_version)

        self.db_data_dict = {}

    def forward(self, image, **kwargs):
        """

        :param image: 载入的一张待识别人脸的图像
        :param kwargs:
                database_path: 已注册过的人脸数据库路径, 必须确保路径存在
                reload_dbdata: 复用已加载过的人脸数据库, 若数据库无变动, 建议使用默认值False
                pos_thr: 人脸特征相似度阈值, 默认0.75, 可在[0.7, 0.9]阈值内调整
        :return:
        """

        database_path = kwargs.get('database_path', '')
        reload_dbdata = kwargs.get('reload_dbdata', False)
        pos_thr = kwargs.get('pos_thr', (0.3 + 1) / 2)

        if reload_dbdata or database_path not in self.db_data_dict.keys():
            db_data = get_database_for_match(database_path)
            self.db_data_dict[database_path] = db_data
        else:
            db_data = self.db_data_dict[database_path]

        output = {'labels_snapshot': None, 'labels': []}
        if db_data is not None:
            faceimg_flm2dx106_pairs, norm_bboxes = face_detect_and_prealign_inference(
                image, self.faceregpres_model
            )
            if len(faceimg_flm2dx106_pairs) != 0:
                feats_query, facechips_ffhq_query = get_aligned_facechips_and_feats(
                    faceimg_flm2dx106_pairs, self.facerecog_model, save_format='png'
                )
                output = face_matching(
                    feats_query,
                    facechips_ffhq_query,
                    norm_bboxes,
                    db_data,
                    pos_thr=pos_thr,
                )

        return output


if __name__ == '__main__':
    inputdir = "/home/iv/Temp/DW/1-需求评估/写真生成/刘亦菲最新写真"

    # database_path 必须确保存在该路径
    # database_path = "./data/register/last.npz"
    database_path = "/home/iv/Temp/DW/1-需求评估/写真生成/刘亦菲最新写真/1/last.npz"
    reload_dbdata = False
    pos_thr = 0.75

    facerec2 = FaceRecogInference()
    for img_path in listdir(inputdir):
        image = cv2.imread(path.join(inputdir, img_path))
        output = facerec2.forward(
            image,
            database_path=database_path,
            reload_dbdata=reload_dbdata,
            pos_thr=pos_thr,
        )
        print(output['labels'])
        if len(output['labels']) > 0:
            cv2.imshow('labels_snapshot', output['labels_snapshot'])
            cv2.waitKey(0)
