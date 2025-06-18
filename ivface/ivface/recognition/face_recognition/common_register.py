import cv2
import numpy as np
from os import path, listdir, makedirs

from ivface.detection.face_lm import PFLDJDFaceLMer
from ivface.detection.face_alignment.infer_model import face_align_2dx106

from ivface.recognition.face_recognition.elasticface import ElasticFaceModel
from ivface.recognition.face_recognition.common_utils import _face_center_crop, resize_maxsize, stacked_img_formatter


def get_database_for_register(exist_db_path, labels):
    """

    :param exist_db_path: 由opt['database_path']确定
    :param labels: 待入库的人脸id名
    :return: db_data (dict)
    """

    if path.exists(exist_db_path) and exist_db_path.split('.')[-1] == 'npz':
        print('载入数据库...')
        db_data = np.load(exist_db_path, allow_pickle=True)
        facechips_all = db_data['facechips'].astype(
            np.uint8
        )  # jpg~[n, ] / png~[n, FACECHIP_SIZE, FACECHIP_SIZE, 3]
        enclens_all = db_data['enclens'].astype(np.int32)  # [n, ]
        feats_all = db_data['feats'].astype(np.float32)  # [n, 512]
        names_all = db_data['names'].astype(str)  # [n, ]

        print('当前数据库ID容量:', len(names_all))
        intersec = set(names_all).intersection(labels)
        if len(intersec) > 0:
            print('数据库已存在ID: {}, 待添加文件存在重复ID: {}'.format(set(names_all), intersec))
            return None
    else:
        print('数据库初始化...')
        facechips_all, enclens_all, feats_all, names_all = None, None, None, None

    return {
        'facechips_all': facechips_all,
        'enclens_all': enclens_all,
        'feats_all': feats_all,
        'names_all': names_all,
    }


def face_detect_and_prealign_register(images, labels, faceregpre_model, min_size=32):
    """

    :param images: load_images输出
    :param labels: 待入库的人脸id名
    :param faceregpre_model: facedet+facelm模型
    :return: 提取出的人脸与关键点 人脸框(yolo格式) 对应的label名称
    """
    # 检测[包括预扩充与裁剪], 支持2dx106关键点检测输出
    faceimg_flm2dx106_pair, names = [], []
    for image, label in zip(images, labels):
        landmarks = faceregpre_model.forward(image, posefilter=False)
        img_abslms = landmarks['img_abslms'] if 'img_abslms' in landmarks else []

        num_faces = min(1, len(img_abslms))
        # remark 将最大人脸进行裁剪, 大图缩放至长边512
        for face_idx in range(num_faces):
            faceimg, lt_pos = _face_center_crop(
                image, img_abslms[face_idx], expand=1.25
            )
            # remark 过滤较小的人脸
            h, w, _ = faceimg.shape
            if h >= min_size and w >= min_size:
                faceimg, scale = resize_maxsize(faceimg, size=512)
                img_abslm = (
                    img_abslms[face_idx] - np.array(lt_pos)[np.newaxis, ...]
                ) * scale
                faceimg_flm2dx106_pair.append([faceimg, img_abslm])
                names.append(label)
            else:
                break

    return faceimg_flm2dx106_pair, names


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


def update_database(
    feats,
    facechips_ffhq,
    names,
    db_data,
    save_dir,
    save_format='jpg',
    num_cols=10,
    facechip_size=256,
):

    #:param save_dir: 由opt['output_dir']/modelfiles确定
    print('########## 更新数据库&数据库缩略图存储 ##########')
    if save_format == 'png':
        # 存储无损格式
        enclens = np.zeros(len(names))  # [n, ]
        facechips_ffhq = np.stack(
            facechips_ffhq, axis=0
        )  # [n, FACECHIP_SIZE, FACECHIP_SIZE, 3]
    else:
        enclens = np.array(
            [len(item) for item in facechips_ffhq], dtype=np.int32
        )  # [n, ]
        facechips_ffhq = np.concatenate(facechips_ffhq, axis=0).astype(np.uint8)
        if facechips_ffhq.ndim == 2:
            facechips_ffhq = facechips_ffhq.squeeze(1)

        # print(facechips_ffhq.shape, facechips_ffhq.dtype, type(facechips_ffhq))
    names = np.array(names, dtype=str)

    facechips_all, enclens_all, feats_all, names_all = (
        db_data['facechips_all'],
        db_data['enclens_all'],
        db_data['feats_all'],
        db_data['names_all'],
    )
    if names_all is None:
        facechips_all = facechips_ffhq
        enclens_all = enclens
        feats_all = feats
        names_all = names
    else:
        facechips_all = np.concatenate(
            (facechips_all, facechips_ffhq), axis=0
        )  # [n', ]
        enclens_all = np.concatenate((enclens_all, enclens), axis=0)  # [n', ]
        feats_all = np.concatenate((feats_all, feats), axis=0)  # [n', 512]
        names_all = np.concatenate((names_all, names), axis=0)  # [n', ]

    makedirs(save_dir, exist_ok=True)
    np.savez(
        path.join(save_dir, 'last.npz'),
        facechips=facechips_all,
        enclens=enclens_all,
        feats=feats_all,
        names=names_all,
    )

    with open(path.join(save_dir, 'names.txt'), 'w') as f:
        f.write(str(len(names_all)) + '\n')
        for name in names_all:
            f.write(name + '\n')

    # 存储已注册人脸缩略图合集
    save_database_snapshot(
        facechips_all, enclens_all, names_all, save_dir, num_cols, facechip_size
    )


def save_database_snapshot(
    facechips_all, enclens_all, names_all, save_dir, num_cols=10, facechip_size=256
):
    if facechips_all.ndim == 1:
        # remark 批量拆分
        facechips_backup = np.split(facechips_all, enclens_all.cumsum(), axis=0)[:-1]
        facechips_viz = [cv2.imdecode(facechip, 1) for facechip in facechips_backup]
    elif facechips_all.ndim == 4:
        facechips_viz = np.split(facechips_all, facechips_all.shape[0], axis=0)
        facechips_viz = [item.squeeze(0) for item in facechips_viz]
    else:
        print('数据库特征维度异常')
        return

    for idx in range(len(names_all)):
        cv2.putText(
            facechips_viz[idx],
            'ID{:03d} {}'.format(idx + 1, names_all[idx]),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(
        path.join(save_dir, 'last_snapshot.jpg'),
        stacked_img_formatter(
            facechips_viz,
            num_cols=min(num_cols, len(facechips_viz)),
            size=facechip_size,
        ),
    )


class FaceRecogRegister(object):
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

    def forward(self, images_key, labels, **kwargs):
        """

        :param images_key: 载入的含有待注册人物人脸的图像(列表)
        :param labels: 每张图像对应的ID名称(列表)
        :param kwargs:
                database_path: 若存在, 则为新增注册(旧表注册新ID); 若不存在, 则为初始化注册
                output_dir: 数据库文件存储文件夹
        :return:
        """

        database_path = kwargs.get('database_path', '')
        output_dir = kwargs.get(
            'output_dir', path.expanduser('~/.cache/ivalgohub/face_recog/register')
        )

        # 新建数据库/加载现有数据库
        db_data = get_database_for_register(database_path, labels)
        if db_data is not None:
            faceimg_flm2dx106_pairs, names_key = face_detect_and_prealign_register(
                images_key, labels, self.faceregpres_model
            )
            feats_key, facechips_ffhq_key = get_aligned_facechips_and_feats(
                faceimg_flm2dx106_pairs, self.facerecog_model, save_format='jpg'
            )
            update_database(
                feats_key,
                facechips_ffhq_key,
                names_key,
                db_data,
                output_dir,
                save_format='jpg',
            )


if __name__ == '__main__':
    # 获取图像文件列表及对应的注册名称列表
    inputdir = "/home/iv/Temp/DW/1-需求评估/写真生成/刘亦菲最新写真/1"
    # inputdir = "/home/nie/4K/Models/FaceRecognition/JailAPI_v2/data/cache/testdata4"
    imgfns = sorted(listdir(inputdir))
    images_key = [
        cv2.imread(path.join(inputdir, imgfn), cv2.IMREAD_COLOR) for imgfn in imgfns
    ]
    labels = [imgfn.split('.')[0] for imgfn in imgfns]  # 示例用文件名作为注册名, 实际最好采用人名, 支持中文存储

    database_path = ""
    # output_dir = "./data/register"
    output_dir = "/home/iv/Temp/DW/1-需求评估/写真生成/刘亦菲最新写真/1"
    # output_dir = "/home/nie/4K/Models/FaceRecognition/JailAPI_v2/data/cache/testdata4/db"

    facerec1 = FaceRecogRegister()
    facerec1.forward(
        images_key,
        labels,
        database_path=database_path,
        output_dir=output_dir
    )
