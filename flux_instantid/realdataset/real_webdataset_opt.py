import os
import webdataset as wds
import numpy as np
from ivface.detection.face_alignment import FaceAlignment
from ivface.recognition.face_recognition import FaceRecogFeatures
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from accelerate.utils import set_seed
from ivface.face_seg import BiSeNetFaceSegModel
import cv2
from pathlib import Path
from typing import List,  Union
from braceexpand import braceexpand  # 用于展开花括号
import itertools
import math
from torchvision import transforms
import functools
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

import threading
_thread_local = threading.local()


IV_FACE_ROOT = os.getenv("IV_FACE_ROOT","/home/user/Algo_new/DengWei/Project/StableDiffusion/ComfyUI/models/ivface")
IMAGE2TEXT_MODEL = os.getenv("IMAGE2TEXT_MODEL","/home/user/Algo_new/Zhouyuanbo/IV_WORKING/Model/gitbase")


# 全局缓存
_global_face_am = None
_global_face_features = None
_global_image2text = None
_global_face_seg = None

def get_face_modules(root: str, device, use_seg_face: bool):
    if not hasattr(_thread_local, "modules"):
        print(">>> Loading FaceAlignment on device {}...".format(device))
        face_am = FaceAlignment(root, device=device)
        face_features = FaceRecogFeatures(root, device=device)
        image2text = Image2Text(IMAGE2TEXT_MODEL, device=device)
        face_seg = BiSeNetFaceSegModel(root, device=device) if use_seg_face else None
        _thread_local.modules = (face_am, face_features, image2text, face_seg)
    return _thread_local.modules

class Image2Text:
    def __init__(self, MODEL_ROOT, device=-1):
        # 标准化设备
        if device==-1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        self.device = device
        print("init Image2Text on {}".format(device))
        # 加载模型和处理器
        self.processor = AutoProcessor.from_pretrained(MODEL_ROOT)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT).to(self.device)
        self.model.eval()

    def forward(self, image):
        # 图像预处理

        inputs = self.processor(Image.fromarray(image[:,:,(2,1,0)]), return_tensors="pt").to(self.device)

        with torch.no_grad():  # 加快推理
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"], max_length=20
            )

        # 解码
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # 如果只有一张图像，返回单个字符串，否则返回列表
        return captions[0] if isinstance(image, (list, tuple)) is False else captions

# def filter_keys(key_set):
#     def _f(dictionary):
#         return {k: v for k, v in dictionary.items() if k in key_set}
#
#     return _f
# 顶层定义，模块最前面
def keep_image_text_orig_size(sample: dict) -> dict:
    return {k: v for k, v in sample.items() if k in ("image", "text", "orig_size")}

# def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
#     """Return function over iterator that groups key, value pairs into samples.
#
#     :param keys: function that splits the key into key and extension (base_plus_ext)
#     :param lcase: convert suffixes to lower case (Default value = True)
#     """
#     current_sample = None
#     for filesample in data:
#         assert isinstance(filesample, dict)
#         fname, value = filesample["fname"], filesample["data"]
#         prefix, suffix = keys(fname)
#         if prefix is None:
#             continue
#         if lcase:
#             suffix = suffix.lower()
#         # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
#         #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
#         #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
#         if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
#             if valid_sample(current_sample):
#                 yield current_sample
#             current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
#         if suffixes is None or suffix in suffixes:
#             current_sample[suffix] = value
#     if valid_sample(current_sample):
#         yield current_sample

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples, skipping malformed entries."""
    current_sample = None
    for filesample in data:
        if not isinstance(filesample, dict):
            continue
        # 跳过缺少 fname 或 data 的样本
        if "fname" not in filesample or "data" not in filesample:
            continue
        try:
            fname = filesample["fname"]
            value = filesample["data"]
        except KeyError:
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # new sample 开始
        if current_sample is None or prefix != current_sample.get("__key__") or suffix in current_sample:
            if current_sample is not None and valid_sample(current_sample):
                yield current_sample
            # 初始化新的 sample，保留 __url__ 供后续使用
            current_sample = {"__key__": prefix, "__url__": filesample.get("__url__", None)}
        # 只保留需要的后缀
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    # 最后一批
    if current_sample is not None and valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class WebdatasetFilter:
    def __init__(self, min_size=1024):
        self.min_size = min_size

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get("width", 0.0) or 0.0) >= self.min_size and x_json.get(
                    "height", 0
                ) >= self.min_size
                return filter_size
            else:
                return False
        except Exception:
            return False

class FaceFilter:
    def __call__(self, x):
        try:
            if "is_valid" in x:
                is_valid = x["is_valid"]
                return is_valid
        except Exception as e:
            print(e)
            return False


def img2tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
    img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0  # to CHW 0~1
    img = img * 2 - 1  # to -1~1
    return img


def face_process(
    example,
    iv_face_root: str,
    device,
    use_seg_face: bool,
    face_shape: int = 128,
):
    """
    example 中已经有 example['image'] 这一 Tensor。
    其它参数：
      - iv_face_root: 模型根目录
      - device: "cpu" 或 GPU id (int)
      - use_seg_face: 是否加载分割模型
      - lock: 多线程的锁
      - face_shape: 输出大小
    """
    # 在这里第一次调用时会创建、后续直接复用
    face_am, face_features, image2text, face_seg = get_face_modules(
        iv_face_root, device, use_seg_face
    )
    try:
        image = example["image"]
        image = ((image.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)[:, :, (2, 1, 0)]  # bgr
        faces, tfs = face_am.forward(image)
        img_abslms = face_am.img_abslms
        text = image2text.forward(image)
        if len(faces) != 1:
            example['is_valid'] = False
        else:
            face = faces[0]
            example['is_valid'] = True

            # get kpimage
            for j in range(len(img_abslms)):
                kpimg = get_kp_image(img_abslms[j])
            kpimg = img2tensor(cv2.resize(kpimg, dsize=(face_shape, face_shape), interpolation=cv2.INTER_LANCZOS4))
            example["image_kp"] = kpimg  # Tensor 3,FACE_SHAPE,FACE_SHAPE

            # get face embedding
            feature = face_features.forward([image], [img_abslms])
            example["face_id_embed"] = torch.from_numpy(feature)

            if face_seg:
                face = cv2.resize(face, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                outimg = np.ones_like(face) * 255
                mask = face_seg.forward(face)
                mask = np.repeat(mask[:, :, None], 3, axis=2)
                face = face * mask + (1 - mask) * outimg
            face = img2tensor(
                cv2.resize(face, dsize=(face_shape, face_shape), interpolation=cv2.INTER_CUBIC))  # To RGB # TO -1~1
            example["image_face"] = face  # Tensor 3,FACE_SHAPE,FACE_SHAPE

            example['text'] = text
            return example
    except Exception as e:
        print(f"[FaceProcess] Error: {e} -> skipping this sample.")
        return None  # 会在 WebDataset pipeline 中被跳过




def draw_kps(kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    out_img = np.zeros([512, 512, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:,0]
        y = kps[index][:,1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)

    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)


    return out_img
def get_kp_image(face_kp):
    kps = np.array([[int(face_kp[104, 0]), int(face_kp[104, 1])],
                    [int(face_kp[105, 0]), int(face_kp[105, 1])],
                    [int(face_kp[54, 0]), int(face_kp[54, 1])],
                    [int(face_kp[96, 0]), int(face_kp[96, 1])],
                    [int(face_kp[100, 0]), int(face_kp[100, 1])]])

    kp_img = draw_kps(kps)  # BGR 0-255

    # 输出转换
    kp_img = kp_img.astype(np.uint8)

    return kp_img


def whole_image_transform(example, resolution=1024):
    image = example["image"]
    # resize
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
    # get crop coordinates
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    image = transforms.functional.crop(image, c_top, c_left, resolution, resolution)

    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.5], [0.5])(image)

    example["image"] = image
    return example



def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    texts = [example["text"] for example in data]
    images_face = torch.stack([example["image_face"] for example in data])
    images_kp = torch.stack([example["image_kp"] for example in data])
    face_id_embeds = torch.cat([example["face_id_embed"] for example in data], dim=0)


    return {
        "pixel_values": images,
        "conditioning_pixel_values": images_kp,
        "face_embedding": face_id_embeds,
        "face_images": images_face,
        'captions': texts
    }


def get_orig_size(json):
    return (int(json.get("width", 0.0)), int(json.get("height", 0.0)))

class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 10,
            face_shape=128,
            use_seg_face=False,
            accelerator=None,

    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))



        image_transform = functools.partial(
            whole_image_transform, resolution=resolution
        )


        if accelerator:
            dev = accelerator.device.index
            print("dataset use device {}".format(accelerator.device))
        else:
            dev = -1
            print("dataset use device {}".format("cpu"))
        # 只传 root / device / flag，让 face_process 自己去拿全局实例
        face_det_transform = functools.partial(
            face_process,
            iv_face_root=IV_FACE_ROOT,
            device=dev,
            use_seg_face=use_seg_face,
            face_shape=face_shape,
        )

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                orig_size="json",
                handler=wds.warn_and_continue,
            ),
            wds.map(keep_image_text_orig_size),
            wds.map_dict(orig_size=get_orig_size),
            wds.map(image_transform),
            wds.map(face_det_transform),
            wds.select(FaceFilter())
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url, deterministic=True),
            tarfile_to_samples_nothrow,
            wds.select(WebdatasetFilter(min_size=128)),
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=collate_fn),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        # each worker is iterating over this
        self.train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)



def tensor2pilimg(tensors):
    images = []
    for tensor in tensors:
        images.append(
            Image.fromarray(((tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)))
    return images


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class Text2ImageDatasetLoader:
    def __init__(
            self,
            input_dir: str,
            per_gpu_batch_size,
            global_batch_size,
            num_workers,
            resolution=512,
            face_shape=128,
            accelerator=None, # if NONE donot use gpu
            use_seg_face=False, # if use, face image will be seg


    ):
        self.input_dir = input_dir
        folder_path = Path(input_dir)
        # 递归地搜索所有 .tar 文件
        tar_files = folder_path.glob("**/*.tar")
        tar_files = [tar_file.__str__() for tar_file in tar_files]

        # 递归地搜索所有 .josn 文件 获取数据集大小
        json_files = folder_path.glob("**/*.json")
        json_files = [json_file.__str__() for json_file in json_files]

        self.number_examples = 0
        for json_file in json_files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                self.number_examples = self.number_examples + int(json_data['image_count'])

        self.dataset = Text2ImageDataset(train_shards_path_or_url=tar_files, num_train_examples=self.number_examples,
                                         per_gpu_batch_size=per_gpu_batch_size, global_batch_size=global_batch_size,
                                         num_workers=num_workers, resolution=resolution,
                                         face_shape=face_shape,use_seg_face=use_seg_face,
                                         accelerator=accelerator)
        self.train_dataloader = wds.WebLoader(
            self.dataset.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        num_worker_batches = math.ceil(self.number_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        # add meta-data to dataloader instance for convenience
        self.train_dataloader.num_batches = num_samples
        self.train_dataloader.num_samples = self.number_examples



if __name__ == '__main__':
    set_seed(2)

    input_dir = "/home/user/images/new_aws_tar_v2/Test"
    dataset_loader = Text2ImageDatasetLoader(input_dir, per_gpu_batch_size=2, global_batch_size=4, face_shape=512,
                                             num_workers=2, use_seg_face=False)
    train_dataloader = dataset_loader.train_dataloader
    print(dataset_loader.number_examples)
    for index, batch in enumerate(train_dataloader):
        images, texts, face_id_embeds, images_face, images_kp = batch['pixel_values'], batch['captions'], batch[
            'face_embedding'], batch['face_images'], batch['conditioning_pixel_values']
        print(images.size(), face_id_embeds.size(), images_face.size())
        print(texts)
        img = image_grid(tensor2pilimg(images), 2, len(images) // 2)
        faceimg = image_grid(tensor2pilimg(images_face), 2, len(images) // 2)
        kpimg = image_grid(tensor2pilimg(images_kp), 2, len(images) // 2)
        cv2.imshow('img', np.array(img)[:, :, (2, 1, 0)])
        cv2.imshow('faceimg', np.array(faceimg)[:, :, (2, 1, 0)])
        cv2.imshow('kpimg', np.array(kpimg)[:, :, (2, 1, 0)])
        cv2.waitKey(0)
