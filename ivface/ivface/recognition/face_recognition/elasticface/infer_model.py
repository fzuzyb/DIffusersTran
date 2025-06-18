import cv2
import numpy as np
import os
from ivface.AppBaseModel import BaseModel


__WEIGHTS_FNS__ = {'v1': 'elasticface.onnx'}


def normalize(input: np.ndarray, p: int = 2, dim: int = 1, eps: float = 1e-12) -> np.ndarray:
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """

    denom = np.linalg.norm(input, ord=p, axis=dim, keepdims=True).clip(min=eps)
    return input / denom


class ElasticFaceModel(BaseModel):
    def __init__(self, IVFACE_ROOT="./weights",version='v1', max_batchsize=16,device=0):
        assert version in list(__WEIGHTS_FNS__.keys()), \
            "version only support {} , your version is {}".format(list(__WEIGHTS_FNS__.keys()), version)

        __WEIGHTS_PATH__ = os.path.join(IVFACE_ROOT, 'recognition/face_recognition/elasticface')
        super(ElasticFaceModel, self).__init__(version=version)
        self.load_model(__WEIGHTS_PATH__, __WEIGHTS_FNS__,device)
        self.alignment_mode = 'arcface'
        self.alignment_size = 112
        self.max_batchsize = max_batchsize

        self.__first_forward__()

    def __first_forward__(self):
        print('initialize FaceRecognition Model >>> elasticface_onnx {} ...'.format(self.version))

        _ = self.model.run([self.model.get_outputs()[0].name],
                           {self.model.get_inputs()[0].name:
                                np.random.randn(1, 3, self.alignment_size, self.alignment_size).astype(np.float32)})

    def resize(self, imgs, insize=112):
        imgs = [cv2.resize(img, (insize, insize), interpolation=cv2.INTER_CUBIC) for img in imgs]
        return imgs

    def input_preprocess(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        inputs = self.resize(imgs)
        inputs = np.stack(inputs, axis=0).astype(np.float32)

        inputs = inputs[:, :, :, (2, 1, 0)]
        inputs = ((inputs.transpose(0, 3, 1, 2) / 255.) - 0.5) / 0.5
        return inputs

    def forward(self, imgs):
        # imgs as list of img
        inputs = self.input_preprocess(imgs)
        num_forwards = int(np.ceil(len(inputs) / self.max_batchsize))

        features = []
        for i in range(num_forwards):
            feas = self.model.run(
                None, {self.model.get_inputs()[0].name:
                           inputs[i * self.max_batchsize: (i + 1) * self.max_batchsize]})[0]
            features.append(normalize(feas))

        features = np.concatenate(features, axis=0)

        return features