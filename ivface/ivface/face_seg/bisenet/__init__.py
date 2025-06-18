#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: __init__.py
#@time: 10/17/23 10:27 AM

from .infer_model import FaceSegModel as BiSeNetFaceSegModel
from .infer_model import classify_face_color
from .infer_model import __WEIGHTS_FNS__ as BiSeNetFaceSegModel_versions

__all__ = ['BiSeNetFaceSegModel', 'BiSeNetFaceSegModel_versions','classify_face_color']