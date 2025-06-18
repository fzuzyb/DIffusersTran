#!/usr/bin/env python
# encoding: utf-8
#@author: ZYB
#@license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
#@file: __init__.py
#@time: 2/27/23 6:11 PM

from .yolov5_widerface import FaceDetecter as YOLOV5FaceDetecter
from .yolov5_widerface import FaceDetecter_versions as YOLOV5FaceDetecter_versions

__all__ = ['YOLOV5FaceDetecter', "YOLOV5FaceDetecter_versions"]