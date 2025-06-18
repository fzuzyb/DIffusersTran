#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: __init__.py
# @time: 2/27/23 6:11 PM

from .infer_model import FaceLMer
from .infer_model import __WEIGHTS_FNS__ as FaceLM_versions

__all__ = ['FaceLMer', 'FaceLM_versions']
