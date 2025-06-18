#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: __init__.py
# @time: 2/28/23 9:51 AM

from .infer_model import FaceDetecter
from .infer_model import __WEIGHTS_FNS__ as FaceDetecter_versions

__all__ = ['FaceDetecter', 'FaceDetecter_versions']
