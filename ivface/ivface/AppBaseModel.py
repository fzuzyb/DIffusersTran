#!/usr/bin/env python
# encoding: utf-8
# @author: ZYB
# @license: (C) Copyright 2016-2021, Node Supply Chain Manager Corporation Limited.
# @file: AppBaseModel.py
# @time: 3/1/23 11:25 AM


import os
import onnxruntime as ort
import torch


class BaseModel:
    def __init__(self, version='v1'):
        file_abs = os.path.abspath(__file__)
        index = file_abs.find('applications')
        self.weights_root = file_abs[:index]
        self.version = version
        self.model = None

    def forward(self):
        pass

    def backward(self):
        pass

    def load_model(self, weights_path, weights_fns, device=0):

        file_endwith = weights_fns[self.version].split('.')[-1]
        assert file_endwith in ['pth', 'pt',
                                'onnx'], "file endwith only support 'pth','pt','onnx', however it is  {}".format(
            file_endwith)
        if file_endwith == "onnx" and self.model is None:
            if device!=-1:
                self.model = ort.InferenceSession(
                    os.path.join(self.weights_root, weights_path, weights_fns[self.version]),
                    providers=[('CUDAExecutionProvider', {'device_id': device}),
                               'CPUExecutionProvider'])
            else:
                self.model = ort.InferenceSession(
                    os.path.join(self.weights_root, weights_path, weights_fns[self.version]),
                    providers=[('CPUExecutionProvider', {'device_id': -1})])
        elif file_endwith in ["pt", 'pth'] and self.model is not None:

            resume_state = torch.load(
                os.path.join(self.weights_root, weights_path, weights_fns[self.version]),
                map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(resume_state, strict=True)
            self.model.eval()
        else:
            raise NotImplemented()
