# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

from mmcv.parallel import MMDistributedDataParallel

from mmseg.models import BaseSegmentor, build_segmentor
from abc import ABCMeta, abstractmethod


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model'])) # change here to build a quantized segmentor for student model

        self.weighted_loss = False
        self.pretrained_ema = None
        self.pretrained_kd = None
        self.dataset = None

        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        
        self.task = cfg['task']

        if 'SAM' not in cfg.keys():
            self.SAM = False
        else:
            self.SAM = cfg['SAM']
        if 'DAM' not in cfg.keys():
            self.DAM = False
        else:
            self.DAM = cfg['DAM']

        # Depth supervision switches (defaults reproduce the original FAMDA behaviour:
        # source depth loss was never backpropagated, target depth self-training was on).
        self.depth_src_sup = cfg.get('depth_src_sup', False)
        self.depth_tgt_st = cfg.get('depth_tgt_st', True)

        self.pretrained_ema = None
        if 'pretrained_ema' in cfg.keys():
            self.pretrained_ema = cfg['pretrained_ema']


    def get_model(self):
        return get_module(self.model)

    def get_kd_model(self):
        return get_module(self.kd_model)

    @abstractmethod
    def get_ema_model(self):
        """Placeholder to get teacher model."""
        pass

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat)
        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True, eval_teacher=False):
        """Simple test with single image."""

        #print("uda_decorator.py simple_test is called")
        #print("eval_teacher: ", eval_teacher, "self.kd: ", self.kd)
        #input("here")

        if eval_teacher:
            return self.get_ema_model().simple_test(img, img_meta, rescale)
        else:
            return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True, eval_teacher=False):
        """Test with augmentations.

        Only rescale=True is supported.
        """

        if eval_teacher == True:
            return self.get_ema_model().simple_test(img, img_meta, rescale)
        else:
            return self.get_model().aug_test(imgs, img_metas, rescale)
