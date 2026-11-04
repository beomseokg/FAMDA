# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy, depth_loss, disparity_loss, norm_loss

MIN_DEPTH, MAX_DEPTH = 1e-3, 80 # in meters
def depth2disp(d):
    return 65536 / (d+1)
def disp2meters(d):
    return 655.36 / d - 1
    
class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 task=None,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.task = task

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.conv_depth = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_norm = nn.Conv2d(channels, 3, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        self.count = 0

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      gt_depth_est,
                      gt_norm_est,
                      train_cfg,
                      seg_weight=None,
                      mask_depth_est=None,
                      mask_norm_est=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses_seg, losses_depth, losses_norm = None, None, None
        seg_logits, seg_logits_resize = None, None
        depth_logits, depth_logits_resize = None, None
        norm_logits, norm_logits_resize = None, None

        multi_task_logits = self.forward(inputs, task=self.task)

        if (multi_task_logits['seg'] is not None) and (gt_semantic_seg is not None):
            losses_seg, seg_logits_resize = self.losses_seg(multi_task_logits['seg'], gt_semantic_seg, seg_weight)

        if (multi_task_logits['depth'] is not None) and (gt_depth_est is not None):
            losses_depth, depth_logits_resize = self.losses_depth(multi_task_logits['depth'], gt_depth_est, mask_depth_est)

        if (multi_task_logits['norm'] is not None) and (gt_norm_est is not None):
            losses_norm, norm_logits_resize = self.losses_norm(multi_task_logits['norm'], gt_norm_est, mask_norm_est)

        return [losses_seg, losses_depth, losses_norm], [seg_logits, depth_logits, norm_logits], [seg_logits_resize, depth_logits_resize, norm_logits_resize]


    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        logits = self.forward(inputs, task=self.task)
        return logits

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def cls_depth(self, feat):
        """regression for each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_depth(feat)
        return output

    def cls_norm(self, feat):
        """regression for each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_norm(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses_seg(self, seg_logit, seg_label, seg_weight=None):

        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss, seg_logit


    @force_fp32(apply_to=('depth_logit', ))
    def losses_depth(self, depth_logit, depth_label, depth_mask):

        """Compute depth estimation loss."""
        loss = dict()

        # depth_logit: (2,1,128,128)
        # depth_label: (2,1,512,512)

        # print("depth_logit, depth_label in decode_head.py: ", depth_logit.shape, depth_label.shape)
        # print("depth_mask in decode_head.py: ", depth_mask.shape)

        depth_logit = resize(
            input=depth_logit,
            size=depth_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # depth_logit = F.sigmoid(depth_logit)
        # depth_label = depth_label
        # depth_label_meter = disp2meters(depth_label) # inverse depth -> depth (meter)

        # depth_label_mn = torch.zeros_like(depth_label) # normalization
        # depth_logit_m = torch.zeros_like(depth_logit)
        # mask = torch.zeros_like(depth_logit)

        # depth_mask *= ((depth_label_meter < 80) * (depth_label_meter > 1e-3))
        # depth_logit_masked = depth_logit * depth_mask
        # depth_logit_meter_masked = depth_label_meter * depth_mask
        # # depth_label_mn *= depth_mask.detach()
        # # depth_label_m *= depth_mask.detach()

        # for i in range(depth_label.shape[0]):
        #     min_depth, max_depth = depth_label_m[i].min(), depth_label_m[i].max()
        #     depth_label_mn[i] = (depth_label_m[i] - min_depth) / (max_depth - min_depth)
        #     depth_logit_m[i] = depth_logit[i] * (max_depth - min_depth) + min_depth

        # #depth_logit_m = disp2meters(depth_logit_dn) # inverse depth -> depth (meter)

        # loss['loss_depth'] = (torch.abs(depth_logit_masked - depth_label_mn) ** 2).mean()
        # loss['RMSE_depth'] = (torch.abs(depth_logit_m - depth_label_m) ** 2).mean()
        # #loss['berhu_depth'] = berhu.berhu(depth_logit, depth_label, None) #rmse.rmse(depth_logit_m, depth_label_m, depth_mask)
        # #loss['l1_depth'] = l1.scaled_l1(depth_logit, depth_label, None) #rmse.rmse(depth_logit_m, depth_label_m, depth_mask)
        # return loss, depth_logit

        depth_logit = F.sigmoid(depth_logit) * 655.35 + 1
        depth_label = depth_label.squeeze(1)

        depth_logit_m = disp2meters(depth_logit) # inverse depth -> depth (meter)
        depth_label_m = disp2meters(depth_label) # inverse depth -> depth (meter)

        #loss['loss_depth'] = depth_loss.rmse_grad(depth_logit_m, depth_label_m, depth_mask)
        loss['loss_depth'] = depth_loss.ssi_rmse(depth_logit_m, depth_label_m, depth_mask)
        loss['RMSE_depth'] = depth_loss.rmse(depth_logit_m, depth_label_m, depth_mask)
        loss['MAE_depth'] = depth_loss.mae(depth_logit_m, depth_label_m, depth_mask)

        return loss, depth_logit


    @force_fp32(apply_to=('norm_logit', ))
    def losses_norm(self, norm_logit, norm_label, norm_mask):

        """Compute depth estimation loss."""
        loss = dict()

        if norm_label.dim() == 5:
            # [B, 1, H, W, 3] -> [B, H, W, 3]
            norm_label = norm_label.squeeze(1)          # [B, H, W, 3]
            # [B, H, W, 3] -> [B, 3, H, W]
            norm_label = norm_label.permute(0, 3, 1, 2) # [B, 3, H, W]

        norm_logit = resize(
            input=norm_logit,
            size=norm_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        loss['loss_norm'] = norm_loss.l1(norm_logit, norm_label, norm_mask)
        loss['RMSE_norm'] = norm_loss.rmse(norm_logit, norm_label, norm_mask)
        loss['mErr_norm'] = norm_loss.merr(norm_logit, norm_label, norm_mask)

        # loss['loss_norm'] = torch.sum(torch.abs(norm_logit - norm_label) * norm_mask.unsqueeze(1)) / torch.sum(norm_mask) # L1 loss
        #loss['loss_norm'] = torch.sum(1 - F.cosine_similarity(norm_logit, norm_label.detach(), dim=1) * norm_mask.squeeze()) / torch.sum(norm_mask) # cos loss

        #loss['RMSE_norm'] = depth_loss.rmse(norm_logit_m, norm_label_m, norm_mask)
        #loss['MAE_norm'] = depth_loss.mae(norm_logit_m, norm_label_m, norm_mask)

        return loss, norm_logit