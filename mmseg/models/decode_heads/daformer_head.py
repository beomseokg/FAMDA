# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import math
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

    def forward_without_cls(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))

        return x
    
    def forward(self, inputs, task):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x

@HEADS.register_module()
class DAFormerMultiHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerMultiHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.local_iter = None

        # self.embed_layers = {}
        # for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
        #                                      embed_dims):
        #     if i == self.in_index[-1]:
        #         self.embed_layers[str(i)] = build_layer(
        #             in_channels, embed_dim, **embed_neck_cfg)
        #     else:
        #         self.embed_layers[str(i)] = build_layer(
        #             in_channels, embed_dim, **embed_cfg)
        # self.embed_layers = nn.ModuleDict(self.embed_layers)

        # self.fuse_layer = build_layer(
        #     sum(embed_dims), self.channels, **fusion_cfg)

        self.embed_layers_seg, self.fuse_layer_seg = {}, {}
        self.embed_layers_depth, self.fuse_layer_depth = {}, {}
        self.embed_layers_norm, self.fuse_layer_norm = {}, {}
        
        for task_name in self.task:
            for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                                embed_dims):
                if i == self.in_index[-1]:
                    if task_name == 'seg':
                        self.embed_layers_seg[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_neck_cfg)
                    elif task_name == 'depth':
                        self.embed_layers_depth[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_neck_cfg)
                    elif task_name == 'norm':
                        self.embed_layers_norm[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_neck_cfg)
                else:
                    if task_name == 'seg':
                        self.embed_layers_seg[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_cfg)
                    elif task_name == 'depth':
                        self.embed_layers_depth[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_cfg)
                    elif task_name == 'norm':
                        self.embed_layers_norm[str(i)] = build_layer(
                            in_channels, embed_dim, **embed_cfg)
            if task_name == 'seg':
                self.embed_layers_seg = nn.ModuleDict(self.embed_layers_seg)
                self.fuse_layer_seg = build_layer(
                    sum(embed_dims), self.channels, **fusion_cfg)
            elif task_name == 'depth':
                self.embed_layers_depth = nn.ModuleDict(self.embed_layers_depth)
                self.fuse_layer_depth = build_layer(
                    sum(embed_dims), self.channels, **fusion_cfg)
            elif task_name == 'norm':
                self.embed_layers_norm = nn.ModuleDict(self.embed_layers_norm)
                self.fuse_layer_norm = build_layer(
                    sum(embed_dims), self.channels, **fusion_cfg)

        ##### channel-wise attention
        # num channels: 64, 128, 256, 512
        # self.channel_att_seg = nn.Sequential(ChannelAtt2(1, 1),
        #                         ChannelAtt2(1, 1),
        #                         ChannelAtt2(320, 320),
        #                         ChannelAtt2(512, 512))
        # self.channel_att_depth = nn.Sequential(ChannelAtt2(1, 1),
        #                           ChannelAtt2(1, 1),
        #                           ChannelAtt2(320, 320),
        #                           ChannelAtt2(512, 512))
        ##### channel-wise attention

        ##### cross-task distillation
        self.cross_task_distill = False
        if self.cross_task_distill:
            self.extract_seg = Extractor(in_ch=self.channels, out_ch=self.channels)
            self.extract_depth = Extractor(in_ch=self.channels, out_ch=self.channels)
            self.sa_seg = SABlock(in_ch=self.channels, out_ch=self.channels//2)
            self.sa_depth = SABlock(in_ch=self.channels, out_ch=self.channels//2)
            self.xtam_seg2depth = SxTAM(in_ch=self.channels)
            self.xtam_depth2seg = SxTAM(in_ch=self.channels)
            self.proj_seg2depth = Project(in_ch=self.channels+self.channels//2, out_ch=self.channels)
            self.proj_depth2seg = Project(in_ch=self.channels+self.channels//2, out_ch=self.channels)
            self.fuse_seg2depth = Project(in_ch=self.channels, out_ch=self.channels)
            self.fuse_depth2seg = Project(in_ch=self.channels, out_ch=self.channels)
        ##### cross-task distillation

    def forward_without_cls(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))

        return x
    
    def forward(self, inputs, task=['seg']):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {'seg': {}, 'depth': {}, 'norm': {}}

        # if self.local_iter > 10000:
        #     self.cross_task_distill = False

        # if self.local_iter % 100 == 0:
        #     import numpy as np
        #     softmax = nn.Softmax(dim=-1)
        #     x2_similarity = x[2].view(2,320,-1) @ x[2].view(2,320,-1).permute(0,2,1)
        #     x3_similarity = x[3].view(2,512,-1) @ x[3].view(2,512,-1).permute(0,2,1)
        #     np.savetxt("channel_wise_similarity/x2_similarity_"+str(self.local_iter)+".csv", x2_similarity[0].cpu().detach())
        #     np.savetxt("channel_wise_similarity/x3_similarity_"+str(self.local_iter)+".csv", x2_similarity[0].cpu().detach())

        for i in self.in_index:
            if 'seg' in task:
                # if i == 2 or i == 3:
                #     att_seg, corr_seg = self.channel_att_seg[i](x[i])                        
                #     _c['seg'][i] = self.embed_layers_seg[str(i)](x[i] + att_seg)
                # else:
                #     _c['seg'][i] = self.embed_layers_seg[str(i)](x[i])
                _c['seg'][i] = self.embed_layers_seg[str(i)](x[i])

                if _c['seg'][i].dim() == 3:
                    _c['seg'][i] = _c['seg'][i].permute(0, 2, 1).contiguous() \
                        .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                if _c['seg'][i].size()[2:] != os_size:
                    _c['seg'][i] = resize(
                        _c['seg'][i],
                        size=os_size,
                        mode='bilinear',
                        align_corners=self.align_corners)

            if 'depth' in task:
                # if i == 2 or i == 3:
                #     att_depth, corr_depth = self.channel_att_depth[i](x[i])
                #     _c['depth'][i] = self.embed_layers_depth[str(i)](x[i] + att_depth)
                # else:
                #     _c['depth'][i] = self.embed_layers_depth[str(i)](x[i])
                _c['depth'][i] = self.embed_layers_depth[str(i)](x[i])

                if _c['depth'][i].dim() == 3:
                    _c['depth'][i] = _c['depth'][i].permute(0, 2, 1).contiguous() \
                        .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                if _c['depth'][i].size()[2:] != os_size:
                    _c['depth'][i] = resize(
                        _c['depth'][i],
                        size=os_size,
                        mode='bilinear',
                        align_corners=self.align_corners)

            if 'norm' in task:
                _c['norm'][i] = self.embed_layers_norm[str(i)](x[i])
                
                if _c['norm'][i].dim() == 3:
                    _c['norm'][i] = _c['norm'][i].permute(0, 2, 1).contiguous()\
                        .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                if _c['norm'][i].size()[2:] != os_size:
                    _c['norm'][i] = resize(
                        _c['norm'][i],
                        size=os_size,
                        mode='bilinear',
                        align_corners=self.align_corners)

        x = {'seg': None, 'depth': None, 'norm': None}

        ##### cross-task attention
        if 'seg' in self.task and 'depth' in self.task and self.cross_task_distill:
            if self.cross_task_distill:
                for i in self.in_index:
                    f_base_seg = _c['seg'][i]
                    f_base_depth = _c['depth'][i]

                    f_dir_seg = self.extract_seg(_c['seg'][i])
                    f_dir_depth = self.extract_depth(_c['depth'][i])

                    sa_seg = self.sa_seg(f_dir_seg)
                    sa_depth  = self.sa_depth(f_dir_depth)

                    s_corr_seg2depth = self.xtam_seg2depth(f_base_depth, f_dir_seg)
                    s_corr_depth2seg = self.xtam_depth2seg(f_base_seg, f_dir_depth)

                    proj_seg2depth = self.proj_seg2depth(torch.cat([sa_seg, s_corr_seg2depth], dim=1))
                    proj_depth2seg = self.proj_depth2seg(torch.cat([sa_depth, s_corr_depth2seg], dim=1))

                    _c['seg'][i] = f_base_seg + self.fuse_depth2seg(proj_depth2seg)
                    _c['depth'][i] = f_base_depth + self.fuse_seg2depth(proj_seg2depth)

        if 'seg' in self.task and 'depth' in self.task and self.cross_task_distill:
            x_seg = self.fuse_layer_seg(torch.cat(list(_c['seg'].values()), dim=1))
            x_depth = self.fuse_layer_depth(torch.cat(list(_c['depth'].values()), dim=1)) # in: (B, 1024, 128, 128), out: (B, 1, 128, 128)

            """
            f_base_seg = x_seg
            f_base_depth = x_depth

            f_dir_seg = self.extract_seg(x_seg)
            f_dir_depth = self.extract_depth(x_depth)
            sa_seg = self.sa_seg(f_dir_seg)
            sa_depth = self.sa_seg(f_dir_depth)
            s_corr_seg2depth = self.xtam_seg2depth(f_base_depth, f_dir_seg)
            s_corr_depth2seg = self.xtam_depth2seg(f_base_seg, f_dir_depth)
            proj_seg2depth = self.proj_seg2depth(torch.cat([sa_seg, s_corr_seg2depth], dim=1))
            proj_depth2seg = self.proj_depth2seg(torch.cat([sa_depth, s_corr_depth2seg], dim=1))

            x_seg = f_base_seg + self.fuse_depth2seg(proj_depth2seg)
            x_depth = f_base_depth + self.fuse_seg2depth(proj_seg2depth)
            """

            f_base_seg = x_seg
            f_base_depth = x_depth

            s_corr_depth2seg = self.xtam_depth2seg(f_base_depth, f_base_seg)
            s_corr_seg2depth = self.xtam_seg2depth(f_base_seg, f_base_depth)

            x_seg = f_base_seg + s_corr_depth2seg
            x_depth = f_base_depth + s_corr_seg2depth

            x_seg = self.cls_seg(x_seg)
            x_depth = self.cls_depth(x_depth) # in: (B, 256, 128, 128), out: (B, 1, 128, 128)
            x['seg'] = x_seg
            x['depth'] = x_depth
        else:
            if 'seg' in task:
                x_seg = self.fuse_layer_seg(torch.cat(list(_c['seg'].values()), dim=1))
                x_seg = self.cls_seg(x_seg)
                x['seg'] = x_seg

            if 'depth' in task:
                x_depth = self.fuse_layer_depth(torch.cat(list(_c['depth'].values()), dim=1))
                x_depth = self.cls_depth(x_depth)
                x['depth'] = x_depth
    
            if 'norm' in task:
                x_norm = self.fuse_layer_norm(torch.cat(list(_c['norm'].values()), dim=1))
                x_norm = self.cls_norm(x_norm)
                x['norm'] = x_norm

        return x


class ChannelAtt(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_ch, out_ch):
        super(ChannelAtt, self).__init__()
        self.conv_q = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_k = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv_v = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        #print("B, C, H, W: ", B, C, H, W)

        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        q, k, v = q.view(B,C,-1), k.view(B,C,-1).permute(0,2,1), v.view(B,C,-1)
        coeff = math.sqrt(H * W)
        corr = self.softmax(q @ k / coeff)

        out = (corr @ v).view_as(x)
        return out

class ChannelAtt2(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_ch, out_ch):
        super(ChannelAtt2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        #print("B, C, H, W: ", B, C, H, W)
        k = self.conv(x)
        coeff = math.sqrt(H*W)
        corr = self.sigmoid(k.view(B,C,-1) @ k.view(B,C,-1).permute(0,2,1) / coeff)
        out = corr @ x.view(B,C,-1)
        out = out.view_as(x)
        return out, corr

# borrowed from https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/vendor/mtlpt/layers.py
class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_ch, out_ch):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


# borrowed from https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/vendor/mtlpt/layers.py
class SxTAM(nn.Module):
    """Spatial cross-Task Attention Module"""
    def __init__(self, in_ch, s=4, use_alpha=True):
        super().__init__()

        ## Projection layers
        self.conv_b = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_c = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(),
                                    nn.Flatten(2))

        self.conv_d = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

        ## Channel-wise weights
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_ch, 1, 1))

        self.down = nn.MaxPool2d(s, s)
        self.d = nn.Upsample(scale_factor=1/s)
        self.u = nn.Upsample(scale_factor=s)

    def forward(self, x, y):
        # downscale and flatten spatial dimension
        x_ = self.d(x)
        B = self.conv_b(x_).transpose(1,2)
        C = self.conv_c(self.down(y))
        D = self.conv_d(self.down(y))

        # compute correlation matrix
        # (b, hw_x, c) @ (b, c, hw_y) = (b, hw_x, hw_y) -T-> (b, hw_y, hw_x)
        coeff = math.sqrt(B.size(2))
        corr = self.softmax(B @ C / coeff).transpose(1,2)

        # (b, c, hw_y) @ (b, hw_y, hw_x) = (b, c, hw_x) -view-> (b, c, h_x, w_x)
        out = self.u((D.flatten(2) @ corr).view_as(x_))

        if self.use_alpha:
            out *= self.alpha

        return out

# borrowed from https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/vendor/mtlpt/layers.py
class Project(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU())

# borrowed from https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/vendor/mtlpt/layers.py
class Extractor(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU())
