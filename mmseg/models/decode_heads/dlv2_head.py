from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
import torch.nn as nn

@HEADS.register_module()
class DLV2Head(BaseDecodeHead):

    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2Head, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, inputs):
        """Forward function."""
        # for f in inputs:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0]
        for i in range(len(aspp_outs) - 1):
            out += aspp_outs[i + 1]
        return out
        
        
@HEADS.register_module()
class DLV2MultiHeadSimple(BaseDecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2MultiHeadSimple, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        
        self.local_iter = None
        
        # Create separate ASPP modules for each task with correct output channels
        self.aspp_modules_seg = ASPPModule(
            dilations,
            self.in_channels,
            self.num_classes,  # 19 channels for segmentation
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
            
        self.aspp_modules_depth = ASPPModule(
            dilations,
            self.in_channels,
            1,  # 1 channel for depth
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
            
        self.aspp_modules_norm = ASPPModule(
            dilations,
            self.in_channels,
            3,  # 3 channels for surface normals
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        
        # Keep backward compatibility - alias the main one for initialization
        self.aspp_modules = self.aspp_modules_seg

    def forward(self, inputs, task=['seg']):
        """Forward function."""
        # for f in inputs:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        x = self._transform_inputs(inputs)
        
        outputs = {'seg': None, 'depth': None, 'norm': None}
        
        if 'seg' in task:
            aspp_outs_seg = self.aspp_modules_seg(x)
            out_seg = aspp_outs_seg[0]
            for i in range(len(aspp_outs_seg) - 1):
                out_seg += aspp_outs_seg[i + 1]
            outputs['seg'] = out_seg
            
        if 'depth' in task:
            aspp_outs_depth = self.aspp_modules_depth(x)
            out_depth = aspp_outs_depth[0]
            for i in range(len(aspp_outs_depth) - 1):
                out_depth += aspp_outs_depth[i + 1]
            outputs['depth'] = out_depth
            
        if 'norm' in task:
            aspp_outs_norm = self.aspp_modules_norm(x)
            out_norm = aspp_outs_norm[0]
            for i in range(len(aspp_outs_norm) - 1):
                out_norm += aspp_outs_norm[i + 1]
            outputs['norm'] = out_norm
            
        return outputs
        
        
@HEADS.register_module()
class DLV2MultiHead(BaseDecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2MultiHead, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        
        self.local_iter = None
        
        # Create separate ASPP modules for each task with correct output channels
        self.aspp_modules_seg = ASPPModule(
            dilations,
            self.in_channels,
            256,  # Use intermediate channels for better feature learning
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
            
        self.aspp_modules_depth = ASPPModule(
            dilations,
            self.in_channels,
            256,  # intermediate channels
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
            
        self.aspp_modules_norm = ASPPModule(
            dilations,
            self.in_channels,
            256,  # intermediate channels
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        
        # Keep backward compatibility - alias the main one for initialization
        self.aspp_modules = self.aspp_modules_seg
        
        # Add stronger layers to combine ASPP outputs
        self.fusion_seg = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.fusion_depth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fusion_norm = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Final output layers
        self.final_seg = nn.Conv2d(256, self.num_classes, 1)
        self.final_depth = nn.Conv2d(128, 1, 1)
        self.final_norm = nn.Conv2d(256, 3, 1)
        
        # Attention modules for better feature weighting
        self.attention_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, task=['seg']):
        """Forward function."""
        x = self._transform_inputs(inputs)
        
        outputs = {'seg': None, 'depth': None, 'norm': None}
        
        if 'seg' in task:
            aspp_outs_seg = self.aspp_modules_seg(x)
            # Weighted combination instead of simple addition
            out_seg = aspp_outs_seg[0].clone()  # Clone to avoid inplace issues
            for i in range(len(aspp_outs_seg) - 1):
                out_seg = out_seg + aspp_outs_seg[i + 1]
            out_seg = out_seg / len(aspp_outs_seg)  # Average instead of sum
            
            # Apply fusion layers
            out_seg = self.fusion_seg(out_seg)
            out_seg = self.final_seg(out_seg)
            outputs['seg'] = out_seg
            
        if 'depth' in task:
            aspp_outs_depth = self.aspp_modules_depth(x)
            # Weighted combination
            out_depth = aspp_outs_depth[0].clone()  # Clone to avoid inplace issues
            for i in range(len(aspp_outs_depth) - 1):
                out_depth = out_depth + aspp_outs_depth[i + 1]
            out_depth = out_depth / len(aspp_outs_depth)
            
            # Apply attention for depth
            attention = self.attention_depth(out_depth)
            out_depth = out_depth * attention
            
            # Apply fusion layers
            out_depth = self.fusion_depth(out_depth)
            out_depth = self.final_depth(out_depth)
            outputs['depth'] = out_depth
            
        if 'norm' in task:
            aspp_outs_norm = self.aspp_modules_norm(x)
            # Weighted combination
            out_norm = aspp_outs_norm[0].clone()  # Clone to avoid inplace issues
            for i in range(len(aspp_outs_norm) - 1):
                out_norm = out_norm + aspp_outs_norm[i + 1]
            out_norm = out_norm / len(aspp_outs_norm)
            
            # Apply fusion layers
            out_norm = self.fusion_norm(out_norm)
            out_norm = self.final_norm(out_norm)
            outputs['norm'] = out_norm
            
        return outputs
        
        
@HEADS.register_module()
class DLV2MultiHeadV2(BaseDecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2MultiHeadV2, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        
        self.local_iter = None
        
        # Create separate ASPP modules for each task (similar to embed_layers in DAFormer)
        self.aspp_modules_seg = ASPPModule(
            dilations,
            self.in_channels,
            256,  # Use intermediate channels
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=False))
            
        self.aspp_modules_depth = ASPPModule(
            dilations,
            self.in_channels,
            256,  # Use intermediate channels
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=False))
            
        self.aspp_modules_norm = ASPPModule(
            dilations,
            self.in_channels,
            256,  # Use intermediate channels
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=False))
        
        # Keep backward compatibility - alias the main one for initialization
        self.aspp_modules = self.aspp_modules_seg
        
        # Fusion layers (similar to fuse_layer in DAFormer).strong layers to combine ASPP outputs
        self.fuse_layer_seg = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.fuse_layer_depth = nn.Sequential(
            nn.Conv2d(256, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        
        self.fuse_layer_norm = nn.Sequential(
            nn.Conv2d(256, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        
        # Classification heads (similar to DAFormer)
        self.conv_seg = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.conv_depth = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.conv_norm = nn.Conv2d(self.channels, 3, kernel_size=1)

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

    def forward_without_cls(self, inputs):
        """Forward function without classification (similar to DAFormer)."""
        x = self._transform_inputs(inputs)
        
        # Process with ASPP modules
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0].clone()
        for i in range(len(aspp_outs) - 1):
            out = out + aspp_outs[i + 1]
        out = out / len(aspp_outs)
        
        # Apply fusion layer
        x = self.fuse_layer_seg(out)
        return x

    def forward(self, inputs, task=['seg']):
        """Forward function (structure similar to DAFormer)."""
        x = self._transform_inputs(inputs)
        
        outputs = {'seg': None, 'depth': None, 'norm': None}
        
        if 'seg' in task:
            # processing for segmentation (original DLV2 style)
            aspp_outs_seg = self.aspp_modules_seg(x)
            out_seg = aspp_outs_seg[0].clone()
            for i in range(len(aspp_outs_seg) - 1):
                out_seg = out_seg + aspp_outs_seg[i + 1]
            
            out_seg = out_seg / len(aspp_outs_seg)  # Average instead of sum            
            # Apply simple fusion layer (just 1x1 conv)
            out_seg = self.fuse_layer_seg(out_seg)
            # Apply classification
            out_seg = self.cls_seg(out_seg)
            outputs['seg'] = out_seg
           
                        
        if 'depth' in task:
            # Sophisticated processing for depth
            aspp_outs_depth = self.aspp_modules_depth(x)
            out_depth = aspp_outs_depth[0].clone()
            for i in range(len(aspp_outs_depth) - 1):
                out_depth = out_depth + aspp_outs_depth[i + 1]
            out_depth = out_depth / len(aspp_outs_depth)
            
            # Apply sophisticated fusion layer
            out_depth = self.fuse_layer_depth(out_depth)
            # Apply classification
            out_depth = self.cls_depth(out_depth)
            outputs['depth'] = out_depth
            
        if 'norm' in task:
            # processing for norm
            aspp_outs_norm = self.aspp_modules_norm(x)
            out_norm = aspp_outs_norm[0].clone()
            for i in range(len(aspp_outs_norm) - 1):
                out_norm = out_norm + aspp_outs_norm[i + 1]
            out_norm = out_norm / len(aspp_outs_norm)
            
            # Apply fusion layer
            out_norm = self.fuse_layer_norm(out_norm)
            # Apply classification
            out_norm = self.cls_norm(out_norm)
            outputs['norm'] = out_norm
            
        return outputs   