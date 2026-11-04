# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support UDA models

import warnings

from copy import deepcopy
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
UDA = MODELS
MTL = MODELS

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_train_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.model.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.model.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    if 'uda' in cfg:
        cfg.uda['model'] = cfg.model
        cfg.uda['max_iters'] = cfg.runner.max_iters

        if 'task' not in cfg.uda.keys():
            cfg.uda['task'] = ['seg']

        if len(cfg.uda['task']) == 1:
            # single-task
            cfg.uda['model']['type'] = 'EncoderMultiDecoder'
            if 'seg' in cfg.uda['task'][0]: 
                cfg.uda['model']['decode_head']['num_classes'] = 19
            elif 'depth' in cfg.uda['task'][0]:
                cfg.uda['model']['decode_head']['num_classes'] = 1
            elif 'norm' in cfg.uda['task'][0]:
                cfg.uda['model']['decode_head']['num_classes'] = 3
            #cfg.uda['model']['decode_head']['type'] = 'DAFormerMultiHead'
            if cfg.model['decode_head']['type'] == 'DLV2Head':
                cfg.model['decode_head']['type'] = 'DLV2MultiHead'        
            else:
                cfg.model['decode_head']['type'] = 'DAFormerMultiHead'
                
            cfg.uda['model']['decode_head']['task'] = cfg.uda['task']
        else: 
            # multi-task
            num_task = len(cfg.uda['task'])
            cfg.uda['model']['type'] = 'EncoderMultiDecoder'
            
            # segmentation as default
            #cfg.uda['model']['decode_head']['type'] = 'DAFormerMultiHead'
            
            if cfg.model['decode_head']['type'] == 'DLV2Head':
                cfg.model['decode_head']['type'] = 'DLV2MultiHead'        
            else:
                cfg.model['decode_head']['type'] = 'DAFormerMultiHead'
            
            cfg.uda['model']['decode_head']['task'] = cfg.uda['task']

            # if num_task == 2 and 'depth' in cfg.uda['task']:
            #     cfg.uda['model']['decode_head_depth'] = cfg.uda['model']['decode_head']
            #     cfg.uda['model']['decode_head_depth']['num_classes'] = 1
            # elif num_task == 2 and 'norm' in cfg.uda['task']:
            #     cfg.uda['model']['decode_head_norm'] = cfg.uda['model']['decode_head']
            #     cfg.uda['model']['decode_head_norm']['num_classes'] = 3
            # elif num_task == 3:
            #     cfg.uda['model']['decode_head_depth'] = cfg.uda['model']['decode_head']
            #     cfg.uda['model']['decode_head_depth']['num_classes'] = 1
            #     cfg.uda['model']['decode_head_norm'] = cfg.uda['model']['decode_head']
            #     cfg.uda['model']['decode_head_norm']['num_classes'] = 3

            # print("(before) cfg.uda['model']: ", cfg.uda['model'])
            # del cfg.uda['model']['decode_head']
            # print("(after) cfg.uda['model']: ", cfg.uda['model'])

        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return SEGMENTORS.build(
            cfg.model,
            default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""

    #input("build_segmentor is called")

    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_multi_task(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '


    if 'uda' in cfg:
        cfg.model['type'] = 'EncoderMultiDecoder'
        
        if cfg.model['decode_head']['type'] == 'DLV2Head':
            cfg.model['decode_head']['type'] = 'DLV2MultiHead'        
        else:
            cfg.model['decode_head']['type'] = 'DAFormerMultiHead'

        if 'task' not in cfg.uda.keys():
            #cfg.model['task'] = ['seg'] # default task - segmentation
            cfg.model['decode_head']['task'] = ['seg']
        else:
            cfg.model['decode_head']['task'] = cfg.uda['task']

        if len(cfg.uda['task']) == 1: # single-task
            if 'seg' in cfg.model['task']:
                cfg.model['decode_head']['num_classes'] = 19
            elif 'depth' in cfg.model['task']:
                cfg.model['decode_head']['num_classes'] = 1
            elif 'norm' in cfg.model['task']:
                cfg.model['decode_head']['num_classes'] = 3

    print("cfg.model: ", cfg.model)
    #input("here")
    return MTL.build(
        cfg.model, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))