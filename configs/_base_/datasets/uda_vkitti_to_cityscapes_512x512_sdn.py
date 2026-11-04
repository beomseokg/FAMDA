# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'VkittiDataset'
data_root = 'data/vkitti2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
vkitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadDepthEstimations'),
    dict(type='LoadNormEstimations'),
    dict(type='Resize', img_scale=(1692, 512)), # original: (1242, 375)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=1, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_depth_est', 'gt_norm_est', 'mask_depth_est', 'mask_norm_est']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadDepthEstimations'),
    dict(type='LoadNormEstimations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=1, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_depth_est', 'gt_norm_est', 'mask_depth_est', 'mask_norm_est']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='VkittiDataset',
            data_root='data/vkitti2/',
            img_dir='rgb',
            ann_dir='classSegmentation_label',
            depth_dir='depth',
            pipeline=vkitti_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            depth_dir='disparity/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        depth_dir='disparity/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        depth_dir='disparity/val',
        pipeline=test_pipeline))
