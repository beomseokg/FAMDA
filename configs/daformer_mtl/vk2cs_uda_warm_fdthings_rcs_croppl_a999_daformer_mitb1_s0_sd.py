# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
#
# WACV #1315 rebuttal baseline: "DAFormer-multitask w/o SAM and w/o DAM".
# No foundation-model guidance at all.
#   seg   : source GT + EMA-teacher pseudo-labels (DACS class-mix), NO SAM refinement
#   depth : source GT only (no DAM, no target self-training)
#
# Depth supervision is controlled by two uda flags (mmseg/models/uda/uda_decorator.py):
#   depth_src_sup : source depth GT loss is backpropagated      (dacs.py L738)
#   depth_tgt_st  : target depth self-training branch is active (dacs.py L1072)
# Their defaults (False / True) reproduce the original FAMDA behaviour, so the existing
# *_SAM_DAM configs are unaffected. Pre-flag code snapshot: mmseg/models/uda/dacs_260818.py
#
# NOTE: uses the *_sdn* dataset base (same as the FAMDA configs). uda_dataset.py
# requires the norm keys unconditionally; dacs.py drops them for a 2-task run.

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb1.py',
    # Virtual KITTI2->Cityscapes Data Loading
    '../_base_/datasets/uda_vkitti_to_cityscapes_512x512_sdn.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    task=['seg', 'depth'],
    # No foundation-model guidance
    SAM=False,
    DAM=False,
    # Depth supervision
    depth_src_sup=True,
    depth_tgt_st=False
    )
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU+RMSE', efficient_test=True)
# Meta Information for Result Analysis
name = 'vk2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb1_s0_sd'
exp = 'basic'
name_dataset = 'vkitti2cityscapes'
name_architecture = 'daformer_sepaspp_mitb1'
name_encoder = 'mitb1'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
