# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.utils import base_kd
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
#from quantization.latency import measure_inference_latency
from mmseg.models.discriminator.discriminator import Discriminator
from mmseg.models.generator.generator import _ResGeneratorT2Net, _MultiscaleDiscriminator
from mmseg.models.utils.image_pool import ImagePool
from ..losses import depth_loss, norm_loss

from mmseg.ops import resize
from depth_anything.dpt import DepthAnything
#from depth_anything_v2.dpt import DepthAnythingV2

#from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as maskUtils
#from fastsam import FastSAM, FastSAMPrompt
from torchvision.utils import save_image
import cv2
import time
import joblib

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/main/models/normals.py
def remove_borders(x):
    """Remove 1 pixel along all 4 borders."""
    x[..., (0,-1), :] = False
    x[..., (0,-1)] = False
    return x

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/main/models/normals.py
def dilate_mask(m, ksize):
    padding = {3: 1, 5: 2}[ksize]
    m = F.conv2d(1 - m, weight=torch.ones(1, 1, ksize, ksize, device=m.device), padding=padding)
    m[m > 0] = 1
    return (1 - m).bool()

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/dataset/base.py#L361
def get_K(fx, fy, u0, v0, u_offset, v_offset, full_res_shape, do_flip):
    if do_flip:
        u0 = full_res_shape[0] - u0
        v0 = full_res_shape[1] - v0

    return np.array([[    fx,        0,  u0 - u_offset,  0],
                    [      0,       fy,  v0 - v_offset,  0],
                    [      0,        0,              1,  0],
                    [      0,        0,              0,  1]], dtype=np.float32)

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/models/normals.py
def unproject(depth, inv_K):
    h, w = depth.shape[-2:]

    meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w))#, indexing='ij')
    pix_coords = torch.stack(meshgrid).flip(0).unsqueeze(0).flatten(2).float().to(depth.device)
    ones = torch.ones(1, 1, h*w, device=depth.device)
    pix_coords = torch.cat((pix_coords, ones), dim=1)

    # print("depth in dacs.py: ", depth.shape)
    # input("here")
    # forward
    cam_points = depth.flatten(2) * (inv_K[:3, :3].unsqueeze(0) @ pix_coords)
    return cam_points.reshape(-1, 3, h, w).contiguous()


# borrowed from: https://github.com/fudan-zvg/Semantic-Segment-Anything/blob/main/scripts/pipeline.py
def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None,
                                 model='segformer'):

    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape
    class_names = []
    class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)

    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]

        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        semantc_mask[valid_mask] = top_1_propose_class_ids

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
    
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks = [], []

    # semantic prediction
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_bitmasks.append(class_mask)
    
    del img
    del anns
    del class_ids
    del semantc_mask

    del class_names
    del semantic_bitmasks


# reference: https://github.com/alopezgit/DESC/blob/master/models/base_model.py
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
        else:
            print('Error freezing')


# reference: https://github.com/alopezgit/DESC/blob/master/utils/dataset_util.py
def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = F.upsample(img, size=(nh, nw), mode='nearest')
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs

def inverse_sigmoid(y):
    return torch.log(y / (1 - y))

def bce_loss(y_pred, y_label):
    y_truth = torch.ones_like(y_pred, device=y_pred.device)*y_label
    return F.binary_cross_entropy_with_logits(y_pred, y_truth)

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


def object_wise_inconsistency(logits_teacher, logits_student, labels=None):
    # logits should be (batch=2, class=19, height=512, width=512)
    if labels != None:
        ignore = labels != 255 # False for label=255, dim=(2,1,512,512)

    teacher_pred_indices = torch.max(logits_teacher, dim=1)[1] # (2,512,512)
    if labels != None:
        one_hot_teacher = ignore * torch.nn.functional.one_hot(teacher_pred_indices, num_classes=19).permute(0,3,1,2) # (2,19,512,512)
    else:
        one_hot_teacher = torch.nn.functional.one_hot(teacher_pred_indices, num_classes=19).permute(0,3,1,2) # (2,19,512,512)
    one_hot_teacher = one_hot_teacher == 1 # (2,19,512,512) make it boolean, True for 1 and False for 0

    student_pred_indices = torch.max(logits_student, dim=1)[1] # (2,512,512)
    if labels != None:
        one_hot_student = ignore * torch.nn.functional.one_hot(student_pred_indices, num_classes=19).permute(0,3,1,2)
    else:
        one_hot_student = torch.nn.functional.one_hot(student_pred_indices, num_classes=19).permute(0,3,1,2)
    ont_hot_student = one_hot_student == 1 # (2,19,512,512) make it boolean, True for 1 and False for 0 

    intersection = torch.logical_and(one_hot_teacher, one_hot_student) # (2,19,512,512)
    intersection = intersection.permute(0,2,3,1).reshape(2,-1,19)
    intersection = torch.sum(intersection, dim=1) # (2,19)

    union = torch.logical_or(one_hot_teacher, one_hot_student) # (2,19,512,512)
    union = union.permute(0,2,3,1).reshape(2,-1,19)
    union = torch.sum(union, dim=1) # (2,19)
    noise_mask = union > int(512*512*1e-3) # ignore inconsistency if union is less than 0.1% of img

    inconsistency = 1 - (intersection / (union + 1e-9)) # (2,19)
    inconsistency = (noise_mask * inconsistency).sum(0)
    object_num = noise_mask.sum(0)

    inconsistency_dict={}
    for i in range(19):
        inconsistency_dict[str(i)] = float(inconsistency[i].cpu())

    object_num_dict={}
    for i in range(19):
        object_num_dict[str(i)] = int(object_num[i].cpu())

    return inconsistency_dict, object_num_dict

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        self.class_inconsistency = {}

        self.ema_model = build_segmentor(deepcopy(cfg['model']))
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.discriminator_depth = None #Discriminator(in_ch=1)
        self.net_s2t = None #_ResGeneratorT2Net(3, 3, 64, 9, 'batch', 'PReLU', 0, False, gpu_ids=[0])
        self.net_Ds2t = None #_MultiscaleDiscriminator(3, 64, 3, 1, 'batch', 'PReLU', gpu_ids=[0]) # third argument: n_layers_D
        self.fake_img_pool = ImagePool(pool_size=20)

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80 # in meters

        if self.DAM == True:            
            # Depth Anythign V1
            depth_anything_encoder = 'vitl' # 'vits', 'vitb', 'vitl'
            self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(depth_anything_encoder)).to('cuda').eval()

            # Depth Anythign V2
            # depth_anything_v2_configs = {
            #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            # }

            # depth_anything_v2_encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
            # self.depth_anything_v2 = DepthAnythingV2(**depth_anything_v2_configs[depth_anything_v2_encoder])
            # self.depth_anything_v2.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{depth_anything_v2_encoder}.pth', map_location='cpu'))
            # self.depth_anything_v2 = self.depth_anything_v2.to('cuda').eval()
            ###

        #self.fastsam = FastSAM('fastsam/ckp/FastSAM-x.pt')

        """
        sam = sam_model_registry["vit_h"](checkpoint='segment_anything/ckp/sam_vit_h_4b8939.pth').to('cuda')
        self.seg_anything = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=128, #if args.dataset == 'foggy_driving' else 64,
            # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode='coco_rle',
        ) # mask_branch_model

        num_params = sum(p.numel() for p in sam.parameters())
        param_size_bytes = sum(p.numel() * p.element_size() for p in sam.parameters())
        total_size_mb = param_size_bytes / (1024 ** 2)
        print("total SAM num_params: ", num_params)
        print("total SAM memory size: ", total_size_mb)
        """

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def get_kd_model(self):
        return get_module(self.kd_model)


    def _load_pretrained_seg_backbone(self):
        # (1) copy pretrained seg model into student
        pretrained_seg = "work_dirs/local-basic/230722_0536_syn2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_bc181/iter_40000.pth"
        checkpoint = load_checkpoint(
            self.get_model(),
            pretrained_seg,
            map_location=self.device,
            revise_keys=[(r'^module\.', ''), ('model.', '')])
        # In revise_keys, 'model' is student during pre-training
        # to load teacher during pre-training, change 'model' to 'ema_model'

    def _init_depth_decoder(self):
        named_mp = list(self.get_model().named_parameters())
        named_mcp = list(self.get_ema_model().named_parameters())

        for i in range(len(named_mcp)):
            if 'decode' in named_mcp[i][0]:
                named_mp[i][1].data = named_mcp[i][1].data.clone()

            if 'backbone' in named_mcp[i][0]:
                named_mcp[i][1].data = named_mp[i][1].data.clone()
        
        for param in self.get_ema_model().parameters():
            param.detach_()

    def _load_pretrained_ema_weights(self):
        # For two-stage, we should load pre-trained large models (either teacher or student)
        # load pretrained teacher
        if self.pretrained_ema != None:
            checkpoint = load_checkpoint(
                self.get_ema_model(),
                self.pretrained_ema,
                map_location=self.device,
                revise_keys=[(r'^module\.', ''), ('ema_model.', '')]) # change 'model' to 'ema_model' to load teacher
            # In revise_keys, 'model' is student during pre-training
            # to load teacher during pre-training, change 'model' to 'ema_model'
            for param in self.get_ema_model().parameters():
                param.detach_()
        else:
            raise ValueError('pretrained large model is required in second stage')           

        """
        checkpoint = load_checkpoint(
            self.get_ema_model2(),
            self.pretrained_ema,
            map_location=self.device,
            revise_keys=[(r'^module\.', ''), ('ema_model.', '')])
        # In revise_keys, 'model' is student during pre-training
        # to load teacher during pre-training, change 'model' to 'ema_model'
        for param in self.get_ema_model2().parameters():
            param.detach_()
        """

        # load pretrained small student
        if self.pretrained_kd != None:
            checkpoint = load_checkpoint(
                self.get_model(),
                self.pretrained_kd,
                map_location=self.device,
                revise_keys=[(r'^module\.', ''), ('kd_model.', '')]) 
            # In revise_keys, 'kd_model' is small student during pre-training in one-stage        

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())

        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def depth2disp(self, d):
        return 65536 / (d+1)

    def disp2meters(self, d):
        return 655.36 / d - 1

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch) # forward_train is called inside here
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs


    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)


    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def epsilon_map(self, pred, task, detach=False, src_gt=None):
        if task == 'depth':
            #d = torch.relu(pred)
            dmin, dmax = src_gt.min(), src_gt.max()
            e = (pred-dmin)/(dmax-dmin)
            dout = self.discriminator_depth(e.detach() if detach else e)

        # elif task == 'seg':
        #     e = prob_2_entropy(pred.softmax(1))
        #     dout = D(e.detach() if detach else e)
        return dout

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, gt_depth_est=None, mask_depth_est=None, 
                      target_gt_depth_est=None, target_mask_depth_est=None,
                      gt_norm_est=None, mask_norm_est=None, 
                      target_gt_norm_est=None, target_mask_norm_est=None):
        
        start = time.time()
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

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        self.device = dev

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()

            # self._load_pretrained_seg_backbone()           
            # self._init_depth_decoder()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        elif self.pretrained_ema == None:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        if self.local_iter == 0 and self.pretrained_ema != None:
            self._load_pretrained_ema_weights()


        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        mix_masks_seg = get_class_masks(gt_semantic_seg)

        self.get_model().decode_head.local_iter = self.local_iter
        self.get_ema_model().decode_head.local_iter = self.local_iter

        # estimate memory and paramters
        """
        total_num_params = sum(p.numel() for p in self.get_ema_model().parameters())
        backbone_num_params = sum(p.numel() for p in self.get_ema_model().backbone.parameters())
        decode_num_params = sum(p.numel() for p in self.get_ema_model().decode_head.parameters())

        total_param_size_bytes = sum(p.numel() * p.element_size() for p in self.get_ema_model().parameters())
        backbone_param_size_bytes = sum(p.numel() * p.element_size() for p in self.get_ema_model().backbone.parameters())
        decode_param_size_bytes = sum(p.numel() * p.element_size() for p in self.get_ema_model().decode_head.parameters())

        total_size_mb = total_param_size_bytes / (1024 ** 2)
        backbone_size_mb = backbone_param_size_bytes / (1024 ** 2)
        decode_size_mb = decode_param_size_bytes / (1024 ** 2)

        print("total, backbone, and decode num_params: ", total_num_params, backbone_num_params, decode_num_params)
        print("total, backbone, and decode size: ", total_size_mb, backbone_size_mb, decode_size_mb)
        input("here")
        """


        # estimate per image latency
        """
        target_img_518 = resize(input=target_img, size=[518,518], mode='bilinear', align_corners=False)
        for i in range(100):
            start = time.time()
            ema_logits, ema_raw_logits = self.get_ema_model().encode_decode(
                            target_img_518[0:1], target_img_metas)
            end = time.time()
            print(i, "-th time: ", end - start)
        input("here")
        """

        # estimate per image latency (SAM)
        """
        target_img_518 = resize(input=target_img, size=[518,518], mode='bilinear', align_corners=False)
        for i in range(100):
            start = time.time()
            with torch.no_grad():
                anns0 = {'annotations': self.seg_anything.generate(target_img_518[0])}
            end = time.time()
            print("1 end - start: ", end - start)

            start = time.time()
            with torch.no_grad():
                anns1 = {'annotations': self.seg_anything.generate(target_img_518[1])}
            end = time.time()
            print("2 end - start: ", end - start)
        """
        
        """
        with torch.no_grad():
            anns0 = {'annotations': self.seg_anything.generate(img[0])}
            anns1 = {'annotations': self.seg_anything.generate(img[1])}

        f = plt.figure(figsize=(10,10))
        plt.imshow(torch.clamp(denorm(img, means, stds), 0, 1)[0].squeeze().cpu().detach().permute(1,2,0))
        plt.savefig("img_0.png")
        plt.close()            

        anns0['annotations'] = sorted(anns0['annotations'], key=lambda x: x['area'], reverse=True)
        ann_idx = 0
        for ann in anns0['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            f = plt.figure(figsize=(10,10))
            plt.imshow(valid_mask.cpu().detach())
            plt.savefig("ann0_" + str(ann_idx) + "png")
            plt.close()
            ann_idx += 1

        f = plt.figure(figsize=(10,10))
        plt.imshow(torch.clamp(denorm(img, means, stds), 0, 1)[1].squeeze().cpu().detach().permute(1,2,0))
        plt.savefig("img_1.png")
        plt.close()            

        anns1['annotations'] = sorted(anns1['annotations'], key=lambda x: x['area'], reverse=True)
        ann_idx = 0
        for ann in anns1['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            f = plt.figure(figsize=(10,10))
            plt.imshow(valid_mask.cpu().detach())
            plt.savefig("ann1_" + str(ann_idx) + "png")
            plt.close()            
            ann_idx += 1

        class_ids = gt_semantic_seg_1024[0].squeeze()
        semantc_mask = class_ids.clone()

        f = plt.figure(figsize=(10,10))
        plt.imshow(semantc_mask.cpu().detach())
        plt.savefig("semantc_mask_before.png")
        plt.close()            

        for ann in anns0['annotations']:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            # get the class ids of the valid pixels
            propose_classes_ids = class_ids[valid_mask]
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantc_mask[valid_mask] = propose_classes_ids[0]
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
            semantc_mask[valid_mask] = top_1_propose_class_ids

        f = plt.figure(figsize=(10,10))
        plt.imshow(semantc_mask.cpu().detach())
        plt.savefig("semantc_mask_after.png")
        plt.close()        
        input("here")
        """

        # (I) training on source domain
        #  single-task learning - semantic segmentation or depth estimation
        if len(self.task) == 1 and self.task[0] == 'seg':
            gt_depth_est, mask_depth_est = None, None
            gt_norm_est, mask_norm_est = None, None
        elif len(self.task) == 1 and self.task[0] == 'depth':
            gt_semantic_seg = None
            gt_norm_est, mask_norm_est = None, None
        elif len(self.task) == 2 and "norm" not in self.task:
            gt_norm_est, mask_norm_est = None, None

        # (I.1) supervision for semantic segmentation + depth estimation
        if 'seg' in self.task:
            clean_losses, src_logits, src_logits_resize = self.get_model().forward_train(
                        img, img_metas, gt_semantic_seg=gt_semantic_seg, gt_depth_est=gt_depth_est, gt_norm_est=gt_norm_est,
                        seg_weight=None, mask_depth_est=mask_depth_est, mask_norm_est=mask_norm_est, return_feat=True)
        else:
            clean_losses, src_logits, src_logits_resize = self.get_model().forward_train(
                        img, img_metas, gt_semantic_seg=None, gt_depth_est=gt_depth_est, gt_norm_est=gt_norm_est,
                        seg_weight=None, mask_depth_est=mask_depth_est, mask_norm_est=mask_norm_est, return_feat=True)


        # NOTE: reduce the importance on depth estimation
        # Also, depth loss is numerically larger than seg loss
        depth_weight = 0.1
        if 'depth' in self.task:
            #clean_losses['decode.loss_depth'] *= 0 # make depth loss from source zero.
            clean_losses['decode.loss_depth'] *= depth_weight
        norm_weight = 0.1
        if 'norm' in self.task:
            clean_losses['decode.loss_norm'] *= norm_weight

        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)

        # NOTE: original loss backword (Daformer)
        #clean_loss.backward(retain_graph=self.enable_fdist)

        # NOTE: our loss backword
        if "seg" in self.task:
            # source domain segmentation loss
            clean_losses['decode.loss_seg'].backward(retain_graph=self.enable_fdist)
        if self.depth_src_sup and "depth" in self.task:
            # source domain depth loss
            clean_losses['decode.loss_depth'].backward(retain_graph=self.enable_fdist)
        if "norm" in self.task:
            clean_losses['decode.loss_norm'].backward(retain_graph=self.enable_fdist)


        # (I.2) predict target depth using DAM
        if 'depth' in self.task and self.DAM == True:
            target_img_518 = resize(input=target_img, size=[518,518], mode='bilinear', align_corners=False)

            with torch.no_grad():
                tgt_depth_anything = self.depth_anything(target_img_518) # (518, 518)
                #tgt_depth_anything = self.depth_anything_v2(target_img_518) # (518, 518)
                tgt_depth_anything_resize = resize(input=tgt_depth_anything.unsqueeze(1), size=[512,512], mode='bilinear', align_corners=False) # (512, 512)
                tgt_depth_anything_resize += 1e-3


        # (I.3) ImageNet feature distance for semantic segmentation
        if 'seg' in self.task: # and len(self.task) == 1:
            if self.enable_fdist:
                feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                        src_feat)
                feat_loss.backward()
                log_vars.update(add_prefix(feat_log, 'src'))
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [
                        p.grad.detach() for p in params if p.grad is not None
                    ]
                    fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')


        # (II) training on target domain
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # (II.1) Generate pseudo-label for semantic segmentation
        if 'seg' in self.task:
            # NOTE: ema_logits and ema_raw_logits are list of logits [logits for seg, depth, and norm]
            ema_logits, ema_raw_logits = self.get_ema_model().encode_decode(
                target_img, target_img_metas)
            ema_softmax_seg = torch.softmax(ema_logits[0].detach(), dim=1)

            pseudo_prob, pseudo_label_seg = torch.max(ema_softmax_seg, dim=1)

            # generate SAM mask on target images - V2 (pre-generated)
            start_mask_time = time.time()
            if self.SAM == True:
                import pickle
                for batch_idx in range(target_img.shape[0]):
                    file_path = target_img_metas[batch_idx]['filename'].replace('/leftImg8bit/', '/sam_mask_small/').replace('.png', '.pkl')
                    # Open the pickle file in binary read mode
                    
                    mask_loading_time = time.time()
                    # with open(file_path, 'rb') as file:
                    #     mask_list = pickle.load(file)
                    mask_list = joblib.load(file_path)
                    mask_loaded_time = time.time()

                    class_ids = pseudo_label_seg[batch_idx].squeeze()
                    semantic_mask = class_ids.clone()

                    for mask_idx in range(len(mask_list)):
                        mask = mask_list[mask_idx]
                        t = time.time()
                        #mask = cv2.resize(mask.astype(np.float32), (1024, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
                        time_resize = time.time() - t

                        crop_bbox = target_img_metas[batch_idx]['crop_bbox']
                        mask = mask[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]]
                        if target_img_metas[batch_idx]['flip'] == True:
                            mask = np.fliplr(mask).copy() # flip left to right
                        time_mask_pre = time.time() - t

                        t = time.time()
                        if mask.sum() > 0: # since we crop the mask, there exist empty mask
                            propose_classes_ids = class_ids[mask]
                            num_class_proposals = len(torch.unique(propose_classes_ids))
                            if num_class_proposals == 1:
                                semantic_mask[mask] = propose_classes_ids[0]
                            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
                            semantic_mask[mask] = top_1_propose_class_ids
                        time_mask_apply = time.time() - t

                    pseudo_label_seg[batch_idx] = semantic_mask

            end_mask_time = time.time()

            # (II.2) Generate pseudo-label for depth estimation
            if 'depth' in self.task:
                if self.DAM == True: # pseudo depth from both DAM and teacher
                    pseudo_label_depth = tgt_depth_anything_resize
                    pseudo_label_depth_teacher = F.sigmoid(ema_logits[1].detach()) * 655.35 + 1

                    # median scaling - matching median of DAM's prediction and teacher's prediction
                    for i in range(2):
                        if torch.median(pseudo_label_depth[i]) != 0:
                            pseudo_label_depth_teacher[i] *= float(torch.median(pseudo_label_depth[i])) / float(torch.median(pseudo_label_depth_teacher[i]))
                else: # pseudo depth only from teacher
                    pseudo_label_depth = F.sigmoid(ema_logits[1].detach()) * 655.35 + 1

            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label_seg.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=dev)

            if self.psweight_ignore_top > 0:
                # Don't trust pseudo-labels in regions with potential
                # rectification artifacts. This can lead to a pseudo-label
                # drift from sky towards building or traffic light.
                pseudo_weight[:, :self.psweight_ignore_top, :] = 0
            if self.psweight_ignore_bottom > 0:
                pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img_seg, mixed_lbl_seg = [None] * batch_size, [None] * batch_size
            mixed_img_depth, mixed_lbl_depth = None, None
            if 'depth' in self.task:
                mixed_img_depth, mixed_lbl_depth = [None] * batch_size, [None] * batch_size
                mixed_lbl_depth_teacher = [None] * batch_size

            mix_masks = get_class_masks(gt_semantic_seg)
            #mix_masks = torch.zeros_like(img) # target only: zero, source only: one

            # (II.3) Generate mixed image for semantic segmentation
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img_seg[i], mixed_lbl_seg[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label_seg[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

                # (II.4) Generate mixed image for depth estimation
                if 'depth' in self.task:
                    # NOTE: mix_masks above differs with mix_masks_depth
                    # mix_masks basically determines which objects to mix between source and target images.
                    # as we cannot simply apply object-wise mixing for depth estimation, due to scale mismatches,
                    # we set mix_masks_depth to have binary elements, which forces - 
                    # e.g., first channel to only have objects from source and second channel for target.

                    mix_masks_depth = torch.zeros_like(img) # target only: zero, source only: one
                    """
                    if self.local_iter > -1:
                        # augmenting source and target
                        mix_masks_depth[0] += 1 # first channel: source, second channel: target
                    else:
                        # augmenting source only
                        mix_masks_depth += 1
                    """
                    strong_parameters['mix'] = mix_masks_depth[i]

                    # we will use two pseudo labels for depth estimation
                    # one from Depth Anything, another one from Teacher network
                    mixed_img_depth[i], mixed_lbl_depth[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((img[i], target_img[i])),
                        target=torch.stack((gt_depth_est[i], pseudo_label_depth[i])))

                    if self.DAM == True:
                        _, mixed_lbl_depth_teacher[i] = strong_transform(
                            strong_parameters,
                            data=torch.stack((img[i], target_img[i])),
                            target=torch.stack((gt_depth_est[i], pseudo_label_depth_teacher[i])))


            mixed_img_seg = torch.cat(mixed_img_seg)
            mixed_lbl_seg = torch.cat(mixed_lbl_seg)
            if 'depth' in self.task:
                mixed_img_depth = torch.cat(mixed_img_depth)
                mixed_lbl_depth = torch.cat(mixed_lbl_depth)
                if self.DAM == True:
                    mixed_lbl_depth_teacher = torch.cat(mixed_lbl_depth_teacher)

            if 'depth' in self.task:
                # NOTE: evaluate RMSE on target and mixed images
                gt_t = self.disp2meters(target_gt_depth_est)
                mask_t = target_mask_depth_est
                tgt_rmse, mix_rmse = {}, {}

                # evaluate teachers on target
                with torch.no_grad():
                    tgt_ema_sigmoid = F.sigmoid(ema_logits[1].detach()) * 655.35 + 1 # from teacher network
                    tgt_ema_sigmoid_m = self.disp2meters(tgt_ema_sigmoid)
                    tgt_ema_rmse = depth_loss.rmse(tgt_ema_sigmoid_m, gt_t, mask_t, median_scaling=True)
                    tgt_rmse['target.teacher.RMSE'] = float(tgt_ema_rmse)

                    tgt_DAM_sigmoid = pseudo_label_depth # from DAM
                    tgt_DAM_sigmoid_m = self.disp2meters(tgt_DAM_sigmoid)
                    tgt_DAM_rmse = depth_loss.rmse(tgt_DAM_sigmoid_m, gt_t, mask_t, median_scaling=True)
                    tgt_rmse['target.DAM.RMSE'] = float(tgt_DAM_rmse)

                # evaluate student on target
                with torch.no_grad():
                    tgt_logits, tgt_raw_logits = self.get_model().encode_decode(
                        target_img, target_img_metas)
                    tgt_stu_sigmoid = F.sigmoid(tgt_logits[1]) * 655.35 + 1
                    tgt_stu_sigmoid_m = self.disp2meters(tgt_stu_sigmoid)
                    tgt_stu_rmse = depth_loss.rmse(tgt_stu_sigmoid_m, gt_t, mask_t, median_scaling=True)
                    tgt_rmse['target.student.RMSE'] = float(tgt_stu_rmse)

                log_vars.update(tgt_rmse)
        
            if 'norm' in self.task:
                # NOTE: evaluate RMSE on target and mixed images
                gt_t = target_gt_norm_est
                if gt_t.dim() == 5:
                    # [B, 1, H, W, 3] -> [B, H, W, 3]
                    gt_t = gt_t.squeeze(1)          # [B, H, W, 3]
                    # [B, H, W, 3] -> [B, 3, H, W]
                    gt_t = gt_t.permute(0, 3, 1, 2) # [B, 3, H, W]
                mask_t = target_mask_norm_est

                tgt_merr = {}

                with torch.no_grad():                
                    tgt_logits, tgt_raw_logits = self.get_ema_model().encode_decode(
                        target_img, target_img_metas)
                    tgt_tea_norm = tgt_logits[2]
    
                    tgt_tea_mErr = 0
                    for batch_idx in range(tgt_logits[2].shape[0]):
                        mErr = torch.rad2deg(2 * torch.atan2(torch.norm(tgt_tea_norm[batch_idx] - gt_t[batch_idx], dim=0), torch.norm(tgt_tea_norm[batch_idx] + gt_t[batch_idx], dim=0)))
                        mErr = mErr * mask_t[batch_idx]
                        mErr = mErr.mean()
                        tgt_tea_mErr += mErr
                    tgt_tea_mErr = tgt_tea_mErr / tgt_logits[2].shape[0]
                    tgt_merr['target.teacher.mErr'] = float(tgt_tea_mErr)


                with torch.no_grad():                
                    tgt_logits, tgt_raw_logits = self.get_model().encode_decode(
                        target_img, target_img_metas)
                    tgt_stu_norm = tgt_logits[2]
    
                    tgt_stu_mErr = 0
                    for batch_idx in range(tgt_logits[2].shape[0]):
                        mErr = torch.rad2deg(2 * torch.atan2(torch.norm(tgt_stu_norm[batch_idx] - gt_t[batch_idx], dim=0), torch.norm(tgt_stu_norm[batch_idx] + gt_t[batch_idx], dim=0)))
                        mErr = mErr * mask_t[batch_idx]
                        mErr = mErr.mean()
                        tgt_stu_mErr += mErr
                    tgt_stu_mErr = tgt_stu_mErr / tgt_logits[2].shape[0]
                    tgt_merr['target.student.mErr'] = float(tgt_stu_mErr)

                log_vars.update(tgt_merr)


        # training only for depth estimation
        elif len(self.task) == 1 and 'depth' in self.task:
            # we cannot estimate pseudo_weight for depth estimation
            # since it is not based on probabilistic prediction

            # NOTE: ema_logits and ema_raw_logits are list of logits [logits for seg, depth, and norm]
            ema_logits, ema_raw_logits = self.get_ema_model().encode_decode(
                target_img, target_img_metas)

            if self.local_iter > -1:
                ema_sigmoid = tgt_depth_anything_resize # from depth anything
            else:
                ema_sigmoid = F.sigmoid(ema_logits[1].detach()) * 655.35 + 1 # from teacher network
            pseudo_label = ema_sigmoid

            pseudo_weight = torch.ones(pseudo_label.shape, device=dev)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = torch.zeros_like(img) # target only: zero, source only: one
            """
            if self.local_iter > -1:
                # augmenting source only
                mix_masks[0] += 1
            """

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_depth_est[i], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            #NOTE: evaluate RMSE on target and mixed images
            gt_t = self.disp2meters(target_gt_depth_est)
            mask_t = target_mask_depth_est
            tgt_rmse, mix_rmse = {}, {}

            # evaluate Teacher on target
            with torch.no_grad():
                #tgt_ema_sigmoid = ema_sigmoid
                tgt_ema_sigmoid = F.sigmoid(ema_logits[1].detach()) * 655.35 + 1
                tgt_ema_sigmoid_m = self.disp2meters(tgt_ema_sigmoid)
                tgt_ema_rmse = depth_loss.rmse(tgt_ema_sigmoid_m, gt_t, mask_t, median_scaling=True)
                tgt_rmse['target.teacher.RMSE'] = float(tgt_ema_rmse)

            # evaluate Student on target
            with torch.no_grad():
                tgt_logits, tgt_raw_logits = self.get_model().encode_decode(
                    target_img, target_img_metas)
                tgt_stu_sigmoid = F.sigmoid(tgt_logits[1]) * 655.35 + 1
                tgt_stu_sigmoid_m = self.disp2meters(tgt_stu_sigmoid)
                tgt_stu_rmse = depth_loss.rmse(tgt_stu_sigmoid_m, gt_t, mask_t, median_scaling=True)
                tgt_rmse['target.student.RMSE'] = float(tgt_stu_rmse)

            log_vars.update(tgt_rmse)


        # (II.2) Train on mixed images for semantic segmentation
        if 'seg' in self.task:
            mix_seg_losses, mix_seg_logits, mix_seg_logits_resize = self.get_model().forward_train(
                mixed_img_seg, img_metas, gt_semantic_seg=mixed_lbl_seg, gt_depth_est=None,
                seg_weight=pseudo_weight, return_feat=True, mask_depth_est=None)
            mix_seg_losses.pop('features')
            mix_seg_losses = add_prefix(mix_seg_losses, 'mix')
            mix_seg_loss, mix_seg_log_vars = self._parse_losses(mix_seg_losses)
            log_vars.update(mix_seg_log_vars)

            # mix_seg_loss.backward() # original loss backward (Daformer)
            mix_seg_losses['mix.decode.loss_seg'].backward()

            if self.depth_tgt_st and 'depth' in self.task:
                if self.DAM == True:
                    alpha_depth_anything = 1.0
                    alpha_teacher = 0.0
                else:
                    alpha_depth_anything = 1.0
                    alpha_teacher = 0.0

                # learning from Depth Anything
                mix_depth_losses, mix_depth_logits, mix_depth_logits_resize = self.get_model().forward_train(
                    mixed_img_depth, img_metas, gt_semantic_seg=None, gt_depth_est=mixed_lbl_depth,
                    seg_weight=None, return_feat=True, mask_depth_est=None)
                mix_depth_losses.pop('features')
                mix_depth_losses = add_prefix(mix_depth_losses, 'mix')

                # NOTE: reduce the importance on depth estimation
                mix_depth_losses['mix.decode.loss_depth'] *= (depth_weight * alpha_depth_anything)

                if self.DAM == True and alpha_teacher != 0:
                    # learning from Teacher network
                    mix_depth_losses_teacher, mix_depth_logits_teacher, mix_depth_logits_resize_teacher = self.get_model().forward_train(
                        mixed_img_depth, img_metas, gt_semantic_seg=None, gt_depth_est=mixed_lbl_depth_teacher,
                        seg_weight=None, return_feat=True, mask_depth_est=None)
                    mix_depth_losses_teacher.pop('features')
                    mix_depth_losses_teacher = add_prefix(mix_depth_losses_teacher, 'mix_teacher')

                    # NOTE: reduce the importance on depth estimation
                    mix_depth_losses_teacher['mix_teacher.decode.loss_depth'] *= (depth_weight * alpha_teacher)

                mix_depth_loss, mix_depth_log_vars = self._parse_losses(mix_depth_losses)
                log_vars.update(mix_depth_log_vars)

                if self.DAM == True and alpha_teacher != 0:
                    mix_depth_loss_teacher, mix_depth_log_vars_teacher = self._parse_losses(mix_depth_losses_teacher)
                    log_vars.update(mix_depth_log_vars_teacher)

                # (II.3) Train on mixed images for depth estimation
                mix_depth_losses['mix.decode.loss_depth'].backward()
                if self.DAM == True and alpha_teacher != 0:
                    mix_depth_losses_teacher['mix_teacher.decode.loss_depth'].backward()


        elif len(self.task) == 1 and self.task[0] == 'depth':
            mix_losses, mix_depth_logits, mix_depth_logits_resize = self.get_model().forward_train(
                mixed_img, img_metas, gt_semantic_seg=None, gt_depth_est=mixed_lbl,
                seg_weight=None, return_feat=True, mask_depth_est=None)
            mix_losses.pop('features')
            mix_losses = add_prefix(mix_losses, 'mix')

            # NOTE: reduce the importance on depth estimation
            mix_losses['mix.decode.loss_depth'] *= depth_weight

            mix_loss, mix_log_vars = self._parse_losses(mix_losses)

            log_vars.update(mix_log_vars)
            mix_losses['mix.decode.loss_depth'].backward()

        self.local_iter += 1

        return log_vars
