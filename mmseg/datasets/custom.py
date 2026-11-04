# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose

from PIL import Image
import cv2
import json
import torch
import torch.nn.functional as F


MIN_DEPTH, MAX_DEPTH = 1e-3, 80

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 depth_dir=None,
                 norm_dir=None,
                 seg_map_suffix='.png',
                 depth_map_suffix='.png',
                 norm_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.depth_dir = depth_dir
        self.norm_dir = norm_dir
        self.seg_map_suffix = seg_map_suffix
        self.depth_map_suffix = depth_map_suffix
        self.norm_map_suffix = norm_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.depth_dir is None or osp.isabs(self.depth_dir)):
                self.depth_dir = osp.join(self.data_root, self.depth_dir)
            if not (self.norm_dir is None or osp.isabs(self.norm_dir)):
                self.norm_dir = osp.join(self.data_root, self.norm_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir, self.depth_dir, self.norm_dir,
                                               self.seg_map_suffix, self.depth_map_suffix, self.norm_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, depth_dir, norm_dir, seg_map_suffix, depth_map_suffix, norm_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    if depth_dir is not None:
                        depth_map = img_name + depth_map_suffix
                        img_info['depth'] = dict(depth_map=depth_map)
                    if norm_dir is not None:
                        norm_map = img_name + norm_map_suffix
                        img_info['norm'] = dict(norm_map=norm_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if depth_dir is not None:
                    depth_map = img.replace(img_suffix, depth_map_suffix)
                    img_info['depth'] = dict(depth_map=depth_map)
                if norm_dir is not None:
                    norm_map = img.replace(img_suffix, norm_map_suffix)
                    img_info['norm'] = dict(norm_map=norm_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_depth_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.img_infos[idx]['depth']

    def get_norm_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.img_infos[idx]['norm']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['depth_fields'] = []
        results['norm_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['depth_prefix'] = self.depth_dir
        results['norm_prefix'] = self.norm_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        if self.depth_dir is not None:
            depth_info = self.get_depth_info(idx)
        else:
            depth_info = None
        if self.norm_dir is not None:
            norm_info = self.get_norm_info(idx)
        else:
            norm_info = None

        results = dict(img_info=img_info, ann_info=ann_info, depth_info=depth_info, norm_info=norm_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_cam_params(self, d_path):
        c_path = d_path.replace('disparity', 'camera').replace('png', 'json')
        with open(c_path) as file:
            calib = json.load(file)
        baseline = calib['extrinsic']['baseline']
        fx = calib['intrinsic']['fx']
        return baseline, fx

    def get_gt_depth_maps(self, efficient_test=False):
        """Get ground truth depth estimation maps for evaluation."""
        efficient_test=False

        gt_depth_maps = []
        mask_depth_maps = []
        for img_info in self.img_infos:

            depth_map = osp.join(self.depth_dir, img_info['depth']['depth_map'])
            if efficient_test:
                gt_depth_map = depth_map
            else:
                if "cityscapes" in self.depth_dir:
                    # for cityscapes
                    gt_depth_map = mmcv.imread(
                        depth_map, flag='unchanged', backend='pillow')

                    gt_depth_map = Image.open(depth_map)
                    #gt_depth_map = gt_depth_map.resize((1024, 512), Image.NEAREST)
                    gt_depth_map = np.asarray(gt_depth_map, np.float32)[None].squeeze()

                    ## Preprocessing
                    # CTRL-UDA https://github.com/susaha/ctrl-uda/blob/234c0ed807252fcf771da58ba3dc5b18396ae8a4/ctrl/dataset/base_dataset.py#L129
                    baseline, fx = self.get_cam_params(depth_map)

                    gt_depth_map = gt_depth_map - 1
                    with np.errstate(invalid='ignore'):
                        mask = gt_depth_map > 0.0
                    gt_depth_map = gt_depth_map / 256

                    # find maximum depth value in map
                    disparity_min = np.amin(gt_depth_map[mask])
                    depth_max = (baseline * fx) / disparity_min # convert from disparity to depth values

                    # can avoid the warning w/ a mask assignment
                    with np.errstate(invalid='ignore'):
                        gt_depth_map = (baseline * fx) / gt_depth_map  # computing the depth
                    gt_depth_map[~mask] = depth_max

                    if True: # inverse_depth
                        gt_depth_map *= 100 # convert to cm depth values
                        gt_depth_map = 65536.0 / (gt_depth_map + 1) # convert to normalized inverse depth
                elif "synthia" in self.depth_dir:
                    gt_depth_map = cv2.imread(depth_map, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    gt_depth_map = (gt_depth_map[None] *.01).squeeze() # to meters

                    if True: # inverse_depth
                        gt_depth_map *= 100 # convert to cm depth values
                        gt_depth_map = 65536.0 / (gt_depth_map + 1) # convert to normalized inverse depth
                    mask = np.ones_like(gt_depth_map)

                elif "vkitti" in self.depth_dir:
                    gt_depth_map = cv2.imread(depth_map, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    gt_depth_map = (gt_depth_map[None] *.01).squeeze() # to meters

                    if True: # inverse_depth
                        gt_depth_map *= 100 # convert to cm depth values
                        gt_depth_map = 65536.0 / (gt_depth_map + 1) # convert to normalized inverse depth

                    # print("gt_depth_map in l379 in custom.py: ", gt_depth_map)
                    # input("here")
                    mask = np.ones_like(gt_depth_map)

            gt_depth_maps.append(gt_depth_map)
            mask_depth_maps.append(mask.astype(int))

        return gt_depth_maps, mask_depth_maps


    def get_gt_norm_maps(self, efficient_test=False):
        """Get ground truth normal estimation maps for evaluation."""
        efficient_test=False

        gt_norm_maps = []
        mask_norm_maps = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for img_info in self.img_infos:

            depth_map = osp.join(self.depth_dir, img_info['depth']['depth_map'])
            if efficient_test:
                gt_depth_map = depth_map
            else:
                if "cityscapes" in self.depth_dir:
                    # Load the depth map on CPU, then move the numeric path to GPU.
                    gt_depth_map = np.asarray(
                        Image.open(depth_map), np.float32)[None].squeeze()
                    baseline, fx = self.get_cam_params(depth_map)
                    gt_depth_map = torch.from_numpy(gt_depth_map).to(device=device)

                    gt_depth_map = gt_depth_map - 1
                    mask = gt_depth_map > 0.0
                    gt_depth_map = gt_depth_map / 256.0

                    disparity_min = torch.amin(gt_depth_map[mask])
                    depth_max = (baseline * fx) / disparity_min

                    safe_gt_depth_map = torch.where(mask, gt_depth_map, torch.ones_like(gt_depth_map))
                    gt_depth_map = (baseline * fx) / safe_gt_depth_map
                    gt_depth_map = torch.where(mask, gt_depth_map, depth_max)

                    gt_m = gt_depth_map.clone()
                    if True:  # inverse_depth
                        gt_depth_map = gt_depth_map * 100.0
                        gt_depth_map = 65536.0 / (gt_depth_map + 1.0)

                    # Estimate surface normal from depth map.
                    if not hasattr(self, '_cityscapes_inv_K_cache'):
                        camera_fx = 2262.52
                        camera_fy = 2265.3017905988554
                        u0 = 1096.98
                        v0 = 513.137
                        K = self.get_K(camera_fx, camera_fy, u0, v0, u_offset=0, v_offset=0)
                        self._cityscapes_inv_K_cache = torch.from_numpy(
                            np.linalg.pinv(K)).float()
                    inv_K = self._cityscapes_inv_K_cache.to(device=device)

                    pointcloud = self.unproject(gt_depth_map, inv_K)
                    gt_norm_map = self.d2normals(pointcloud, n_bases=4)

                    mask = ((MIN_DEPTH < gt_m) * (gt_m < MAX_DEPTH)).float()
                    dmask = self.dilate_mask(self.remove_borders(mask), ksize=3)
                    nmask = torch.isnan(gt_norm_map).logical_not().all(1, True)
                    normals_mask = nmask * dmask

                    gt_norm_np = gt_norm_map[0].permute(1, 2, 0).detach().cpu().numpy()
                    normals_mask_np = normals_mask[0, 0].detach().cpu().numpy().astype(bool)
                    if normals_mask_np.dtype == bool:
                        normals_mask_np = normals_mask_np.astype(np.uint8)
                    

                elif "synthia" in self.depth_dir:
                    gt_norm_map = None
                    print("here")

                elif "vkitti" in self.depth_dir:
                    gt_norm_map = None
                    print("here")

            gt_norm_maps.append(gt_norm_np)
            mask_norm_maps.append(normals_mask_np)
        return gt_norm_maps, mask_norm_maps

    # borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/dataset/base.py#L361
    def get_K(self, fx, fy, u0, v0, u_offset, v_offset):
        return np.array([[     fx,        0,  u0 - u_offset,  0],
                         [      0,       fy,  v0 - v_offset,  0],
                         [      0,        0,              1,  0],
                         [      0,        0,              0,  1]], dtype=np.float32)

    # borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/models/normals.py
    """
    def unproject(self, depth, inv_K):
        h, w = depth.shape[-2:]

        meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w))#, indexing='ij')
        pix_coords = torch.stack(meshgrid).flip(0).unsqueeze(0).flatten(2).float().to('cuda')
        ones = torch.ones(1, 1, h*w, device='cuda')
        pix_coords = torch.cat((pix_coords, ones), dim=1)

        # forward
        cam_points = depth.flatten(2) * (inv_K[:, :3, :3] @ pix_coords)
        return cam_points.reshape(-1, 3, h, w).contiguous()
    """

    def unproject(self, depth, inv_K):
        # 1) Convert depth to a torch tensor if it is a NumPy array
        #    (NumPy does not support flatten(start_dim=2), which caused the original error)
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)

        # 2) Ensure depth has shape [B, 1, H, W]
        #    Depth often arrives as (H, W); this expands batch and channel dimensions
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)   # (H, W) → (1, 1, H, W)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(1)                # (B, H, W) → (B, 1, H, W)

        B, _, h, w = depth.shape
        device = depth.device
        depth = depth.float()

        # 3) Convert inverse intrinsics to torch tensor if needed
        #    and ensure it has shape [B, 4, 4] or [1, 4, 4]
        if isinstance(inv_K, np.ndarray):
            inv_K = torch.from_numpy(inv_K)
        inv_K = inv_K.to(device).float()

        if inv_K.ndim == 2:
            inv_K = inv_K.unsqueeze(0)  # (4×4) → (1, 4, 4)

        # Extract the 3×3 upper-left portion
        inv_K3 = inv_K[:, :3, :3]

        # 4) Create pixel coordinate grid (x, y, 1)
        ys = torch.arange(h, device=device)
        xs = torch.arange(w, device=device)

        # ★ PyTorch 구버전에서는 indexing='ij' 옵션이 없으므로 인자 없이 호출해야 함
        yy, xx = torch.meshgrid(ys, xs)

        pix_coords = torch.stack((xx, yy), dim=0).view(1, 2, -1).float()

        # Append a "1" to each coordinate to form homogeneous coordinates
        ones = torch.ones(1, 1, h * w, device=device)
        pix_coords = torch.cat((pix_coords, ones), dim=1)  # Shape: [1, 3, H*W]

        # 5) Flatten depth to match the coordinates: [B, 1, H*W]
        #    This works now because depth is guaranteed to be a torch tensor
        depth_flat = depth.flatten(2)

        # 6) Unproject pixels: multiply depth with inv_K * pixel_coords
        cam_points = depth_flat * (inv_K3 @ pix_coords)  # [B, 3, H*W]

        # 7) Reshape back into [B, 3, H, W]
        return cam_points.view(B, 3, h, w).contiguous()


    # borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/models/normals.py
    def d2normals(self, points, guide=None, n_bases=1):
        """
            Bases are defined as:
            . → (w0, h0) - basis vector is defined with [(w1, h1), (w0, h0)]
            ↓
            (w1, h1)

            Up to four bases can be used, which sums up to a total of 8 neighbors:
            | (i)  ↑     | (ii)       | (iii)    ↗    | (iv) ↖
            |      . →   |       ← .  |         .     |        .
            |            |         ↓  |          ↘    |      ↙
        """
        # wrapper lambdas for pts and guide
        w = lambda w, h: guide.roll(-w, -1).roll(-h, -2) - guide
        v = lambda w, h: points.roll(-w, -1).roll(-h, -2) - points

        bases = [[( 1,  0), ( 0, -1)],
                [(-1,  0), ( 0,  1)],
                [( 1,  1), ( 1, -1)],
                [(-1, -1), (-1,  1)]]

        normals = []
        for (w1, h1), (w0, h0) in bases[:n_bases]:
            n = torch.cross(v(w1, h1), v(w0, h0))
            w = color_sim(w(w1, h1))*color_sim(w(w0, h0)) if guide else 1
            normals.append(w*n)

        # stack over estimations, normalize vectors, channel axis
        normals = torch.stack(normals).sum(0)
        normals = F.normalize(normals, dim=-3)

        # flip normals pointing outwards on z-axis.
        # tensor is shaped (b,c,h,w) and c is ordered as RGB: flip B
        normals[:, -1].abs_()

        return normals

        
    def remove_borders(self, x):
        """Remove 1 pixel along all 4 borders."""
        x[..., (0,-1), :] = False
        x[..., (0,-1)] = False
        return x

    """
    def dilate_mask(self, m, ksize):
        padding = {3: 1, 5: 2}[ksize]
        m = F.conv2d(1 - m, weight=torch.ones(1, 1, ksize, ksize, device=m.device), padding=padding)
        m[m > 0] = 1
        return (1 - m).bool()
    """
    
    def dilate_mask(self, m, ksize):
        # 1) If m is a NumPy array, convert it to a torch tensor
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m)

        # 2) Ensure m is a float tensor on some device (CPU or GPU)
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"Expected mask to be a torch.Tensor or np.ndarray, got {type(m)}")

        device = m.device
        m = m.float()

        # 3) Make sure m has shape [N, C, H, W] for conv2d
        #    Current shape is likely [H, W], so we add batch and channel dims.
        if m.dim() == 2:           # [H, W]
            m = m.unsqueeze(0).unsqueeze(0)   # → [1, 1, H, W]
        elif m.dim() == 3:         # [N, H, W]
            m = m.unsqueeze(1)               # → [N, 1, H, W]
        # If m is already 4D, we assume it's [N, C, H, W] and leave it as is.

        padding = {3: 1, 5: 2}[ksize]

        # 4) Apply dilation via conv2d
        kernel = torch.ones(1, 1, ksize, ksize, device=device)
        m = F.conv2d(1 - m, weight=kernel, padding=padding)
        m[m > 0] = 1

        # 5) Convert back to boolean mask and squeeze batch/channel dims to [H, W]
        m = (1 - m).bool()          # [N, 1, H, W] or [1, 1, H, W]
        m = m.squeeze(0).squeeze(0) # → [H, W]

        return m


    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate_miou(self,
                      results,
                      logger=None,
                      efficient_test=False):
        metric = 'mIoU'

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

    def evaluate_rmse(self,
                      results,
                      logger=None,
                      efficient_test=False):
        metric = 'RMSE'
        eval_results = {}
        gt_depth_maps, mask_depth_maps = self.get_gt_depth_maps(efficient_test)

        #if "cityscapes" in self.img_dir:
        if True:
            ret_metrics = eval_metrics(
                results,
                (gt_depth_maps, mask_depth_maps),
                None, # num_classes
                self.ignore_index,
                metric,
                )

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'RMSE':
                    eval_results[key] = value

            print_log('RMSE evaluation on target domain RMSE: ' + str(eval_results['RMSE']), logger)

            if mmcv.is_list_of(results, str):
                for file_name in results:
                    os.remove(file_name)

        return eval_results

    def evaluate_merr(self,
                      results,
                      logger=None,
                      efficient_test=False):
        metric = 'mErr'
        eval_results = {}
        gt_norm_maps, mask_norm_maps = self.get_gt_norm_maps(efficient_test)

        #if "cityscapes" in self.img_dir:
        if True:
            ret_metrics = eval_metrics(
                results,
                (gt_norm_maps, mask_norm_maps),
                None, # num_classes
                self.ignore_index,
                metric,
                )

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'mErr':
                    eval_results[key] = value

            print_log('mErr evaluation on target domain mErr: ' + str(eval_results['mErr']), logger)

            if mmcv.is_list_of(results, str):
                for file_name in results:
                    os.remove(file_name)

        return eval_results


    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        efficient_test = True

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'RMSE', 'mErr', 'mIoU+RMSE', 'mIoU+RMSE+mErr']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}

        if metric == ['mIoU+RMSE']:
            def merge(dict1, dict2):
                return(dict2.update(dict1))
            eval_results_depth = self.evaluate_rmse(results[1], logger, efficient_test)
            eval_results_seg = self.evaluate_miou(results[0], logger, efficient_test)
            merge(eval_results_seg, eval_results_depth) # merging dicts into eval_results_depth
            return eval_results_depth

        elif metric == ['mIoU+RMSE+mErr']:
            def merge(dict1, dict2):
                return(dict2.update(dict1))
            eval_results_norm = self.evaluate_merr(results[2], logger, efficient_test)
            eval_results_seg = self.evaluate_miou(results[0], logger, efficient_test)
            eval_results_depth = self.evaluate_rmse(results[1], logger, efficient_test)
            merge(eval_results_seg, eval_results_depth) # merging dicts into eval_results_depth
            merge(eval_results_depth, eval_results_norm)
            return eval_results_norm

        elif metric == ['mErr']:
            gt_depth_maps, mask_depth_maps = self.get_gt_depth_maps(efficient_test)
            if "cityscapes" in self.img_dir:
                ret_metrics = eval_metrics(
                    results[2],
                    (gt_depth_maps, mask_depth_maps), # gt_norm_maps will be estimated internally
                    None, # num_classes
                    self.ignore_index,
                    metric,
                    )

                # summary table
                ret_metrics_summary = OrderedDict({
                    ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })

                # each metric dict
                for key, value in ret_metrics_summary.items():
                    if key == 'mErr':
                        eval_results[key] = value

                print_log('mErr evaluation on target domain: ' + str(eval_results['mErr']), logger)

                if mmcv.is_list_of(results[1], str):
                    for file_name in results[1]:
                        os.remove(file_name)
                return eval_results

        elif metric == ['RMSE']:
            gt_depth_maps, mask_depth_maps = self.get_gt_depth_maps(efficient_test)
            if "cityscapes" in self.img_dir:
                ret_metrics = eval_metrics(
                    results[1],
                    (gt_depth_maps, mask_depth_maps),
                    None, # num_classes
                    self.ignore_index,
                    metric,
                    )

                # summary table
                ret_metrics_summary = OrderedDict({
                    ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })

                # each metric dict
                for key, value in ret_metrics_summary.items():
                    if key == 'RMSE':
                        eval_results[key] = value

                print_log('RMSE evaluation on target domain: ' + str(eval_results['RMSE']), logger)

                if mmcv.is_list_of(results[1], str):
                    for file_name in results[1]:
                        os.remove(file_name)
                return eval_results

        elif metric == ['mIoU']:
            gt_seg_maps = self.get_gt_seg_maps(efficient_test)
            if self.CLASSES is None:
                num_classes = len(
                    reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
            else:
                num_classes = len(self.CLASSES)
            
            ret_metrics = eval_metrics(
                results[0],
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)

            if self.CLASSES is None:
                class_names = tuple(range(num_classes))
            else:
                class_names = self.CLASSES

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            # each class table
            ret_metrics.pop('aAcc', None)
            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)

            # for logger
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                if key == 'aAcc':
                    summary_table_data.add_column(key, [val])
                else:
                    summary_table_data.add_column('m' + key, [val])

            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'aAcc':
                    eval_results[key] = value / 100.0
                else:
                    eval_results['m' + key] = value / 100.0

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                eval_results.update({
                    key + '.' + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                })

            if mmcv.is_list_of(results, str):
                for file_name in results:
                    os.remove(file_name)
            return eval_results
