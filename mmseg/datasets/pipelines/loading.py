# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import torch
import torch.nn.functional as F
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



from PIL import Image
import cv2
import json
@PIPELINES.register_module()
class LoadDepthEstimations(object):
    """Load a depth estimation from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __get_cam_params(self, d_path):
        c_path = d_path.replace('disparity', 'camera').replace('png', 'json')
        with open(c_path) as file:
            calib = json.load(file)
        baseline = calib['extrinsic']['baseline']
        fx = calib['intrinsic']['fx']
        return baseline, fx

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['depth_info']['depth_map'])
        else:
            filename = results['depth_info']['depth_map']


        """
        code below is copied from DenseMTL:
            https://github.com/astra-vision/DenseMTL/blob/main/dataset/cityscapes.py
        """
        if "cityscape" in filename:
            #filename = filename.replace("leftImg8bit_small", "disparity")
            filename = filename.replace("leftImg8bit", "disparity")
            
            gt_depth_est = Image.open(filename)
            #gt_depth_est = gt_depth_est.resize((1024, 512), Image.NEAREST)
            gt_depth_est = np.asarray(gt_depth_est, np.float32)[None].squeeze()

            ## Preprocessing
            # CTRL-UDA https://github.com/susaha/ctrl-uda/blob/234c0ed807252fcf771da58ba3dc5b18396ae8a4/ctrl/dataset/base_dataset.py#L129
            baseline, fx = self.__get_cam_params(filename)

            gt_depth_est = gt_depth_est - 1
            with np.errstate(invalid='ignore'):
                mask = gt_depth_est > 0.0
            gt_depth_est = gt_depth_est / 256

            # find maximum depth value in map
            disparity_min = np.amin(gt_depth_est[mask])
            depth_max = (baseline * fx) / disparity_min # convert from disparity to depth values

            # can avoid the warning w/ a mask assignment
            with np.errstate(invalid='ignore'):
                gt_depth_est = (baseline * fx) / gt_depth_est  # computing the depth
            gt_depth_est[~mask] = depth_max

            if True: # inverse_depth
                gt_depth_est *= 100 # convert to cm depth values
                gt_depth_est = 65536.0 / (gt_depth_est + 1) # convert to normalized inverse depth

            # print("(cityscapes) gt_depth_est in loading.py: ", gt_depth_est, gt_depth_est.max(), gt_depth_est.min())
            # input("here")

            # img = np.where(mask, (baseline * fx) / img, depth_max)
            results['gt_depth_est'] = gt_depth_est
            results['mask_depth_est'] = mask.astype(int)
        else:
            gt_depth_est = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            gt_depth_est = (gt_depth_est[None] *.01).squeeze() # to meters

            if True: # inverse_depth
                gt_depth_est *= 100 # convert to cm depth values
                gt_depth_est = 65536.0 / (gt_depth_est + 1) # convert to normalized inverse depth


            results['gt_depth_est'] = gt_depth_est
            results['mask_depth_est'] = np.ones_like(gt_depth_est)

        results['depth_fields'].append('gt_depth_est')
        results['depth_fields'].append('mask_depth_est')
        return results



MIN_DEPTH, MAX_DEPTH = 1e-3, 80

def depth2disp(d):
    return 65536 / (d+1)

def disp2meters(d):
    return 655.36 / d - 1


from PIL import Image
import cv2
import json
@PIPELINES.register_module()
class LoadNormEstimations(object):
    """Load a depth estimation from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __get_cam_params(self, d_path):
        c_path = d_path.replace('disparity', 'camera').replace('png', 'json')
        with open(c_path) as file:
            calib = json.load(file)
        baseline = calib['extrinsic']['baseline']
        fx = calib['intrinsic']['fx']
        return baseline, fx

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['depth_info']['depth_map'])
        else:
            filename = results['depth_info']['depth_map']


        """
        code below is copied from DenseMTL:
            https://github.com/astra-vision/DenseMTL/blob/main/dataset/cityscapes.py
        """

        if "cityscape" in filename:
            # (i) Load depth map
            filename = filename.replace("leftImg8bit", "disparity")
            
            gt_depth_est = Image.open(filename)
            #gt_depth_est = gt_depth_est.resize((1024, 512), Image.NEAREST)
            gt_depth_est = np.asarray(gt_depth_est, np.float32)[None].squeeze()

            ## Preprocessing
            # CTRL-UDA https://github.com/susaha/ctrl-uda/blob/234c0ed807252fcf771da58ba3dc5b18396ae8a4/ctrl/dataset/base_dataset.py#L129
            baseline, fx = self.__get_cam_params(filename)

            gt_depth_est = gt_depth_est - 1
            with np.errstate(invalid='ignore'):
                mask = gt_depth_est > 0.0
            gt_depth_est = gt_depth_est / 256

            # find maximum depth value in map
            disparity_min = np.amin(gt_depth_est[mask])
            depth_max = (baseline * fx) / disparity_min # convert from disparity to depth values

            # can avoid the warning w/ a mask assignment
            with np.errstate(invalid='ignore'):
                gt_depth_est = (baseline * fx) / gt_depth_est  # computing the depth
            gt_depth_est[~mask] = depth_max

            gt_m = gt_depth_est.copy()
            if True: # inverse_depth
                # depth -> disparity
                gt_depth_est *= 100 # convert to cm depth values
                gt_depth_est = 65536.0 / (gt_depth_est + 1) # convert to normalized inverse depth

            # (ii) Estimate surface normal from depth map
            # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
            camera_fx = 2262.52
            camera_fy = 2265.3017905988554
            u0 = 1096.98
            v0 = 513.137

            K = self.get_K(camera_fx, camera_fy, u0, v0, u_offset=0, v_offset=0)
            inv_K = np.linalg.pinv(K)

            pointcloud = self.unproject(gt_depth_est, inv_K) # depth_gt is inverse depth
            gt_norm_est = self.d2normals(pointcloud, n_bases=4)

            mask = (MIN_DEPTH < gt_m)*(gt_m < MAX_DEPTH)
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.float()
            dmask = self.dilate_mask(self.remove_borders(mask), ksize=3)
            nmask = torch.isnan(gt_norm_est).logical_not().all(1, True)
            normals_mask = nmask*dmask

            if isinstance(gt_norm_est, torch.Tensor):
                gt_norm_np = gt_norm_est[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                gt_norm_np = gt_norm_est

            if isinstance(normals_mask, torch.Tensor):
                normals_mask_np = normals_mask[0, 0].detach().cpu().numpy().astype(bool)
            else:
                normals_mask_np = normals_mask

            # ★ 여기서 bool → uint8(0/1)로 변환
            if normals_mask_np.dtype == bool:
                normals_mask_np = normals_mask_np.astype(np.uint8)

            results['gt_norm_est'] = gt_norm_np
            results['mask_norm_est'] = normals_mask_np

        elif "synthia" in filename:
            gt_depth_est = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            gt_depth_est = (gt_depth_est[None] *.01).squeeze() # to meters
            gt_m = gt_depth_est.copy()


            if True: # inverse_depth
                # depth -> disparity
                gt_depth_est *= 100 # convert to cm depth values
                gt_depth_est = 65536.0 / (gt_depth_est + 1) # convert to normalized inverse depth

            # (ii) Estimate surface normal from depth map
            # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
            camera_fx = 532.740352
            camera_fy = 532.740352
            u0 = 640
            v0 = 380

            K = self.get_K(camera_fx, camera_fy, u0, v0, u_offset=0, v_offset=0)
            inv_K = np.linalg.pinv(K)

            # disparity -> pointcloud
            pointcloud = self.unproject(gt_depth_est, inv_K) # depth_gt is inverse depth
            gt_norm_est = self.d2normals(pointcloud, n_bases=4)

            mask = (MIN_DEPTH < gt_m)*(gt_m < MAX_DEPTH)
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.float()
            dmask = self.dilate_mask(self.remove_borders(mask), ksize=3)
            nmask = torch.isnan(gt_norm_est).logical_not().all(1, True)
            normals_mask = nmask*dmask

            if isinstance(gt_norm_est, torch.Tensor):
                gt_norm_np = gt_norm_est[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                gt_norm_np = gt_norm_est

            if isinstance(normals_mask, torch.Tensor):
                normals_mask_np = normals_mask[0, 0].detach().cpu().numpy().astype(bool)
            else:
                normals_mask_np = normals_mask

            # ★ 여기서 bool → uint8(0/1)로 변환
            if normals_mask_np.dtype == bool:
                normals_mask_np = normals_mask_np.astype(np.uint8)

            results['gt_norm_est'] = gt_norm_np
            results['mask_norm_est'] = normals_mask_np # np.ones_like(gt_norm_est)

        elif "vkitti" in filename:
            gt_depth_est = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            gt_depth_est = (gt_depth_est[None] *.01).squeeze() # to meters
            gt_m = gt_depth_est.copy()

            if True: # inverse_depth
                gt_depth_est *= 100 # convert to cm depth values
                gt_depth_est = 65536.0 / (gt_depth_est + 1) # convert to normalized inverse depth

            # (ii) Estimate surface normal from depth map
            # reference: https://github.com/astra-vision/DenseMTL/blob/main/dataset/vkitti2.py
            camera_fx = 725.0087
            camera_fy = 725.0087
            u0 = 620.5
            v0 = 187

            K = self.get_K(camera_fx, camera_fy, u0, v0, u_offset=0, v_offset=0)
            inv_K = np.linalg.pinv(K)

            pointcloud = self.unproject(gt_depth_est, inv_K) # depth_gt is inverse depth
            gt_norm_est = self.d2normals(pointcloud, n_bases=4)

            mask = (MIN_DEPTH < gt_m)*(gt_m < MAX_DEPTH)

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.float()
            dmask = self.dilate_mask(self.remove_borders(mask), ksize=3)
            nmask = torch.isnan(gt_norm_est).logical_not().all(1, True)
            normals_mask = nmask*dmask

            if isinstance(gt_norm_est, torch.Tensor):
                gt_norm_np = gt_norm_est[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                gt_norm_np = gt_norm_est

            if isinstance(normals_mask, torch.Tensor):
                normals_mask_np = normals_mask[0, 0].detach().cpu().numpy().astype(bool)
            else:
                normals_mask_np = normals_mask

            # ★ 여기서 bool → uint8(0/1)로 변환
            if normals_mask_np.dtype == bool:
                normals_mask_np = normals_mask_np.astype(np.uint8)

            results['gt_norm_est'] = gt_norm_np
            results['mask_norm_est'] = normals_mask_np

        results['norm_fields'].append('gt_norm_est')
        results['norm_fields'].append('mask_norm_est')
        return results


    # borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/dataset/base.py#L361
    def get_K(self, fx, fy, u0, v0, u_offset, v_offset):
        return np.array([[     fx,        0,  u0 - u_offset,  0],
                         [      0,       fy,  v0 - v_offset,  0],
                         [      0,        0,              1,  0],
                         [      0,        0,              0,  1]], dtype=np.float32)

    # borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/models/normals.py
    """
    def unproject(self, depth, inv_K):
        print("depth shape: ", depth.shape)
        h, w = depth.shape[-2:]

        meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w))#, indexing='ij')
        pix_coords = torch.stack(meshgrid).flip(0).unsqueeze(0).flatten(2).float()#.to('cuda')
        ones = torch.ones(1, 1, h*w)#, device='cuda')
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

    def get_normal_mask(self, mask, depth, normals, train):
        if not train:
            mask *= (MIN_DEPTH < depth)*(depth < MAX_DEPTH)

        # get initial mask from invalid values in depth map
        dmask = dilate_mask(remove_borders(mask.float()), ksize=3)

        # get invalid values from normals computation
        nmask = torch.isnan(normals).logical_not().all(1, True)

        # combine both masks
        normals_mask = nmask*dmask

        return dmask, normals_mask
        
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
