# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor, build_multi_task
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

MIN_DEPTH, MAX_DEPTH = 1e-3, 80 # in meters
def depth2disp(d):
    return 65536 / (d+1)
def disp2meters(d):
    return 655.36 / d - 1


def disp2meters_torch(d):
    return 655.36 / d - 1

def get_cam_params(depth_path):
    cam_path = depth_path.replace('disparity', 'camera').replace('png', 'json')
    with open(cam_path) as file:
        calib = json.load(file)
    baseline = calib['extrinsic']['baseline']
    fx = calib['intrinsic']['fx']
    return baseline, fx

def get_gt_depth_maps(depth_path):
    """Get ground truth depth estimation maps for evaluation."""

    gt_depth_maps = []
    mask_depth_maps = []

    if "cityscapes" in depth_path:
        # for cityscapes
        gt_depth_map = Image.open(depth_path)
        gt_depth_map = np.asarray(gt_depth_map, np.float32)[None].squeeze()

        ## Preprocessing
        # CTRL-UDA https://github.com/susaha/ctrl-uda/blob/234c0ed807252fcf771da58ba3dc5b18396ae8a4/ctrl/dataset/base_dataset.py#L129
        baseline, fx = get_cam_params(depth_path)

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

    elif "synthia" in depth_path:
        gt_depth_map = cv2.imread(depth_map, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
        gt_depth_map = (gt_depth_map[None] *.01).squeeze() # to meters

        if True: # inverse_depth
            gt_depth_map *= 100 # convert to cm depth values
            gt_depth_map = 65536.0 / (gt_depth_map + 1) # convert to normalized inverse depth
        mask = np.ones_like(gt_depth_map)

    elif "vkitti" in depth_path:
        gt_depth_map = cv2.imread(depth_map, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
        gt_depth_map = (gt_depth_map[None] *.01).squeeze() # to meters

        if True: # inverse_depth
            gt_depth_map *= 100 # convert to cm depth values
            gt_depth_map = 65536.0 / (gt_depth_map + 1) # convert to normalized inverse depth

        mask = np.ones_like(gt_depth_map)

    gt_depth_maps.append(gt_depth_map)
    mask_depth_maps.append(mask.astype(int))
    return gt_depth_maps, mask_depth_maps

def rmse(pred, target, mask=None, median_scaling=True):
    # this function is used only for validation (or test)

    if mask is not None:
        mask_target = mask.squeeze() * ((1e-3 < target.squeeze()) * (target.squeeze() < 80))
    elif mask is None:
        mask_target = np.ones_like(pred).squeeze() * ((1e-3 < target.squeeze()) * (target.squeeze() < 80))

    pred = pred.squeeze() * mask_target
    target = target.squeeze() * mask_target

    if median_scaling:
        if np.median(pred) != 0:
            pred *= np.median(target) / np.median(pred)
        pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)

    absdiff = np.abs(target.squeeze() - pred.squeeze())
    loss = np.sqrt((absdiff ** 2).mean())
    return loss


def rmse_torch(pred, target, mask=None, median_scaling=True):
    if mask is not None:
        mask_target = mask.squeeze() * ((1e-3 < target.squeeze()) * (target.squeeze() < 80)).float()
    else:
        mask_target = ((1e-3 < target.squeeze()) * (target.squeeze() < 80)).float()

    pred = pred.squeeze() * mask_target
    target = target.squeeze() * mask_target

    if median_scaling:
        pred_median = torch.median(pred)
        if pred_median > 0:
            pred = pred * (torch.median(target) / pred_median)
        pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)

    absdiff = (target.squeeze() - pred.squeeze()).abs()
    return torch.sqrt((absdiff ** 2).mean())

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--quantization', type=bool, default=False)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def merr(pred, target, mask):
    pred = pred.squeeze()
    target = target.squeeze()

    # NOTE: pred, gt, mask should be:  torch.Size([3, 512, 1024]) torch.Size([3, 512, 1024]) torch.Size([1, 512, 1024])
    error = torch.rad2deg(2 * torch.atan2(torch.norm(pred - target, dim=0), torch.norm(pred + target, dim=0)))
    error = error * mask
    mErr = error.mean()
    return mErr

import matplotlib.pyplot as plt
def save_normal_map(pred_norm_est, save_path="normal_map.png"):
    """
    pred_norm_est: torch tensor, shape [1, 3, H, W]
    save_path: output PNG file path
    """
    # 1) detach + cpu + numpy
    normal = pred_norm_est[0].detach().cpu().numpy()  # [3, H, W]

    # 2) [-1,1] → [0,1] normalize
    normal_vis = (normal + 1) / 2.0
    normal_vis = np.clip(normal_vis, 0, 1)  # numerical stability
    
    # 3) [3, H, W] → [H, W, 3]
    normal_vis = np.transpose(normal_vis, (1, 2, 0))

    # 4) matplotlib 저장
    plt.figure(figsize=(12, 6))
    plt.imshow(normal_vis)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    args = parse_args()
    args.eval_options['imgfile_prefix'] += '_dam'

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')


    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                if "trainvaltest" in cfg.data.test[k]:
                    # ACDC dataset
                    cfg.data.test[k] = cfg.data.test[k].replace('/val', '/test')
                else:
                    cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # for pre-trained models from github repo
    #cfg.data.test['data_root'] = "../" + cfg.data.test['data_root']

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    #model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')) # default for segmentation
    model = build_multi_task(cfg, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('FP model size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')

    print("cfg: ", cfg)

    """
    specify gpu_ids here
    """
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    #"""
    avg_rmse = 0
    avg_merr = 0    
    img_count = 0

    norm_path = args.checkpoint.replace("iter_40000.pth", "normal_estimation/")
    file_path = norm_path + "eval_score.txt"
    os.makedirs(norm_path, exist_ok=True)

    for i, data in enumerate(data_loader):
        depth_path = data['img_metas'][0].data[0][0]['filename'] # 'data/cityscapes/leftImg8bit/test/berlin/berlin_000176_000019_leftImg8bit.png'
        depth_path = depth_path.replace('leftImg8bit', 'disparity')
        gt_depth_maps, mask_depth_maps = get_gt_depth_maps(depth_path)

        with torch.no_grad():
            device = torch.device('cuda', torch.cuda.current_device())
            gt_depth_tensor = torch.as_tensor(gt_depth_maps[0], device=device, dtype=torch.float32)
            mask_t = torch.as_tensor(mask_depth_maps[0], device=device, dtype=torch.float32)
            pred_depth_logits = torch.as_tensor(outputs[1][i], device=device, dtype=torch.float32)

            gt_t = disp2meters_torch(gt_depth_tensor)
            pred_depth_maps = torch.sigmoid(pred_depth_logits) * 655.35 + 1
            result_t = disp2meters_torch(pred_depth_maps)

            # Legacy CPU/Numpy path kept for reference.
            #
            # gt_t = disp2meters(gt_depth_maps[0])
            # mask_t = mask_depth_maps[0]
            # pred_depth_maps = sigmoid(outputs[1][i]) * 655.35 + 1
            # result_t = disp2meters(pred_depth_maps)
            # camera_fx = 2262.52
            # camera_fy = 2265.3017905988554
            # u0 = 1096.98
            # v0 = 513.137
            # K = get_K(camera_fx, camera_fy, u0, v0, u_offset=0, v_offset=0)
            # inv_K = np.linalg.pinv(K)
            # gt_pointcloud = unproject(gt_depth_maps[0], inv_K)
            # gt_norm_est = d2normals(gt_pointcloud, n_bases=4)
            # pred_pointcloud = unproject(pred_depth_maps, inv_K)
            # pred_norm_est = d2normals(pred_pointcloud, n_bases=4)
            # mask = torch.from_numpy(mask_depth_maps[0]).float()
            # dmask = dilate_mask(remove_borders(mask), ksize=3)
            # nmask = torch.isnan(gt_norm_est).logical_not().all(1, True)
            # normals_mask = nmask * dmask
            # normals_mask_np = normals_mask[0, 0].detach().cpu().numpy().astype(np.uint8)
            # loss_norm = merr(pred_norm_est, gt_norm_est, normals_mask_np)
            # loss = rmse(result_t, gt_t, mask_t)

            # See https://www.cityscapes-dataset.com/file-handling/?packageID=8
            camera_fx = 2262.52
            camera_fy = 2265.3017905988554
            u0 = 1096.98
            v0 = 513.137

            K = torch.tensor(
                [[camera_fx, 0, u0, 0],
                 [0, camera_fy, v0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],
                device=device,
                dtype=torch.float32)
            inv_K = torch.linalg.pinv(K)

            gt_pointcloud = unproject(gt_depth_tensor, inv_K)  # depth_gt is inverse depth
            gt_norm_est = d2normals(gt_pointcloud, n_bases=4)
            pred_pointcloud = unproject(pred_depth_maps, inv_K)
            pred_norm_est = d2normals(pred_pointcloud, n_bases=4)

            dmask = dilate_mask(remove_borders(mask_t.clone()), ksize=3)
            nmask = torch.isnan(gt_norm_est).logical_not().all(1, True)
            normals_mask = (nmask * dmask).float()

            loss_norm = merr(pred_norm_est, gt_norm_est, normals_mask)

            if i < 10:
                save_normal_map(pred_norm_est * normals_mask, save_path=norm_path + f"pred_norm_est_{i}.png")
                save_normal_map(gt_norm_est * normals_mask, save_path=norm_path + f"gt_norm_est_{i}.png")

            loss = rmse_torch(result_t, gt_t, mask_t)
            avg_rmse += loss.item()
            avg_merr += loss_norm.item()
            img_count += 1


            if False: 
                gt_norm = gt_t * mask_t
                result_norm = result_t * mask_t

                gt_log = np.log1p(gt_t * mask_t)   # log(1+x) to handle zeros safely
                gt_norm = (gt_log - gt_log.min()) / (gt_log.max() - gt_log.min())

                result_log = np.log1p(result_t * mask_t)
                result_norm = (result_log - result_log.min()) / (result_log.max() - result_log.min())
            
            if False: #"munster_000136_000019" in depth_path: 
                gt_log = np.log1p(gt_t * mask_t)   # log(1+x) to handle zeros safely
                gt_norm = (gt_log - gt_log.min()) / (gt_log.max() - gt_log.min())

                result_log = np.log1p(result_t * mask_t)
                result_norm = (result_log - result_log.min()) / (result_log.max() - result_log.min())

                print("gt_norm, result_norm: ", gt_norm.shape, result_norm.shape)

                plt.imsave("gt_norm.png", gt_norm, cmap="viridis")
                plt.imsave("result_norm.png", result_norm[0], cmap="viridis")
                plt.imsave("gt_t.png", gt_t, cmap="viridis")
                plt.imsave("result_t.png", result_t[0], cmap="viridis")
                input("here")

        print("avg_rmse: ", avg_rmse / img_count)
        print("avg_merr: ", avg_merr / img_count)

        if i == 0:
            mode = "w"   # 첫 줄: 파일 새로 쓰기
        else:
            mode = "a"   # 이후: append
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(f"{i}-th averaged RMSE {avg_rmse / img_count} mErr {avg_merr / img_count}\n")

    #"""

    rank, _ = get_dist_info()

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
         
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)



import torch
import torch.nn.functional as F
import numpy as np

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/dataset/base.py#L361
def get_K(fx, fy, u0, v0, u_offset, v_offset):
    return np.array([[     fx,        0,  u0 - u_offset,  0],
                        [      0,       fy,  v0 - v_offset,  0],
                        [      0,        0,              1,  0],
                        [      0,        0,              0,  1]], dtype=np.float32)

# borrowed from: https://github.com/astra-vision/DenseMTL/blob/288425088f497fc9a5a22727bff1124ff857bd0f/models/normals.py
def unproject(depth, inv_K):
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
def d2normals(points, guide=None, n_bases=1):
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

def get_normal_mask(mask, depth, normals, train):
    if not train:
        mask *= (MIN_DEPTH < depth)*(depth < MAX_DEPTH)

    # get initial mask from invalid values in depth map
    dmask = dilate_mask(remove_borders(mask.float()), ksize=3)

    # get invalid values from normals computation
    nmask = torch.isnan(normals).logical_not().all(1, True)

    # combine both masks
    normals_mask = nmask*dmask

    return dmask, normals_mask
    
def remove_borders(x):
    """Remove 1 pixel along all 4 borders."""
    x[..., (0,-1), :] = False
    x[..., (0,-1)] = False
    return x


def dilate_mask(m, ksize):
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

if __name__ == '__main__':
    main()
