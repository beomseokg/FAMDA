# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
#from mmseg.models.losses.rmse import rmse

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

def merr(pred, target, mask=None):
    # NOTE: pred, gt, mask:  torch.Size([3, 512, 1024]) torch.Size([3, 512, 1024]) torch.Size([1, 512, 1024])
    error = torch.rad2deg(2 * torch.atan2(torch.norm(pred - target, dim=1), torch.norm(pred + target, dim=1)))

    # mse = error.pow(2).mean()
    # thr1 = (error < 11.25).float().mean()
    # thr2 = (error < 22.5 ).float().mean()
    # thr3 = (error < 30   ).float().mean()
    return error.mean()#, error.median(), mse, thr1, thr2, thr3    

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
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
    h, w = depth.shape[-2:]

    meshgrid = torch.meshgrid(torch.arange(h), torch.arange(w))#, indexing='ij')
    pix_coords = torch.stack(meshgrid).flip(0).unsqueeze(0).flatten(2).float()#.to(depth.device)
    ones = torch.ones(1, 1, h*w)#, device=depth.device)
    pix_coords = torch.cat((pix_coords, ones), dim=1)

    # forward
    cam_points = depth.flatten(2) * (inv_K[:3, :3].unsqueeze(0) @ pix_coords)
    return cam_points.reshape(-1, 3, h, w).contiguous()

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


def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    num_imgs = len(results)

    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                results[i], gt_seg_maps[i], num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label

def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


MIN_DEPTH, MAX_DEPTH = 1e-3, 80 # in meters
def depth2disp(d):
    return 65536 / (d+1)
def disp2meters(d):
    return 655.36 / d - 1

def eval_metrics(results,
                 gt_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'RMSE', 'mErr']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    if metrics == ['mErr']:
        gt_normal_maps, mask_normal_maps = gt_maps[0], gt_maps[1]
        num_images = len(results)
        assert len(gt_normal_maps) == num_images
        assert len(mask_normal_maps) == num_images

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 2 if device.type == 'cuda' else 1
        total_merr = torch.zeros((), dtype=torch.float32, device=device)

        def _load_pred_norm_map(pred):
            if isinstance(pred, str):
                pred = np.load(pred)
            pred = np.asarray(pred)
            if pred.ndim == 4 and pred.shape[0] == 1:
                pred = pred[0]
            if pred.ndim != 3:
                raise ValueError(f'Expected predicted normal map with 3 dims, got shape {pred.shape}')
            if pred.shape[0] == 3:
                return pred
            if pred.shape[-1] == 3:
                return np.transpose(pred, (2, 0, 1))
            raise ValueError(f'Expected predicted normal map with channel size 3, got shape {pred.shape}')

        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)

            pred_batch = []
            gt_batch = []
            mask_batch = []
            for i in range(start, end):
                pred_batch.append(np.ascontiguousarray(_load_pred_norm_map(results[i])))
                gt_batch.append(np.ascontiguousarray(np.transpose(gt_normal_maps[i], (2, 0, 1))))
                mask_batch.append(np.ascontiguousarray(mask_normal_maps[i]))

            pred_batch = torch.from_numpy(np.stack(pred_batch)).to(device=device, dtype=torch.float32)
            gt_batch = torch.from_numpy(np.stack(gt_batch)).to(device=device, dtype=torch.float32)
            mask_batch = torch.from_numpy(np.stack(mask_batch)).to(device=device, dtype=torch.float32)

            error = torch.rad2deg(
                2 * torch.atan2(
                    torch.norm(pred_batch - gt_batch, dim=1),
                    torch.norm(pred_batch + gt_batch, dim=1)))
            batch_merr = (error * mask_batch).mean(dim=(1, 2))
            total_merr += batch_merr.sum()

        total_merr = (total_merr / num_images).detach().cpu().numpy()
        ret_metrics = OrderedDict({'mErr': total_merr})

        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }

        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })

        return ret_metrics

    elif metrics == ['RMSE']:
        gt_depth_maps, mask_depth_maps = gt_maps[0], gt_maps[1]
        num_images = len(results)
        assert len(gt_depth_maps) == num_images
        assert len(mask_depth_maps) == num_images

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 8 if device.type == 'cuda' else 1
        total_rmse = torch.zeros((), dtype=torch.float32, device=device)

        def _load_depth_map(depth):
            if isinstance(depth, str):
                depth = np.load(depth)
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            if depth.ndim != 2:
                raise ValueError(f'Expected depth map with 2 dims, got shape {depth.shape}')
            return depth

        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)

            pred_batch = []
            gt_batch = []
            mask_batch = []
            for i in range(start, end):
                pred_batch.append(np.ascontiguousarray(_load_depth_map(results[i])))
                gt_batch.append(np.ascontiguousarray(_load_depth_map(gt_depth_maps[i])))
                mask_batch.append(np.ascontiguousarray(_load_depth_map(mask_depth_maps[i])))

            pred_batch = torch.from_numpy(np.stack(pred_batch)).to(device=device, dtype=torch.float32)
            gt_batch = torch.from_numpy(np.stack(gt_batch)).to(device=device, dtype=torch.float32)
            mask_batch = torch.from_numpy(np.stack(mask_batch)).to(device=device, dtype=torch.float32)

            pred_depth = torch.sigmoid(pred_batch) * 655.35 + 1.0
            gt_m = 655.36 / gt_batch - 1.0
            pred_m = 655.36 / pred_depth - 1.0

            valid_mask = mask_batch * ((MIN_DEPTH < gt_m) & (gt_m < MAX_DEPTH)).float()
            pred_m = pred_m * valid_mask
            gt_m = gt_m * valid_mask

            pred_m_flat = pred_m.view(pred_m.shape[0], -1)
            gt_m_flat = gt_m.view(gt_m.shape[0], -1)

            pred_median = pred_m_flat.median(dim=1).values
            gt_median = gt_m_flat.median(dim=1).values
            scale = torch.ones_like(pred_median)
            nonzero = pred_median != 0
            scale[nonzero] = gt_median[nonzero] / pred_median[nonzero]
            pred_m = torch.clamp(pred_m * scale.view(-1, 1, 1), min=MIN_DEPTH, max=MAX_DEPTH)

            absdiff = torch.abs(gt_m - pred_m)
            batch_rmse = torch.sqrt((absdiff ** 2).mean(dim=(1, 2)))
            total_rmse += batch_rmse.sum()

        total_rmse = (total_rmse / num_images).detach().cpu().numpy()
        ret_metrics = OrderedDict({'RMSE': total_rmse})

        ret_metrics = {
            metric: value
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

    else: # ['mIoU'], ['mIoU', 'RMSE'], ['mIoU', 'RMSE', 'mErr']
        gt_seg_maps = gt_maps 
        total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results, gt_seg_maps, num_classes, ignore_index, label_map,
                reduce_zero_label)
        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
            

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
