from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset

import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class VkittiDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(VkittiDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_labelTrainIds.png',
            depth_map_suffix='.png',
            split=None,
            **kwargs)

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
                        # seg_map = img_name + seg_map_suffix
                        seg_map = img_name.replace('rgb/', 'classSegmentation/').replace('rgb_', 'classgt_') + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    if depth_dir is not None:
                        # depth_map = img_name + depth_map_suffix
                        depth_map = img_name.replace('rgb/', 'depth/').replace('rgb_', 'depth_') + depth_map_suffix
                        img_info['depth'] = dict(depth_map=depth_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    # seg_map = img.replace(img_suffix, seg_map_suffix)
                    seg_map = img.replace('rgb/', 'classSegmentation/').replace('rgb_', 'classgt_').replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if depth_dir is not None:
                    # depth_map = img.replace(img_suffix, depth_map_suffix)
                    depth_map = img.replace('rgb/', 'depth/').replace('rgb_', 'depth_').replace(img_suffix, depth_map_suffix)
                    img_info['depth'] = dict(depth_map=depth_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos