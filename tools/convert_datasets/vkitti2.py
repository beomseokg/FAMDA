# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import os
import cv2
import mmcv
import numpy as np
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

def plot_ones_mask(data, save_path="mask_plot.png"):
    mask = (data == 1).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask: value == 1")
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mask image saved to {save_path}")

def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(file, cv2.IMREAD_UNCHANGED)#[:, :, -1]

    # original classes: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, sky, person, rider, car, truck, bus, train, motorbike, bike
    
    # reference: L40-L55 https://github.com/astra-vision/DenseMTL/blob/main/dataset/semseg/labels.py
    # vk-cs setup (8 classes): road, building, pole, traffic light, traffic sign, tree+vegetation, sky, car+truck+van
    id_to_trainid = {
        5: 0,   # road
        # 0: 1,   # sidwalk: not present in vkitti
        4: 2,   # building
        # 0: 3,   # wall: not present in vkitti
        # 0: 4,   # fence: not present in vkitti
        9: 5,  # pole
        8: 6,   # traffic light
        7: 7,   # traffic sign
        2: 8,   # tree -> vegetation
        3: 8,   # vegetation -> vegetation
        # 1: 9,   # terrain - not considered in vk-cs
        1: 10,  # sky
        # 0: 11,  # person - not present in vkitti
        # 0: 12,  # rider - not present in vkitti
        12: 13, # car -> car
        11: 13, # truck -> car
        13: 13, # van -> car
        # 0: 15,  # bus - not present in vkitti
        # 0: 16,  # train - not present in vkitti
        # 0: 17,  # motorbike - not present in vkitti
        # 0: 18   # bike - not present in vkitti
    }

    # class VKitti2:
    #     #            name             id         vk colors        vk-cs colors
    #     labels = [('Terrain'        ,  0,    [210,   0, 200],   [             ] ),
    #             ('Sky'            ,  1,    [ 90, 200, 255],   [ 70, 130, 180] ),
    #             ('Tree'           ,  2,    [  0, 199,   0],   [107, 142,  35] ),
    #             ('Vegetation'     ,  3,    [ 90, 240,   0],   [107, 142,  35] ),
    #             ('Building'       ,  4,    [140, 140, 140],   [ 70,  70,  70] ),
    #             ('Road'           ,  5,    [100,  60, 100],   [128,  64, 128] ),
    #             ('GuardRail'      ,  6,    [250, 100, 255],   [             ] ),
    #             ('TrafficSign'    ,  7,    [255, 255,   0],   [220, 220,   0] ),
    #             ('TrafficLight'   ,  8,    [200, 200,   0],   [250, 170,  30] ),
    #             ('Pole'           ,  9,    [255, 130,   0],   [153, 153, 153] ),
    #             ('Misc'           , 10,    [ 80,  80,  80],   [             ] ),
    #             ('Truck'          , 11,    [160,  60,  60],   [  0,   0, 142] ),
    #             ('Car'            , 12,    [255, 127,  80],   [  0,   0, 142] ),
    #             ('Van'            , 13,    [  0, 139, 139],   [  0,   0, 142] )]


    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n

    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VKITTI2 annotations to TrainIds')
    parser.add_argument('vkitti_path', help='vkitti data path')
    parser.add_argument('--gt-dir', default='classSegmentation_label', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def convert_RGB_to_id(file):
    # official color map for Virtual Kitti 2 (reference: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)
    # Category r g b    
    # Terrain 210 0 200
    # Sky 90 200 255
    # Tree 0 199 0
    # Vegetation 90 240 0
    # Building 140 140 140
    # Road 100 60 100
    # GuardRail 250 100 255
    # TrafficSign 255 255 0
    # TrafficLight 200 200 0
    # Pole 255 130 0
    # Misc 80 80 80
    # Truck 160 60 60
    # Car 255 127 80
    # Van 0 139 139
    # Undefined 0 0 0

    # color to label map (reference: https://github.com/astra-vision/DenseMTL/blob/main/dataset/semseg/labels.py)
    #                          name             id         vk colors        vk-cs colors
    vkitti_color_id_map = [('Terrain'         ,  0,    [210,   0, 200],   [             ] ),
                            ('Sky'            ,  1,    [ 90, 200, 255],   [ 70, 130, 180] ),
                            ('Tree'           ,  2,    [  0, 199,   0],   [107, 142,  35] ),
                            ('Vegetation'     ,  3,    [ 90, 240,   0],   [107, 142,  35] ),
                            ('Building'       ,  4,    [140, 140, 140],   [ 70,  70,  70] ),
                            ('Road'           ,  5,    [100,  60, 100],   [128,  64, 128] ),
                            ('GuardRail'      ,  6,    [250, 100, 255],   [             ] ),
                            ('TrafficSign'    ,  7,    [255, 255,   0],   [220, 220,   0] ),
                            ('TrafficLight'   ,  8,    [200, 200,   0],   [250, 170,  30] ),
                            ('Pole'           ,  9,    [255, 130,   0],   [153, 153, 153] ),
                            ('Misc'           , 10,    [ 80,  80,  80],   [             ] ),
                            ('Truck'          , 11,    [160,  60,  60],   [  0,   0, 142] ),
                            ('Car'            , 12,    [255, 127,  80],   [  0,   0, 142] ),
                            ('Van'            , 13,    [  0, 139, 139],   [  0,   0, 142] ),
                            ('Undefined'      , 14,    [  0,   0,   0],   [             ])]

    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    
    color_label = cv2.imread(file, cv2.IMREAD_UNCHANGED) # BGR-based
    id_label = np.zeros(color_label.shape[:2], dtype=np.uint8)[:, :, None]

    for name, id, vk_color, _ in vkitti_color_id_map:
        bgr_vk_color = np.array([vk_color[2], vk_color[1], vk_color[0]])
        id_mask = np.all(color_label == bgr_vk_color, axis=-1)
        id_label[id_mask] = id
        print("name: ", name, int(np.sum(id_mask)))

    new_file = file.replace('vkitti2/classSegmentation', 'vkitti2/classSegmentation_label')
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    Image.fromarray(id_label[:,:,-1]).save(new_file)

def main():
    args = parse_args()
    vkitti_path = args.vkitti_path
    out_dir = args.out_dir if args.out_dir else vkitti_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(vkitti_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    # convert RGB-based classSemSegmentation files into id-based one (if there is no classSegmentation_label)
    if args.nproc > 1:
        mmcv.track_parallel_progress(
            convert_RGB_to_id, poly_files, args.nproc)
    else:
        mmcv.track_progress(convert_RGB_to_id, poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
