import torch
import numpy as np
import glob
import shutil
import os
import pickle
import cv2

def main():
    # Get the list of all files matching the pattern
    directory_path = 'data/cityscapes/sam_mask/train/*/*'
    filenames = glob.glob(os.path.join(directory_path))
    for filename in filenames:
        sub_directory = filename.split('/')
        des_directory = filename.replace(sub_directory[-1], "").replace("/sam_mask/", "/sam_mask_small/")
        des_filename = des_directory + sub_directory[-1]

        if not os.path.exists(des_directory):
            os.makedirs(des_directory)
        
        with open(filename, 'rb') as file:
            mask_list = pickle.load(file)

        mask_resize_list = []
        for mask_idx in range(len(mask_list)):
            mask = mask_list[mask_idx]
            mask_resize = cv2.resize(mask.astype(np.float32), (1024, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_resize_list.append(mask_resize)

        with open(des_filename, 'wb') as f:
            pickle.dump(mask_resize_list, f)

        print("src_filename: ", filename)
        print("des_filename: ", des_filename)


if __name__ == "__main__":
    main()
    print("here")