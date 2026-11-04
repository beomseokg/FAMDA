# https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth

import numpy
import torch
import torch.nn.functional as F

def berhu(input, target, mask, apply_log=False, threshold=.2):
    if apply_log:
        input, target = torch.log(1 + input), torch.log(1 + target)

    absdiff = torch.abs(target - input)
    if mask is not None:
        absdiff = absdiff * mask
    C = threshold * torch.max(absdiff).item()
    loss = torch.mean(torch.where(absdiff <= C,
                                  absdiff,
                                  (absdiff * absdiff + C * C) / (2 * C)))
    return loss