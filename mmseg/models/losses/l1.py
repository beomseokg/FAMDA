import torch


def l1(input, target, mask):
    num_images = input.shape[0] # (i.e., num_batches)
    absdiff = torch.abs(target.squeeze() - input.squeeze())
    if mask is not None:
        absdiff = absdiff * mask.squeeze()

    loss = 0
    for i in range(num_images):
        loss += absdiff[i].mean()
    loss = loss / num_images
    return loss

# reference: equation (1) - (3) in https://arxiv.org/pdf/2401.10891
def scaled_l1(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    loss = 0
    for i in range(num_images):
        median_target = target[i].median()
        median_pred = pred[i].median()
        scale_target = torch.abs(target[i] - median_target).mean()
        scale_pred = torch.abs(pred[i] - median_pred).mean()
        
        normalized_target = (target[i] - median_target) / scale_target
        normalized_pred = (pred[i] - median_pred) / scale_pred

        loss += torch.abs(normalized_pred - normalized_target).mean()

    loss = loss / num_images
    return loss