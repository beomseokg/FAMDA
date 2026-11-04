import torch

MIN_DEPTH=1e-3
MAX_DEPTH=80

def rmse(pred, target, mask, median_scaling=False):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))
    else:
        mask_target = torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))
    pred = pred.squeeze() * mask_target
    target = target.squeeze() * mask_target

    if median_scaling:
        for i in range(num_images):
            if torch.median(pred[i]) != 0:
                pred[i] *= float(torch.median(target[i]).detach()) / float(torch.median(pred[i]).detach())
        pred = torch.clamp(pred, min=MIN_DEPTH, max=MAX_DEPTH)

    loss = 0
    absdiff = torch.abs(pred.squeeze() - target.squeeze())
    for i in range(num_images):
        loss += torch.sqrt((absdiff[i] ** 2).mean())
    loss = loss / num_images
    return loss

def ssi_mse(pred, target, mask, median_scaling=False):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    print("mask_target: ", mask_target.shape)
    input("here")

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        loss += (((pred_ssi - target_ssi) ** 2) * mask_target[i]).mean()
    loss = loss / num_images

    return loss

def ssi_mse_grad(pred, target, mask, median_scaling=False):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    grad = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        loss += (((pred_ssi - target_ssi) ** 2) * mask_target[i]).mean() + grad
    loss = loss / num_images

    return loss

def ssi_mae(pred, target, mask, median_scaling=False):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        loss += (torch.abs(pred_ssi - target_ssi) * mask_target[i]).mean()
    loss = loss / num_images

    return loss

def ssi_mae_grad(pred, target, mask, median_scaling=False):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        loss += (torch.abs(pred_ssi - target_ssi) * mask_target[i]).mean()
    loss = loss / num_images

    return loss