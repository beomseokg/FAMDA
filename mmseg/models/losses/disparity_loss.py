import torch

MIN_DEPTH=1e-3
MAX_DEPTH=80


def mse(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        diff = pred - target
        image_loss = ((torch.abs(diff) ** 2) * mask_target[i]).sum() / mask_target[i].sum()

        loss += image_loss
    loss = loss / num_images

    return loss

def ssi_mse(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        diff = pred_ssi - target_ssi
        image_loss = ((torch.abs(diff) ** 2) * mask.squeeze()[i]).sum() / mask.squeeze()[i].sum()

        loss += image_loss
    loss = loss / num_images

    return loss

def ssi_mse_grad(pred, target, mask, alpha=0.5):
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
        diff = pred_ssi - target_ssi
        image_loss = ((torch.abs(diff) ** 2) * mask_target[i]).sum() / mask_target[i].sum()

        grad_x = torch.abs(diff[:,1:] - diff[:,:-1])
        mask_x = torch.mul(mask_target[i,:,1:], mask_target[i,:,:-1])
        grad_x = torch.mul(mask_x, grad_x)
        grad_y = torch.abs(diff[1:,:] - diff[:-1,:])
        mask_y = torch.mul(mask_target[i,1:,:], mask_target[i,:-1,:])
        grad_y = torch.mul(mask_y, grad_y)
        grad_loss = (grad_x.sum() + grad_y.sum()) / mask_target[i].sum()

        loss += image_loss + alpha * grad_loss
    loss = loss / num_images

    return loss

def mae(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    if mask is not None:
        mask_target = (mask.squeeze() * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()
    else:
        mask_target = (torch.ones_like(pred.squeeze()) * ((MIN_DEPTH < target.squeeze()) * (target.squeeze() < MAX_DEPTH))).detach()

    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        diff = pred - target
        image_loss = (torch.abs(diff) * mask_target[i]).sum() / mask_target[i].sum()

        loss += image_loss
    loss = loss / num_images

    return loss

def ssi_mae(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    mask = mask.squeeze()
    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        diff = pred_ssi - target_ssi
        image_loss = (torch.abs(diff) * mask[i]).sum() / mask[i].sum()

        loss += image_loss
    loss = loss / num_images

    return loss

def ssi_mae_grad(pred, target, mask, alpha=0.5):
    num_images = pred.shape[0] # (i.e., num_batches)

    mask = mask.squeeze()
    pred = pred.squeeze()
    target = target.squeeze()

    loss = 0
    for i in range(num_images):
        pred_ssi = (pred[i] - pred[i].median()) / torch.abs(pred[i] - pred[i].median()).mean()
        target_ssi = (target[i] - target[i].median()) / torch.abs(target[i] - target[i].median()).mean()
        diff = pred_ssi - target_ssi
        image_loss = (torch.abs(diff) * mask[i]).sum() / mask[i].sum()

        grad_x = torch.abs(diff[:,1:] - diff[:,:-1])
        mask_x = torch.mul(mask[i,:,1:], mask[i,:,:-1])
        grad_x = torch.mul(mask_x, grad_x)
        grad_y = torch.abs(diff[1:,:] - diff[:-1,:])
        mask_y = torch.mul(mask[i,1:,:], mask[i,:-1,:])
        grad_y = torch.mul(mask_y, grad_y)
        grad_loss = (grad_x.sum() + grad_y.sum()) / mask[i].sum()

        loss += image_loss + alpha * grad_loss
    loss = loss / num_images

    return loss