import torch

def l1(pred, target, mask):
    pred = pred.squeeze() * mask.unsqueeze(1).detach()
    target = target.squeeze() * mask.unsqueeze(1).detach()
    
    loss = torch.sum(torch.abs(pred - target)) / torch.sum(mask)
    return loss

def rmse(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    pred = pred.squeeze() * mask.unsqueeze(1).detach()
    target = target.squeeze() * mask.unsqueeze(1).detach()

    loss = 0

    absdiff = torch.abs(pred.squeeze() - target.squeeze())
    for i in range(num_images):
        loss += torch.sqrt((absdiff[i] ** 2).mean())

    loss = loss / num_images
    return loss

def merr(pred, target, mask):
    num_images = pred.shape[0] # (i.e., num_batches)

    pred = pred.squeeze()
    target = target.squeeze()

    mErr, thr1, thr2, thr3 = 0, 0, 0, 0
    for i in range(num_images):
        # NOTE: pred, gt, mask should be:  torch.Size([3, 512, 1024]) torch.Size([3, 512, 1024]) torch.Size([1, 512, 1024])
        error = torch.rad2deg(2 * torch.atan2(torch.norm(pred[i] - target[i], dim=0), torch.norm(pred[i] + target[i], dim=0)))
        error = error * mask[i]
        mErr += error.mean()

        thr1 += (error < 11.25).float().mean()
        thr2 += (error < 22.5 ).float().mean()
        thr3 += (error < 30   ).float().mean()

    mErr = mErr / num_images
    thr1 = thr1 / num_images
    thr2 = thr2 / num_images
    thr3 = thr3 / num_images

    return mErr
