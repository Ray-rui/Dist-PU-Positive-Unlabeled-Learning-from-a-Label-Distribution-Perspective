import numpy as np
import torch
from torch.nn import functional as F

# adapted from https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch
def mixup_two_targets(x, y, alpha=1.0, device='cuda', is_bias=False):
    """
        Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam * x + (1-lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_ce_loss_with_softmax(preds, targets_a, targets_b, lam):
    """ 
        mixed categorical cross-entropy loss
    """
    mixup_loss_a = -torch.mean(torch.sum(F.softmax(targets_a,1)* F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(F.softmax(targets_b,1)* F.log_softmax(preds, dim=1), dim=1))

    mixup_loss = lam * mixup_loss_a + (1-lam) * mixup_loss_b
    return mixup_loss

def mixup_bce(scores, targets_a, targets_b, lam):
    mixup_loss_a = F.binary_cross_entropy(scores, targets_a)
    mixup_loss_b = F.binary_cross_entropy(scores, targets_b)
    
    mixup_loss = lam * mixup_loss_a + (1-lam) * mixup_loss_b
    return mixup_loss

def get_mix_up_loss(x, y, model, device):
    mixed_x, y_a, y_b, lam = mixup_two_targets(x, y, alpha=1.0, device=device, is_bias=False)
    return mixup_ce_loss_with_softmax(model(mixed_x), y_a, y_b, lam)
