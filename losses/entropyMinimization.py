import torch
import torch.nn.functional as F

def loss_entropy(scores):
    return -torch.mean(scores*torch.log(scores)+(1-scores)*torch.log(1-scores))