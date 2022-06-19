from .distributionLoss import *
from .entropyMinimization import *

CLASS_PRIOR = {
    'cifar-10': 0.4,
    'fmnist': 0.4,
    'alzheimer': 0.5
}

def create_loss(args):
    prior = CLASS_PRIOR[args.dataset]
    print('prior: {}'.format(prior))
    
    if args.loss == 'Dist-PU':
        base_loss = LabelDistributionLoss(prior=prior, device=args.device)
    else:
        raise NotImplementedError("The loss: {} is not defined!".format(args.loss))

    def loss_fn_entropy(outputs, labels):
        scores = torch.sigmoid(torch.clamp(outputs, min=-10, max=10))
        return base_loss(outputs, labels) + args.co_mu * loss_entropy(scores[labels!=1])
    
    if args.entropy == 1:
        return loss_fn_entropy
    
    return base_loss