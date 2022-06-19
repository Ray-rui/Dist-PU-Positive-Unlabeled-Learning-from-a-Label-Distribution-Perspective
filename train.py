# ## import modules
import argparse
import math
import torch
from torch.utils.data import DataLoader

# customized modules
from dataTools.factory import *
from models.factory import *
from losses.factory import *
from losses.entropyMinimization import loss_entropy
from utils import fix_random_seed, validate, train
from dataTools.mixupDataset import MixupDataset
from customized.mixup import mixup_two_targets, mixup_bce
# ================

# ## define program arguments
def get_params():
    parser = argparse.ArgumentParser(description='Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective')

    parser.add_argument('--device', type=int, default=2, help='GPU index')
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['cifar-10', 'fmnist', 'alzheimer'])
    parser.add_argument('--datapath', type=str, default='') # TODO: fill in the datapath
    parser.add_argument('--num-labeled', type=int, default=1000, help='num of labeled positives in training set')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--loss', type=str, default='Dist-PU', choices=['Dist-PU'], help='PU risk')
    
    parser.add_argument('--warm-up-lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warm-up-weight-decay', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'])
    parser.add_argument('--schedular', type=str, default='cos-ann', choices=['cos-ann'])
    parser.add_argument('--entropy', type=int, default=1, choices=[0, 1])

    parser.add_argument('--co-mu', type=float, default=2e-3, help='coefficient of L_ent')
    parser.add_argument('--co-entropy', type=float, default=0.004)

    parser.add_argument('--alpha', type=float, default=6.0)
    parser.add_argument('--co-mix-entropy', type=float, default=0.04)
    parser.add_argument('--co-mixup', type=float, default=5.0)

    parser.add_argument('--warm-up-epochs', type=int, default=60)
    parser.add_argument('--pu-epochs', type=int, default=60)

    parser.add_argument('--random-seed', type=int, default=0,
                        help='initial conditions for generating random variables')

    global args
    args = parser.parse_args()
    print(args)

    return args
# ================

# ## experiment set up
def set_up_for_warm_up():
    fix_random_seed(args.random_seed)

    # set device
    global device
    device = torch.device('cuda:{}'.format(args.device) 
        if torch.cuda.is_available() else "cpu")
    args.device = device

    # obtain data
    global dataset_train, dataset_test, pu_dataset, test_loader
    dataset_train, dataset_test = create_dataset(args.dataset, args.datapath)
    pu_dataset = create_pu_dataset(dataset_train, args.num_labeled)
    test_loader = DataLoader(
        dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True
    )

    global train_loader
    train_loader = DataLoader(
        pu_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=True
    )

    # obtain model
    global model
    model = create_model(args.dataset)
    model = model.to(device)

    # loss fn
    global loss_fn
    loss_fn = create_loss(args)

    # obtain optimizer
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.warm_up_lr,
            weight_decay=args.warm_up_weight_decay
        )
    else:
        raise NotImplementedError("The optimizer: {} is not defined!".format(args.optimizer))

    # obtain schedular
    global schedular
    if args.schedular == 'cos-ann':
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs)
    else:
        raise NotImplementedError("The schedular: {} is not defined!".format(args.schedular))

    return

def set_up_for_dist_pu():
    # mixup dataset
    global mixup_loader, mixup_dataset
    mixup_loader = DataLoader(
        pu_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True
    )

    mixup_dataset = MixupDataset()
    mixup_dataset.update_psudos(mixup_loader, model, device)

    # label distribution loss
    global base_loss
    args.entropy = 0
    base_loss = create_loss(args)

    global co_entropy
    co_entropy = 0

    # obtain optimizer
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError("The optimizer: {} is not defined!".format(args.optimizer))

    # obtain schedular
    global schedular
    if args.schedular == 'cos-ann':
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs, 0.7*args.lr)
    else:
        raise NotImplementedError("The schedular: {} is not defined!".format(args.schedular))

    return
# ================

# ## train script with mixup
def train_mixup():
    model.train()
    loss_total = 0
    for _, (index, Xs, Ys) in enumerate(train_loader):
        Xs = Xs.to(device)
        Ys = Ys.to(device)
        psudos = mixup_dataset.psudo_labels[index].to(device)
        psudos[Ys==1] = 1

        mixed_x, y_a, y_b, lam = mixup_two_targets(Xs, psudos, args.alpha, device)
        outputs = model(mixed_x).squeeze()
        outputs = torch.clamp(outputs, min=-10, max=10)
        scores = torch.sigmoid(outputs)

        outputs_ = torch.clamp(model(Xs).squeeze(), min=-10, max=10)
        scores_ = torch.sigmoid(outputs_)

        loss = ( base_loss(outputs_, Ys.float())
            + co_entropy*loss_entropy(scores_[Ys!=1]) 
            + args.co_mix_entropy*loss_entropy(scores)
            + args.co_mixup*mixup_bce(scores, y_a, y_b, lam))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mixup_dataset.psudo_labels[index] = scores_.detach()

        loss_total = loss_total + loss.item()

    
    schedular.step()
    return loss_total / len(train_loader)
# ================

# ## functions used in main
def warm_up():
    for epoch in range(args.warm_up_epochs):
        train_loss = train(train_loader, model, device, loss_fn, optimizer, schedular)
        test_loss, acc, precision, recall, f1, auc, ap = validate(test_loader, model, device, loss_fn)
        print('epoch:{}; loss: {:.4f}, {:.4f}; \
            acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
                epoch, train_loss, test_loss, acc, precision, recall, f1, auc, ap
            ))
    
    print('Final result is %g', acc)
        

def update_co_entropy(epoch):
    global co_entropy
    co_entropy = (1-math.cos((float(epoch)/args.pu_epochs) * (math.pi/2))) * args.co_entropy

def dist_PU():
    test_loss, acc, precision, recall, f1, auc, ap = validate(test_loader, model, device, base_loss)
    print('pretrained; loss: {:.4f}; \
        acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
            test_loss, acc, precision, recall, f1, auc, ap
        ))

    best_acc = 0
    for epoch in range(args.pu_epochs):
        update_co_entropy(epoch)
        print('==> updating co-entropy: {:.5f}'.format(co_entropy))

        print('==> training with mixup')
        train_loss = train_mixup()
        test_loss, acc, precision, recall, f1, auc, ap = validate(test_loader, model, device, base_loss)
        print('epoch:{}; loss: {:.4f}, {:.4f}; \
            acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
                epoch, train_loss, test_loss, acc, precision, recall, f1, auc, ap
            ))
        
        if acc > best_acc:
            best_acc = acc
    print('Best result is %g', best_acc)
    print('Final Result is %g', acc)
# ================

# ## main
if __name__ == '__main__':
    try:
        get_params()

        print('====> warm up')
        set_up_for_warm_up()
        warm_up()

        print('====> Dist-PU')
        set_up_for_dist_pu()
        dist_PU()
    except Exception as exception:
        print(exception)
        raise
# ================