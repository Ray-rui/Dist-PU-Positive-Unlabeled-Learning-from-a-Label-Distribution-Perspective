import torch
from torchvision import transforms

from .PUdataset import PUdataset
from .cifar10 import get_cifar10, binarize_cifar_class
from .fmnist import get_mnist_fashion, binarize_mnist_fashion_class
from .alzheimer import get_ad, ad_transform

def create_dataset(dataset, datapath):
    if dataset == 'cifar-10':
        (X_train, Y_train), (X_test, Y_test) = get_cifar10(datapath)
        Y_train, Y_test = binarize_cifar_class(Y_train, Y_test)
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    elif dataset == 'fmnist':
        (X_train, Y_train), (X_test, Y_test) = get_mnist_fashion(datapath)
        Y_train, Y_test = binarize_mnist_fashion_class(Y_train, Y_test)
        transform = None
    elif dataset == 'alzheimer':
        (X_train, Y_train), (X_test, Y_test) = get_ad(datapath)
        transform = ad_transform
    else:
        raise NotImplementedError("The dataset: {} is not defined!".format(dataset))

    return BCDataset(X_train, Y_train, transform), BCDataset(X_test, Y_test, transform)

def create_pu_dataset(dataset_train, num_labeled):
    return PUdataset(dataset_train.X, dataset_train.Y, num_labeled, dataset_train.transform)

class BCDataset(torch.utils.data.Dataset):
    """
    BCDataset - Supervised Binary Classification dataset

    members:
        X - features
        Y - labels
    """

    def __init__(self, X, Y, transform=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __len__(self):
        return(len(self.Y))

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        return index, img, self.Y[index]