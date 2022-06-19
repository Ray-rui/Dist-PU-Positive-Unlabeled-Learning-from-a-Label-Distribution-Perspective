from .modelForCIFAR10 import CNN
from .modelForFMNIST import MultiLayerPerceptron
from torchvision.models import resnet50
from torch import nn

def create_model(dataset):
    if dataset.startswith('cifar'):
        return CNN()
    elif dataset.startswith('fmnist'):
        return MultiLayerPerceptron(28*28)
    elif dataset == 'alzheimer':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 1)
        return model
    else:
        raise NotImplementedError("The model: {} is not defined!".format(dataset))