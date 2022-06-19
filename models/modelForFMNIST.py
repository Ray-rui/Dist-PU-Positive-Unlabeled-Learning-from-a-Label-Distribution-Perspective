import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()
        
        self.l1 = nn.Linear(dim, 300, bias=False)
        self.bn1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.bn2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.l4 = nn.Linear(300, 300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1, 300)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.l5(x)
        return x