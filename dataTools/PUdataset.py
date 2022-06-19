import torch
import numpy as np

class PUdataset(torch.utils.data.Dataset):
    """
    members:
        X_train - feature matrix (num_instance, ) of training set
        Y_train - groud-truth labels of training set (1 for positive, -1 for negative)
        
        P_indexes - indexes of labeled positives
        Y_PU - PU labels of training set (1 for labeled positive, -1 for unlabeled)
    """

    def __init__(self, X_train, Y_train, num_labeled=1000, transform=None):
        super().__init__()

        self.X_train = X_train
        self.Y_train = Y_train

        self.transform = transform

        # simulate PU
        self.P_indexes = np.where(Y_train==1)[0]
        np.random.shuffle(self.P_indexes)
        self.P_indexes = self.P_indexes[:num_labeled]
        self.U_indexes = np.arange(len(Y_train))

        negative_mark = Y_train[Y_train!=1][0]

        # make U ~ pi*P+(1-pi)*N
        self.Y_PU = negative_mark * np.ones(len(Y_train)+num_labeled, dtype=np.int) 
        self.Y_PU[self.P_indexes] = 1
        
        self.X_train = np.concatenate((X_train, X_train[self.P_indexes]), axis=0)

    def __len__(self):
        return len(self.Y_PU)
    
    def __getitem__(self, index):
        img = self.X_train[index]
        if self.transform is not None:
            img = self.transform(img)
        return index, img, self.Y_PU[index]

if __name__ == '__main__':
    X_train = np.ones((4,10,10,3))
    Y_train = np.array([1,1,-1,-1])
    num_labeled = 1
    pu_dataset = PUdataset(X_train, Y_train, num_labeled)

    print(len(pu_dataset))
    print(pu_dataset[0])
    print(pu_dataset[1])