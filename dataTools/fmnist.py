from scipy.io import loadmat
# from sklearn.datasets import fetch_openml
import numpy as np

def get_mnist_fashion(datapath):
    x_tr, y_tr = load_mnist(datapath, 'train')
    x_te, y_te = load_mnist(datapath, 't10k')
    return (x_tr, y_tr), (x_te, y_te)

def binarize_mnist_fashion_class(_trainY, _testY, negative_mark=0):
    trainY = negative_mark * np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = 1
    trainY[_trainY == 2] = 1
    trainY[_trainY == 4] = 1
    trainY[_trainY == 6] = 1
    testY = negative_mark * np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = 1
    testY[_testY == 2] = 1
    testY[_testY == 4] = 1
    testY[_testY == 6] = 1
    return trainY, testY


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images.copy().astype(np.float32), labels.copy()

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = get_mnist_fashion()
    Y_train, Y_test = binarize_mnist_fashion_class(Y_train, Y_test)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_train[:5])
    print(Y_train[:5])