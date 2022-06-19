import numpy as np
import pickle
import os

def get_cifar10(datapath):
    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'

    x_tr = np.zeros((50000, 32, 32, 3), dtype='uint8')
    y_tr = np.zeros(50000, dtype='int32')

    for ii, fname in enumerate(train_filenames):
        cur_images, cur_labels = _load_datafile(os.path.join(datapath, fname))
        x_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_images
        y_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_labels

    x_te, y_te = _load_datafile(os.path.join(datapath, eval_filename))
    return (x_tr, y_tr), (x_te, y_te)

def binarize_cifar_class(_trainY, _testY, negative_mark=0):
    trainY = negative_mark * np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = 1
    trainY[_trainY == 1] = 1
    trainY[_trainY == 8] = 1
    trainY[_trainY == 9] = 1
    testY = negative_mark * np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = 1
    testY[_testY == 1] = 1
    testY[_testY == 8] = 1
    testY[_testY == 9] = 1
    return trainY, testY

def _load_datafile(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        #print(data_dict.keys())
        assert data_dict[b'data'].dtype == np.uint8
        image_data = data_dict[b'data']
        image_data = image_data.reshape((image_data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
        return image_data, np.array(data_dict[b'labels'])