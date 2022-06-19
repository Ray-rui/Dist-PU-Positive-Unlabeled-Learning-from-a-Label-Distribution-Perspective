# Dist-PU: Positive and Unlabeled Learning From a Label Disrtibution Perspective
This is a Pytorch implementation of Dist-PU.

# Environment
GPU:
- Geoforce RTX 3090
- cuda 11.1

OS:
- ubuntu 18.04.5

Python Related:
- python 3.7
- pytorch 1.8.1
- torchvision 0.9.1
- numpy 1.19.2
- sklearn 0.24.1

# Data Preparation 
1. Download *CIFAR-10 python version* (163MB) from http://www.cs.utoronto.ca/~kriz/cifar.html to your machine.
2. Decompress the downloaded file *cifar-10-python.tar.gz* from the first step.
3. Usually the second step would result in a new directory like '*/cifar-10-batches-py/' with files in it including:
- data_batch_[1-5]
- test_batch
- batches.meta
- readme.html

# Command
python train.py --device *GPUID* --datapath *DATAPATH*

# bibtex
@InProceedings{Zhao_2022_CVPR,
    author    = {Zhao, Yunrui and Xu, Qianqian and Jiang, Yangbangyan and Wen, Peisong and Huang, Qingming},
    title     = {Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14461-14470}
}