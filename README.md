This project prvides the Torch solution for the paper [Recurrent Convolutional Neural Network for object recognition](http://www.xlhu.cn/papers/Liang15-cvpr.pdf). The code is heavily inspired by [facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

#Requirements

1. A GPU machine with Torch and its cudnn bindings. See [Installing Torch](http://torch.ch/docs/getting-started.html#_).

2. Download Torch version cifar10 and svhn datasets, and put them to rcnn/data/cifar/ and rcnn/data/svhn/, respectively.

#How to use
`CUDA_VISIBLE_DEVICES=0,1 th main.lua`
