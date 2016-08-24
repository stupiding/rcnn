This project prvides the Torch solution for the paper [Recurrent Convolutional Neural Network for object recognition](http://www.xlhu.cn/papers/Liang15-cvpr.pdf). The code is heavily inspired by [facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

#Requirements

1. A GPU machine with Torch and its cudnn bindings. See [Installing Torch](http://torch.ch/docs/getting-started.html#_).

2. Download Torch version cifar10 and svhn datasets, and put them to rcnn/data/cifar/ and rcnn/data/svhn/, respectively.

#How to use
Run main.lua with options to train RCNN models.

An cifar10 example with error rate 4.59% is:

`CUDA_VISIBLE_DEVICES=0,1 th main.lua -dataset cifar10 -model rcl3_large -nGPU 2 -nThreads 4 -lr 0.1 -nChunks 100 -batchSize 64`

An svhn example with error rate 1.5% is:

`CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -dataset svhn -model rcl3 -nGPU 4 -nThreads 8 -lr 0.1 -nChunks 50 -batchSize 256`

To see all options and their default value, run:

`th main.lua -help`

#Code introduction

1. main.lua: Overall procedure to run the code.

2. dataset.lua: Prepare mini-batchs from specified datasets, including possible data augmentation.

3. data.lua: Initiate the dataset and setup multi-thread data loaders.

4. model.lua: Initiate the network models. Model files are placed in rcnn/models/.

5. train.lua: Train and test network models.

6. parse.lua: Parse the input options.
