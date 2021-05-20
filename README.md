# CIFAR 10 PyTorch

A PyTorch implementation for training a medium sized convolutional neural network on CIFAR-10 dataset. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is a subset of the 80 million tiny image dataset (taken down). Each image in CIFAR-10 dataset has a dimension of 32x32. There are 60000 coloured images in the dataset. 50,000 images form the training data and the remaining 10,000 images form the test data.  The training data is divided into 5 batches each with 10,000 images. However, I have combined all the batches to form a single training set which allows us to have custom batch sizes. CIFAR-10 has 10 categories. Each category having 6000 images. 

## Model Architecture

I have used a medium sized Convolutional Neural Network to train and classify the images in CIFAR-10. I will briefly describe the convnet architecture here.

The model consists of five convolutional layers with dropout and maxpool layers inserted in between. Following the Conv layers are the fully connected layers. These layers are responsible for reading the feature map generated by the conv layers and convert them to class scores. The class scores are given by the Softmax classifier. 

```python
self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,stride=1,kernel_size=(3,3),padding=1)
self.conv2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1)

self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
self.fc2 = nn.Linear(in_features=256,out_features=128)
self.fc3 = nn.Linear(in_features=128,out_features=64)
self.fc4 = nn.Linear(in_features=64,out_features=10)

self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
self.dropout = nn.Dropout2d(p=0.5)
```

First, conv1 takes in a batch of input data and performs the conv operation and outputs 8 filters. The conv2 layer takes the 8 channeled input and outputs 32 channels. The spatial dimensions for inputs to these channels will be 32x32. Stride=1 and Pad=1 keeps the spatial dimensions same.

We then half our spatial dimensions by passing through a maxpool layer over a 2x2 window and at stride 2. Next the 16x16 output of maxpool layer will be convolved again to produce 64 channels in the conv3 layer. We then perform successive conv operations in layers conv3 and conv4. We then reduce our dimensions again to 8x8 by passing through a maxpool layer. The final conv layer takes in these and produces a 256 channeled output with the spatial dimensions now being 6x6.

Finally, we flatten these feature maps and pass them through fully connected layers to get the final class scores.

## Training

CIFAR-10 dataset is a hard dataset. After training for a long time, the model could achieve an Top-1 accuracy of 82% and a Top-3 accuracy of % and a Top-5 accuracy of 98% on the test data. Training for longer time will improve the Top-1 and Top-3 scores. Two models are available.

Total Training Time took about 4hrs on RTX 3080 10GB GPU. This includes model inference after every epoch as well as model checkpointing when we achieve a lower test loss than the previous best.

## Requirements

1. numpy
2. torch
3. torchvision
4. matplotlib
5. pillow

Download PyTorch from their website depending on the hardware configurations you have.

## Execution

In terminal, type the following command to start the training process.

```bash
$ python CIFAR10.py
```

Alternatively, you can also train the model in the notebook verison of this project. In the notebook version, I have explained some of the steps taken in training the model which includes things like data visualization, hyperparameter optimization, setting up training + evaluation pipelines and much more. Do check it out. 

*Note : If you are on windows, please use the python scripts to train the model. Training on Jupyter Notebook is very slow.*

## Citations

```
@article{,
title= {CIFAR-10 (Canadian Institute for Advanced Research)},
journal= {},
author= {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
year= {},
url= {http://www.cs.toronto.edu/~kriz/cifar.html},
abstract= {The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. },
keywords= {Dataset},
terms= {}
}
```