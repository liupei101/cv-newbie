# Fast R-CNN

Ross Girshick, ICCV 2015

## 介绍

目标检测任务的复杂程度是相对较大的，主要表现在两个方面。首先，非常多的目标位置候选框必须要生成出来。其次，这些候选框仅仅提供了粗略的定位信息，需要进一步微调以达到更加准确的定位。解决这些问题，通常需要在计算速度、准确率、简便性之间综合考虑。

RCNN方法和SPPNet方法的模型训练都是多阶段的（CNN，SVM，Bounding-box Regressor）。由于对每张图片的每个候选框都需要提取特征进行计算，所以RCNN的训练和检测都非常慢。而SPPNet虽然使用了共享计算来加快每个候选框的特征提取，但是也是多阶段训练的。

本文提出了一种新的训练算法，该算法解决了R-CNN和SPPnet存在的问题，同时提升了它们的速度和准确率。其优点如下：
- 比R-CNN和SPPnet更高的检测质量（mAP指标）
- 训练是单阶段的，使用多任务目标函数
- 训练可以更新所有的网络层参数
- 不需要硬盘存储用作Feature caching

## 原理

Fast R-CNN架构如下图所示。模型将整张图片和一系列候选目标作为输入。模型首先处理整张图片，经过多层的卷积和池化得到一个特征图。然后，对每个候选目标，一个RoI (region of interest) pooling层可以从得到的特征图中提取固定长度的特征向量。每个特征向量接着经过几个FC层进行处理，然后作为分类器（FC + softmax）和回归器（FC + bbox-regressor）的输入。所以，模型对每个RoI都有两个输出向量：softmax概率（K个类+背景类），每个类的bbox回归偏移值（四个值表示bbox的位置）。

### RoI pooling layer

RoI定义为一个矩形框界定的感兴趣的目标区域，使用四元组(r, c, j, w)表示。和SPPnet中的处理方式类似，我们把任意尺寸h x w的RoI分为 H x W （固定的值）个格子，然后使用最大池化计算每个格子的响应值，最后得到固定大小的特征输入。不同之处在于，我们仅仅使用了一次固定输出大小的池化(7 x 7)。

### 使用预训练模型初始化

当一个预训练的网络初始化一个Fast R-CNN网络时，它需要经历以下三个过程：
- 首先，将最后一层的最大池化层使用一个RoI pooling层进行代替（H = W = 7 for VGG16）。
- 然后，网络最后的FC和softmax层使用两个分支网络代替（即前面描述到的分类器和回归器）。
- 最后，网络的输入修改为：图片，以及图片中的RoIs。

### 检测网络的微调

**多任务目标函数**


**Mini-batch采样**


**经过RoI池化层的反向传播**


**SGD超参数**

分类器和回归器的FC层参数使用零均值的高斯分布进行初始化（标准差分别为0.01和0.001）。Bias被初始化为0。

> All layers use a per-layer learning rate of 1 for weights and 2 for biases and a global learning rate of 0.001. A momentum of 0.9 and parameter decay of 0.0005 (on weights and biases) are used.


# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

Shaoqing Ren, NIPS 2015

