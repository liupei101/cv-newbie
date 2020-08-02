# Feature Pyramid Networks for Object Detection

CVPR 2017

## 介绍

在计算机视觉领域，从不同的尺度识别目标是一个基本的挑战。基于图像金字塔构建的特征金字塔构成了标准解决方案的基础。特征化的图像金字塔在手动特征设计的领域特别常见，因为很多检测器需要有密集的尺度采样来得到好的结果。同样地，在ConvNets中，很多目标检测方案都使用多尺度的图片测试。对图像金字塔每一层进行特征化的好处是它可以产生多尺度的特征表达，这些表达带有很强的语义信息。

尽管如此，对图像金字塔每一层进行特征化存在许多的限制。如果直接针对不同的尺度图片设计网络进行端到端的训练会由于内存限制而不可行，同时也会增加很多时间消耗。然而，图像金字塔不是唯一一种可以产生多尺度特征的方式。我们可以利用ConvNets固有的、经过计算的、多尺度层级的特征来得到特征金字塔，但是需要注意不同层产生的语义特征是有差异的。

Single Shot Detector (SSD)是最先尝试利用ConvNets金字塔特征层级的框架之一，但是它没有重新使用更高分辨率的特征映射（这对小目标的检测非常重要）。

本文的目标是很自然地使用ConvNets固有的特征层级的金字塔形状，同时创建一个在不同尺度都有很强语义特征的特征金字塔。To achieve this goal, we rely on an architecture that combines low-resolution, semantically strong features with high-resolution, semantically weak features via a top-down pathway and lateral connections。本文提出的FPN结构如下图所示，每一层的表示不同尺度下，不同语义级别的特征，每一层的特征都会被用于做预测。特别注意，高分辨率的特征对应网络的前面几层，它们对小目标的检测非常重要。

![FPN](tools/fpn-1.png)
![FPN-Architecture](tools/fpn-2.png)

如上图所示的FPN，Bottom-up结构中，每一组有相同输出大小的特征层都对应一个金字塔特征层。比如在ResNet中使用FPN，我们利用了4个层级的特征(C_2, C_3, C_4, C_5)，分别对应(4, 8, 16, 32)的stride。C_1没有被包括因为它会造成太大的内存消耗。Top-down结构中，使用了1 x 1的卷积来降低通道数。C_5直接经过1 x 1 的卷积得到分辨率最低的映射，其余都是和相应的上采样特征进行了融合（元素逐个相加）。最后，对于每一个融合后的特征映射，我们使用3 x 3的卷积得到最终的金字塔层级的映射(P_2, P_3, P_4, P_5)，这是为了减少上采样的混叠效果。

## 应用

如何在RPN和Fast R-CNN中使用FPN？

RPN中，我们将单个的特征图使用FPN来代替，即得到多个特征图（不同空间尺寸，相同通道数）。FPN中的每一层输出的特征映射仅使用一个anchor尺度(同样的3种aspect ratio)，因为每一层特征已经代表了不同的尺度信息。注意，**FPN所有层的特征映射共享classifier/regressor网络权值**。

Fast R-CNN中，把原来单个的RoI Pooling层换成多个RoI Pooling层即可，每一个RoI Pooling针对FPN不同层的输出映射。但是实际上需要注意，不同大小的RoI应该使用不同级别的金字塔特征（小的RoI应该使用分辨率高的金字塔特征，这样利于小物体的检测）。同样的，每个RoI Pooling的输出所经过的classifier/regressor网络也是共享的。

## 实验

更多的实验结果见原文，该部分也是很值得探究的。

作者提出的FPN（Feature Pyramid Network）算法同时利用低层特征高分辨率和高层特征的高语义信息，通过融合这些不同层的特征达到预测的效果。并且预测是在每个融合后的特征层上单独进行的，这和常规的特征融合方式不同。

FPN已经成为很多目标检测算法的组件，如Faster R-CNN，Mask R-CNN以及RetinaNet等。
