# ResNets Variants

The networks described below include these papers:
- Residual Attention Network for Image Classification (CVPR 2017)
- SENet: Squeeze-and-Excitation Networks (CVPR 2018)
- SCNet: Improving Convolutional Networks with Self-Calibrated Convolutions (CVPR 2020)

## Architecture

### Residual Attention Network

RAN（Residual Attention Network）可以使用多个Attention Module堆叠起来，最后以一种端到端的方式训练。每个Attention Module示意图如下：

![Attention Network](tools/Attention-1.png)

![The receptive field comparison between mask branch and trunk branch.](tools/Attention-2.png)

下面对Attention Module的特点进行以下说明：
- 每个Attention模型被分为两个分支，mask branch 和 trunk branch。其中trunk branch用来做特征处理（输出为T(x)），可以使用任何模型来代替。本文使用了ResNet、ResNeXt、Inception作为基本网络单元来构建trunk分支。而Mask branch使用bottom-up top-down 结构去学习和T(x)相同尺寸的Mask用来对T(x)进行加权处理，即Attention mechanism。Mask branch结构类似于HighWay Network。
- 关于Mask Branch。在前向传播时，它可以当做特征Mask；反向传播时，它可以作为梯度更新的过滤器（由于梯度乘上了M(x)）。所以它可以让Attention模块变得更加抗扰动（有相应的实验证实）。每一层的Mask Branch都在提取这一层特有的Mask值，这在复杂的场景中特别有用。
- 由于M(x)是0-1内的小数，随着深度的增加，会不断地乘上M(x)，这会削弱深层的特征。所以我们使用了H(x) = (1 + M(x)) T(x)结构对网络进行了改进。最终，M(x)起到了提升特征表达以及抑制来自trunk特征的噪声（反向传播时）。
- Mask branch contains fast feed-forward sweep and top-down feedback steps。前面的feed-forward操作快速地收集了整张图片的全局信息，后面的操作将全局信息和原始特征映射进行结合。这两个步骤展开为自下而上，自上而下，完全卷积的结构。
- 论文尝试了三种不同的Attention模式，即对Mask Branch输出进行处理的三种不同方法：Mixed Attention （每个通道每个坐标点进行sigmoid运算）、Channel Attention（对每个坐标点沿着通道轴向进行L2正则化，X_{i,j}/||X||）、Spatial Attention（每个通道的特征图先进行标准化处理，然后进行sigmoid运算）。最终的实验结果表明Mixed Attention模式（不添加任何约束，自适应特征）更好。

实验结果：在CIFAR-10和CIFAR-100数据上，均是当前最佳效果。在ImageNet上，得到了和ResNet-200相当的效果，同时仅用了46%的深度和69%的FLOPs。


## SENet



## SCNet


