## 介绍

该代码来自斯坦福CS20-TensorFlow教程，主要演示了如何借助TensorFlow库构建CNN模型进行手写数字识别任务。

## 动机

作者在代码中使用TensorFlow非常清晰地实现了构建一个深度学习模型的标准流程：
- `self.get_data()`，构建数据流，使用TensorFlow标准的数据流（`tf.data` API）构建；
- `self.inference()`，构建网络结构，实现整个前向传播过程；
- `self.loss()`，定义损失函数计算；
- `self.optimize()`，定义模型优化器；
- `self.eval()`，定义模型性能评估指标；
- `self.summary()`，创建summary在TensorBoard上查看（`tf.summary` API）；
- `train()`，训练常规的深度学习模型。

还提供了两种实现思路：
- `07_convnet_mnist.py`，使用更加低级的接口构建网络结构。
- `o7_convnet_layers.py`，使用`tf.layers`高阶的API构建网络结构。

所以该示例代码作为**使用TensorFlow实现深度学习模型的代码框架参考**记录在此，方便查阅。
