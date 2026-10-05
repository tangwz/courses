# 正则化方法：L1、L2、Dropout

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow/regularization-l1-l2-dropout)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当模型过度学习训练数据（包括其中的噪声和特定模式）时，会导致在新数据上的性能不佳。正则化 (regularization)方法旨在通过限制模型的复杂性来解决此问题，促使模型学习更普遍的模式。三种常见方式是：L1、L2 和 Dropout。

### L1 和 L2 正则化 (regularization)（权重 (weight)正则化）

L1和L2正则化通过在模型的损失函数 (loss function)中加入一个惩罚项来起作用。此惩罚项依据网络权重的数值大小（$w$）。思路是，权重过大的模型通常更复杂，且易于过拟合 (overfitting)，因为大权重可能导致输入微小变化时输出产生剧烈变化。通过惩罚大权重，我们鼓励模型找到更简洁且泛化能力更好的解决方案。

修改后的损失函数如下所示：


$$
\text{新损失} = \text{原始损失} + \lambda \times \text{正则化项}
$$


这里，$\lambda$ (lambda) 是正则化强度超参数 (parameter) (hyperparameter)。较大的 $\lambda$ 会施加更强的惩罚。

#### L1 正则化（Lasso）

L1正则化添加的惩罚项与权重的绝对值成正比：


$$
\text{L1惩罚项} = \lambda \sum_{i} |w_i|
$$


L1正则化一个显著效果是它促使*稀疏性*。它倾向于将一些权重推至精确的零，通过消除不重要输入的影响，有效地执行一种自动特征选择。

#### L2 正则化（Ridge或权重衰减）

L2正则化添加的惩罚项与权重的平方成正比：


$$
\text{L2惩罚项} = \lambda \sum_{i} w_i^2
$$


L2正则化在深度学习 (deep learning)中通常比L1更常见。它促使权重小且分布，防止任何单个权重变得过大，但它通常不会强制权重变为零。这通常被称为“权重衰减”，因为在梯度下降 (gradient descent)过程中，它会添加一个将权重推向零的项。

#### 在Keras中实现L1/L2

你可以使用`kernel_regularizer`、`bias_regularizer`和`activity_regularizer`参数轻松地为Keras层添加L1或L2正则化惩罚。`kernel_regularizer`对层的主要权重（即核）进行惩罚，而`bias_regularizer`对偏置 (bias)项进行惩罚。`activity_regularizer`对层的输出（激活）进行惩罚，这不太常见。

以下是在`Dense`层中添加L2正则化的示例：

```python
import os
import torch # 导入torch，以备他处需要时确认其可用性

# 将Keras后端设置为PyTorch（必须在导入Keras之前完成）
os.environ["KERAS_BACKEND"] = "torch"

# 现在导入Keras组件
import keras
from keras import layers
from keras import regularizers

# L2正则化示例
model = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001), # 对权重应用L2惩罚
        input_shape=(784,)
    ),
    layers.Dense(
        10,
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001) # 这里也应用L2惩罚
    )])

# 你也可以使用L1或L1/L2组合
# regularizers.l1(0.01)
# regularizers.l1_l2(l1=0.01, l2=0.001)

# 照常编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 可选：验证后端并运行快速测试 ---
print(f"\nKeras后端已确认: {keras.backend.backend()}")
import numpy as np
print("正在使用虚拟数据运行快速测试步骤...")
(x_test, y_test) = (np.random.rand(10, 784).astype(np.float32),
                    keras.utils.to_categorical(np.random.randint(10, size=10), num_classes=10))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试评估完成 (损失: {loss:.4f}, 准确度: {acc:.4f})")
```

传递给`regularizers.l1()`、`regularizers.l2()`或`regularizers.l1_l2()`的值是正则化因子 $\lambda$。选择合适的值通常需要尝试，并且通常通过超参数调整来确定。值通常在0.1到0.0001之间。

### Dropout

Dropout是一种不同但非常有效的正则化 (regularization)方法，专为神经网络 (neural network)开发。Dropout不是修改损失函数 (loss function)，而是在训练期间修改网络本身。

在每个训练步骤中，对于应用了Dropout的给定层，其一部分输出单元（神经元）会被随机地暂时“舍弃”——这意味着它们的输出被设为零。要舍弃的单元比例由*dropout率*决定，这是一个通常设置为0.1到0.5之间的超参数 (parameter) (hyperparameter)。

> Dropout示意图。在不同的训练步骤中，不同的神经元（显示为灰色虚线）会被随机停用。

为什么这样做有效？

1. **避免协同适应：** 神经元不能过分依赖特定其他神经元的存在，因为它们可能随时被舍弃。这促使它们学习更独立的特征。
2. **集成效应：** 使用dropout训练网络有点像同时训练大量共享权重 (weight)的小型网络集成。每个训练步骤实际上是在网络的某个“精简”版本上运行。

在*推断*阶段（评估或预测时），dropout被关闭，所有神经元都被使用。然而，为了弥补比训练时更多的神经元处于活跃状态这一事实，dropout层的输出通常会按等于dropout率的因子进行缩放（或者等效地，训练期间激活值被放大——这被称为“倒置dropout”，是常见的实现方式）。

#### 在Keras中实现Dropout

Dropout在Keras中通过`Dropout`层实现。你可以在要正则化其输出的层之间插入它。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3), # 应用30%的dropout率
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # 再次应用dropout
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

`Dropout`层的参数是dropout率（要舍弃的单元比例）。一种常见做法是在激活函数 (activation function)之后应用dropout，通常在网络中更密集的区域。最佳比率通常需要调整。

### 选择正则化 (regularization)方法

- **L2正则化：** 一个好的起点，使用广泛，且通常效果好。
- **Dropout：** 对大型深层网络尤其有用。在实践中非常常见。
- **L1正则化：** 在一般深度学习 (deep learning)中不如L2常见，但如果需要特征选择或高稀疏性时会很有用。

结合使用不同方法也很常见，例如在同一网络中同时使用L2权重 (weight)正则化和Dropout。正则化强度（L1/L2的 $\lambda$，dropout率）是重要的超参数 (parameter) (hyperparameter)，通常需要根据验证集性能进行调整。

通过应用这些方法，你可以显著减少过拟合 (overfitting)，并构建在新数据上泛化能力更好的模型，这是机器学习 (machine learning)中的一个重要目标。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7 and 8
  本书提供了深度神经网络中L1、L2（权重衰减）和Dropout正则化技术的全面理论与实践论述。
- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](https://www.jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 15(56); Pages: 1929-1958
  这篇开创性论文介绍了Dropout作为神经网络正则化技术，阐释了其机制以及在防止共适应方面的有效性。
- [Keras Regularizers API](https://keras.io/api/layers/regularizers/) — Keras Team (2024)
  官方Keras文档，用于将L1、L2和L1_L2正则化应用于层，包含代码示例和参数详情。
- [Keras Dropout Layer](https://keras.io/api/layers/regularization_layers/dropout/) — Keras Team (2024)
  官方Keras文档，解释了`Dropout`层、其用法和参数，用于实现Dropout正则化。

---

[上一节](01-%E7%90%86%E8%A7%A3%E8%BF%87%E6%8B%9F%E5%90%88%E5%92%8C%E6%AC%A0%E6%8B%9F%E5%90%88.md) · [下一节](03-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA.md)
