---
course: "introduction-to-neural-networks"
chapter: "improving-network-performance"
lesson: "dropout-regularization"
sourceId: 2032
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-6-improving-network-performance/dropout-regularization"
title: "Dropout：随机停用神经元"
description: "了解dropout如何防止神经元共同适应。"
order: 4
plots: []
sourceHash: "54fc588459bb9e542fa53089467621b29381f1c2b9cca74f8ff91b80f51c072c"
sourceCorrections: []
---

Dropout，由 Srivastava 等人于2014年提出，是一种正则化 (regularization)技术，它采取了一种根本不同的方法来防止神经网络 (neural network)过拟合 (overfitting)。像 $L1$ 和 $L2$ 这样的正则化方法通过根据网络权重 (weight)的幅度向损失函数 (loss function)添加惩罚来发挥作用。然而，Dropout 在*训练期间*修改网络结构本身，为每个训练步骤随机移除网络中的单元（及其连接）。

### 思路：训练稀疏网络

想象你有一个大型神经网络 (neural network)。在每次训练迭代中（通常是每个mini-batch），dropout会随机“丢弃”或停用层中的一部分神经元。这意味着这些神经元，对于特定的训练迭代，不参与前向传播（它们的输出实际为零），也不参与反向传播 (backpropagation)步骤（不为其计算梯度）。

考虑一个包含神经元的层。对于每个神经元，会做出一个随机决定：它以概率 $p$（即“保留概率”）保持活跃，或者以概率 $1-p$ 被停用（丢弃）。这个过程会为每个训练样本或mini-batch重复，因此活跃的特定神经元集合持续变化。

> 一个网络层在单个dropout训练步骤之前和期间的简化视图。灰色节点（H2, H4）在此迭代中被随机停用。

### Dropout为何有效？

Dropout作为一种有效的正则化 (regularization)器，原因如下：

1. **避免共同适应：** 由于任何神经元都可能被随机丢弃，其他神经元无法过度依赖它的存在。这迫使每个神经元学习在其他神经元的不同随机子集中更有用的特征。它阻止了复杂的共同适应，即多个神经元仅以非常特定的方式协同工作，这可能是训练数据过拟合 (overfitting)的迹象。
2. **鼓励冗余：** 网络学习冗余表示。如果某个通路或神经元被丢弃，其他通路或神经元仍应能够产生合理的输出。
3. **隐式集成：** 使用dropout训练网络可以解释为训练大量“稀疏”网络（移除了不同子集的神经元的网络）。在测试时，使用完整网络近似于对这个由稀疏网络组成的广泛集成进行预测平均，这通常能提升性能和泛化能力。

### 实现：保留概率和反向Dropout

Dropout由一个超参数 (parameter) (hyperparameter)控制，通常是“保留概率” $p$，即任何给定神经元保持活跃的概率。或者，实现中可能使用“dropout率”（$1-p$），即神经元被丢弃的概率。对于隐藏层，$p$ 的常见取值在 0.5 到 0.8 之间。$p=1.0$ 表示不应用dropout。

一个重要之处在于如何在评估或预测（推理 (inference)时间）期间处理网络。我们希望此时使用完整的网络能力，因此我们不随机丢弃神经元。然而，简单地使用所有神经元将意味着层在推理期间的预期输出幅度会高于训练期间（因为更多的神经元处于活跃状态）。

为了弥补这一点，常用一种称为**反向Dropout**的技术。在*训练*阶段，应用dropout后（将一些激活值设为零），剩余*活跃*神经元的激活值通过除以保留概率 $p$ 来进行缩放。


$$
\text{缩放后的激活值} = \frac{\text{原始激活值}}{p} \quad (\text{如果神经元保留})
$$


$$
\text{缩放后的激活值} = 0 \quad (\text{如果神经元丢弃})
$$


通过在训练期间进行这种缩放，激活值的预期总和在训练和推理之间保持一致。在测试时，Dropout被简单地关闭，并且不需要缩放，因为权重 (weight)在训练时已经考虑了平均而言只有 $p$ 比例的神经元活跃这一情况。

### Dropout的实际运用

Dropout通常应用于隐藏层的输出，通常是网络中的全连接层，因为这些层往往拥有最多的参数 (parameter)且容易过拟合 (overfitting)。较少见（但并非没有）将其直接应用于输入层（它可能会丢弃输入特征）或卷积层（其中专门的dropout变体有时表现更好）。

大多数深度学习 (deep learning)框架都提供了简单的实现：

```python
# 使用Keras-like API的示例
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(0.5), # 应用dropout，速率为0.5 (p=0.5)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5), # 再次应用dropout
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 在model.fit()期间，dropout是活跃的。
# 在model.predict()或model.evaluate()期间，dropout会自动禁用。
```

总而言之，dropout是一种有效且计算开销小的正则化 (regularization)技术，它通过在训练期间随机禁用神经元来防止过拟合，迫使网络学习更少相互依赖的特征。通过反向dropout的实现确保了训练和推理 (inference)阶段之间的一致性。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting Neural Networks](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  介绍 Dropout 正则化技术、其理论动机和实证结果的基础性论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 262-269
  一本全面教科书，在深度学习的广泛背景下详细解释了包括 Dropout 在内的正则化方法。
- [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) — TensorFlow Developers (2024)
  TensorFlow Keras API 中 Dropout 层的官方文档，展示了其实际应用和配置。
