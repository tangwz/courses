---
course: "getting-started-with-tensorflow"
chapter: "training-evaluating-models"
lesson: "compiling-optimizers"
sourceId: 1732
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/compiling-optimizers"
title: "编译模型：优化器"
description: "理解与选择Adam、SGD、RMSprop等优化器用于模型训练。"
order: 2
plots: []
sourceHash: "62e2bcf62d3c05b99137c376a214782d492abc2fc4f6007038e2389284b2a7f5"
sourceCorrections: []
---

通过 `model.compile()` 配置模型训练过程时，损失函数 (loss function)和**优化器**都是主要组成部分。损失函数用于量化 (quantization)模型预测与真实目标之间的偏差。如果说损失函数告诉我们模型*错得有多离谱*，那么优化器则决定了*模型该如何调整自身*才能减少这种偏差。

可以将神经网络 (neural network)的训练想象成在一个复杂的高维空间 (high-dimensional space)中寻找最低点，其中任意点的高度代表给定模型参数 (parameter)（权重 (weight)和偏置 (bias)）下的损失值。优化器就是你在这个空间中移动并向最低损失点下降的策略。

### 核心思想：梯度下降 (gradient descent)

深度学习 (deep learning)中的大多数优化策略都是**梯度下降**的变体。其基本原理很简单：

1. 计算损失函数 (loss function)相对于每个模型参数 (parameter)的**梯度**。梯度是一个向量 (vector)，它指向损失*增加*最快的方向。
2. 为了*减少*损失，我们需要沿着梯度的*反方向*移动。
3. 通过减去每个参数对应梯度的一部分来更新它。

这个过程迭代重复，理想情况下指导参数朝向使损失最小化的值。第3步中提到的“一部分”由一个重要的超参数 (hyperparameter)控制，它被称为**学习率**（通常表示为 $\alpha$ 或 $\eta$）。


$$
参数_{新} = 参数_{旧} - 学习率 \times 梯度
$$


学习率决定了下降过程中每步的大小。

- **小学习率**会导致收敛缓慢，但不太可能越过最小值。
- **大学习率**开始时能加快收敛，但有越过最小值的风险，可能导致损失震荡甚至发散。

选择一个合适的学习率对成功训练非常重要。

### Keras中常用的优化器

虽然基本的梯度下降 (gradient descent)奠定了基础，但人们已经开发出几种更复杂的优化器来提高收敛速度和稳定性。TensorFlow的Keras API提供了对其中许多优化器的方便使用。以下是一些最常用的优化器：

#### 1. SGD (随机梯度下降)

这是一个经典的优化器。SGD不是使用整个数据集（计算成本高昂）来计算梯度，而是使用称为**小批量**的少量随机数据子集来估计梯度。

- **优点：** 计算效率高，原理简单。
- **缺点：** 更新有噪音，可能收敛缓慢，对学习率选择敏感，比自适应方法更容易陷入局部最小值或鞍点。

Keras的`SGD`优化器通常包括增强功能：

- **动量：** 这引入了“速度”成分。更新在一致的方向上累积动量，有助于加速收敛，尤其是在平坦区域或浅沟壑中，并抑制震荡。
- **Nesterov动量：** 动量的一个细微变体，在实践中通常提供更快的收敛。它在动量更新方向上稍微“向前看”来计算梯度。

```python
import tensorflow as tf

# 基本SGD
sgd_optimizer_basic = tf.keras.optimizers.SGD(learning_rate=0.01)

# 带有动量的SGD
sgd_optimizer_momentum = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# 带有Nesterov动量的SGD
sgd_optimizer_nesterov = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

#### 2. Adam (自适应矩估计)

由于其有效性和相对易用性，Adam通常是许多深度学习 (deep learning)任务的默认选择。它是一个**自适应学习率**优化器，这意味着它为不同的参数 (parameter)计算各自的学习率。

它通过追踪过去梯度的指数衰减平均值（一阶矩，类似于动量）和过去平方梯度的指数衰减平均值（二阶矩，捕捉方差）来实现这一点。

- **优点：** 通常收敛速度快，在各种问题上表现良好。
- **缺点：** 与经过精细调整的带有动量的SGD相比，有时会收敛到次优解，需要更多内存来存储移动平均值。

重要的超参数 (hyperparameter)包括 `learning_rate`，`beta_1`（一阶矩的衰减率），`beta_2`（二阶矩的衰减率）和 `epsilon`（一个防止除以零的小值）。

```python
import tensorflow as tf

# 带有默认参数的Adam优化器 (learning_rate=0.001)
adam_optimizer_default = tf.keras.optimizers.Adam()

# 带有自定义学习率的Adam优化器
adam_optimizer_custom = tf.keras.optimizers.Adam(learning_rate=0.0005)

# 带有自定义beta值的Adam优化器
adam_optimizer_betas = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
```

#### 3. RMSprop (均方根传播)

RMSprop是另一种自适应学习率方法，它也维护着平方梯度的移动平均值。它将学习率除以该平均值的平方根。这有效地为每个参数自适应地调整学习率，对于大梯度的参数减小学习率，对于小梯度的参数增加学习率。

- **优点：** 在实践中表现良好，特别适用于循环神经网络 (neural network)（RNN）。Adagrad的一个良好替代品（Adagrad可能面临学习率变得过小的问题）。
- **缺点：** 作为默认选择不如Adam常用，但仍然非常有效。

```python
import tensorflow as tf

# 带有默认参数的RMSprop优化器 (learning_rate=0.001)
rmsprop_optimizer_default = tf.keras.optimizers.RMSprop()

# 带有自定义学习率和动量的RMSprop优化器
rmsprop_optimizer_custom = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.1)
```

#### 其他优化器

Keras提供了其他优化器，如 `Adagrad`、`Adadelta`、`Adamax`和 `Nadam`。虽然它们作为初始选择不如Adam或SGD常用，但它们具有某些特性，可能对特定类型的数据或网络结构有利（例如，Adagrad适用于稀疏数据）。

### 选择优化器

选择最佳优化器通常需要一些实验：

- **从Adam开始：** 它的自适应特性和普遍优秀的性能使其成为大多数问题的极好起点。开始时使用默认学习率（0.001）。
- **尝试带有动量的SGD：** 如果Adam没有产生令人满意的结果，或者你怀疑它收敛过快到次优最小值，请尝试带有动量的SGD（例如，momentum=0.9）。这通常需要更仔细地调整学习率。你可能需要尝试0.1、0.01、0.001等值。
- **考虑RMSprop：** 它是一个不错的替代方案，特别是如果你遇到Adam的问题或正在使用RNN时。
- **学习率调度：** 你可以使用学习率调度，而不是固定的学习率，它在训练过程中随时间降低学习率。这有助于在训练过程后期实现更精细的收敛。Keras回调或 `tf.keras.optimizers.schedules` 可以实现这一点。

### 在 `model.compile()` 中指定优化器

在编译模型时集成所选的优化器。你可以使用其字符串标识符（如果使用默认参数 (parameter)）来指定优化器，或者通过创建一个优化器实例（如果需要自定义学习率等参数）来指定。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 假设 'model' 是一个已定义的Keras模型 (例如, Sequential 或 Functional)
# model = keras.Sequential([...])

# 使用字符串标识符（使用默认参数）
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 使用带有自定义学习率的优化器实例
custom_adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=custom_adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 使用带有动量的SGD
custom_sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=custom_sgd,
              loss='mean_squared_error', # 回归示例
              metrics=['mae']) # 平均绝对误差指标
```

通过选择合适的优化器并配置其参数（尤其是学习率），你为模型有效地从数据中学习并最小化所选的损失函数 (loss function)提供了机制。连同损失和指标，优化器完成了在通过`model.fit()`启动训练过程之前所需的核心配置。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的资源，其中有专门的章节讨论梯度下降、SGD、动量、Adam和RMSprop等优化算法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化算法的原始研究论文，解释了其利用梯度一阶和二阶矩的自适应学习率机制。
- [tf.keras.optimizers module](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) — TensorFlow Developers (2024)
  TensorFlow中Keras优化器的官方文档，详细介绍了SGD、Adam和RMSprop等算法的使用、参数及实际实现方法。
