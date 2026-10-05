# 实践：构建图像分类CNN

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/practice-cnn-image-classification)

[返回章节目录](README.md) · [返回课程目录](../README.md)

卷积神经网络 (neural network)（CNN）由`Conv2D`和`MaxPooling2D`等层组成。构建、训练和评估一个用于图像分类任务的CNN，使用Keras。CIFAR-10数据集，作为图像分类模型常用的参考数据集，提供了数据。

### CIFAR-10数据集

CIFAR-10包含60,000张32x32像素的彩色图像，分属于10个不同类别（例如，飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。每个类别有6,000张图像，其中50,000张用于训练，10,000张用于测试。我们的目标是训练一个能够正确分类这些小图像的CNN。

### 1. 设置与数据加载

首先，我们需要从Keras导入必要的库并加载数据集。Keras方便地提供了对多个标准数据集的访问，包括CIFAR-10。

```python
import numpy as np
import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical

# 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of training samples: {x_train.shape[0]}")
print(f"Number of test samples: {x_test.shape[0]}")
```

您应该会看到显示数据形状的输出：50,000张训练图像和10,000张测试图像，每张都是32x32像素，带有3个颜色通道（RGB）。标签最初是0到9的整数。

### 2. 数据预处理

神经网络 (neural network)通常在输入数据进行适当缩放时表现更好。我们将像素值从[0, 255]的范围归一化 (normalization)到[0, 1]。此外，由于我们有10个不同的类别，并将使用分类交叉熵作为损失函数 (loss function)，我们需要将整数标签转换为独热编码格式。

```python
# 将像素值归一化到0到1之间
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 将类别向量转换为二进制类别矩阵（独热编码）
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Sample training label (original): {y_train[0].argmax()}") # 原始标签示例索引
print(f"Sample training label (one-hot): {y_train[0]}") # 独热编码标签示例
```

归一化有助于稳定训练过程，独热编码将每个标签表示为一个向量 (vector)，其中只有对应于类别索引的元素为1，其余均为0。`categorical_crossentropy`损失函数需要这种格式。

### 3. 构建CNN模型

现在，让我们使用Keras Sequential API定义CNN的结构。我们将堆叠`Conv2D`和`MaxPooling2D`层，然后是`Flatten`和`Dense`层用于分类。

```python
input_shape = x_train.shape[1:] # Should be (32, 32, 3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # 添加Dropout进行正则化
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
```

我们来分析一下这个结构：

- `keras.Input(shape=input_shape)`: 定义模型预期的输入形状。
- `Conv2D(32, kernel_size=(3, 3), activation="relu")`: 第一个卷积层使用32个3x3大小的滤波器。它应用ReLU激活函数 (activation function)。默认填充为'valid'，这意味着输出的空间维度可能会略微缩小。
- `MaxPooling2D(pool_size=(2, 2))`: 将特征图的空间维度（高度和宽度）减半。
- `Conv2D(64, kernel_size=(3, 3), activation="relu")`: 第二个卷积层使用64个滤波器。通常的做法是，随着空间维度的减小，在更深层中增加滤波器数量。
- `MaxPooling2D(pool_size=(2, 2))`: 另一个池化层，用于进一步下采样。
- `Flatten()`: 将池化层的2D特征图转换为1D向量 (vector)，为全连接层做准备。
- `Dropout(0.5)`: 一种正则化 (regularization)技术，在每个更新周期中，随机选择的神经元在训练期间被忽略（设置为0）。这通过减少神经元之间的相互依赖来帮助防止过拟合 (overfitting)。0.5的比例意味着一半的输入单元被丢弃。
- `Dense(num_classes, activation="softmax")`: 最终输出层。它有`num_classes`（10）个单元，每个类别一个。`softmax`激活函数输出类别上的概率分布，表示模型对每个类别的置信度。

`model.summary()`的输出提供了层、其输出形状以及可训练参数 (parameter)数量的有用概览。

### 4. 编译模型

在训练之前，我们需要使用`compile()`方法配置学习过程。这包括指定优化器、损失函数 (loss function)以及在训练期间要监控的任何指标。

```python
# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

- `loss="categorical_crossentropy"`: 此损失函数适用于标签为独热编码的多类别分类问题。它衡量预测概率分布（来自softmax）与真实分布之间的差异。
- `optimizer="adam"`: Adam是一种流行且通常有效的优化算法，它在训练期间调整学习率。
- `metrics=["accuracy"]`: 我们要求模型在训练和评估期间报告分类准确率。

### 5. 训练模型

现在我们使用`fit()`方法训练模型，提供训练数据、批次大小、周期数和验证数据。

```python
batch_size = 128
epochs = 15

print("开始训练...")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print("训练完成。")
```

- `x_train`, `y_train`: 训练图像及其对应的独热编码标签。
- `batch_size=128`: 每次迭代（梯度更新）处理的样本数量。更大的批次大小可以加快训练速度，但可能需要更多内存。
- `epochs=15`: 模型将遍历整个训练数据集的次数。
- `validation_split=0.1`: 为了避免在训练期间使用单独的测试集进行验证（这可能会导致间接拟合测试集），我们预留10%的*训练*数据作为验证集。这些数据不用于训练模型权重 (weight)，而是在每个周期结束时进行评估，以监控模型在未见过数据上的表现并检测潜在的过拟合 (overfitting)。

`fit()`方法返回一个`History`对象，其中包含每个周期的训练和验证损失及指标的记录。

### 6. 评估模型

训练后，我们将在模型从未见过的保留测试集（`x_test`、`y_test`）上评估模型的性能。

```python
# 在测试数据上评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")
```

这提供了测试集上的最终损失和准确率，对模型的泛化能力给出了无偏估计。您应该会看到测试准确率低于训练准确率，特别是如果发生了过拟合 (overfitting)。对于这个简单的CNN和有限的周期，准确率可能在65-75%左右。后续讨论的更复杂的架构和方法（如数据增强和更精密的正则化 (regularization)）可以显著提升这一点。

### 7. 可视化训练历史

绘制训练和验证的准确率和损失随周期的变化图有助于理解训练动态并识别过拟合 (overfitting)。



[交互图表：模型训练历史](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/practice-cnn-image-classification#plot-dmvws8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "模型训练历史",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "准确率"
    },
    "legend": {
      "title": "指标"
    },
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "训练准确率",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
      ],
      "y": [
        0.38,
        0.53,
        0.6,
        0.64,
        0.67,
        0.69,
        0.71,
        0.73,
        0.74,
        0.75,
        0.76,
        0.77,
        0.78,
        0.79,
        0.8
      ],
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "验证准确率",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
      ],
      "y": [
        0.49,
        0.58,
        0.63,
        0.66,
        0.68,
        0.69,
        0.7,
        0.71,
        0.71,
        0.72,
        0.72,
        0.72,
        0.73,
        0.73,
        0.73
      ],
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 15个周期中训练集和验证集的准确率曲线。



[交互图表：模型训练历史](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/practice-cnn-image-classification#plot-1eyaa21)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "模型训练历史",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "损失"
    },
    "legend": {
      "title": "指标"
    },
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "训练损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
      ],
      "y": [
        1.7,
        1.32,
        1.15,
        1.04,
        0.96,
        0.9,
        0.84,
        0.8,
        0.76,
        0.73,
        0.7,
        0.67,
        0.64,
        0.61,
        0.59
      ],
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "验证损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
      ],
      "y": [
        1.42,
        1.19,
        1.07,
        0.99,
        0.94,
        0.9,
        0.88,
        0.86,
        0.85,
        0.83,
        0.82,
        0.82,
        0.81,
        0.81,
        0.81
      ],
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 15个周期中训练集和验证集的损失曲线。

理想情况下，训练和验证准确率都应增加，而损失应减少。如果验证准确率趋于平稳或开始下降，而训练准确率持续上升（且验证损失增加），这便是过拟合的迹象。我们的示例图表显示了典型行为：初期快速提升，随后增长放缓。训练和验证指标之间的差距表明存在一定程度的过拟合，`Dropout`层有助于缓解这一点。第6章的方法将讨论如何进一步改善泛化能力。

本次实践练习体现了使用Keras进行图像分类的CNN端到端过程。您已加载数据、对其进行预处理、构建了一个标准CNN结构、编译、训练并评估了其性能。这为处理更复杂的图像相关问题奠定了坚实的基础。

## 参考资料

- [Keras documentation](https://keras.io/) — Keras Team (2024)
  Keras深度学习框架的官方文档，提供了层、模型和训练API的详细信息。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的理论和实践，包括CNN、数据预处理和训练策略。
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) — Fei-Fei Li, Ehsan Adeli, Justin Johnson, Zane Durante (2015)
  一门备受推崇的在线课程，提供卷积神经网络及其在计算机视觉中应用的深入讲座和材料。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化算法，该算法因其自适应学习率特性而广泛用于训练深度神经网络。
- [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  一篇基础性论文，介绍了LeNet-5，这是一种开创性的卷积神经网络架构，确立了许多现代CNN原则。

---

[上一节](07-%E4%BA%86%E8%A7%A3%E7%89%B9%E5%BE%81%E5%9B%BE.md) · [下一节](../05-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28RNN%29/01-%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E6%A6%82%E8%BF%B0.md)
