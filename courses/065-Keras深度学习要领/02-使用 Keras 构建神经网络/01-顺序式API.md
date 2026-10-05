# 顺序式API

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/sequential-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Keras提供了简洁易懂的方式来定义神经网络 (neural network)的结构。对于许多标准的深度学习 (deep learning)任务，网络由直接的序列层组成，其中一个层的输出直接作为下一个层的输入。对于这类常见情形，Keras提供了`Sequential` API，一种简单而优雅地逐层构建模型的方法。

可以将`Sequential`模型想象成线性堆叠的煎饼。您将层逐一堆叠起来，数据按照它们被添加的顺序流经各层。由于其简洁性，它是Keras中开始构建模型最常用的方法。

### 创建顺序模型

创建`Sequential`模型主要有两种方式：

1. **向构造函数传递层列表：** 当您预先知道整个架构时，这通常是最简洁的方法。
2. **使用`.add()`方法：** 这使您能够逐步添加层，在编程构建模型或进行试验时非常有用。

我们来看看这两种方法的实际应用。首先，请确保导入必要的组件：

```python
# 导入Sequential模型类型
from keras.models import Sequential

# 导入Dense层类型（全连接）
from keras.layers import Dense
```

**方法一：层列表**

您可以直接将层实例列表传递给`Sequential`构造函数来定义您的模型：

```python
# 定义一个包含两个Dense层的模型
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)), # 第一层需要input_shape
    Dense(units=10, activation='softmax')                  # 输出层
])

# 打印模型层和参数的摘要
model.summary()
```

**方法二：使用`.add()`**

或者，您可以实例化一个空的`Sequential`模型，并使用`.add()`方法逐一添加层：

```python
# 创建一个空的Sequential模型
model = Sequential()

# 添加第一层（指定输入形状）
model.add(Dense(units=64, activation='relu', input_shape=(784,)))

# 添加第二层（输出层）
model.add(Dense(units=10, activation='softmax'))

# 打印摘要
model.summary()
```

这两种方法都会得到完全相同的模型架构。选择哪种通常取决于个人偏好或编码风格。基于列表的方法更紧凑，而`.add()`方法对于复杂的序列或有条件添加层的情况可能更易读。

### 指定输入形状

请注意，两个示例中*第一个* `Dense`层都有`input_shape=(784,)`参数 (parameter)。这很重要。Keras需要知道模型应预期输入的形状，以便为第一层创建必要的权重 (weight)。

- **为什么只在第一层？** 一旦定义了第一层并知道了其输入形状，Keras就能自动推断所有后续层的输入形状。层 $N$ 的输出形状会成为层 $N+1$ 的输入形状。
- **`(784,)`是什么意思？** 这表示模型预期输入样本，其中每个样本是包含784个特征的向量 (vector)。这是展平的MNIST图像（28x28像素 = 784）的常见形状。如果您处理的是包含10个特征的表格数据，您可以使用`input_shape=(10,)`。对于未展平的图像数据，您可能会使用`(高度, 宽度, 通道)`等形状，例如`(28, 28, 1)`。

您只需为添加到`Sequential`模型中的第一个层指定输入形状。

### 数据线性流的可视化

`Sequential`模型强制执行简单、线性的数据流：

> 此图表说明了数据如何通过一个包含两个`Dense`层的简单`Sequential`模型进行线性流动。输入数据依次经过第1层和第2层，生成最终输出。

### 不使用顺序模型的情形

尽管`Sequential` API对许多任务都很有用，但它也有局限性。它严格设计用于每个层只有一个输入张量和一个输出张量，且这些层线性排列的模型。您不能使用`Sequential` API来构建以下模型：

- 有多个输入或多个输出的模型。
- 共享层（在图中多次使用同一个层实例）的模型。
- 有非线性连接，例如残差连接或分支的模型。

对于这些更复杂的架构，Keras提供了**函数式API**，我们将在后面的章节中介绍。

目前，`Sequential` API提供了一种简洁明了的方式，让您开始构建许多常见类型的神经网络 (neural network)，例如简单的前馈分类器和回归器。在接下来的章节中，我们将更仔细地查看这些示例中使用的`Dense`层和激活函数 (activation function)。

## 参考资料

- [Sequential model](https://keras.io/api/models/sequential/) — Keras team (2024)
  Keras官方文档，提供Sequential模型的详细API规范和使用示例。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras创建者撰写的权威书籍，涵盖了Keras的核心概念和实践应用，包括使用Sequential API构建模型。
- [Build a Keras model with the Sequential API](https://www.tensorflow.org/guide/keras/sequential_model) — fchollet (2024)
  Publisher: TensorFlow
  TensorFlow官方指南，探讨如何使用Keras Sequential API构建模型，并提供实际代码示例。

---

[上一节](../01-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9F%A5%E8%AF%86%E6%A6%82%E8%A7%88/06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%8E%AF%E5%A2%83%E9%AA%8C%E8%AF%81.md) · [下一节](02-%E5%B8%B8%E8%A7%81%E5%B1%82%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82.md)
