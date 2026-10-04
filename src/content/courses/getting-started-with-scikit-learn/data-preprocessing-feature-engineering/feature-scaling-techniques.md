---
course: "getting-started-with-scikit-learn"
chapter: "data-preprocessing-feature-engineering"
lesson: "feature-scaling-techniques"
sourceId: 1879
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/feature-scaling-techniques"
title: "特征缩放技术"
description: "了解StandardScaler、MinMaxScaler和RobustScaler。"
order: 2
plots: []
sourceHash: "3edf622adc1aabb5907b26605131f2bfe1ccac5ee7490816ac5ef0073dc86bdf"
sourceCorrections: []
---

当数值输入特征处于相似尺度时，许多机器学习 (machine learning)算法的表现会明显更好。例如，考虑K近邻 (KNN) 等计算距离的算法，以及线性模型和神经网络 (neural network)中使用的梯度下降 (gradient descent)等优化方法。如果一个特征的范围是0到1，而另一个特征的范围是0到1,000,000，算法可能会无意中给予范围更大的特征更大的权重 (weight)，仅仅因为其尺度，而不是其重要性。特征缩放通过转换数据来解决这个问题，使特征具有可比较的范围或分布。

Scikit-learn提供了多种特征缩放工具，主要通过其在`sklearn.preprocessing`模块中的转换器API实现。让我们了解最常用的技术：

### 标准化 (StandardScaler)

标准化重新调整数据，使其均值 ($\mu$) 为0，标准差 ($\sigma$) 为1。这个过程常被称为Z-分数归一化 (normalization)。对于每个特征值 $x$，标准化后的值 $z$ 计算如下：


$$
z = \frac{x - \mu}{\sigma}
$$


这里，$\mu$ 是特征值的均值，$\sigma$ 是标准差。

标准化不会将值限制在特定范围（如[0, 1]），这可能是一个误解。它将数据中心化到零并根据标准差进行缩放。这种方法被广泛使用，对于那些假设数据以零为中心或遵循高斯分布的算法通常很有益（尽管标准化本身不能保证高斯分布）。它受异常值的影响小于Min-Max缩放，但显著的异常值仍可能影响计算出的均值和标准差。

**常见应用场景：**

- 主成分分析 (PCA)
- 线性模型（线性回归、逻辑回归），特别是带有正则化 (regularization)（岭回归、Lasso回归）的
- 支持向量 (vector)机 (SVM)
- 神经网络 (neural network)

### Min-Max缩放 (MinMaxScaler)

Min-Max缩放，通常简称为归一化 (normalization)，通过将每个特征缩放到给定范围（通常是[0, 1]或[-1, 1]）来转换特征。缩放到[0, 1]的转换公式如下：


$$
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$


其中 $X_{min}$ 和 $X_{max}$ 分别是特征的最小值和最大值。

这种方法保证所有特征具有完全相同的尺度。然而，它对异常值相当敏感。单个非常大或非常小的值会大幅压缩数据的其余部分到[0, 1]范围的非常小一部分。

**常见应用场景：**

- 需要特征在特定有界区间内的算法。
- 图像处理，其中像素强度通常缩放到[0, 1]。
- 某些神经网络 (neural network)激活函数 (activation function)期望输入在这个范围内。

### 鲁棒缩放 (RobustScaler)

当您的数据集包含显著异常值时，标准化和Min-Max缩放都可能存在问题。异常值会严重影响均值/标准差（对于标准化）或最小值/最大值（对于Min-Max缩放）。`RobustScaler` 使用对异常值更具抵抗力的统计量。

它通过移除中位数并根据四分位距 (IQR) 缩放数据来工作。IQR是第一四分位数（第25百分位数）和第三四分位数（第75百分位数）之间的范围。转换公式为减去中位数($Q_2$)并除以IQR($Q_3 - Q_1$):


$$
X_{scaled} = \frac{X - Q_2(X)}{Q_3(X) - Q_1(X)}
$$


通过使用中位数和IQR，`RobustScaler` 将数据中心化并进行缩放，而不会过度受到分布尾部极端值的影响。

**常见应用场景：**

- 已知或怀疑含有异常值的数据集。
- 当您希望缩放方法对极端值不如`StandardScaler`或`MinMaxScaler`敏感时。

### 选择合适的缩放器

没有一种缩放器适用于所有情况。

- `StandardScaler` 通常是一个好的默认选择，特别是对于那些假设数据以零为中心或遵循高斯分布的算法。
- `MinMaxScaler` 当您需要数据限制在特定范围内时很有用，但要留意其对异常值的敏感性。
- `RobustScaler` 是处理含有显著异常值的数据时的优选方案。

不同缩放技术的性能影响可能不同，取决于所使用的算法和数据的具体特征。通常有益于尝试不同的缩放器，作为模型开发过程的一部分。

在下一节，我们将看到如何使用Scikit-learn的`fit`和`transform`方法来应用这些缩放器。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  提供Scikit-learn中特征缩放方法的详细信息和示例。
- [Introduction to Machine Learning with Python: A Guide for Data Scientists](https://www.oreilly.com/library/view/introduction-to-machine-learning/9781449369415/) — Andreas C. Müller, Sarah Guido (2016)
  Publisher: O'Reilly Media
  提供使用Scikit-learn进行机器学习任务的实用指导，包括数据预处理。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125966/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  涵盖Scikit-learn的实用数据预处理和特征缩放技术。
