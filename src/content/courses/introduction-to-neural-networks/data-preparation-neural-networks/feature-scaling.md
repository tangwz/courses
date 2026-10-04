---
course: "introduction-to-neural-networks"
chapter: "data-preparation-neural-networks"
lesson: "feature-scaling"
sourceId: 1930
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks/feature-scaling"
title: "特征缩放：归一化与标准化"
description: "理解输入特征缩放的重要性及方法。"
order: 2
plots: ["plots/1930-0.json", "plots/1930-1.json"]
sourceHash: "e60d1dfd5200ae458ce6c96097d38afa844d689d2cd582d2aff0188c70ea47d0"
sourceCorrections: []
---

如前所述，神经网络 (neural network)对输入数据进行数学运算。数据的尺度和分布会显著影响网络有效学习的能力。设想一个数据集，其中包含两个特征：一个范围从 0 到 1，另一个范围从 1,000 到 100,000。在训练期间，特别是在使用基于梯度的优化方法时，值较大的特征可能不成比例地影响网络权重 (weight)的更新，这可能减缓收敛速度，甚至阻止网络找到一个好的解决方案。特征缩放通过转换数据来处理这个问题，使得所有特征在学习过程中能更平均地发挥作用。

两种常用的特征缩放技术是归一化 (normalization)（常被称为最小-最大缩放）和标准化（或 Z-分数归一化）。

### 归一化 (normalization) (最小-最大缩放)

归一化将特征重新缩放到一个固定范围，通常是 [0, 1]，有时是 [-1, 1]。缩放到 [0, 1] 的公式如下：


$$
x' = \frac{x - min(x)}{max(x) - min(x)}
$$


这里，$x$ 是原始特征值，$min(x)$ 是该特征在数据集中的最小值，$max(x)$ 是最大值。特征列中的每个值都被转换为介于0到1之间的新值 $x'$。

**优点：**

- 保证数据落在特定范围 ([0, 1]) 内。这对于期望输入在此范围内的算法或激活函数 (activation function)（如 Sigmoid）会有帮助。
- 易于理解和实现。

**缺点：**

- 对异常值高度敏感。一个非常大或非常小的值可以显著地将其余数据压缩到 [0, 1] 范围内的狭窄区间，可能降低缩放的实用性。
- 如果目标是根据特征的分布来比较它们，则它不能很好地处理变化的标准差。

考虑一个特征，其值域从 10 到 110。最小值为 10，最大值为 110。

- 值 60 将被归一化为：$(60 - 10) / (110 - 10) = 50 / 100 = 0.5$。
- 最小值 10 变为：$(10 - 10) / (110 - 10) = 0 / 100 = 0$。
- 最大值 110 变为：$(110 - 10) / (110 - 10) = 100 / 100 = 1$。

如果出现一个异常值 1000，最大值将变为 1000。现在，值 60 归一化为 $(60 - 10) / (1000 - 10) = 50 / 990 \approx 0.05$，显著压缩了原始范围。

### 标准化 (Z-分数归一化 (normalization))

标准化重新缩放特征，使其平均值 ($\mu$) 为 0，标准差 ($\sigma$) 为 1。公式如下：


$$
x' = \frac{x - \mu}{\sigma}
$$


这里，$x$ 是原始值，$\mu$ 是特征值的平均值，$\sigma$ 是特征值的标准差。结果值 $x'$ 表示原始值 $x$ 偏离平均值的标准差数量。

**优点：**

- 与归一化相比，受异常值影响较小，因为它使用平均值和标准差，这本质上对极端值不如最小/最大值敏感。
- 将数据中心化到零附近。这通常对神经网络 (neural network)训练中使用的梯度下降 (gradient descent)等优化算法有益。
- 不将数据限制在特定范围，如果异常值有意义且不应过度压缩，这可能更好。

**缺点：**

- 不会产生有界区间内的值，这可能是某些特定算法或情况的要求（尽管对大多数神经网络层来说通常不是问题）。

如果一个特征的平均值为 50，标准差为 15：

- 值 65 将被标准化为：$(65 - 50) / 15 = 15 / 15 = 1$。（比平均值高 1 个标准差）
- 值 35 将被标准化为：$(35 - 50) / 15 = -15 / 15 = -1$。（比平均值低 1 个标准差）
- 平均值 50 变为：$(50 - 50) / 15 = 0 / 15 = 0$。

下面的可视化图表显示了归一化和标准化对具有不同尺度的合成数据集的影响。



![特征缩放效果 (原始数据)](plots/1930-0.json)



> 原始数据的散点图，其中特征 1 大致范围在 20-80，特征 2 大致范围在 1千-1万。请注意轴上的不同尺度。



![特征缩放效果对比](plots/1930-1.json)



> 归一化（右上）将数据缩放到 [0, 1] 区间，标准化（左下）将数据中心化到 (0, 0) 并使方差为 1。由于异常值的影响，相对位置略有变化。

### 归一化 (normalization)与标准化之间的选择

没有一个确定的答案，但这里有一些指导原则：

- **标准化** 在神经网络 (neural network)训练中通常更受青睐。梯度下降 (gradient descent)等技术在输入特征中心化到零且具有可比方差时，通常表现更好。它对异常值也更具鲁棒性。许多标准网络初始化和激活函数 (activation function)都隐式地假定输入是标准化的。
- **归一化** 可以在以下情况考虑：
  - 您需要数据严格在 [0, 1] 或 [-1, 1] 范围内，这可能是由于特定的激活函数选择（尽管像 ReLU 这样的现代激活函数敏感度较低）。
  - 您使用的算法明确要求输入在一个有界范围内。
  - 您确定没有显著的异常值，或者它们已经单独处理。

在实际操作中，标准化 ($x' = \frac{x - \mu}{\sigma}$) 是为深度神经网络准备数据时更常见的选择。

### 重要实施说明

一个常见错误是在将数据集划分为训练集、验证集和测试集之前，计算整个数据集的缩放参数 (parameter)（最小/最大值或均值/标准差）。这会引入数据泄露，即验证集和测试集的信息影响了应用于训练集的转换，导致过于乐观的性能评估。

**正确的步骤是：**

1. 将您的数据划分为训练集、验证集和测试集。
2. **仅** 从**训练集**计算缩放参数（$min$，$max$，$\mu$，$\sigma$）。
3. 将*相同*的转换（使用从训练集获得的参数）应用于训练集、验证集和测试集。

这保证了在验证集和测试集上的模型评估能够反映模型在仅使用训练数据获得的知识处理后的新、未见过的数据上的表现。像 scikit-learn 这样的库提供了工具（`StandardScaler`、`MinMaxScaler`），它们通过其 `fit`（在训练数据上）和 `transform`（在所有数据分割上）方法来协助这一正确的流程。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本教材全面介绍了深度学习概念，包括特征缩放等数据预处理技术对网络训练的重要性。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  统计学习的经典参考文献，广泛涵盖了基本的数据预处理和缩放方法。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  官方文档，详细介绍了各种预处理技术，包括 `StandardScaler` 和 `MinMaxScaler`，以及应用它们以避免数据泄露的最佳实践。
- [Notes on Normalization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_m-0CcbldX1AWg2L9AQ2jSRRd_gDPi9fSot8TIxck2ubJD3Z6qhuIL3ivDE-2Rzx0DyIzOFCV6guTBV-UlqMiA_R_hcdZzjLyqMsBxBprDZ06xX-nGik23t01TeL0bcBJxTk=) — Andrew Ng (2023)
  Publisher: DeepLearning.AI
  来自一位领先的深度学习教育者的简明笔记，解释了与神经网络相关的数据归一化技术。
