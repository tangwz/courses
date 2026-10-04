---
course: "introduction-to-deep-learning"
chapter: "building-training-dnns"
lesson: "data-preparation-preprocessing"
sourceId: 5054
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/data-preparation-preprocessing"
title: "神经网络的数据准备"
description: "讨论数据格式化、特征缩放（归一化/标准化）和数据划分。"
order: 3
plots: ["plots/5054-0.json"]
sourceHash: "3fbfdd92ec6133717e3505aa12340ce05aaabd2f9cfe390215b879bc3c65f77b"
sourceCorrections: []
---

神经网络 (neural network)能有效学习之前，输入数据必须具备正确的形态和格式。原始数据集常包含尺度不一的特征、类别值或缺失条目，这些都无法直接用于网络内部的数学运算。正确的数据准备是深度学习 (deep learning)流程中的一个基本环节，它直接影响模型的收敛速度和整体表现。可以将其比作烹饪前的食材准备；没有妥善准备，最终的菜肴很难成功。

本节涵盖了使用现代框架将原始数据转换为适合训练深度神经网络的基本方法。我们将侧重于数据格式化、数值特征缩放以及数据用于训练和评估的划分。

### 数据格式化：张量与结构

PyTorch和TensorFlow等深度学习 (deep learning)框架主要处理名为张量的多维数组。您通常会将输入数据（特征）和目标数据（标签或目标值）表示为张量。

- **特征 ($X$):** 通常是一个二维张量，其中行表示单个样本（例如，图像、用户档案、传感器读数），列表示不同的特征（例如，像素值、用户年龄、温度）。形状通常为（样本数，特征数）。对于图像或序列等特定数据类型，维度可能会增加（例如，图像为（样本数，高度，宽度，通道数））。
- **标签 ($y$):** 通常是用于回归或单输出分类的一维张量，或用于多类别分类（通常在独热编码后）的二维张量。形状通常与样本数对齐 (alignment)，例如（样本数，）或（样本数，类别数）。

大多数框架都提供从NumPy数组或Python列表等常见数据结构到其原生张量格式的便捷转换。例如，在PyTorch中：

```python
import torch
import numpy as np

# 示例：将NumPy数组转换为PyTorch张量
numpy_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
feature_tensor = torch.from_numpy(numpy_array).float() # 确保神经网络使用浮点类型

print(feature_tensor)
# 输出：
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])
```

### 特征缩放：归一化 (normalization)与标准化

神经网络 (neural network)对输入特征的尺度很敏感。值较大的特征可能会主导学习过程，导致收敛速度变慢或阻碍网络有效学习。梯度下降 (gradient descent)法作为核心优化算法，在特征尺度相近时表现通常更佳。两种常见的缩放技术是归一化和标准化。

#### 归一化（最小-最大缩放）

归一化将特征重新缩放到一个固定范围，通常是[0, 1]或[-1, 1]。缩放到[0, 1]的公式是：


$$
X_{\text{缩放后}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$


其中$X_{\min}$和$X_{\max}$是特征在*训练*数据集中的最小值和最大值。

- **优点：** 保证数据落在特定范围内。适用于需要有界区间输入的算法，或像素值通常缩放到[0, 1]的图像数据。
- **缺点：** 对异常值敏感。单个极端值会大幅压缩其余数据到一个很小的范围。

#### 标准化（Z-分数归一化）

标准化将特征重新缩放，使其均值（$\mu$）为0，标准差（$\sigma$）为1。公式是：


$$
X_{\text{缩放后}} = \frac{X - \mu}{\sigma}
$$


此处，$\mu$和$\sigma$是从*训练*数据集中计算出的均值和标准差。

- **优点：** 相较于归一化，对异常值不那么敏感。常受梯度下降等算法青睐，这类算法假设特征以零为中心。
- **缺点：** 不保证有界范围。缩放后的值可以是正数或负数，并可能超出[-1, 1]。



![数据缩放比较](plots/5054-0.json)



> 原始数据点与经过最小-最大归一化和标准化后的位置对比。请注意异常值(15, 30)如何压缩归一化数据，而标准化则以不同方式处理。

#### 正确应用缩放器

初学者常忽略的一点是，当您拥有独立的训练、验证和测试数据集时，如何应用缩放。

1. **仅在训练数据上拟合缩放器：** 仅使用训练样本计算$X_{\min}$、$X_{\max}$、$\mu$和$\sigma$。这能防止验证集或测试集的信息“泄露”到训练过程中，从而导致评估出现偏差。
2. **使用已拟合的缩放器转换所有数据集（训练、验证、测试）：** 将*相同的*缩放转换（使用从训练数据中学到的参数 (parameter)）应用于训练、验证和测试集。

scikit-learn等库为此提供了便捷工具：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch

# 示例数据（请替换为您的实际数据）
X = np.array([[10.0, 0.1], [12.0, 0.2], [15.0, 0.15], [9.0, 0.3], [11.0, 0.05], [18.0, 0.25]])
y = np.array([0, 0, 1, 0, 1, 1])

# 1. 首先划分数据
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42) # 60% 训练
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 20% 验证, 20% 测试

# 2. 仅在训练数据上初始化并拟合缩放器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 拟合并转换

# 3. 使用相同的已拟合缩放器转换验证和测试数据
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 将缩放后的数据转换为PyTorch张量（示例）
X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(y_train).long() # 整数标签使用long类型

print("原始训练数据样本：\n", X_train[0])
print("缩放后的训练数据样本：\n", X_train_scaled[0])
print("\n缩放后的验证数据样本：\n", X_val_scaled[0])

# 示例输出：
# 原始训练数据样本：
#  [11.    0.05]
# 缩放后的训练数据样本：
#  [-0.3380617   -1.34164079]

# 缩放后的验证数据样本： # (注意：使用了从训练数据学到的缩放参数)
#  [ 0.16903085 -0.4472136 ]
```

### 数据划分：训练集、验证集和测试集

训练之前，您必须将数据集划分为不同的子集：

1. **训练集：** 数据中最大的部分，模型通过反向传播 (backpropagation)和优化算法调整其权重 (weight)和偏差，从而学习潜在模式。
2. **验证集（或开发集）：** 在训练期间定期使用，用于调整超参数 (parameter) (hyperparameter)（如学习率、层数、正则化 (regularization)强度）并决定模型架构。它提供了一个对模型在*开发过程中*未经训练数据上表现的公正估计。这有助于及早发现过拟合 (overfitting)（使用诸如早停等技术）。
3. **测试集：** 在开发过程的最后阶段才完全保留。一旦您根据验证集上的表现选择了最终模型架构和超参数，您将最后一次评估其在测试集上的表现。这能最真实地估计模型在实际中对新颖、未见数据的表现。

**为何严格区分？** 如果您根据测试集表现调整超参数，您实际上是将模型选择过程隐式地拟合到该特定测试数据。那么，您的测试集表现估计将过于乐观，模型可能无法很好地泛化到真正未见的数据。常见划分包括70%训练、15%验证、15%测试，或80%训练、10%验证、10%测试，但最佳比例可能取决于数据集总大小。对于非常大的数据集，验证集和测试集有时可以是更小的百分比。

> 将数据划分为训练集、验证集和测试集的典型流程。测试集在最终评估前保持原样。

妥善的数据准备，包括格式化、缩放和划分，不仅仅是初步步骤，更是构建成功深度学习 (deep learning)模型不可或缺的一部分。正确执行此项工作能确保您的网络高效学习，并获得对其表现的可靠估计。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对深度学习概念进行了学术论述，包括数据表示（张量）、预处理策略以及训练/验证/测试集对模型评估的重要性。
- [Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch Core Team (2024)
  官方文档，解释了PyTorch中的数据结构（张量），包括创建、类型和操作，这对于深度学习数据格式化很重要。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  官方文档，详细介绍了各种数据预处理技术，例如缩放（最小-最大归一化、标准化），以及使用scikit-learn在训练/验证/测试集上应用这些技术。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本受到认可的实践指南，提供了清晰的解释和代码示例，涵盖数据预处理，包括特征缩放和数据划分，与深度学习工作流程相关（第三版）。
