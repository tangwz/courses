---
course: "applied-data-science"
chapter: "practical-feature-engineering"
lesson: "categorical-encoding"
sourceId: 1857
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-2-practical-feature-engineering/categorical-encoding"
title: "高效编码类别变量"
description: "比较和实现类别数据的不同编码方法（独热编码、目标编码）。"
order: 2
plots: []
sourceHash: "c83a363d59136bcfffad76acf3323128714c4c9c9614d34b8e3eff26d009c357"
sourceCorrections: []
---

大多数机器学习 (machine learning)算法都处理数值数据。它们能理解数字、距离和梯度，但不能直接理解“红色”、“蓝色”、“管理员”或“用户”这样的类别。当您的数据集包含类别特征（表示不同群组或标签的变量）时，您需要将它们转换为算法可以高效处理的数值格式。这种转换过程是特征工程的一个基本组成部分。选择正确的编码策略非常重要，因为它能直接影响模型性能。

下面我们来看一些常用且有效的类别变量编码技术。

### 独热编码

独热编码（One-Hot Encoding）可能是处理*标称*类别特征（类别之间没有固有顺序）最常用的技术。它的工作原理是为原始特征的每个独特类别创建一个新的二元（0或1）列。对于给定的观测值，与其类别对应的列获得1，而所有其他新列获得0。

考虑一个具有“红色”、“绿色”和“蓝色”类别的特征`Color`。独热编码将其单列转换为三列：

| 原始颜色 | 颜色\_红 | 颜色\_绿 | 颜色\_蓝 |
| --- | --- | --- | --- |
| Red | 1 | 0 | 0 |
| Green | 0 | 1 | 0 |
| Blue | 0 | 0 | 1 |
| Red | 1 | 0 | 0 |

**实现：**

您可以使用Pandas库轻松实现这一点：

```python
import pandas as pd

# 示例DataFrame
data = {'ID': [1, 2, 3, 4], 'Color': ['Red', 'Green', 'Blue', 'Red']}
df = pd.DataFrame(data)

# 应用独热编码
df_encoded = pd.get_dummies(df, columns=['Color'], prefix='Color')

print(df_encoded)
#    ID  Color_Blue  Color_Green  Color_Red
# 0   1           0            0          1
# 1   2           0            1          0
# 2   3           1            0          0
# 3   4           0            0          1
```

或者，Scikit-learn的`OneHotEncoder`提供了更多控制，尤其是在ML管道中：

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 示例数据 (需要是二维)
colors = np.array(['Red', 'Green', 'Blue', 'Red']).reshape(-1, 1)

# 初始化并拟合编码器
encoder = OneHotEncoder(sparse_output=False) # sparse=False 给出密集数组
encoded_colors = encoder.fit_transform(colors)

print(encoded_colors)
# [[0. 0. 1.]  # -> 蓝 绿 红 (顺序取决于拟合)
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
print(encoder.categories_) # 显示映射
# [array(['Blue', 'Green', 'Red'], dtype='<U5')]
```

**优点：**

- 避免在类别之间引入人为的顺序。
- 与大多数算法配合良好，特别是线性模型。

**缺点：**

- 可能会大幅增加特征数量（维度），特别是如果原始变量具有许多独特类别（高基数）时。这可能导致稀疏性，并可能影响性能或内存使用（“维度灾难”）。
- 如果保留所有新列，可能会出现完美的共线性（例如，如果您知道`Color_Red`和`Color_Green`为0，则`Color_Blue`*必须*为1）。一些实现（如`pd.get_dummies(..., drop_first=True)`)或模型会自动处理此问题。

### 标签编码

标签编码为每个类别分配一个独特的整数。对于我们的`Color`示例，它可能看起来像这样：“红色” -> 0，“绿色” -> 1，“蓝色” -> 2。

| 原始颜色 | 编码颜色 |
| --- | --- |
| Red | 0 |
| Green | 1 |
| Blue | 2 |
| Red | 0 |

**实现：**

Scikit-learn的`LabelEncoder`常用：

```python
from sklearn.preprocessing import LabelEncoder

# 示例数据 (一维数组)
colors = ['Red', 'Green', 'Blue', 'Red']

# 初始化并拟合编码器
label_encoder = LabelEncoder()
encoded_colors = label_encoder.fit_transform(colors)

print(encoded_colors)
# [2 1 0 2]  # <- 顺序取决于字母顺序或首次出现顺序
print(label_encoder.classes_) # 显示映射
# ['Blue' 'Green' 'Red']
```

**优点：**

- 易于实现。
- 不增加特征数量。

**缺点：**

- 引入了*人为的序数关系*。模型可能会将“蓝色”（2）解释为“大于”“绿色”（1），这对于颜色等标称数据通常是无意义的。这可能误导依赖距离或幅度的算法（例如，线性回归、逻辑回归、支持向量 (vector)机、K-Means）。
- 基于树的模型（如决策树、随机森林、梯度提升）通常对此人为顺序的敏感度较低，因为它们根据阈值（$特征 \le 值$）进行分割，但它有时仍会影响分割选择。

**何时使用标签编码：**

- 对于类别具有有意义顺序的*真正的序数*数据（例如，“低”、“中”、“高”）。
- 有时即使对于标称数据，也会谨慎地与基于树的模型一起使用，但对于标称特征，通常更倾向于使用独热编码。

### 目标编码（均值编码）

目标编码（Target Encoding）是一种更高级的技术，特别适用于高基数类别特征。它不是创建虚拟变量或任意数字，而是用属于该类别的观测值的目标变量均值来编码每个类别。

例如，如果您正在预测客户流失（流失则目标为1，否则为0），并且有一个“城市”特征，那么“纽约”的目标编码将是训练数据中纽约客户的平均流失率。

**原理：**

1. 按类别特征对数据进行分组。
2. 计算每个组的目标变量均值。
3. 用这个计算出的均值替换类别标签。

**优点：**

- 可以直接在一个特征中捕获类别与目标变量之间关系的信息。
- 避免了独热编码在高基数特征上的高维度问题。

**缺点：**

- **过拟合 (overfitting)风险高：** 编码直接源自目标变量。如果实施不当，它会将目标信息泄露到特征中，导致训练数据上性能评估过于乐观，泛化能力不佳。
- 需要仔细的验证策略（例如，仅在训练折叠上计算编码、应用平滑、添加噪声）来缓解过拟合。
- 可能对稀有类别敏感，这些类别基于少量样本可能具有不可靠的目标均值。

目标编码功能强大，但需要仔细实现，通常涉及在交叉验证期间学到的技术或专门用于可靠处理它的库。

### 其他技术（简要说明）

- **二元编码：** 一种针对高基数特征的折中方案。类别首先被编码为整数（类似于标签编码），然后这些整数被转换为二进制代码。每个二进制数字都获得自己的列。这比独热编码创建的列少，但比标签编码多，同时不强加纯粹的线性顺序。
- **哈希编码：** 使用哈希函数将类别（即使是未见过的类别）映射到固定数量的输出列。可以处理非常高的基数和在线学习场景，但可能发生冲突（不同的类别映射到相同的哈希值）。

### 选择合适的策略

最佳编码方法取决于几个因素：

1. **类别数据类型：** 它是标称型（无序）还是序数型（固有顺序）？对于标称型使用独热编码，对于序数型考虑标签编码（或自定义映射）。
2. **基数：** 有多少独特的类别？独热编码对于非常高的基数会变得有问题。目标编码、二元编码或哈希编码可能是更好的替代方案。
3. **机器学习 (machine learning)算法：** 线性模型对标签编码引入的人为顺序很敏感。基于树的模型敏感度较低，但通常仍倾向于对标称数据使用独热编码，除非基数非常高。
4. **期望维度：** 您是否需要保持特征数量较低？标签编码或目标编码可以做到这一点，但有注意事项。

下面是一个简单的图示，说明了“颜色”特征的独热编码和标签编码之间的结构差异：

> 单个类别特征使用独热编码与标签编码的转换。独热编码创建多个二元列，而标签编码创建单个整数列。

高效编码类别变量是构建可靠机器学习模型的先决条件。进行实验并理解不同方法之间的权衡对于为您的特定数据集和建模任务找到最佳方法非常重要。编码后，这些特征以及前面讨论的数值特征，将构成算法的输入矩阵。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  Publisher: scikit-learn
  scikit-learn 中数据预处理的官方文档，包含 `OneHotEncoder` 和 `LabelEncoder` 的详细信息和示例。
- [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) — pandas development team (2023)
  `pandas.get_dummies` 函数的官方文档，提供独热编码的示例和参数说明。
- [Feature Engineering for Machine Learning: Principles and Techniques for Building Powerful Predictors](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本关于特征工程的综合书籍，涵盖多种分类编码方法、其理论基础和实际考虑。
- [A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems](https://doi.org/10.1145/507533.507538) — Daniele Micci-Barreca (2001)
  Journal: ACM SIGKDD Explorations Newsletter; Publisher: Association for Computing Machinery; Volume: 3; Pages: 27-32; DOI: [10.1145/507533.507538](https://doi.org/10.1145/507533.507538)
  一篇基础论文，描述了使用目标变量统计数据编码高基数分类特征的方法，并讨论了解决过拟合的方案。
