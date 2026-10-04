---
course: "introduction-to-neural-networks"
chapter: "data-preparation-neural-networks"
lesson: "data-preprocessing-practical"
sourceId: 1942
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks/data-preprocessing-practical"
title: "实践操作：预处理样本数据"
description: "使用Python库对样本数据集应用缩放和编码技术。"
order: 6
plots: ["plots/1942-0.json"]
sourceHash: "4982fb17e996d19ab1e3c84ae94c0a5420f36f69a2e67f0139dfef54099bb053"
sourceCorrections: []
---

数据准备涉及数值特征的缩放和类别特征的编码。一个典型的数据预处理流程将得到介绍，使用Python，重点讲解常用的`pandas`库（用于数据处理）和`scikit-learn`库（用于预处理工具）。

假设我们有一个小型数据集，旨在预测学生是否通过考试，依据他们的学习时长、之前相关测验的得分以及主要学习方式。

### 设置样本数据

首先，我们使用`pandas`创建样本数据集。这模拟了你在实际项目中如何从CSV文件或数据库加载数据。

```python
import pandas as pd
import numpy as np

# 创建样本数据
data = {
    'Study_Hours': [2.5, 5.1, 1.3, 8.7, 4.5, 9.8, 3.3, 6.1, 7.0, 0.5],
    'Previous_Score': [65, 82, 50, 95, 75, 88, 68, 85, 91, 45],
    'Study_Method': ['Solo', 'Group', 'Solo', 'Group', 'Group', 'Group', 'Solo', 'Solo', 'Group', 'Solo'],
    'Passed_Exam': [0, 1, 0, 1, 1, 1, 0, 1, 1, 0] # 目标变量 (0=未通过, 1=通过)
}

df = pd.DataFrame(data)

print("原始DataFrame:")
print(df)
```

这给我们提供了一个DataFrame `df`，其中包含我们的特征（`Study_Hours`、`Previous_Score`、`Study_Method`）以及我们希望预测的目标变量（`Passed_Exam`）。

### 识别特征类型

在进行任何转换之前，我们将特征与目标变量分离，并识别哪些特征是数值型，哪些是类别型。

```python
# 分离特征 (X) 和目标 (y)
X = df.drop('Passed_Exam', axis=1)
y = df['Passed_Exam']

# 识别数值特征和类别特征
numerical_features = ['Study_Hours', 'Previous_Score']
categorical_features = ['Study_Method']

print("\n特征 (X):")
print(X.head())
print("\n目标 (y):")
print(y.head())
```

### 缩放数值特征

正如在*特征缩放*部分讨论的，数值特征通常受益于缩放，特别是对于像神经网络 (neural network)这样对特征量级敏感的算法。我们使用`scikit-learn`的`StandardScaler`来应用标准化（Z-score缩放）。这会转换数据，使其均值为0，标准差为1（$x' = \frac{x - \mu}{\sigma}$）。

```python
from sklearn.preprocessing import StandardScaler

# 初始化缩放器
scaler = StandardScaler()

# 将缩放器拟合到数值数据并进行转换
X_numerical_scaled = scaler.fit_transform(X[numerical_features])

# 转换回DataFrame以提高可读性（可选）
X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X.index)

print("\n缩放前的数值特征:")
print(X[numerical_features].describe())

print("\n标准化后的数值特征:")
print(X_numerical_scaled_df.describe())

# 注意：缩放后均值约等于0，标准差约等于1
```

你会注意到，在缩放后，`Study_Hours`和`Previous_Score`的`mean`非常接近零，而`std`（标准差）接近一。

我们使用直方图可视化`Previous_Score`缩放前后的分布。



![标准化前后Previous Score的分布](plots/1942-0.json)



> 'Previous\_Score'特征在应用StandardScaler前（蓝色）和后（橙色）的分布比较。缩放后的分布以零为中心。

或者，我们可以使用`MinMaxScaler`将数据缩放到特定范围，通常是[0, 1]，公式为$x' = \frac{x - min(x)}{max(x) - min(x)}$。选择哪种方法通常取决于具体数据和网络架构，但`StandardScaler`是一个常用的默认选项。

### 编码类别特征

我们的`Study_Method`列包含文本类别（'Solo'、'Group'）。神经网络 (neural network)需要数值输入，所以我们需要对这个特征进行编码。独热编码是名义类别变量（没有内在顺序的变量）的一种标准技术。它为每个类别创建新的二元（0或1）列。

我们将使用`scikit-learn`的`OneHotEncoder`。

```python
from sklearn.preprocessing import OneHotEncoder

# 初始化编码器
# handle_unknown='ignore' 防止在后续（例如，测试数据中）出现未见类别时出错
# sparse_output=False 返回一个密集Numpy数组而不是稀疏矩阵
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 将编码器拟合到类别数据并进行转换
X_categorical_encoded = encoder.fit_transform(X[categorical_features])

# 获取编码器生成的新特征名称
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# 转换为DataFrame（可选，用于提高可读性）
X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names, index=X.index)

print("\n编码前的类别特征:")
print(X[categorical_features].head())

print("\n独热编码后的类别特征:")
print(X_categorical_encoded_df.head())
```

正如你所见，单个`Study_Method`列被两个新列取代：`Study_Method_Group`和`Study_Method_Solo`。列中的“1”表示给定样本存在该类别。

### 组合已处理特征

现在，我们将缩放后的数值特征和编码后的类别特征组合成一个单独的数组或DataFrame。这个组合数据集就是我们通常会馈送给神经网络 (neural network)的数据。

```python
# 连接缩放后的数值特征和编码后的类别特征
X_processed = pd.concat([X_numerical_scaled_df, X_categorical_encoded_df], axis=1)

print("\n最终处理后的特征 (X_processed):")
print(X_processed.head())

# 如果特定框架需要，转换为NumPy数组
X_processed_np = X_processed.to_numpy()
print("\n最终处理后的NumPy数组特征（前5行）:")
print(X_processed_np[:5])
print("\n最终处理后的特征形状:", X_processed_np.shape)
```

我们最终处理的特征集`X_processed`现在有4列（2个缩放后的数值特征，2个独热编码的类别特征），并且完全是数值型，可用于训练。

### 拆分数据用于训练和评估

尽管本节侧重于预处理转换，但下一步通常是将数据拆分为训练集、验证集和测试集，正如之前讨论的。这确保我们可以训练模型、调整其超参数 (parameter) (hyperparameter)，并在未见数据上评估其最终性能。`scikit-learn`为此提供了`train_test_split`函数。

```python
from sklearn.model_selection import train_test_split

# 拆分*已处理*的特征 (X_processed) 和原始目标 (y)
# 通常拆分为训练集和一个临时集，然后将临时集拆分为验证集和测试集
# 为简化起见，这里我们只进行一次训练集和测试集的拆分。
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_np, # 使用最终的NumPy数组
    y.to_numpy(),   # 将目标Series转换为NumPy数组
    test_size=0.2,  # 预留20%作为测试集
    random_state=42 # 确保拆分的可复现性
)

print("\n训练特征的形状:", X_train.shape)
print("测试特征的形状:", X_test.shape)
print("训练目标的形状:", y_train.shape)
print("测试目标的形状:", y_test.shape)
```

### 小结

在本次实践操作中，我们处理了一个原始样本数据集，并应用了重要的预处理步骤：

1. 识别了数值和类别特征。
2. 对数值特征应用了`StandardScaler`，使其以零为中心，单位方差。
3. 对类别特征应用了`OneHotEncoder`，将其转换为数值格式。
4. 将转换后的特征组合成一个单一数据集。
5. 对处理后的数据进行了训练集和测试集拆分。

这些经过处理和拆分的数据（`X_train`、`y_train`、`X_test`、`y_test`）现在已是适合用于训练和评估神经网络 (neural network)的格式，这将在后续章节中讲解。掌握这些预处理步骤对于构建有效的机器学习 (machine learning)模型非常重要。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  这本实践指南涵盖机器学习概念和实现，包含使用scikit-learn进行数据预处理的许多内容。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  scikit-learn官方数据预处理模块文档，详细介绍多种缩放和编码技术。
- [Model selection and evaluation](https://scikit-learn.org/stable/modules/model_selection.html) — scikit-learn developers (2024)
  Publisher: scikit-learn developers
  scikit-learn官方模型选择模块文档，包含`train_test_split`等数据分割工具。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论的基础教科书，包含数据预处理对神经网络的重要性及方法。
