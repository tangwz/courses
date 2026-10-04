---
course: "getting-started-with-scikit-learn"
chapter: "data-preprocessing-feature-engineering"
lesson: "encoding-categorical-features"
sourceId: 1881
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/encoding-categorical-features"
title: "分类特征编码"
description: "将分类数据转换为数值格式的技术：独热编码和序数编码。"
order: 4
plots: []
sourceHash: "dc1c28a6948e79e832fef0d9c462cf9f4c4a90491199b2a270d20f8ae5471b76"
sourceCorrections: []
---

机器学习 (machine learning)算法基于数学原理运作，通常需要数值输入。原始数据集经常包含分类特征，表示'颜色'、'城市'、'产品类别'或'客户群'等定性属性。这些特征通常表现为文本字符串或特定标识符，无法直接输入到大多数Scikit-learn估计器中。因此，我们需要方法将这些分类描述转换为算法能够理解的数值格式。这个过程被称为**分类特征编码**。

目标是将非数值分类转换为数值，同时不丢失重要信息，更重要的是，避免在分类之间引入误导性的关系。Scikit-learn在其`preprocessing`模块中提供了多种转换器来处理此项工作。我们将侧重介绍两种广泛应用的方法：独热编码和序数编码。

### 独热编码

独热编码是一种常用技术，特别适用于**名义型分类特征**，即类别之间没有固有的顺序或等级（例如，'Red'、'Green'、'Blue'）。它通过为原始特征中的每个独特类别创建一个新的二进制（0或1）特征来运作。对于给定的观测值，与其类别对应的列将取值为1，而该原始特征的所有其他新列将取值为0。

考虑一个名为'Color'的特征，它有三种可能的类别：'Red'、'Green'和'Blue'。独热编码会将这一个特征转换为三个新特征：'Color\_Red'、'Color\_Green'和'Color\_Blue'。

- 'Color'为'Red'的观测值将表示为 [1, 0, 0]。
- 'Color'为'Green'的观测值将表示为 [0, 1, 0]。
- 'Color'为'Blue'的观测值将表示为 [0, 0, 1]。

> 使用独热编码对'Color'特征进行转换。每个类别都成为一个新的二进制列。

**优点：**

- 避免在类别之间引入人为的序数关系。
- 适用于线性模型、基于距离的算法（如KNN）以及许多其他算法。

**缺点：**

- 如果原始分类特征包含许多独特类别，可能会显著增加特征数量（维度）。这有时被称为“维度灾难”，可能导致计算更密集或过拟合 (overfitting)。
- 结果特征是稀疏的（大部分为零）。

在Scikit-learn中，`sklearn.preprocessing.OneHotEncoder`是进行此转换的主要工具。它旨在在Scikit-learn管道中运行，并提供参数 (parameter)来处理训练期间未见的类别（`handle_unknown='ignore'`），或控制输出格式（`sparse_output=True`为默认值，这对于高维稀疏数据来说是内存高效的）。

### 序数编码

序数编码为每个类别分配一个唯一的整数。例如，如果特征'Size'有['Small', 'Medium', 'Large']这些类别，序数编码可能会将它们映射为[0, 1, 2]。

**优点：**

- 简单且不增加特征数量。

**缺点：**

- 主要缺点是它隐式地强加了一种序数关系（0 < 1 < 2）。这种假设对于名义数据（'Red' < 'Green' < 'Blue'在数值上没有意义）可能不正确，并可能误导将这些数值解释为具有大小或顺序的算法。
- 它通常只适用于**序数型分类特征**，即类别具有有意义的、固有的顺序（如'低'、'中'、'高'或'学士学位'、'硕士学位'、'博士学位'）。

Scikit-learn提供了`sklearn.preprocessing.OrdinalEncoder`。你可以使用`categories`参数 (parameter)明确定义类别的顺序；否则，它会根据拟合期间遇到的顺序（通常是字母顺序）来确定映射。

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# 示例数据
data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Red'],
    'Size': ['Medium', 'Large', 'Small', 'Medium', 'Large']
})

# --- 独热编码示例 ---
# 选择用于独热编码的分类列
ohe_cols = ['Color']
# 初始化编码器
# sparse_output=False 会返回一个密集型的 numpy 数组，此处便于查看
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# 拟合并转换
ohe_transformed = ohe.fit_transform(data[ohe_cols])
# 获取新列的特征名称
ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
# 创建包含新特征的 DataFrame
ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_feature_names, index=data.index)

print("原始数据:\n", data)
print("\n独热编码后的'Color':\n", ohe_df)

# --- 序数编码示例 ---
# 选择用于序数编码的分类列
ord_cols = ['Size']
# 定义'Size'所需的顺序
size_categories = ['Small', 'Medium', 'Large']
# 使用指定类别初始化编码器
ordinal_encoder = OrdinalEncoder(categories=[size_categories]) # 注意：categories 需要一个列表的列表
# 拟合并转换
ord_transformed = ordinal_encoder.fit_transform(data[ord_cols])
# 创建 DataFrame（可选，通常直接用作 numpy 数组）
ord_df = pd.DataFrame(ord_transformed, columns=['Size_Encoded'], index=data.index)

print("\n序数编码后的'Size':\n", ord_df)

# 合并结果（排除原始分类列）
final_df = pd.concat([data[['ID']], ohe_df, ord_df], axis=1)
print("\n合并后的DataFrame:\n", final_df)
```

### 选择合适的编码器

- 对名义型特征（无固有顺序）使用**独热编码**。注意可能增加的维度，特别是对于高基数特征（包含许多独特类别的特征）。可能需要特征选择或使用能处理高维数据的模型等方法。
- 对序数型特征使用**序数编码**，其中数值顺序应反映类别间的关系。确保分配的整数与内在顺序相符。
- 一些基于树的模型（如决策树、随机森林）有时可以直接处理分类特征，或者对序数编码和独热编码之间的区别不那么敏感，但使用Scikit-learn编码器可以确保不同模型类型之间以及管道内部的兼容性。

尽管Pandas提供了方便的`pd.get_dummies`函数来创建虚拟/指示变量（类似于独热编码），但在机器学习 (machine learning)工作流程中，通常更推荐在`Pipeline`（第6章将讲解）中使用Scikit-learn的`OneHotEncoder`。这可以确保训练期间识别出的相同类别在测试或预测时保持一致使用，从而避免由于未见类别或不同的列集引起的错误。

分类数据编码是数据预处理的一个基本步骤。方法的选择取决于数据本身的特性以及你计划使用的机器学习算法的要求。在此处的仔细考虑对构建有效的模型有重要贡献。

## 参考资料

- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) — scikit-learn developers (2024)
  Scikit-learn `OneHotEncoder` 的官方文档，详细介绍了其参数和用于名义型类别特征编码的用法。
- [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) — scikit-learn developers (2024)
  Scikit-learn `OrdinalEncoder` 的官方文档，概述了其参数和用于序数型类别特征编码的用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098122461/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实践指南，涵盖数据预处理，包括使用 Scikit-learn 实现的各种类别编码技术的讨论和示例。
