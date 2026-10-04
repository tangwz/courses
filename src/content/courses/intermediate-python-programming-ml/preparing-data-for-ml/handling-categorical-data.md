---
course: "intermediate-python-programming-ml"
chapter: "preparing-data-for-ml"
lesson: "handling-categorical-data"
sourceId: 2199
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-5-preparing-data-for-ml/handling-categorical-data"
title: "处理分类数据"
description: "应用独热编码和标签编码等技术，将分类特征转换为数值格式。"
order: 3
plots: []
sourceHash: "c866203d5f62f35482debb1b5502a3885559cbb2d89e27bf6b35f3e4e007adff"
sourceCorrections: []
---

机器学习 (machine learning)算法，特别是那些基于数学方程的，例如线性回归、逻辑回归、支持向量 (vector)机，或基于距离计算的，例如k近邻，都作用于数值数据。它们无法直接理解原始数据集中存在的文本标签或类别。因此，将分类特征转换为合适的数值格式是数据准备流程中一项标准且必需的步骤。

分类特征表示属于不同组或类别的数据点。这些可以是：

1. **名义型：** 没有内在顺序或排名的类别（例如，“颜色”：红色、蓝色、绿色；“城市”：伦敦、巴黎、东京）。
2. **序数型：** 具有有意义的顺序或排名，但类别之间的量级不一定是固定的类别（例如，“大小”：小、中、大；“教育水平”：高中、学士、硕士）。

我们的目标是将这些基于文本的类别转换为机器学习模型能够理解的数字，同时不引入误导信息。让我们来看一下常见策略。

### 标签编码 / 序数编码

最直接的方法是为每个类别分配一个唯一的整数。这通常被称为标签编码。

例如，如果我们有一个序数特征“大小”：

| 原始值 | 编码值 |
| --- | --- |
| small | 0 |
| medium | 1 |
| large | 2 |

这对**序数型**数据效果良好，因为数值顺序（0 < 1 < 2）反映了类别固有的顺序（小 < 中 < 大）。

然而，将标签编码直接应用于**名义型**数据可能会有问题。考虑一个“颜色”特征：

| 原始值 | 编码值 (任意) |
| --- | --- |
| red | 0 |
| blue | 1 |
| green | 2 |

在这里，编码暗示了一种顺序（红色 < 蓝色 < 绿色）和实际不存在的关系（例如，蓝色在某种程度上“介于”红色和绿色之间）。这种人为的排序会混淆那些根据量级或距离解释数值的算法。

**实现：**

你可以使用 Pandas 执行简单的标签编码：

```python
import pandas as pd

data = {'size': ['medium', 'large', 'small', 'medium']}
df = pd.DataFrame(data)

# 为序数数据定义顺序
size_mapping = {'small': 0, 'medium': 1, 'large': 2}
df['size_encoded'] = df['size'].map(size_mapping)

print(df)
#      size  size_encoded
# 0  medium             1
# 1   large             2
# 2   small             0
# 3  medium             1

# 对于名义数据（如果需要，但不常推荐）
# df['color_encoded'] = df['color'].astype('category').cat.codes
```

为了将其集成到Scikit-learn管道中，尤其是在处理多个序数型列或需要训练集和测试集之间保持一致性时，`OrdinalEncoder` 更受推荐。

```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# 假设df[['size', 'education']]包含序数特征
# 为每个特征定义正确的类别顺序
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large'],
                                   ['high school', 'bachelor', 'master']])

# 在训练数据上拟合并转换
# train_encoded = encoder.fit_transform(df_train[['size', 'education']])

# 使用*相同*的已拟合编码器转换测试数据
# test_encoded = encoder.transform(df_test[['size', 'education']])
```

请注意，Scikit-learn的`LabelEncoder`通常用于编码*目标变量* (y)，而非输入特征 (X)。对于特征，请使用`OrdinalEncoder`。

### 独热编码

为了避免在名义特征中引入人为顺序，**独热编码**是标准技术。它将每个类别值转换为一个新的二进制列（0或1）。

考虑带有“红色”、“蓝色”、“绿色”类别的“颜色”特征。独热编码将创建三个新列：

| 原始颜色 | color\_red | color\_blue | color\_green |
| --- | --- | --- | --- |
| red | 1 | 0 | 0 |
| blue | 0 | 1 | 0 |
| green | 0 | 0 | 1 |
| blue | 0 | 1 | 0 |

现在，每一行在对应其原始类别的列中有一个“1”，其他列为“0”。这在数值上表示了类别归属，而不暗示任何顺序。

> 使用独热编码对名义型“颜色”特征进行转换。

**潜在问题：维度**
如果分类特征有许多唯一值（高基数），独热编码会显著增加数据集的列数。这有时会导致性能问题或某些算法的“维度灾难”。

**潜在问题：多重共线性**
生成的列是完全多重共线的（例如，如果你知道`color_red`和`color_blue`，那么`color_green`可以被完美预测，因为`color_green = 1 - color_red - color_blue`）。对于某些模型（如未正则化 (regularization)的线性回归），这会造成问题。一种常见做法是删除其中一个独热编码列（在Pandas/Scikit-learn中使用`drop='first'`或`drop='if_binary'`）。然而，许多现代算法（特别是基于树的模型或正则化回归）会在内部处理这种多重共线性，因此删除列并非总是必需的，甚至可能会对某些模型略微减少信息。

**实现：**

Pandas 提供了一个方便的函数 `get_dummies`：

```python
import pandas as pd

data = {'color': ['red', 'blue', 'green', 'blue'],
        'value': [10, 15, 12, 15]}
df = pd.DataFrame(data)

# 创建独热编码列，以原始列名作为前缀
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color', drop_first=False)

print(df_encoded)
#    value  color_blue  color_green  color_red
# 0     10           0            0          1
# 1     15           1            0          0
# 2     12           0            1          0
# 3     15           1            0          0
```

在Scikit-learn管道中使用时，`OneHotEncoder`是标准工具。它在`fit`步骤中学习类别，并一致地应用转换。

```python
from sklearn.preprocessing import OneHotEncoder

# 示例数据
data = [['red', 10], ['blue', 15], ['green', 12], ['blue', 15]]
df = pd.DataFrame(data, columns=['color', 'value'])

# 选择分类列
categorical_features = ['color']
# 使用 handle_unknown='ignore' 以避免测试数据中出现未见类别时出错
# 使用 sparse_output=False 以获取密集型 numpy 数组（通常更易于使用）
# 如果模型需要处理多重共线性，请使用 drop='first'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 在训练数据上拟合并转换
# 注意：OneHotEncoder 期望一个二维数组，因此是 df[categorical_features]
# train_encoded_array = encoder.fit_transform(df_train[categorical_features])

# 获取编码后的列名（有助于创建 DataFrame）
# feature_names = encoder.get_feature_names_out(categorical_features)
# train_encoded_df = pd.DataFrame(train_encoded_array, columns=feature_names, index=df_train.index)

# 转换测试数据
# test_encoded_array = encoder.transform(df_test[categorical_features])
# test_encoded_df = pd.DataFrame(test_encoded_array, columns=feature_names, index=df_test.index)

# 示例：拟合和转换样本df
encoded_array = encoder.fit_transform(df[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)
df_encoded_sklearn = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)

# 与数值特征组合
df_final = pd.concat([df.drop(columns=categorical_features), df_encoded_sklearn], axis=1)
print(df_final)
#    value  color_blue  color_green  color_red
# 0     10         0.0          0.0        1.0
# 1     15         1.0          0.0        0.0
# 2     12         0.0          1.0        0.0
# 3     15         1.0          0.0        0.0
```

### 其他编码技术

尽管标签/序数编码和独热编码是最常见的，但针对特定情况也存在其他技术：

- **二值编码：** 一种针对高基数特征的折衷方案。类别首先被整数编码，然后这些整数被转换为二进制，二进制位形成新的列。与独热编码相比，它减少了维度，但可解释性较差。
- **频率编码：** 将类别替换为其在数据集中的计数或频率。简单，但会丢失类别本身的信息。
- **目标编码（均值编码）：** 将类别替换为该类别目标变量的平均值。它具有很强的预测性，但如果实现不小心，容易出现过拟合 (overfitting)和数据泄露（通常涉及在与预测折叠不同的折叠上计算均值）。

### 选择合适的策略

- **序数型数据：** 使用**序数编码**（或经过仔细映射的标签编码）来保留其内在顺序。
- **名义型数据（低基数）：** 使用**独热编码**。这是标准做法，避免了人为顺序，并且在类别数量可管理时（例如，少于15-20个，尽管这只是一个指导原则）效果良好。
- **名义型数据（高基数）：** 如果独热编码创建了太多特征，请考虑替代方案。二值编码、频率编码或更高级的技术（如目标编码）可能是选项，具体取决于特定问题和模型。有时，特征工程（例如，将稀有类别分组到“其他”类别中）在编码前也有效。

始终将从**训练数据**中学习到的编码一致地应用于**测试数据**。使用Scikit-learn转换器，如`OneHotEncoder`和`OrdinalEncoder`，特别是在`Pipeline`或`ColumnTransformer`中，有助于确保这种一致性，防止数据泄露和错误。

```python
# 使用 ColumnTransformer 应用不同预处理的示例
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # 示例模型
import pandas as pd

# 样本数据
data = {
    'size': ['medium', 'large', 'small', 'medium', 'large'],
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'amount': [100, 150, 80, 120, 200],
    'target': [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
X = df[['size', 'color', 'amount']]
y = df['target']

# 将数据拆分为训练/测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义列类型
ordinal_features = ['size']
nominal_features = ['color']
numerical_features = ['amount']

# 为序数特征定义顺序
ordinal_categories = [['small', 'medium', 'large']]

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ('nom', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_features) # drop='first' 可选
    ],
    remainder='passthrough' # 保留其他列（如果有的话，本例中没有）
)

# 创建包含模型的完整管道
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression())])

# 现在你可以在训练数据上拟合管道
# model_pipeline.fit(X_train, y_train)
# 并在测试数据上进行预测/评估
# predictions = model_pipeline.predict(X_test)
# score = model_pipeline.score(X_test, y_test)

# 拟合并转换样本数据以查看预处理结果
X_processed = preprocessor.fit_transform(X)
print("处理后的形状:", X_processed.shape)
# 注意：使用 ColumnTransformer 时，get_feature_names_out 可能会很复杂，
# 但列对应于按定义顺序的缩放数值、编码序数、
# 和独热编码的名义特征。
print("处理后的数据（摘录）:\n", X_processed[:3])
# Shape after processing: (5, 4)
# Processed Data (excerpt):
# [[-0.56266888  1.          0.          1.        ] <--- [缩放后的 amount, 编码后的 size, color_green, color_red] (blue 被删除)
#  [ 0.56266888  2.          0.          0.        ]
#  [-1.12533775  0.          1.          0.        ]]
```

正确处理分类数据是为机器学习 (machine learning)准备数据集的基本步骤。通过根据数据性质（序数型与名义型）和类别数量选择合适的编码策略，可以为算法提供有意义的数值输入，从而提高它们有效学习模式的能力。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  Scikit-learn中数据转换的基本指南，涵盖了OrdinalEncoder、OneHotEncoder以及在管道中使用ColumnTransformer进行分类特征处理。
- [Working with categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) — pandas development team (2023)
  详细解释了如何在Pandas中处理分类数据类型，包括使用get_dummies进行映射和独热编码的方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  第三版实践指南，涵盖数据预处理，包括使用Python的Scikit-learn和Pandas库进行分类特征编码技术。
- [Categorical Feature Encoding for Tabular Data: A Survey](https://ieeexplore.ieee.org/document/9906646) — Meng Zhang, Yanan Song, Wenjun Mao, Guohua Cao, and Shijie Hao (2023)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 35; Pages: 9940-9961; DOI: [10.1109/TKDE.2022.3204918](https://doi.org/10.1109/TKDE.2022.3204918)
  对各种分类特征编码方法进行了全面回顾，讨论了它们的优缺点以及在不同机器学习场景中的适用性。
