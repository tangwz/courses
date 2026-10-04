---
course: "intro-feature-engineering"
chapter: "encoding-categorical-features"
lesson: "practice-encoding-techniques"
sourceId: 1336
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-3-encoding-categorical-features/practice-encoding-techniques"
title: "动手实践：应用编码技术"
description: "使用 Pandas 和 Scikit-learn 在数据集上实现并比较各种编码方法。"
order: 10
plots: ["plots/1336-0.json"]
sourceHash: "186838ffb10569fd8d1b134e1ba91288df31ec7dbe89847e39ef2160523568b0"
sourceCorrections: []
---

Python 用于应用将分类数据转换为数值表示的技术。Pandas 执行数据处理，而 Scikit-learn 结合 `category_encoders` 库则应用各种编码策略。

首先，我们来配置环境，导入必要的库并创建一个示例数据集。这个数据集包含我们常见的各种分类特征。

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 示例数据框
data = {
    'color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green', 'Yellow'],
    'size': ['M', 'L', 'S', 'M', 'L', 'S', 'M'],
    'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East'],
    'city': ['NYC', 'LA', 'Boston', 'SF', 'Chicago', 'Miami', 'Austin'],
    'temperature': [25, 30, 22, 28, 20, 35, 26], # 示例数值特征
    'outcome': [1, 0, 1, 0, 1, 0, 1] # 示例目标变量
}
df = pd.DataFrame(data)

print("原始数据框：")
print(df)
```

### 独热编码

独热编码适用于名义分类特征，其中不存在序数关系。它为每个独特类别创建一个新的二元列。我们可以使用 Pandas 的 `get_dummies` 或 Scikit-learn 的 `OneHotEncoder`。尽管 `get_dummies` 便于快速查看，但 `OneHotEncoder` 更能与 Scikit-learn 流水线结合，特别是在测试期间处理未见过类别时。

**使用 Pandas 的 `get_dummies`：**

```python
# 将 get_dummies 应用于 'color' 列
df_one_hot_pandas = pd.get_dummies(df, columns=['color'], prefix='color', drop_first=False)

print("\n独热编码后数据框（Pandas）：")
print(df_one_hot_pandas)
```

注意 'color' 列如何被多个 `color_*` 列替换。设置 `drop_first=True` 可以删除一个类别以避免多重共线性，这有时对模型有用。

**使用 Scikit-learn 的 `OneHotEncoder`：**

在机器学习 (machine learning)工作流程中，通常更推荐使用 `OneHotEncoder`。它需要根据训练数据进行“拟合”，然后用于“转换”训练和测试数据。

```python
# 初始化 OneHotEncoder
# handle_unknown='ignore' 可防止测试数据中出现未见过类别时出错
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 拟合并转换 'color' 列
# 注意：Scikit-learn 编码器需要二维数组，因此使用 df[['color']]
color_encoded = ohe.fit_transform(df[['color']])

# 创建一个包含新特征名称的数据框
color_encoded_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['color']))

# 与原始数据框拼接（删除原始 'color' 列）
df_one_hot_sklearn = pd.concat([df.drop('color', axis=1), color_encoded_df], axis=1)

print("\n独热编码后数据框（Scikit-learn）：")
print(df_one_hot_sklearn)
```

结果相似，但使用 Scikit-learn 转换器可以确保数据分割之间的一致应用。独热编码的主要缺点是如果原始类别具有许多独特值（高基数），则可能生成非常多的特征。

### 序数编码

序数编码用于类别具有有意义顺序的情况。我们需要明确定义这个顺序。

```python
# 定义 'size' 列的显式顺序
size_mapping = {'S': 0, 'M': 1, 'L': 2}
ordered_categories = [['S', 'M', 'L']] # OrdinalEncoder 需要的列表的列表

# 使用定义的顺序初始化 OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=ordered_categories)

# 将序数编码应用于 'size' 列
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size']])

print("\n序数编码后数据框：")
# 显示相关列
print(df[['size', 'size_encoded']])

# 如果继续，则删除原始 'size' 列
# df = df.drop('size', axis=1)
```

这里，'S'、'M' 和 'L' 分别映射到 0、1 和 2，保留了固有的顺序。将此应用于名义数据（如 'color'）将错误地暗示类别之间存在虚假的顺序和距离关系。

### 目标编码

目标编码将每个类别替换为该类别目标变量的平均值。它对于高基数特征特别有用，但存在过拟合 (overfitting)的风险，特别是如果某些类别不常见。正则化 (regularization)或平滑技术很重要。我们将使用 `category_encoders` 库中的 `TargetEncoder`。

```python
# 初始化 TargetEncoder
# 平滑有助于防止过拟合，特别是对于不常见类别
target_encoder = TargetEncoder(cols=['region'], smoothing=1.0)

# 使用 'outcome' 目标拟合并转换 'region' 列
df['region_target_encoded'] = target_encoder.fit_transform(df['region'], df['outcome'])

print("\n目标编码后数据框：")
# 显示相关列
print(df[['region', 'outcome', 'region_target_encoded']])

# 如果继续，则删除原始 'region' 列
# df = df.drop('region', axis=1)
```

观察每个区域现在如何用数值表示，该数值来源于该区域 'outcome' 的平均值。例如，'North' 出现两次，结果为 1，因此其编码值比 'South' 更接近 1，后者出现两次，结果为 0。平滑将这些值稍微调 (fine-tuning)整到全局平均值，减轻样本量少类别的影响。请注意，此编码器使用来自目标变量的信息，因此需要仔细验证（例如，仅从训练集学习编码的应用）以避免目标泄漏。

### 二进制编码

二进制编码是独热编码和目标编码之间针对高基数特征的一种折衷方法。它首先为每个类别分配一个序数整数，然后将这些整数转换为二进制代码。二进制代码中的每个位置都成为一个单独的特征列。

```python
# 初始化 BinaryEncoder
binary_encoder = BinaryEncoder(cols=['city'], return_df=True)

# 拟合并转换 'city' 列
df_binary_encoded = binary_encoder.fit_transform(df)

print("\n二进制编码后数据框（'city'）：")
print(df_binary_encoded)
```

注意 'city' 列（具有 7 个独特值）被 `city_0`、`city_1` 和 `city_2` 替换。因为 $2^2 < 7 \le 2^3$，我们需要 3 个二进制特征。这明显少于独热编码会创建的 7 个特征。二进制编码在不产生独热编码高维度的情况下捕捉了一些独特性，但生成的特征缺乏直接的可解释性。

### 哈希编码器

哈希编码器使用哈希函数将类别（表示为字符串）映射到固定数量的输出特征。它内存效率高，并能自然处理新类别，使其适用于非常大或流式数据集。主要缺点是由于哈希冲突（不同类别映射到相同的哈希值）可能导致信息丢失。

```python
# 初始化 HashingEncoder
# n_components 指定所需的输出特征数量（哈希空间大小）
hashing_encoder = HashingEncoder(cols=['city'], n_components=4)

# 拟合并转换 'city' 列
df_hashed = hashing_encoder.fit_transform(df)

print("\n哈希编码后数据框（'city'，4 个分量）：")
print(df_hashed)
```

'city' 列被 4 个新的哈希特征（`col_0` 到 `col_3`）替换。分量数量（`n_components`）是一个超参数 (parameter) (hyperparameter)；数量越大，碰撞概率越低，但维度会增加。哈希处理计算效率高，但由于冲突而牺牲了可解释性。

### 比较和选择合适的方法

编码方法的选择很大程度上取决于特定数据集、分类特征的基数以及正在使用的机器学习 (machine learning)模型。

| 方法 | 优点 | 缺点 | 最适合 |
| --- | --- | --- | --- |
| 独热 | 保留所有信息；无隐含顺序；适用于线性模型 | 高基数时维度高；默认对新类别处理不佳 | 低基数名义特征 |
| 序数 | 简单；保留顺序 | 假设存在有意义的顺序；如果顺序任意，可能误导模型 | 序数特征 |
| 目标 | 捕捉目标关系；比独热编码更好地处理高基数 | 易过拟合 (overfitting)；存在目标泄漏风险；需要仔细验证 | 高基数特征；树模型 |
| 二进制 | 高基数时维度低于独热编码 | 与独热编码相比信息丢失；可解释性较低 | 中高基数名义特征 |
| 哈希 | 内存效率高；处理新类别；适用于在线学习 | 通过冲突丢失信息；可解释性较低；对 `n_components` 敏感 | 非常高基数；流数据 |

以下是不同方法为 'city' 列（7 个独特值）生成的特征数量的视觉比较：



![为 'city'（7 个独特值）生成的特征数量](plots/1336-0.json)



> 对应用于 'city' 列的不同编码技术的特征维度比较。

### 将编码器集成到流水线中

在实际操作中，你通常会对不同的列应用不同的编码器。Scikit-learn 的 `ColumnTransformer` 在这方面非常适用，让你能够构建预处理流水线，从而正确且一致地处理各种数据类型。

```python
# 重新创建原始数据框用于演示
df = pd.DataFrame(data)

# 定义序数编码的映射
size_categories = [['S', 'M', 'L']]

# 为不同列类型定义预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['temperature']), # 保持数值特征不变（或进行缩放）
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['color']),
        ('ordinal', OrdinalEncoder(categories=size_categories), ['size']),
        ('binary', BinaryEncoder(), ['region']), # 以 'region' 列使用二进制编码为例
        ('target', TargetEncoder(smoothing=1.0), ['city']) # 以 'city' 列使用目标编码为例
    ],
    remainder='drop' # 删除未指定的列（例如如果 'outcome' 只是目标）
)

# 创建一个流水线（可选，但推荐做法）
# 通常在预处理后添加模型步骤
pipeline = Pipeline(steps=[('preprocess', preprocessor)])

# 拟合并转换数据（不包括目标 'outcome'）
# 注意：TargetEncoder 在拟合时需要 y
X = df.drop('outcome', axis=1)
y = df['outcome']

encoded_data = pipeline.fit_transform(X, y) # 为 TargetEncoder 传递 y

# 转换后获取特征名称（需要 Scikit-learn 1.1+）
# 对于旧版本或复杂转换器，可能需要手动构造名称
feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()

# 创建一个包含编码数据和正确列名的数据框
df_encoded_pipeline = pd.DataFrame(encoded_data, columns=feature_names)

print("\n应用 ColumnTransformer 流水线后的数据框：")
print(df_encoded_pipeline.head())
```

这个实际应用表明了如何使用标准 Python 库选择和应用各种编码技术。请记住，编码方法的有效性通常取决于下游的机器学习 (machine learning)模型。需要通过实验和验证来确定解决特定问题的最佳方法。

## 参考资料

- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) — Pedregosa, Fabian, et al. (2025)
  Scikit-learn 的官方文档，详细介绍了 `OneHotEncoder`、`OrdinalEncoder`、`ColumnTransformer`、`Pipeline` 以及其他用于分类特征编码的预处理工具。
- [category_encoders documentation](http://contrib.scikit-learn.org/category_encoders/) — Sainburg, Daniel, et al. (2023)
  `category_encoders` 库的官方文档，提供了 `TargetEncoder`、`BinaryEncoder` 和 `HashingEncoder` 的详细信息及使用示例。
- [Feature Hashing for Large Scale Multitask Learning](https://doi.org/10.1145/1553374.1553512) — Kilian Q. Weinberger, Anirban Dasgupta, Josh Attenberg, John Langford, Alex Smola (2009)
  Journal: Proceedings of the 26th International Conference on Machine Learning; Publisher: ACM; Pages: 1169-1176; DOI: [10.1145/1553374.1553512](https://doi.org/10.1145/1553374.1553512)
  介绍了特征哈希技术，它是 HashingEncoder 的理论基础，讨论了其在高维数据处理中的效率。
- [A Comprehensive Survey of Categorical Feature Encoding for Machine Learning](https://arxiv.org/abs/2002.06277) — Mohsen Jamshidian, Alireza Tavakoli (2020)
  Journal: arXiv preprint arXiv:2002.06277; Publisher: arXiv; DOI: [https://doi.org/10.48550/arXiv.2002.06277](https://doi.org/10.48550/arXiv.2002.06277)
  一份调查报告，概述了各种分类特征编码技术，比较了它们的特点、优缺点，包括与目标编码及其挑战相关的讨论。
