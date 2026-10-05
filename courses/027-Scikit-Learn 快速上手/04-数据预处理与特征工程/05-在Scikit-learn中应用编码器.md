# 在Scikit-learn中应用编码器

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/applying-encoders)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将类别特征转换为数值表示是机器学习 (machine learning)中常见的必要步骤。这里将演示如何使用Scikit-learn的专用工具实现这些转换。Scikit-learn提供了`Transformer`类，例如`OneHotEncoder`和`OrdinalEncoder`，它们遵循标准的`fit`和`transform` API模式，保持了库内部的统一性。

### 使用OneHotEncoder

当类别特征没有内在顺序时，独热编码（One-Hot Encoding）是合适的。它将每个类别转换为一个新的二元特征（0或1）。Scikit-learn的`OneHotEncoder`是完成此任务的主要工具。

我们来考虑一个带有类别特征的简单数据集：

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 示例数据
data = pd.DataFrame({'color': ['Red', 'Green', 'Blue', 'Green'], 
                     'size': ['M', 'L', 'S', 'M']})

print("原始数据:")
print(data)

# 选择要编码的类别列
categorical_features = ['color'] # 我们先编码 'color'
```

要对'color'列应用独热编码：

1. **实例化编码器：** 创建`OneHotEncoder`的一个实例。
2. **拟合并转换：** 对选定的数据使用`fit_transform`方法。此方法首先学习存在的唯一类别（`fit`），然后执行转换（`transform`）。

```python
# 1. 实例化编码器
# 默认情况下，它返回一个稀疏矩阵。为了演示，我们请求一个密集数组。
# handle_unknown='ignore' 可以防止在转换过程中出现新类别时报错
encoder_color = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 

# 2. 拟合并转换 'color' 列
# 编码器需要2D输入，所以我们使用双括号 [['color']]
encoded_colors = encoder_color.fit_transform(data[['color']])

print("\n已编码的 'color' 特征 (独热编码):")
print(encoded_colors)

# 获取新创建的特征名称
feature_names_color = encoder_color.get_feature_names_out(['color'])
print("\n特征名称:")
print(feature_names_color)

# 与原始数据合并（可选，仅用于演示）
encoded_df_color = pd.DataFrame(encoded_colors, columns=feature_names_color, index=data.index)
data_encoded = pd.concat([data.drop(columns=['color']), encoded_df_color], axis=1)

print("\n已对 'color' 进行独热编码的数据:")
print(data_encoded)
```

**输出:**

```
原始数据:
   color size
0    Red    M
1  Green    L
2   Blue    S
3  Green    M

已编码的 'color' 特征 (独热编码):
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]

特征名称:
['color_Blue' 'color_Green' 'color_Red']

已对 'color' 进行独热编码的数据:
  size  color_Blue  color_Green  color_Red
0    M         0.0          1.0        0.0
1    L         1.0          0.0        0.0
2    S         0.0          0.0        1.0
3    M         1.0          0.0        0.0
```

**`OneHotEncoder`的参数 (parameter)：**

- `sparse_output`：默认为`True`，返回一个SciPy稀疏矩阵。当类别很多（因此产生许多大部分为零的列）时，稀疏矩阵能有效节省内存。将其设置为`False`可获得标准的NumPy密集数组，这可能在初始检查时更方便，但对于高基数特征，可能会占用大量内存。
- `handle_unknown`：控制在`transform`期间遇到在`fit`期间未见的类别时的行为。默认的`'error'`会引发错误。使用`'ignore'`会将该未知类别的所有独热编码列赋值为零，这通常是一种实用的方法。
- `categories`：默认情况下（`'auto'`），类别是从训练数据中推断的。如果需要，您可以提供一个特定的类别列表。
- `drop`：可以设置为`'first'`或特定的类别值，以删除每个特征的一个二元列。这有助于在某些线性模型中避免多重共线性，但可能会使解释变得不那么直接。

### 使用OrdinalEncoder

当类别具有有意义的顺序时（例如，'low'、'medium'、'high'），`OrdinalEncoder`更为合适。它根据每个类别在指定顺序中的位置，为其分配一个整数值。

我们来使用之前示例中的'size'列，假设顺序为 S < M < L。

```python
from sklearn.preprocessing import OrdinalEncoder

# 定义 'size' 特征的期望顺序
size_order = ['S', 'M', 'L']

# 1. 实例化编码器，提供类别顺序
# 'categories' 参数期望一个列表的列表，其中每个列表对应一个要编码的特征
encoder_size = OrdinalEncoder(categories=[size_order]) 

# 2. 拟合并转换 'size' 列
# 同样，使用双括号表示2D输入
encoded_sizes = encoder_size.fit_transform(data[['size']])

print("\n已编码的 'size' 特征 (序数编码):")
print(encoded_sizes)

# 将序数编码的列添加回DataFrame
data['size_encoded'] = encoded_sizes.astype(int) # 将浮点数输出转换为整数

print("\n已对 'size' 进行序数编码的数据:")
print(data) 
```

**输出:**

```
已编码的 'size' 特征 (序数编码):
[[1.]
 [2.]
 [0.]
 [1.]]

已对 'size' 进行序数编码的数据:
   color size  size_encoded
0    Red    M             1
1  Green    L             2
2   Blue    S             0
3  Green    M             1
```

请注意，'S'被映射为0，'M'被映射为1，'L'被映射为2，这遵循了我们提供的顺序。

**`OrdinalEncoder`的参数 (parameter)：**

- `categories`：这是最重要的参数。它应该是一个列表，其中包含一个列表，对应于每个被编码的特征。每个内部列表指定了该特征的期望类别顺序。如果省略，编码器会根据数据中遇到的唯一值来确定顺序，这通常会导致序数数据的数值映射是任意且可能没有意义的。当为具有内在顺序的特征使用`OrdinalEncoder`时，请务必明确指定`categories`。

### 编码器在实际中的应用

这些示例展示了如何将编码器应用于单个列或列的子集。在典型的机器学习 (machine learning)工作流程中，您通常会遇到多个数值特征和类别特征。对不同的列应用不同的预处理步骤（例如，对数值特征进行缩放，对类别特征进行编码）是一个常见的需求。Scikit-learn的`ColumnTransformer`（通常在`Pipeline`中使用，第6章会介绍）正是为此类情况而设计的，它允许您以结构化的方式将不同的转换器应用于不同的列子集。

选择正确的编码器（`OneHotEncoder`用于名义型特征，`OrdinalEncoder`用于序数型特征）并使用Scikit-learn的转换器API正确应用它们，是为有效模型训练准备数据的基本步骤。请记住，`OneHotEncoder`会显著增加数据的维度，而`OrdinalEncoder`则会强加一个顺序，如果`categories`参数 (parameter)使用不当，该顺序可能不存在或无法被正确反映。

## 参考资料

- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) — scikit-learn developers (2024)
  Scikit-learn官方OneHotEncoder文档，提供其功能和参数的全面详细信息。
- [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) — scikit-learn developers (2024)
  Scikit-learn官方OrdinalEncoder文档，详细说明其用法、参数以及如何定义类别顺序。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125965/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广受认可的实用指南，涵盖数据预处理，包括使用Scikit-learn的分类编码技术，并提供实际示例。

---

[上一节](04-%E5%88%86%E7%B1%BB%E7%89%B9%E5%BE%81%E7%BC%96%E7%A0%81.md) · [下一节](06-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC.md)
