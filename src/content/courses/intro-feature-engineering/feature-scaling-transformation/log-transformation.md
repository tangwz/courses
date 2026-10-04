---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "log-transformation"
sourceId: 1341
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/log-transformation"
title: "对偏斜数据的对数变换"
description: "应用对数变换以减少正偏斜数据中的偏斜度。"
order: 5
plots: ["plots/1341-0.json"]
sourceHash: "349154c1844d4bc2c4e0734b4d1531abdb0388287725a5e85307219aaa76e50e"
sourceCorrections: []
---

许多机器学习 (machine learning)算法，尤其是线性模型和那些假设误差呈正态分布的算法，在处理高度偏斜的数值特征时，表现可能不理想。偏斜度指的是数据分布的不对称性。一种常见类型是右偏（或正偏），其分布的右侧尾部比左侧长或厚。这常出现在表示计数、频率或货币数值的特征中，大多数观测值集中在较低值，但少数极高值使分布拉伸。

以“收入”特征为例。大多数人的收入可能在一个特定范围，但少数人的收入可能明显更高，从而形成一个长长的右尾。这些极端值可能不成比例地影响模型参数 (parameter)或距离计算。

### 使用对数变换处理偏斜

处理右偏数据最常用且有效的方法之一是**对数变换**。应用自然对数（底数为 $e$，常表示为 $\ln$）或以10为底的对数（$\log_{10}$）会压缩数据范围，尤其是在高端。

变换定义如下：
$y = \log(x)$
这里 $x$ 是原始特征值，$y$ 是变换后的值。

对数的作用是，大值比小值被更显著地缩小。例如，$\log(10) \approx 2.3$，$\log(100) \approx 4.6$，$\log(1000) \approx 6.9$。虽然100和1000之间的绝对差是900，但它们自然对数之间的差异只有大约2.3。这种压缩有助于使分布更对称，常更接近正态分布。

### 处理零值和负值

标准对数变换的一个明显限制是它只对正值（$x > 0$）定义。零的对数未定义，负数的对数会得到复数，这些复数通常不能直接用于标准机器学习 (machine learning)模型。

如果您的数据包含零但没有负值，一个常用的变通方法是在应用对数之前，将一个小的常数（通常是1）加到所有值上。这被称为 **log(1+x)** 变换或 **log1p**：

$y = \log(1 + x)$

这种变换有一个方便的属性，即 $\log(1+0) = \log(1) = 0$，它在允许对非负数据进行变换的同时，保留了零的含义。许多数值计算库，包括NumPy，都提供一个专用函数 `np.log1p()`，即使对于非常小的 $x$ 值，它也能准确计算 $\log(1+x)$。

如果您的数据包含负值，对数变换（即使是 `log1p`）也不能直接应用。在这种情况下，您可能需要考虑其他变换，例如Yeo-Johnson变换（本章稍后会提到），或者如果这在您的应用背景下合理，可以根据特征的符号来拆分特征。

### 使用NumPy和Pandas实现

在Python中使用NumPy应用对数变换很简单。假设您的数据在Pandas DataFrame `df` 中，并且您想变换名为 `feature_skewed` 的列：

```python
import numpy as np
import pandas as pd

# 样本偏斜数据（例如，模拟收入）
np.random.seed(42)
data_skewed = np.random.exponential(scale=10000, size=1000)
# 引入一些零值
data_skewed[::10] = 0
df = pd.DataFrame({'feature_skewed': data_skewed})

# 在应用对数前检查负值
if (df['feature_skewed'] < 0).any():
    print("警告：特征包含负值。对数变换不适用。")
else:
    # 应用log1p变换以处理可能存在的零值
    df['feature_log_transformed'] = np.log1p(df['feature_skewed'])

# 显示原始数据和变换后数据的前几行
print(df[['feature_skewed', 'feature_log_transformed']].head())

# 显示基本统计信息
print("\n原始数据统计信息:")
print(df['feature_skewed'].describe())
print("\n变换后数据统计信息:")
print(df['feature_log_transformed'].describe())
```

### 效果可视化

对数变换的影响通常通过可视化能更好地理解。让我们比较原始偏斜特征及其对数变换后的版本分布。



![原始分布与对数变换后的分布比较](plots/1341-0.json)



> 原始正偏斜数据（蓝色）和应用 `log1p` 变换后的数据（绿色）的概率密度直方图比较。变换后的数据显示出更对称的钟形。

如图所示，原始分布在零附近高度集中，并有一个长尾延伸向更高的值。在 `log1p` 变换后，分布变得更加对称和分散，更接近正态分布。这种变换后的特征通常更适合对尺度和分布形状敏感的算法。

### 注意事项

- **可解释性：** 应用对数变换会改变特征的解释方式。变换后特征的单位变化对应于原始特征的乘性变化。使用对数变换特征的模型中的系数需要相应地进行解释。
- **并非总是完美：** 尽管对数变换对于右偏数据通常有效，但它不能保证数据呈完美正态分布。始终要在变换后检查分布。
- **替代方法：** 对于并非严格为正的数据或不同类型的分布问题，Box-Cox或Yeo-Johnson等其他变换可能更适合，我们将在接下来看到。

对数变换是特征工程工具包中一个简单但有效的工具，对于处理真实数据集中常见的具有指数增长模式或乘性影响的特征特别有用。

## 参考资料

- [Applied Predictive Modeling](https://www.springer.com/book/9781461468486) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer
  本书全面介绍了数据预处理技术，包括各种变换（如对数变换），并解释了它们在提升模型性能方面的作用。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 582
  一本使用Python的Pandas和NumPy库进行数据处理的实践指南，与实现对数变换等数值变换直接相关。
- [numpy.log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html) — NumPy Developers (2023)
  NumPy `log1p` 函数的官方文档，详细说明其数学定义、用法以及为何适用于包含零值的非负数据。
- [An Analysis of Transformations](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFgsgL5uEh94qsdj2K5MJNYOZUvWN1Ck8Xg72mkJysArLnJSmJEHQMEDOulgoMXkcp0XWwl7p77-KrMzBEKgULB0gG5cwfPiiioBBGR0tS7KEv4AUmPIH3rbav5HXuR1xu9gcF5e8spFpK0QdG43V4f1MGWmhyG5_P0ebksROOBLgV73sRNb4pgH4FxHApD_8nmBKc7ig3Ah7D_hab9C3rDfFphopI9YR6OATVaRWrp) — George E. P. Box and David R. Cox (1964)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Oxford University Press; Volume: 26; Pages: 211-252; DOI: [10.1111/j.2517-6161.1964.tb00553.x](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)
  这篇基础论文介绍了Box-Cox变换，该变换泛化了幂变换（包括对数变换），为改善数据正态性提供了统计框架。
