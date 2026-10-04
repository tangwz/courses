---
course: "intermediate-python-programming-ml"
chapter: "preparing-data-for-ml"
lesson: "feature-engineering-concepts"
sourceId: 2196
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-5-preparing-data-for-ml/feature-engineering-concepts"
title: "特征工程原理"
description: "了解特征工程的重要性以及创建或调整输入特征的常用技术。"
order: 2
plots: []
sourceHash: "6357f5e5cef8090ff16c9074a771b995601cba1b1bc247c79af35f7d1d019897"
sourceCorrections: []
---

如章前所述，机器学习 (machine learning)模型通常需要干净、结构良好的数值输入数据。原始数据常包含有信息量的特征，但其格式可能不适合算法学习模式。特征工程是借助于领域知识和数据处理技术，从现有原始数据中创建新输入特征（预测变量）的过程。目的是通过提供更相关、信息量更大或尺度适当的底层信息表示，来提升模型性能。

可以将其比作是为机器学习配方精制更好的原料。与其直接将生蔬菜（原始特征）放入锅中，不如切碎、组合，甚至生成新的元素（工程特征），让最终的菜肴（模型预测）美味许多。设计良好的特征可以大大简化建模任务，有时能让更简单、更易于解释的模型达到高精度。反之，即使是复杂的模型，如果输入特征未能充分表示数据中的相关模式，也可能表现不佳。

### 为何特征工程很重要？

输入特征的质量和格式直接影响模型学习的能力。以下是特征工程是重要步骤的原因：

1. **提升模型性能：** 这是主要动机。能更好表示数据底层关系的特征有助于模型做出更准确的预测。例如，线性模型无法表示非线性关系，除非你提供能表示该非线性的特征（如多项式特征）。
2. **增强模型可解释性：** 有时，工程特征比原始输入更直观。例如，将“总债务”和“年收入”组合成“债务收入比”，可能比单独使用这两个原始数字更能直接解释贷款违约风险的预测。
3. **降低复杂度：** 好的特征工程有时能让你有效使用更简单的模型，这些模型通常训练更快，也更容易理解和维护。
4. **算法兼容性：** 有些算法对输入数据有特定要求。例如，线性模型会从与目标变量近似呈线性关系的特征中获得益处。特征工程可以帮助转换特征，以更好地满足这些假设。

### 常用特征工程技术

特征工程通常是一个创造性过程，它由数据分析和领域专业知识指导。然而，有几种常用技术适用范围很广：

#### 1. 创建交互特征

交互特征表示两个或多个特征的组合效应。如果你怀疑一个特征的影响依赖于另一个特征的值，创建交互项会很有帮助。

- **数值 x 数值：** 将两个数值特征相乘。例如，如果预测房价，`number_of_bedrooms`（卧室数量）的影响可能与`square_footage`（面积）关联。你可以创建`bedrooms_x_sqft = number_of_bedrooms * square_footage`。
- **类别 x 类别：** 组合两个特征的类别。例如，组合`region`（区域）和`product_type`（产品类型）可能会创建一个更具体的特征，如`region_product`。
- **数值 x 类别：** 基于类别创建不同的数值特征。例如，`income_in_region_A`（区域A的收入），`income_in_region_B`（区域B的收入）。

```python
import pandas as pd

# 示例DataFrame
data = {'price': [10, 15, 12], 'quantity': [5, 8, 6]}
df = pd.DataFrame(data)

# 创建交互特征：total_revenue (总收入)
df['total_revenue'] = df['price'] * df['quantity']

print(df)
#    price  quantity  total_revenue
# 0     10         5             50
# 1     15         8            120
# 2     12         6             72
```

#### 2. 多项式特征

如果你怀疑特征与目标变量之间存在非线性关系，可以通过添加原始特征的幂（例如，$x^2$, $x^3$）或特征间的交互项（例如，$x_1 x_2$）来创建多项式特征。这对于假设线性关系的线性模型尤其有用。

Scikit-learn为此提供了一个方便的转换器：`sklearn.preprocessing.PolynomialFeatures`。

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 示例数据 (1个特征)
X = np.array([[2], [3], [4]]) # 对于PolynomialFeatures，需要是二维数组

# 创建最高二次的特征（如果存在多个输入特征，则包含交互项）
# include_bias=False 避免添加全1列（截距项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"原始特征:\n{X}")
print(f"\n多项式特征 (2次):\n{X_poly}")
# 原始特征:
# [[2]
#  [3]
#  [4]]
#
# 多项式特征 (2次):
# [[ 2.  4.]  -> 对于输入2，是 x, x^2
#  [ 3.  9.]  -> 对于输入3，是 x, x^2
#  [ 4. 16.]] -> 对于输入4，是 x, x^2
```

使用高次数时要小心，因为这可能导致特征数量过多，并增加过拟合 (overfitting)的风险。

#### 3. 分箱 (离散化)

分箱是将连续数值特征通过将值分组到区间或“箱”中，转换为离散类别特征。

- **为何分箱？**
  - 有助于处理连续数据困难的算法（尽管现在较少见）。
  - 可以表示非线性效应。例如，收入在低、中、高水平时对购买行为可能产生不同影响，分箱可以表示这些效应。
  - 可以使模型对异常值更具鲁棒性（极端值会落入最高或最低的箱中）。

Pandas的`cut`函数对于创建箱很有用。

```python
import pandas as pd

# 示例数据
ages = pd.Series([22, 35, 58, 19, 41, 73, 28])

# 定义分箱边界和标签
bins = [0, 18, 35, 60, 100]
labels = ['<18', '18-35', '36-60', '>60']

# 创建分箱特征
ages_binned = pd.cut(ages, bins=bins, labels=labels, right=False) # right=False 表示 [0, 18), [18, 35) 等

print(ages_binned)
# 0    18-35
# 1    36-60  <- 注意：35 落在 [36, 60) 区间，因为 right=False
# 2    36-60
# 3    18-35
# 4    36-60
# 5      >60
# 6    18-35
# dtype: category
# Categories (4, object): ['<18' < '18-35' < '36-60' < '>60']

# 如果使用 right=True (默认):
# ages_binned_right = pd.cut(ages, bins=bins, labels=labels, right=True)
# print(ages_binned_right)
# 0    18-35  -> (18, 35]
# 1    18-35  -> (18, 35]
# ...
```

#### 4. 变换

应用对数、平方根或倒数等数学变换可以帮助稳定方差、使分布更接近正态，或使关系线性化。例如，对高度偏斜的特征（如收入或人口）取对数（$log(x)$）通常会得到更对称的分布，这对某些模型会有益处。

#### 5. 处理日期和时间特征

日期和时间特征常包含有价值的信息，但其原始格式（例如，'2023-10-27 10:30:00'）无法直接使用。你可以提取多种特征：

- 年、月、日、周几、一年中的第几天、一年中的第几周
- 时、分、秒
- 是否是周末？是否是节假日？
- 自特定事件发生以来经过的时间。

Pandas通过针对日期时间对象的Series上的`.dt`访问器，提供强大的日期时间处理能力。

```python
import pandas as pd

# 示例日期时间数据
dates = pd.Series(pd.to_datetime(['2023-01-15', '2023-05-20', '2024-12-25']))

# 提取特征
df_dates = pd.DataFrame({
    'year': dates.dt.year,
    'month': dates.dt.month,
    'dayofweek': dates.dt.dayofweek, # 周一=0，周日=6
    'is_weekend': dates.dt.dayofweek >= 5
})

print(df_dates)
#    year  month  dayofweek  is_weekend
# 0  2023      1          6        True
# 1  2023      5          5        True
# 2  2024     12          2       False
```

#### 6. 运用领域知识

特征工程中影响最大但自动化程度最低的方面，是运用你对问题领域的理解。如果你在预测客户流失，你可能知道`support_calls`（支持电话数量）与`contract_duration`（合同期限）之比是一个有意义的指标。如果分析传感器数据，变化率（`value_t - value_{t-1}`）可能比原始值更有信息量。这通常涉及根据专家知识而非纯粹的统计模式来组合现有特征。

### 特征工程的应用场景

特征工程并非独立进行。它是数据准备流程中不可或缺的一部分，常需要迭代地进行回溯和调整：

1. **分析数据：** 了解分布、关系、缺失值。
2. **清理数据：** 处理缺失值，修正错误。
3. **工程特征：** 根据数据分析的发现和领域知识创建新特征。
4. **选择特征：** 选择最相关的特征（如有必要）。
5. **缩放/编码：** 为特定的模型算法准备特征（本章稍后讨论）。
6. **训练模型：** 构建并训练模型。
7. **评估模型：** 评估性能。如果结果不满意，你可能需要重新审视特征工程，以创建不同或更好的数据表示。

这种结合数据分析、领域专业知识和实验的迭代过程，是构建有效机器学习 (machine learning)模型的根本。这里讨论的技术为将原始数据转换为强大的预测特征提供了支持。

## 参考资料

- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本全面指导特征工程的书，涵盖理论概念和实践技术。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn development team (2023)
  Journal: scikit-learn.org
  scikit-learn数据预处理模块的官方文档，包括多项式特征等特征转换和其他缩放方法。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://books.google.com/books?id=z1_zvgAACAAJ) — Wes McKinney (2017)
  Publisher: O'Reilly Media
  第3版。Python数据操作的基础资源，包含使用Pandas处理类别、时间序列和数值数据的技术。
