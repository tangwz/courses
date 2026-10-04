---
course: "intro-feature-engineering"
chapter: "handling-missing-data"
lesson: "missing-value-indicators"
sourceId: 1314
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data/missing-value-indicators"
title: "创建缺失值指示器"
description: "生成二元指示器列，以标记原始缺失数据的存在。"
order: 4
plots: []
sourceHash: "6e29dcc68ef79cf45163c00c9b7f5c4b8543cc1d2abbc05acccdfc09bdc80883"
sourceCorrections: []
---

尽管均值、中位数或众数填充等方法能够补全缺失数据留下的空白，但它们本质上会丢弃一条信息：某个值最初是缺失的这个事实。有时，数据本身的缺失就带有预测能力。例如，如果未提供年龄的客户行为倾向不同，那么了解年龄信息缺失本身可能对您的模型来说是一个有用的信号。

这就是缺失值指示器发挥作用的地方。指示器本质上是一个添加到数据集中的新二元特征，它标志着特定列中的原始值是否缺失。通常，如果数据缺失，它取值为1；否则取值为0。

### 为何使用指示器？

主要目的是保留缺失模式所传达的信息。请考虑以下几种情况：

1. **非随机缺失 (MNAR)：** 如果某个值缺失的原因与目标变量或其他特征存在简单填充无法捕获的关联，指示器变量会明确地将此信号提供给模型。
2. **补充填充：** 您可以将指示器与填充方法*一同*使用。首先，创建指示器特征。然后，使用均值、中位数、KNN或其他方法填充原始列中的缺失值。这种做法使模型获得一个估计值，*同时*也得知该值最初是缺失的。基于树的模型，如随机森林或梯度提升，通常能有效利用这种组合。
3. **简便性：** 创建指示器计算开销小，且实现起来很简单。

### 使用 Pandas 实现

使用 Pandas 在 Python 中创建指示器变量很直接。`.isnull()` 方法会返回一个布尔序列（数据缺失处为 True，否则为 False），该序列随后可以转换为整数（True 为 1，False 为 0）。

让我们通过一个例子来说明。假设我们有一个在 `Age` 和 `Income` 列中存在缺失值的 DataFrame：

```python
import pandas as pd
import numpy as np

# 示例 DataFrame
data = {'ID': [1, 2, 3, 4, 5],
        'Age': [25, 30, np.nan, 35, np.nan],
        'Income': [50000, 60000, 75000, np.nan, 90000],
        'Score': [85, 90, 78, 92, 88]}
df = pd.DataFrame(data)

print("原始 DataFrame：")
print(df)

# 识别有缺失值的列
cols_with_na = df.columns[df.isnull().any()].tolist()

# 创建指示器特征
for col in cols_with_na:
    df[col + '_missing_indicator'] = df[col].isnull().astype(int)

print("\n带有缺失值指示器的 DataFrame：")
print(df)
```

运行这段代码会产生以下输出：

```
原始 DataFrame：
   ID   Age   Income  Score
0   1  25.0  50000.0     85
1   2  30.0  60000.0     90
2   3   NaN  75000.0     78
3   4  35.0      NaN     92
4   5   NaN  90000.0     88

带有缺失值指示器的 DataFrame：
   ID   Age   Income  Score  Age_missing_indicator  Income_missing_indicator
0   1  25.0  50000.0     85                      0                         0
1   2  30.0  60000.0     90                      0                         0
2   3   NaN  75000.0     78                      1                         0
3   4  35.0      NaN     92                      0                         1
4   5   NaN  90000.0     88                      1                         0
```

如您所见，`Age_missing_indicator` 和 `Income_missing_indicator` 这两列新列已被添加。它们在原始 `Age` 或 `Income` 列中对应值为 `NaN` 的地方包含 1，否则为 0。

### 指示器与填充结合使用

一种常见且通常有效的方法是将指示器与填充方法结合使用。

1. **创建指示器：** 为具有缺失值的特征添加二元指示器列，如上所示。
2. **执行填充：** 应用一种填充技术（例如，均值、中位数、众数、KNNImputer、IterativeImputer）来填充*原始*列中的 `NaN` 值。

```python
from sklearn.impute import SimpleImputer

# 假设 df 是添加指示器后的 DataFrame
# 使用中位数填充原始 'Age' 列中的缺失值
median_imputer_age = SimpleImputer(strategy='median')
df['Age'] = median_imputer_age.fit_transform(df[['Age']])

# 使用均值填充原始 'Income' 列中的缺失值
mean_imputer_income = SimpleImputer(strategy='mean')
df['Income'] = mean_imputer_income.fit_transform(df[['Income']])

print("\n填充后的 DataFrame（带指示器）：")
print(df)
```

结果 DataFrame：

```
填充后的 DataFrame（带指示器）：
   ID   Age   Income  Score  Age_missing_indicator  Income_missing_indicator
0   1  25.0  50000.0     85                      0                         0
1   2  30.0  60000.0     90                      0                         0
2   3  30.0  75000.0     78                      1                         0
3   4  35.0  68750.0     92                      0                         1
4   5  30.0  90000.0     88                      1                         0
```

现在，`Age` 和 `Income` 列已没有缺失值，但相应的 `_missing_indicator` 列保留了原始缺失值位置的信息。您的机器学习 (machine learning)模型有可能从填充值和缺失模式中同时获取有用信息。

### 考量因素

虽然有用，但请记住以下几点：

- **维度：** 添加指示器会增加数据集中特征的数量。如果您有许多列存在缺失数据，这可能对对高维度敏感的模型构成问题。
- **模型选择：** 除非明确创建交互项，线性模型可能难以解释填充值及其指示器所组合的信息。基于树的模型通常能更自然地处理这种相互作用。
- **MCAR（完全随机缺失）：** 如果数据确实是完全随机缺失 (MCAR)，指示器变量可能无法提供任何额外的预测价值，因为缺失模式是纯粹随机的，与其他的变量或目标无关。

创建缺失值指示器是特征工程工具集中一种有用的方法。它实现起来简单，并能确保关于数据缺失的潜在有用信息在填充过程中不被丢失。尤其当您怀疑缺失模式本身可能提供有用信息或在使用基于树的模型时，可以考虑使用它。

## 参考资料

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953242/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  本书提供了创建有效特征的实用策略，包括处理缺失数据和使用指示变量的方法。
- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) — scikit-learn developers (2023)
  `sklearn.impute`模块的官方文档，描述了各种插补策略和`add_indicator`参数。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  本书提供了构建预测模型的实用指南，包含数据预处理和缺失值处理的章节。
- [A Review on Missing Data Imputation in Machine Learning](https://link.springer.com/article/10.1007/s42979-021-00762-7) — Md. Imran Hossain, Md. Asraful Islam, Md. Abdul Mukit, Sheikh Abujar, and Mohammad Shidujaman (2021)
  Journal: SN Computer Science; Publisher: Springer; Volume: 2; Pages: 1-22; DOI: [10.1007/s42979-021-00762-7](https://doi.org/10.1007/s42979-021-00762-7)
  这篇综述文章考察了各种缺失数据插补技术，提供了指示变量在哪些情境下有益的见解。
