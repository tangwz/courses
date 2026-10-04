---
course: "intro-feature-engineering"
chapter: "feature-creation"
lesson: "binning-numerical-features"
sourceId: 1379
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation/binning-numerical-features"
title: "数值特征分箱"
description: "通过将连续数值特征的值分组到不同的箱中，将其转换为分类特征。"
order: 5
plots: ["plots/1379-0.json", "plots/1379-1.json"]
sourceHash: "5a8fa7d8036c27ec98190d613d0ad068210a6783ddb64b7b83278d199de9c098"
sourceCorrections: []
---

虽然交互特征和多项式特征有助于捕捉*现有*特征之间复杂的关系，但有时将单个数值特征转换为分类特征会很有益。此过程称为**分箱**或**离散化**。分箱涉及将连续数值变量的范围分组到不同的区间，从而有效地将其转换为分类特征。

我们为什么要用更宽泛的类别来代替精确的数值呢？原因有以下几点：

1. **处理非线性影响：** 有些关系是非线性的。例如，非常年轻和非常年长的人群收入可能低于中年群体。将年龄分箱为“儿童”、“青年”、“中年”、“老年”等类别，可以使模型（特别是线性模型）比直接使用原始年龄更容易处理这种U型关系。
2. **减少噪声和异常值影响：** 数值变量的微小波动可能是噪声。分箱可以消除这些波动。同样，极端异常值可能落入第一个或最后一个箱中，从而防止它们过度影响模型参数 (parameter)，尤其是在基于距离的算法或线性模型中。
3. **提高模型可解释性：** 分箱后的特征有时更容易解释。例如，说明收入在“$50k-$75k”区间的客户行为不同，通常比解释连续收入变量的一个小系数更直观。
4. **与某些模型的兼容性：** 某些算法可能天生就能更好地处理分类特征或需要它们。

接下来，我们来看一下对数值特征进行分箱的常见策略。

### 等宽分箱

这可能是最直接的方法。我们将数值特征的范围（从最小值到最大值）划分为预设数量的箱，每个箱具有相同的宽度。

例如，如果我们的“年龄”特征范围是0到100，并且我们想要5个箱，则每个箱将覆盖$(100 - 0) / 5 = 20$年的范围。这些箱将是[0, 20]、(20, 40]、(40, 60]、(60, 80]和(80, 100]。

在Pandas中，您可以使用`cut`函数轻松实现这一点。

```python
import pandas as pd
import numpy as np

# 示例数据
data = {'age': np.random.randint(0, 85, size=100)}
df = pd.DataFrame(data)

# 显式定义箱的边界
bin_edges = [0, 18, 35, 60, 85]
bin_labels = ['0-18', '19-35', '36-60', '61+']

# 使用指定边界应用等宽分箱
df['age_bin_fixed'] = pd.cut(df['age'], bins=bin_edges, labels=bin_labels, right=True, include_lowest=True)

# 或者，只指定箱的数量（Pandas 会计算等宽）
df['age_bin_fixed_auto'] = pd.cut(df['age'], bins=4, labels=False) # labels=False 返回箱的整数编码

print(df[['age', 'age_bin_fixed', 'age_bin_fixed_auto']].head())
#    age age_bin_fixed  age_bin_fixed_auto
# 0   78           61+                   3
# 1   12          0-18                   0
# 2   45         36-60                   2
# 3   68           61+                   3
# 4   29         19-35                   1
```



![等宽分箱前后的分布](plots/1379-0.json)



> 原始“年龄”的分布与等宽箱内的计数进行比较。注意连续分布如何被分组到离散类别中。

**优点：** 易于理解和实现。
**缺点：** 对异常值敏感。如果数据存在极端值，大多数数据点可能只集中在少数几个箱中，而其他箱则稀疏或为空。它没有考虑到数据的底层分布。

### 基于分位数的分箱

不同于等宽，基于分位数的分箱旨在创建包含大致相同数量观测值的箱。它使用百分位数（分位数）来定义箱的边界。例如，如果我们想要4个箱（四分位数），则边界将设置为最小值、第25百分位数、第50百分位数（中位数）、第75百分位数和最大值。

当数据偏斜时，通常更受欢迎此方法，因为它确保每个箱都有合理数量的数据。

Pandas 为此提供了 `qcut` 函数。

```python
import pandas as pd
import numpy as np

# 示例偏斜数据（例如，收入）
np.random.seed(42)
income_data = np.random.exponential(scale=20000, size=200) + 15000 # 正偏斜
df_income = pd.DataFrame({'income': income_data})

# 应用基于分位数的分箱（例如，分成4个四分位数箱）
df_income['income_bin_quantile'] = pd.qcut(df_income['income'], q=4, labels=False) # 4个箱

# 也可以指定标签
df_income['income_bin_quantile_labeled'] = pd.qcut(df_income['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

print(df_income[['income', 'income_bin_quantile', 'income_bin_quantile_labeled']].head())
#          income  income_bin_quantile income_bin_quantile_labeled
# 0  32338.414699                    1                      Medium
# 1  22500.455803                    0                         Low
# 2  58575.039677                    2                        High
# 3  84972.969536                    3                   Very High
# 4  18131.939134                    0                         Low

# 检查值计数以查看它们是否大致相等
print("\nCounts per quantile bin:")
print(df_income['income_bin_quantile_labeled'].value_counts())
# Counts per quantile bin:
# Low          50
# Medium       50
# High         50
# Very High    50
# Name: income_bin_quantile_labeled, dtype: int64
```



![分位数分箱前后分布（偏斜数据）](plots/1379-1.json)



> 原始偏斜的“收入”分布与分位数箱内的计数对比。注意分位数分箱如何使每个类别中的计数大致相等，尽管存在偏斜。

**优点：** 很好地处理偏斜数据，确保每个箱都有代表性。通常显示与排名或相对位置相关的模式。
**缺点：** 箱的宽度可能差异很大，如果离百分位数边界很近，可能会合并不同的数值。对箱的解释仅取决于排名，而非绝对值范围（除非手动创建反映范围的标签）。损失了关于绝对值差异的信息。

### 手动和领域专业知识分箱

有时，分箱特征的最佳方式是基于外部信息或领域专业知识。例如，标准年龄组（0-17、18-64、65+）、既定的所得税等级或在某个领域中已知具有重要意义的特定阈值（例如，临床测量）。这通常会产生最易于解释且可能最具预测性的箱，但这需要数据本身以外的知识。

```python
# 示例：基于常见年龄类别的手动分箱
manual_bins = [0, 17, 64, np.inf] # 使用 np.inf 作为上限
manual_labels = ['Child/Teen', 'Adult', 'Senior']
df['age_bin_manual'] = pd.cut(df['age'], bins=manual_bins, labels=manual_labels, right=True)

print(df[['age', 'age_bin_manual']].head())
#    age age_bin_manual
# 0   78         Senior
# 1   12     Child/Teen
# 2   45          Adult
# 3   68         Senior
# 4   29          Adult
```

### 使用 Scikit-learn：KBinsDiscretizer

Scikit-learn 提供了 `KBinsDiscretizer` 转换器，它很好地融入机器学习 (machine learning)管道。它支持不同的策略来确定箱：

- `strategy='uniform'`：等同于等宽分箱（当指定箱的数量时，与 `pd.cut` 类似）。
- `strategy='quantile'`：等同于基于分位数的分箱（与 `pd.qcut` 类似）。
- `strategy='kmeans'`：使用一维K-均值聚类算法来根据数据密度寻找箱的边界。

```python
from sklearn.preprocessing import KBinsDiscretizer

# 为 Scikit-learn 重塑数据（需要二维数组）
age_data = df[['age']].values

# 初始化离散化器（例如，5个分位数箱，输出序数整数）
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None) # subsample=None 以避免在新版本上的警告

# 拟合并转换
df['age_bin_sklearn'] = kbd.fit_transform(age_data)

print(df[['age', 'age_bin_sklearn']].head())
#    age  age_bin_sklearn
# 0   78              4.0
# 1   12              0.0
# 2   45              2.0
# 3   68              4.0
# 4   29              1.0

# 可以检查计算出的箱边界
print("\nSklearn 箱边界:", kbd.bin_edges_[0])
# Sklearn Bin Edges: [ 0.  17.4 34.8 51.2 68.6 84. ]
```

默认情况下，`KBinsDiscretizer` 输出箱的数值表示（0、1、2...）。如果您的后续模型需要分类特征（如独热编码），您可能需要设置 `encode='onehot-dense'` 或在离散化后应用单独的编码器。

### 注意事项

- **箱的数量：** 选择箱的数量 ($k$) 很重要。箱的数量过少可能会过度简化数据并丢失重要信息。箱的数量过多可能无法比原始连续特征带来更多好处，并可能导致稀疏类别（过拟合 (overfitting)）。这种选择通常涉及实验或领域知识。
- **信息损失：** 请记住，分箱本质上是一个有损过程。您放弃精确的数值，转而使用一个范围。请确保其好处（处理非线性、鲁棒性）超过这种损失，以符合您的具体问题和模型。
- **视为分类特征：** 分箱后，请将生成的特征视为分类特征。这可能涉及应用适当的编码技术（如独热编码或序数编码，在第3章中讨论），具体取决于所选模型，尽管基于树的模型通常可以有效地处理分箱特征而无需显式编码。

分箱提供了一种转换数值特征的方法，可以展现非线性模式或使模型更强大。与其他特征工程技术一样，最佳方法（等宽、分位数、手动）和最佳箱数通常取决于数据分布和分析目标。

## 参考资料

- [sklearn.preprocessing.KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) — scikit-learn developers (2023)
  Scikit-learn 的 KBinsDiscretizer 官方文档，详细说明其用法、策略（均匀、分位数、KMeans）和参数。
- [pandas.cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) — pandas development team (2023)
  pandas.cut 的官方文档，该函数用于实现固定宽度分箱和自定义分箱边界。
- [pandas.qcut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html) — pandas development team (2023)
  Publisher: pandas project
  pandas.qcut 的官方文档，该函数用于实现基于分位数的分箱，以创建包含大致相等观测值的箱。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本关于各种特征工程技术的综合书籍，包含对离散化方法及其对模型性能影响的详细讨论。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125969/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本广泛使用的机器学习教科书，涵盖特征工程实践，包括处理数值和分类特征的策略。第四版。
