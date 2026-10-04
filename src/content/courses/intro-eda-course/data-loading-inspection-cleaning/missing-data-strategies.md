---
course: "intro-eda-course"
chapter: "data-loading-inspection-cleaning"
lesson: "missing-data-strategies"
sourceId: 1145
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/missing-data-strategies"
title: "缺失数据的处理策略：填充与删除"
description: "比较处理缺失数据的方法，包括简单填充（均值、中位数、众数）和删除。"
order: 5
plots: []
sourceHash: "64ba1bdcefad65ff9a17a06d9901bb906dd9f24930ca718f810085b315f44405"
sourceCorrections: []
---

您已使用上一节中的方法，识别出数据集中存在的缺失值（`NaN`）。现在面临一个决定：您该如何处理它们？忽略缺失数据通常不是一个好的选择，因为大多数分析算法和可视化工具都无法很好地处理 `NaN` 值。概括来说，您有两种主要的处理方法：移除缺失数据（删除）或用估计值进行补充（填充）。选择哪种方法并不总是显而易见的，它很大程度上取决于具体情况、缺失数据的数量以及您的分析目标。

### 删除策略：移除缺失数据

最直接的方法是直接移除包含缺失值的行或列。

#### 行删除（列表式删除）

这涉及到移除包含*任何*列中*任何*缺失值的整行（观测值）。

- **何时适用：** 当含缺失数据的行数与数据集总大小相比非常少（例如，少于5%）时，此策略最为合理。如果数据是完全随机缺失（MCAR），指数据的缺失与缺失值本身或其他观测值无关，那么删除可能不会带来明显偏差。
- **潜在缺点：** 主要不足是会丢失被删除行中非缺失列包含的潜在有用信息。如果很大比例的行存在缺失值，这会大幅减少数据集的大小和统计功效。此外，如果数据*并非*完全随机缺失，删除行可能给分析带来偏差，导致结果不准确。

在Pandas中，您可以使用 `.dropna()` 方法轻松删除含有任何 `NaN` 值的行：

```python
# 假设 'df' 是您的 DataFrame
df_cleaned_rows = df.dropna()

# 检查删除前后的形状
print(f"原始形状: {df.shape}")
print(f"删除含 NaN 行后的形状: {df_cleaned_rows.shape}")
```

#### 列删除

或者，如果某一特定列（特征）含有非常高比例的缺失值，它可能给您的分析提供很少有用信息。在这种情况下，删除整个列可能是一个合理选项。

- **何时适用：** 如果一列缺失了其值的相当大比例（例如，> 50-70%），可以考虑此方法。保留这样的列可能需要大量填充，这比直接删除它可能引入更多噪音或偏差。
- **潜在缺点：** 您会丢失与该特征相关的所有信息，即使不完整，这些信息也可能相关。删除的阈值是主观的，并取决于该特征被认为的重要性。

在Pandas中删除缺失值列时，可以在 `.dropna()` 方法中指定 `axis=1`。您还可以使用 `thresh` 参数 (parameter)来保留至少有特定数量非 `NaN` 值的列。

```python
# 删除所有值为 NaN 的列
df_cleaned_cols_all_nan = df.dropna(axis=1, how='all')

# 删除缺失值超过 30% 的列
threshold = len(df) * 0.7 # 保留至少有 70% 非 NaN 值的列
df_cleaned_cols_thresh = df.dropna(axis=1, thresh=threshold)

print(f"原始形状: {df.shape}")
# print(f"Shape after dropping columns with all NaN: {df_cleaned_cols_all_nan.shape}")
print(f"删除缺失值超过 30% 的列后的形状: {df_cleaned_cols_thresh.shape}")
```

### 填充策略：补充缺失数据

填充是指用替代值替换缺失值。这可以保留您的样本量，但会引入人造数据点，可能影响数据的原始分布和关联。简单填充方法在初始 EDA 阶段很常见。

#### 均值填充

用数值列中观测值的均值替换该列的缺失值。

- **优点：** 实现简单。保留列的均值。
- **缺点：** 降低列的方差。扭曲变量之间的关联（相关性）。对异常值高度敏感，这会使均值不准确。仅适用于数值数据。

```python
# 用均值填充 'numerical_col' 中的缺失值
mean_value = df['numerical_col'].mean()
df['numerical_col_mean_imputed'] = df['numerical_col'].fillna(mean_value)
```

#### 中位数填充

用数值列中观测值的中位数替换缺失值。

- **优点：** 实现简单。比均值填充对异常值更有效。
- **缺点：** 与均值填充类似，它会降低方差并扭曲相关性。仅适用于数值数据。

```python
# 用中位数填充 'numerical_col' 中的缺失值
median_value = df['numerical_col'].median()
df['numerical_col_median_imputed'] = df['numerical_col'].fillna(median_value)
```

#### 众数填充

用列中观测值的众数（最常出现的值）替换缺失值。

- **优点：** 适用于数值和分类数据。简单。
- **缺点：** 可能引入偏差，特别是当某个值压倒性地占优时。降低方差。如果存在多个众数，可能会随意选择一个。扭曲关联。

Pandas 的 `.mode()` 方法返回一个 Series（因为可能存在多个众数）。通常，您会使用第一个众数（`[0]`）。

```python
# 用众数填充 'categorical_col' 中的缺失值
mode_value = df['categorical_col'].mode()[0]
df['categorical_col_mode_imputed'] = df['categorical_col'].fillna(mode_value)

# 也可以应用于数值列，尽管不常见
# numerical_mode_value = df['numerical_col'].mode()[0]
# df['numerical_col_mode_imputed'] = df['numerical_col'].fillna(numerical_mode_value)
```

#### 常数值填充

用预定义的常数值替换缺失值，例如 0、-1、“未知”或“缺失”。

- **优点：** 简单。有时可以表明该值最初是缺失的（例如，使用“未知”）。
- **缺点：** 引入任意值，这会显著扭曲统计量（均值、方差、相关性）和分布，除非所选常数在该情况下有有意义的解释（例如，对于某些特征，0可能表示“缺失”）。

```python
# 用 0 填充数值 NaN
df['numerical_col_zero_imputed'] = df['numerical_col'].fillna(0)

# 用 'Unknown' 填充分类 NaN
df['categorical_col_unknown_imputed'] = df['categorical_col'].fillna('Unknown')
```

### 选择正确的方法：删除还是填充

那么，您是该删除还是填充？没有普遍答案。考虑以下因素：

1. **缺失数据量：**
   - **少量缺失值（<5%）：** 如果数据集较大，删除（列表式）可能是可接受的。简单填充也不太可能导致重大扭曲。
   - **中等量缺失值（5-30%）：** 由于信息丢失，删除的风险更高。填充通常更受推荐，但方法的选择更重要。考虑对方差和相关性的潜在影响。
   - **大量缺失值（>30%）：** 列表式删除通常不适用。如果特定特征大部分缺失，可以考虑列删除。填充变得必要但可能有问题；简单方法可能严重扭曲数据。更精细的填充方法可能在之后需要。
2. **数据和变量的性质：**
   - **数值型：** 均值或中位数填充是常见选择。对于倾斜分布或含异常值的数据，中位数更受青睐。
   - **分类型：** 众数填充或用“未知”等常数填充是典型方法。
3. **缺失机制（数据为何缺失？）：** 尽管对此的审视是在后期阶段进行，但要考虑缺失是否看起来是随机的，或者是否可能与其他变量有关。如果缺失具有系统性（例如，高收入者不报告收入），简单填充或删除可能导致有偏差的结论。
4. **分析目标：** 对于快速审视，简单填充甚至删除可能足够。为了构建预测模型，缺失数据的处理变得更重要，通常涉及与模型训练结合的策略（例如仅基于训练数据进行填充）。

**实用建议：**

首先量化 (quantization)每列和整体缺失数据的程度。如果缺失数据极少，删除可能是最简单的方法。如果更普遍，简单填充（数值型用中位数，分类型用众数）是初始 EDA 的合理起点，让您在进行分析的同时注意可能引入的扭曲。始终记录您选择的方法，因为这是数据准备过程中的重要步骤。如果简单方法不足以应对后续任务（如建模），可以在之后考虑更复杂的填充方法。

## 参考资料

- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, and Jian Pei (2011)
  Publisher: Elsevier
  详细介绍了数据预处理方法的综合性教科书，包括处理缺失数据的策略、优点和缺点。
- [Missing Data Analysis: A Practical Guide to Multiple Imputation](https://doi.org/10.1007/978-1-4614-4018-5) — John W. Graham (2012)
  Publisher: Springer; DOI: [10.1007/978-1-4614-4018-5](https://doi.org/10.1007/978-1-4614-4018-5)
  讨论缺失数据分析，介绍了缺失数据机制的理论背景以及简单删除和填充方法的局限性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning-with-scikit-learn-keras-and-tensorflow/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  处理数据预处理步骤，包括管理缺失数据，并在机器学习项目中提供策略和考量。
- [Missing Data](https://methods.sagepub.com/book/missing-data) — Paul D. Allison (2001)
  Journal: Quantitative Applications in the Social Sciences; Publisher: SAGE Publications, Inc.; Volume: 136; Pages: 104; DOI: [10.4135/9781412984183](https://doi.org/10.4135/9781412984183)
  一部经典指南，提供了对缺失数据问题和基本处理方法的理解。
