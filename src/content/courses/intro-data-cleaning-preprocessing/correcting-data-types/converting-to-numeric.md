---
course: "intro-data-cleaning-preprocessing"
chapter: "correcting-data-types"
lesson: "converting-to-numeric"
sourceId: 4010
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-4-correcting-data-types/converting-to-numeric"
title: "转换为数字类型 (整数, 浮点数)"
description: "学习将数字的字符串表示形式转换为实际数字类型的技术。"
order: 4
plots: []
sourceHash: "58b231ac064ff64ad7bb65a058354801509ade164b83b825383f5b2cf0e02ad2"
sourceCorrections: []
---

正如我们在章节引言中讨论过的，计算机根据数据分配的类型对数据进行不同处理。执行数学计算需要数据为*数字*格式。如果你的数据集包含以文本（字符串）形式存储的数字，例如'100'或'98.6'，你将无法直接使用它们进行加、减、求平均值或更复杂的分析。尝试计算$5 + '10'$很可能导致错误或意外结果，而非你期望的$15$。本节侧重于将这些基于文本的数字转换为正确的数字类型：整数和浮点数。

### 理解数字类型：整数和浮点数

在转换之前，让我们快速明确你将遇到的两种主要数字类型：

1. **整数 (int):** 它们是正数或负数的整数，不包含任何小数点。例子包括 $0$，$-5$，$100$，$2024$。它们适用于表示计数、ID 或任何本质上为整数的数量。
2. **浮点数 (float):** 这些数字包含小数点，允许出现分数。例子包括 $3.14$，$-0.5$，$98.6$，$100.0$。浮点数用于测量、百分比或任何需要整数之外精度的值。

### 为什么需要转换

想象一个表示产品价格的列，但其值却以字符串形式存储，例如'$5.99'、'12.00'、'€8.50'。你无法直接从这些文本条目计算平均价格或总销售额。同样，排序可能会按字母顺序进行（'$12.00'可能会排在'$5.99'$之前），而非按数值大小。将这些字符串转换为数字格式（如浮点数）对于正确执行这些操作来说非常重要。

### 转换过程

大多数数据分析工具都提供用于尝试这种转换的函数。一般思路是指示工具读取字符串，将其解释为数字，并以适当的数字类型（整数或浮点数）存储。

Pandas 提供了一个名为 `pd.to_numeric()` 的函数，它专门为此任务设计。

**基本转换**

假设你有一个名为 `prices_text` 的 pandas Series（DataFrame 中的一列），其中包含字符串：

```
0     '5.99'
1    '12.00'
2     '8.50'
Name: prices_text, dtype: object
```

`dtype: object` 在 pandas 中通常表示字符串或混合类型。要将其转换为数字类型，你可以使用：

```python
# 假设 'prices_text' 是你的 pandas Series
numeric_prices = pd.to_numeric(prices_text)
print(numeric_prices)
```

输出将如下所示：

```
0     5.99
1    12.00
2     8.50
Name: prices_text, dtype: float64
```

注意 `dtype: float64`。Pandas 识别了小数点并选择了浮点类型。如果字符串表示整数，例如'100'、'25'、'0'，`pd.to_numeric()` 可能会选择整数类型（`int64`）。

**处理潜在问题**

如果列中包含*无法*解释为数字的值，会发生什么？常见例子包括：

- 货币符号（`$`，`€`）
- 作为千位分隔符的逗号（`1,000`）
- 文本标注（`N/A`，`Missing`，`5 units`）
- 意外字符或拼写错误

如果你尝试直接转换包含此类值的列，`pd.to_numeric()` 函数通常会停止并引发错误，因为它不知道如何处理非数字条目。

例如，尝试转换 `['5.99', '$12.00', '8.50']` 很可能会在第二个元素处失败。

**使用 `errors='coerce'`**

一种非常常见的策略是告诉转换函数将任何有问题的值替换为缺失数据的特殊标记 (token)，通常表示为 `NaN`（非数字）。在 pandas 中，你可以通过使用 `errors='coerce'` 参数 (parameter)来实现这一点：

```python
# 包含问题值的示例 Series
mixed_values = pd.Series(['100', '55.5', 'N/A', '2,000', '-5'])

# 尝试转换，将错误强制转换为 NaN
numeric_values = pd.to_numeric(mixed_values, errors='coerce')
print(numeric_values)
```

输出将是：

```
0    100.0
1     55.5
2      NaN  # 'N/A' 变为 NaN
3      NaN  # '2,000'（带逗号）变为 NaN
4     -5.0
Name: values, dtype: float64
```

以下是发生的情况：

- '100' 和 '-5' 被转换为浮点数（$100.0$ 和 $-5.0$）。
- '55.5' 被转换为浮点数 $55.5$。
- 'N/A' 无法被解释为数字，因此 `errors='coerce'` 将其变为 `NaN`。
- '2,000' 包含逗号，在此上下文 (context)中这不是数字表示的标准形式，因此它也变为 `NaN`。

请注意，即使一些原始值是整数（'100'，'-5'），整个列的数据类型也变成了 `float64`。这是因为 `NaN` 本身在技术上是一个浮点值，因此它的存在会迫使列变为浮点类型以适应它。

使用 `errors='coerce'` 通常是一个很好的第一步，因为它会为有效数字执行转换，并将有问题条目标记为缺失数据（`NaN`）。然后你可以决定如何处理这些 `NaN` 值（例如，检查原始数据，进一步清洗字符串，或使用第 2 章中介绍的填充技术）。

### 选择整数还是浮点数

转换时，你应该目标是整数还是浮点数？

- 如果你的数据表示测量值、百分比或任何可以包含小数部分的值，请使用**浮点数**。当转换过程中可能引入 `NaN` 值时，它也是默认选择。
- 如果你的数据表示计数或整数，*并且*你确定清洗后没有小数部分或缺失值，则目标是**整数**。在 pandas 中，如果你希望即使存在 NaN 也得到整数，可能需要在 `pd.to_numeric` 之后额外一步（例如使用 `.astype(pd.Int64Dtype())`，它支持整数和缺失值）。对于入门来说，当存在 NaN 时，让 pandas 默认为浮点数通常是最简单的。

正确地将数据转换为数字类型是基础一步。它使你能够执行计算、比较和定量分析，让你更接近于从数据中获取有意义的结论。请记住，在转换前后始终检查数据类型（`.dtype`），以确保过程按预期进行。

## 参考资料

- [pandas.to_numeric](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html) — pandas development team (2024)
  `pd.to_numeric`函数的官方文档，详细说明其用法、参数（如`errors='coerce'`）和行为。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用pandas进行数据操作的综合指南，包含数据类型转换和缺失数据处理的章节。
