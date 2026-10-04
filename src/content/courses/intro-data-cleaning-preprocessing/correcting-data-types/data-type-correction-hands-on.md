---
course: "intro-data-cleaning-preprocessing"
chapter: "correcting-data-types"
lesson: "data-type-correction-hands-on"
sourceId: 4022
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-4-correcting-data-types/data-type-correction-hands-on"
title: "数据类型修正：动手实践"
description: "应用数据类型转换技术到一个包含混合或错误类型数据的示例数据集。"
order: 8
plots: ["plots/4022-0.json"]
sourceHash: "e173a49f0211b5dcd7c4f892a6dcf694954b819bb81d95b8be31a19d357b3f54"
sourceCorrections: []
---

让我们将本章学到的知识付诸实践。我们了解了不同的数据类型，以及为什么确保列具有正确类型对分析很重要。现在，我们将介绍一个常见情境：你收到混合或错误类型的数据，并应用所讨论的技术来修正它们。

我们将使用一个代表产品订单的小数据集。设想这些数据来自不同来源的合并或手动录入，导致不一致。我们将使用 `pandas` 库，一个用于Python数据操作的标准工具。如果你之前没有使用过它，`pandas` 提供了名为 DataFrame 的结构（类似于表格）和名为 Series 的结构（类似于列），以及用于处理它们的功能。

### 设置示例

首先，让我们创建我们的示例 DataFrame。在实际项目中，你会从文件（如 CSV）加载数据，但为了清晰起见，我们将在此直接定义它。

```python
import pandas as pd
import numpy as np # 我们需要 numpy 来处理 NaN

data = {
    'OrderID': [1, 2, 3, 4, 5],
    'Product': ['Apple', 'Banana', 'Milk', 'Bread', 'Apple'],
    'Category': ['Fruit', 'Fruit', 'Dairy', 'Bakery', 'Fruit'],
    'Quantity': ['10', '5', '2', '1', '5'], # 以字符串形式存储
    'Unit_Price': ['$0.50', '$0.30', '$3.5O', '$2.50', '$0.50'], # 带有符号和错误的字符串
    'Order_Date': ['01/15/2023', '01/16/2023', '01/16/2023', '2023-01-17', '01/18/2023'] # 以字符串形式存储的混合格式
}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

### 查看初始数据类型

在进行更改之前，让我们检查 `pandas` 为每列分配的当前数据类型。`info()` 方法对此非常有用，提供类型和非空计数。

```python
print("\n初始数据类型:")
df.info()
```

你可能会看到类似如下的输出：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   OrderID     5 non-null      int64
 1   Product     5 non-null      object
 2   Category    5 non-null      object
 3   Quantity    5 non-null      object
 4   Unit_Price  5 non-null      object
 5   Order_Date  5 non-null      object
dtypes: int64(1), object(5)
memory usage: 368.0+ bytes
```

请注意，`OrderID` 被正确识别为整数 (`int64`)，但 `Product`、`Category`、`Quantity`、`Unit_Price` 和 `Order_Date` 都被列为 `object` 类型。虽然 `Product` 和 `Category` 为 `object` 类型（在 `pandas` 中通常意味着字符串）可能一开始可以接受，但 `Quantity`、`Unit_Price` 和 `Order_Date` 肯定需要修正才能进行准确分析（例如，计算总成本或分析随时间变化的销售情况）。

### 转换为数值类型

我们来处理 `Quantity` 和 `Unit_Price`。

**1. Quantity:** 此列包含以字符串形式存储的数字。我们可以使用 `pd.to_numeric` 直接转换它。

```python
df['Quantity'] = pd.to_numeric(df['Quantity'])

print("\n转换 Quantity 后的数据类型:")
print(df.dtypes)
```

输出现在应该显示 `Quantity` 为 `int64` 或 `float64`。

**2. Unit\_Price:** 此列比较棘手。它包含美元符号 (`$`) 和可能存在的其他非数字字符（例如 '\$3.5O' 中的字母 'O' 而不是零）。

首先，我们需要清理字符串：删除 `$` 符号。我们可以使用 `pandas` Series 中可用于字符串数据的 `.str.replace()` 方法。

```python
df['Unit_Price'] = df['Unit_Price'].str.replace('$', '', regex=False)

print("\n删除 '$' 后的 Unit_Price:")
print(df['Unit_Price'])
```

现在，尝试转换为数值类型。'\$3.5O' 条目会发生什么？

```python
# 尝试直接转换（可能会导致错误）
# df['Unit_Price'] = pd.to_numeric(df['Unit_Price']) # 取消注释此行将引发错误

# 使用 errors='coerce' 处理无效值
df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')

print("\n使用 errors='coerce' 转换后的 Unit_Price:")
print(df['Unit_Price'])

print("\n转换 Unit_Price 后的数据类型:")
print(df.dtypes)
```

使用 `errors='coerce'` 会告诉 `pandas` 将任何无法转换为数字的值替换为 `NaN`（非数字），这表示缺失数据。你会在原来 '\$3.5O' 的位置看到 `NaN`。这很有用，因为它避免了整个转换失败。然后你需要决定如何处理这个新的缺失值，也许是调查来源，或者使用第2章中讨论的插补技术。目前，我们将其保留为 `NaN`。`Unit_Price` 列现在应该为 `float64` 类型。

### 转换为日期时间类型

`Order_Date` 列包含以字符串形式存储的日期，甚至有混合格式。`pandas` 有一个实用的 `pd.to_datetime` 函数，通常可以自动识别格式。

```python
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

print("\n转换 Order_Date 后的数据类型:")
print(df.dtypes)
```

如果格式更复杂或不明确，你可能需要使用 `pd.to_datetime` 中的 `format` 参数 (parameter)来指定格式字符串。例如，如果所有日期都像 '01/15/2023'，你可以使用 `pd.to_datetime(df['Order_Date'], format='%m/%d/%Y')`。然而，`pandas` 在推断常见格式方面做得很好，如这里所示。`Order_Date` 列现在应该为 `datetime64[ns]` 类型。

### 转换为分类类型

`Category` 列包含代表不同组的文本（'Fruit'、'Dairy'、'Bakery'）。虽然将其保留为 `object`（字符串）类型也可以，但转换为 `category` 数据类型可以更节省内存，有时还能加快操作速度，尤其是在存在许多重复值（例如 'Fruit' 多次出现）的情况下。

```python
df['Category'] = df['Category'].astype('category')

print("\n所有转换后的最终数据类型:")
df.info()
```

`info()` 输出现在反映了这些变化：`Quantity` 是数值型，`Unit_Price` 是数值型（浮点型），其中一个缺失值是由强制转换引入的，`Order_Date` 是日期时间型，`Category` 是分类型。

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   OrderID     5 non-null      int64
 1   Product     5 non-null      object # 仍为 object（字符串）类型，这没有问题
 2   Category    5 non-null      category
 3   Quantity    5 non-null      int64
 4   Unit_Price  4 non-null      float64 # 注意：由于强制转换，非空计数为 4
 5   Order_Date  5 non-null      datetime64[ns]
dtypes: category(1), datetime64[ns](1), float64(1), int64(2), object(1)
memory usage: 593.0+ bytes # 由于转换为 category 类型，内存使用量可能会略有减少
```

以下是我们修正前后数据类型的可视化比较：



![修正前后数据类型分布](plots/4022-0.json)



> 在对 DataFrame 应用类型修正前后，每种数据类型存在的列数。

### 实践步骤总结

在此动手实践部分，我们：

1. **检查**了使用 `df.info()` 得到的初始数据类型。
2. **转换**了字符串列 (`Quantity`) 为数值类型，使用 `pd.to_numeric()`。
3. **清理**了字符串列 (`Unit_Price`)，通过 `.str.replace()` 移除了不需要的字符 (`$`)。
4. **转换**了清理后的列为数值类型，使用 `pd.to_numeric(errors='coerce')` 优雅地处理无效条目，将其变为 `NaN`。
5. **转换**了包含日期信息的字符串列 (`Order_Date`) 为正确的日期时间格式，使用 `pd.to_datetime()`。
6. **转换**了具有重复值的字符串列 (`Category`) 为节省内存的 `category` 类型，使用 `.astype('category')`。
7. **验证**了最终的数据类型。

修正数据类型是数据预处理中的一个基本步骤。这能保证你的数据得到恰当存储，防止错误发生，并能在后续步骤中进行准确的计算、比较和分析。通过应用这些技术，你的数据便可用于可视化、统计分析或机器学习 (machine learning)模型训练。

## 参考资料

- [pandas User Guide: Working with data types](https://pandas.pydata.org/docs/user_guide/dtypes.html) — The pandas development team (2023)
  Publisher: The pandas development team
  pandas官方文档中关于DataFrame和Series数据类型的章节，涵盖如何检查和更改数据类型。对理解类型纠正至关重要。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本关于使用Python和pandas进行数据处理的入门书籍。其中关于数据加载、清洗和准备的章节提供了处理各种数据类型和常见预处理任务的实用见解。
- [Data Cleaning: A Practical Guide](https://www.morganclaypool.com/doi/10.2200/S00913ED1V01Y201903DMK016) — Ihab Ilyas, Xu Chu (2019)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S00913ED1V01Y201903DMK016](https://doi.org/10.2200/S00913ED1V01Y201903DMK016)
  系统概述了数据清洗技术，包括数据类型和一致性的重要性，其范围超越了特定的编程库。
