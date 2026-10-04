---
course: "intro-eda-course"
chapter: "data-loading-inspection-cleaning"
lesson: "understanding-data-types"
sourceId: 1139
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/understanding-data-types"
title: "理解数据类型（dtypes）"
description: "学习 Pandas 中不同的数据类型，以及如何检查和解释每列的数据类型。"
order: 3
plots: []
sourceHash: "9e4cda6ad64467b5b48ee1a97507dc9b6042e378a53a0cc1e744b4ae3d3d3b0d"
sourceCorrections: []
---

加载数据并使用 `.head()`、`.tail()` 和 `.shape()` 等方法初窥数据后，下一步是弄清每列中存储的数据的*种类*。在 Pandas 中，此信息由 DataFrame 中每个 Series（列）关联的数据类型，即 `dtype` 捕获。了解数据类型非常重要，因为它决定了你如何操作、分析和可视化数据。例如，对文本数据执行数学运算没有意义，通常会导致错误。

### 什么是数据类型（dtypes）？

Pandas DataFrame 中的每一列都有特定的数据类型。Pandas 主要使用 NumPy 数据类型，并在此基础上添加了一些扩展，例如用于分类或日期时间数据的特定类型。理解这些类型对于有效分析和高效内存使用很重要。

你会遇到的常见 `dtypes` 包括：

- `object`：最常见的类型。通常表示文本或字符串数据。但它有时也可能表示混合数据类型（例如，字符串和数字）的列，这通常表明存在需要检查的数据质量问题。
- `int64`：整数（整型数字）。适用于计数、标识符或离散数值。（`int8`、`int16`、`int32` 也可用于较小的整数范围，以节省内存）。
- `float64`：浮点数（带小数的数字）。用于测量、百分比或连续数值。（`float32` 是精度较低的替代方案）。
- `bool`：布尔值，表示 `True` 或 `False`。
- `datetime64[ns]`：日期和时间值的特定类型。一旦时间序列数据被正确确定为这种 `dtype`，Pandas 就会提供强大的工具来处理它。`[ns]` 表示纳秒精度。
- `timedelta[ns]`：表示两个日期时间值之间的时间长度或差值。
- `category`：一种专门的 Pandas 类型，用于高效表示分类数据，尤其是在唯一值数量有限（例如，'Low'、'Medium'、'High'）时。

### 检查 DataFrame 中的数据类型

Pandas 提供了直接的方法来检查列的 `dtypes`。

1. **使用 `.dtypes` 属性**：此属性返回一个 Series，其索引是列名，值是该列的数据类型。

   ```python
   # 假设 'df' 是你的 DataFrame
   print(df.dtypes)
   ```

   这可能会产生类似以下内容的输出：

   ```
   PassengerId      int64
   Survived         int64
   Pclass           int64
   Name            object
   Sex             object
   Age            float64
   SibSp            int64
   Parch            int64
   Ticket          object
   Fare           float64
   Cabin           object
   Embarked        object
   dtype: object
   ```
2. **使用 `.info()` 方法**：此方法提供更全面的汇总，包括每列的非空条目数*及其*数据类型，以及内存使用情况。这在初步检查时通常更有用，因为它结合了类型信息和缺失值计数。

   ```python
   # 假设 'df' 是你的 DataFrame
   df.info()
   ```

   输出可能看起来与此类似：

   ```
   <class 'pandas.core.frame.DataFrame'>
   RangeIndex: 891 entries, 0 to 890
   Data columns (total 12 columns):
    #   Column       Non-Null Count  Dtype
   ---  ------       --------------  -----
    0   PassengerId  891 non-null    int64
    1   Survived     891 non-null    int64
    2   Pclass       891 non-null    int64
    3   Name         891 non-null    object
    4   Sex          891 non-null    object
    5   Age          714 non-null    float64
    6   SibSp        891 non-null    int64
    7   Parch        891 non-null    int64
    8   Ticket       891 non-null    object
    9   Fare         891 non-null    float64
    10  Cabin        204 non-null    object
    11  Embarked     889 non-null    object
   dtypes: float64(2), int64(5), object(5)
   memory usage: 83.7+ KB
   ```

### 数据类型为何重要

理解已分配的 `dtypes` 不仅仅是形式，它直接影响你的分析：

- **合适的运算**：`dtype` 决定了哪些运算有效。你可以计算 `int64` 或 `float64` 列的平均值（`.mean()`），但通常不能计算 `object` 列的平均值，除非它包含需要先转换的数值数据。可视化也取决于类型；直方图适用于数值数据，而条形图则适用于表示类别的分类或 `object` 类型。
- **内存使用**：选择正确的 `dtype` 可以显著影响内存消耗。一个只包含字符串的 `object` 列比包含相同数据的 `category` 列消耗更多的内存，尤其是在有许多重复字符串值的情况下。同样，如果你的整数值总是很小（例如，在 0 到 100 之间），使用 `int8` 而不是默认的 `int64` 可以在大型数据集中节省大量内存。
- **找出数据问题**：通常，不正确的 `dtype` 分配表明存在潜在问题。如果你期望的某列是数值类型，但它显示为 `object`，则可能因为它包含非数值字符（如货币符号 `$`、逗号 `,` 或占位符文本如 'Unknown'）。及早发现这一点会促使你在分析进行之前采取必要的数据清理步骤。例如，如果可能存在小数年龄，`Age` 列显示为 `float64` 可能在意料之中，但 `PassengerId` 显示为 `float64` 则不寻常，需要检查。

### 处理不正确或低效的数据类型

在 EDA 期间，你经常会发现有些列未分配最合适的 `dtype`。Pandas 提供了 `.astype()` 方法将列转换为不同的类型。

- **将对象转换为数字**：如果某列包含存储为字符串的数值数据（例如，'1,200'、'\$50.75'），你需要首先清理字符串（删除逗号、货币符号），然后使用 `.astype(int)` 或 `.astype(float)` 进行转换。

  ```python
  # 示例：转换存储为 object 类型（例如，'$1,234.56'）的价格列
  # df['Price'] = df['Price'].str.replace('[$,]', '', regex=True).astype(float)
  ```
- **转换为日期时间**：包含存储为字符串的日期或时间信息的列理想情况下应转换为 `datetime64`。Pandas 的 `pd.to_datetime()` 函数对此非常有效。

  ```python
  # 示例：转换日期字符串列
  # df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # errors='coerce' 将解析错误转换为 NaT（非时间）
  ```
- **转换为类别**：如果 `object` 甚至 `int` 列表示具有相对较少唯一值的不同类别（例如，'Low'、'Medium'、'High' 或调查响应 1-5），将其转换为 `category` `dtype` 通常对内存和性能有益，并清晰地向分析库表明其分类性质。

  ```python
  # 示例：将 'Pclass'（乘客等级）转换为 category
  # df['Pclass'] = df['Pclass'].astype('category')
  ```

在数据加载后，使用 `.dtypes`、`.info()` 和 `.astype()` 检查并可能更正数据类型是一个标准流程。它能确保你的 DataFrame 结构正确，以便进行后续的数据清理、分析和可视化步骤，从而防止错误并实现高效计算。

## 参考资料

- [User Guide: Data types (dtypes)](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) — pandas development team (2024)
  官方文档，涵盖了Pandas数据类型、其用法和转换方法。理解Pandas中dtypes的首要参考资料。
- [NumPy Reference: Data types](https://numpy.org/doc/stable/reference/arrays.dtypes.html) — NumPy Contributors (2024)
  NumPy官方文档，提供了NumPy数组数据类型的核心原理和详细信息，这些类型构成了Pandas dtypes的基础。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者的著作，全面涵盖了数据类型、数据加载、检查和清洗的实践操作。
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  本书讨论了不同数据类型在统计分析和数据建模准备中的实际重要性和影响。
