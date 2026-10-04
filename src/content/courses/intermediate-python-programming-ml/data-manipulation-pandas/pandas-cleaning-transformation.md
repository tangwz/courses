---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "pandas-cleaning-transformation"
sourceId: 2153
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-cleaning-transformation"
title: "数据清洗与数据转换技巧"
description: "应用函数、映射值、重命名列以及执行其他数据清洗和转换任务。"
order: 5
plots: ["plots/2153-0.json"]
sourceHash: "f418ebcd9bc3231eede4273a81808475f8bc38c663944a8317a6fdd91780ce63"
sourceCorrections: []
---

调整数据集的结构和内容是数据准备的主要任务之一。数据通常包含不一致之处，需要调整格式，或者需要从现有数据派生出新特征。这里介绍 Pandas 中用于清洗和转换数据的基本方法，使数据适合进行分析和作为机器学习 (machine learning)模型的输入。

我们将着重介绍应用函数、修改列名、处理重复项、替换值、对连续数据进行分箱、操作字符串数据以及转换数据类型。

### 对数据应用函数

Pandas 提供了灵活的方式来对数据应用函数。`.apply()` 方法在对 DataFrame 的一个轴（行或列）或 Series 中的每个元素应用函数时特别有用。

假设我们有一个包含摄氏温度数据的 DataFrame `df`，我们想将其转换为华氏温度。

```python
import pandas as pd
import numpy as np

data = {'Location': ['City A', 'City B', 'City C'],
        'Temp_Celsius': [25.0, 18.5, 30.2]}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# 定义一个转换函数
def celsius_to_fahrenheit(c):
  return (c * 9/5) + 32

# 对 'Temp_Celsius' 列（一个 Series）应用此函数
df['Temp_Fahrenheit'] = df['Temp_Celsius'].apply(celsius_to_fahrenheit)

print("\nDataFrame with Fahrenheit:")
print(df)

# 使用 lambda 函数完成相同任务
df['Temp_Fahrenheit_lambda'] = df['Temp_Celsius'].apply(lambda c: (c * 9/5) + 32)

print("\nDataFrame with Fahrenheit (using lambda):")
print(df)
```

您也可以将 `.apply()` 用于 DataFrame 的行或列。例如，如果我们有每行的最小/最大值，可以计算温度范围：

```python
# 包含最小/最大温度的示例 DataFrame
data_range = {'Location': ['City A', 'City B', 'City C'],
              'Min_Temp_C': [15.0, 12.5, 22.8],
              'Max_Temp_C': [30.0, 21.0, 35.5]}
df_range = pd.DataFrame(data_range)

print("\nDataFrame with Temp Ranges:")
print(df_range)

# 跨行应用函数 (axis=1)
df_range['Temp_Range'] = df_range.apply(lambda row: row['Max_Temp_C'] - row['Min_Temp_C'], axis=1)

print("\nDataFrame with Calculated Range:")
print(df_range)
```

虽然 `.apply()` 灵活，但对于整列上的简单算术或逻辑操作，使用 NumPy 函数或 Pandas 内置方法的向量 (vector)化操作通常更快。当需要按行或按列应用更复杂的自定义逻辑时，使用 `.apply()`。

对于 Series 上的逐元素操作，`.map()` 很方便，尤其是在根据字典替换值或应用简单函数时。对于整个 DataFrame 上的逐元素操作，存在 `.applymap()`，但其性能通常不如 `.apply()` 或向量化 (quantization)方法。

### 重命名列和索引

清晰一致的列名对代码可读性和数据理解很重要。`.rename()` 方法是更改列名或索引标签的标准方法。

```python
# 接着使用温度 DataFrame
print("\nOriginal Columns:", df.columns)

# 使用字典重命名特定列
df_renamed = df.rename(columns={'Temp_Celsius': 'Celsius', 'Temp_Fahrenheit': 'Fahrenheit'})

print("\nRenamed Columns:", df_renamed.columns)
print(df_renamed)

# 重命名也可以在原地完成
# df.rename(columns={'Temp_Celsius': 'Celsius'}, inplace=True)
```

您可以使用 `.rename()` 中的 `index` 参数 (parameter)类似地重命名索引标签。如果您需要将一列设为 DataFrame 的索引，请使用 `.set_index('ColumnName')`。

### 处理重复项

重复的行会使分析和模型训练产生偏差。Pandas 使识别和删除它们变得简单。

```python
data_duplicates = {'col1': ['A', 'B', 'C', 'A', 'B', 'D'],
                   'col2': [1, 2, 3, 1, 2, 4]}
df_dup = pd.DataFrame(data_duplicates)

print("\nDataFrame with Duplicates:")
print(df_dup)

# 检查重复行（返回布尔型 Series）
print("\nAre rows duplicates?")
print(df_dup.duplicated())

# 默认保留第一次出现的重复项
print("\nKeeping first duplicate:")
print(df_dup.drop_duplicates(keep='first'))

# 保留最后一次出现的重复项
print("\nKeeping last duplicate:")
print(df_dup.drop_duplicates(keep='last'))

# 删除所有重复行
print("\nDropping all duplicates:")
print(df_dup.drop_duplicates(keep=False))

# 添加另一列
df_dup['col3'] = [10, 20, 30, 15, 25, 40]
print("\nDataFrame with added column:")
print(df_dup)
# 根据列的子集检查重复项
print("\nDuplicates based on 'col1' and 'col2':")
print(df_dup.drop_duplicates(subset=['col1', 'col2'], keep='first'))
```

`keep` 参数 (parameter)控制保留哪个重复项（如果有）。`subset` 允许您仅根据特定列定义重复项。

### 替换值

通常，您需要在数据中替换特定值。这可能是为了更正数据输入错误、标准化类别或将数字代码映射到标签。`.replace()` 方法对此用途广泛。

```python
s = pd.Series([1, 2, 3, 1, 4, 5, -99]) # -99 可能是缺失值的占位符
print("\nOriginal Series:")
print(s)

# 替换单个值
print("\nReplace -99 with NaN:")
print(s.replace(-99, np.nan))

# 使用列表替换多个值
print("\nReplace 1 and 2 with 0:")
print(s.replace([1, 2], 0))

# 使用字典进行映射替换
print("\nReplace 1 with 100, 2 with 200:")
print(s.replace({1: 100, 2: 200}))

# 也可以使用 regex=True 进行字符串模式匹配
df_text = pd.DataFrame({'codes': ['A-101', 'B-202', 'C-303', 'A-101-extra']})
print("\nDataFrame with Codes:")
print(df_text)
print("\nReplace prefix using regex:")
# 将字符串开头 (^) 的 'A-'、'B-'、'C-' 替换为 ''
print(df_text['codes'].replace(to_replace=r'^[ABC]-', value='', regex=True))
```

`.replace()` 对于 Series 或 DataFrame 中的目标值替换功能强大。

### 离散化和分箱

有时，将连续的数值数据转换为离散的箱或类别会很有用。这个过程称为离散化或分箱，可以帮助简化模型或捕获非线性效应。Pandas 提供了 `pd.cut` 和 `pd.qcut`。

`pd.cut` 根据值范围将数据分段到箱中。您可以指定箱的数量（等宽）或提供明确的箱边缘。

```python
ages = pd.Series([22, 35, 58, 19, 41, 65, 30, 28, 50, 72])
print("\nOriginal Ages:")
print(ages)

# 将年龄分到 4 个等宽的箱中
age_bins = pd.cut(ages, bins=4)
print("\nAges binned into 4 intervals:")
print(age_bins)
print("\nValue counts per bin:")
print(age_bins.value_counts())

# 定义自定义箱边缘和标签
bin_edges = [0, 18, 35, 60, 100]
bin_labels = ['Youth', 'Young Adult', 'Middle Aged', 'Senior']
age_bins_labeled = pd.cut(ages, bins=bin_edges, labels=bin_labels, right=False) # right=False 表示箱为 [a, b) 区间
print("\nAges with custom bins and labels:")
print(age_bins_labeled)
print("\nValue counts for custom bins:")
print(age_bins_labeled.value_counts())
```

`pd.qcut` 根据分位数将数据分段，这意味着每个箱将包含大约相同数量的数据点。

```python
# 将年龄分到 4 个基于分位数的箱（四分位数）中
age_qbins = pd.qcut(ages, q=4)
print("\nAges binned into quartiles:")
print(age_qbins)
print("\nValue counts per quartile bin:")
print(age_qbins.value_counts())

# 带有标签的分位数箱
qbin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
age_qbins_labeled = pd.qcut(ages, q=4, labels=qbin_labels)
print("\nAges with quartile labels:")
print(age_qbins_labeled)
print("\nValue counts per labeled quartile:")
print(age_qbins_labeled.value_counts())
```



![年龄组分布（自定义箱）](plots/2153-0.json)



> 个体在自定义年龄组中的分布。

分箱是一种常见的特征工程方法，用于在将数据输入到某些类型的机器学习 (machine learning)模型之前。

### 处理字符串数据

文本数据通常需要大量清洗。Pandas Series 有一个特殊的 `.str` 访问器，它提供向量 (vector)化字符串操作函数，类似于 Python 的内置字符串方法，但能在整个 Series 上高效操作。

```python
# 包含杂乱字符串数据的示例 Series
text_data = pd.Series(['  Apple ', 'banana', ' Orange', ' GRAPES! ', 'Pineapple '])
print("\nOriginal Text Series:")
print(text_data)

# 转换为小写
print("\nLowercase:")
print(text_data.str.lower())

# 移除前后空白
print("\nStripped Whitespace:")
print(text_data.str.strip())

# 链式操作
print("\nStripped and Uppercase:")
print(text_data.str.strip().str.upper())

# 检查字符串是否包含模式
print("\nContains 'apple' (case-insensitive):")
print(text_data.str.lower().str.contains('apple'))

# 替换字符串的一部分
print("\nReplace '!' with '':")
print(text_data.str.replace('!', '', regex=False)) # 对于字面替换使用 regex=False

# 分割字符串
print("\nSplit by space (yields lists):")
print(text_data.str.strip().str.split(' '))

# 使用正则表达式提取模式（需要捕获组）
product_codes = pd.Series(['SKU-123-A', 'SKU-456-B', 'ITEM-789-C', 'SKU-000-D'])
print("\nProduct Codes:")
print(product_codes)
print("\nExtract numeric part:")
# 提取 'SKU-' 后的一位或多位数字 (\d+)
print(product_codes.str.extract(r'SKU-(\d+)-'))
```

`.str` 访问器对于清洗以文本形式存储的分类特征、为自然语言处理准备文本，或从非结构化字符串中提取结构化信息很重要。

### 类型转换

确保数据以正确的数据类型（整数、浮点数、布尔值、类别、日期时间等）存储，对于内存效率和操作正确性很重要。您可以使用 `.dtypes` 属性检查类型，并使用 `.astype()` 方法转换类型。

```python
df_types = pd.DataFrame({'ID': ['1', '2', '3', '4'],
                         'Value': ['10.5', '20.1', '30.9', '40.0'],
                         'Category': ['A', 'B', 'A', 'C'],
                         'Count': [5, 10, 15, 20]})

print("\nOriginal DataFrame with mixed types:")
print(df_types)
print("\nOriginal Data Types:")
print(df_types.dtypes)

# 将 ID 从对象（字符串）转换为整数
df_types['ID'] = df_types['ID'].astype(int)

# 将 Value 从对象（字符串）转换为浮点数
df_types['Value'] = df_types['Value'].astype(float)

# 将 Category 转换为专用的 'category' 类型（对于少量独特字符串内存高效）
df_types['Category'] = df_types['Category'].astype('category')

# 将 Count（已为 int）转换为浮点数
df_types['Count'] = df_types['Count'].astype(float)

print("\nDataFrame after Type Conversion:")
print(df_types)
print("\nNew Data Types:")
print(df_types.dtypes)

# 处理转换过程中可能出现的错误
s_error = pd.Series(['1', '2', 'three', '4'])
print("\nSeries with non-numeric value:")
print(s_error)

# 使用 errors='coerce' 会将无法转换的值变为 NaT（日期时间）或 NaN（数值）
numeric_s = pd.to_numeric(s_error, errors='coerce')
print("\nConverted to numeric with coercion:")
print(numeric_s)
```

使用合适的数据类型，特别是对于低基数字符串使用 `category` 类型，或在适用时使用 `int` 子类型（`int8`、`int16` 等），可以大幅减少 DataFrame 的内存占用。`pd.to_numeric`、`pd.to_datetime` 和 `pd.to_timedelta` 函数提供更专门的转换选项，尤其是 `errors='coerce'` 用于优雅地处理无法转换的值。

这些清洗和转换方法构成了数据准备过程的核心部分。熟练掌握它们，您就能将原始数据重塑为干净、结构化的格式，为后续的分析阶段做好准备，例如分组、聚合和合并，我们将在后面介绍这些内容。

## 参考资料

- [Pandas User Guide: Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html) — pandas development team (2024)
  官方文档，涵盖了核心 DataFrame 和 Series 操作，包括应用函数、重命名、处理重复项、替换值和类型转换。
- [Pandas User Guide: Working with text data](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html) — pandas development team (2024)
  官方文档，详细介绍了通过 .str 访问器实现的向量化字符串方法，用于高效的文本操作。
- [Pandas User Guide: Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) — pandas development team (2024)
  官方文档，解释了分类数据类型以及使用 pd.cut 和 pd.qcut 对连续数据进行分箱的方法。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本基础性著作，由 Pandas 的创建者撰写，提供了使用 Pandas 进行数据处理和准备的全面指导。
