---
course: "intro-data-cleaning-preprocessing"
chapter: "basic-data-formatting"
lesson: "formatting-practice"
sourceId: 4036
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-5-basic-data-formatting/formatting-practice"
title: "格式化实操"
description: "将文本规范化和简单格式化方法应用于示例数据。"
order: 6
plots: []
sourceHash: "c7875ba7f33413cdc9847362785ed92d600b650243b9cb1fe37b6e2065989355"
sourceCorrections: []
---

让我们将本章介绍的方法用于实际。你已经学习了如何规范文本大小写、去除多余空格、进行简单替换以及处理基本的单位转换。现在，我们将使用一个数据集来演练，以巩固这些技能。运用这些方法是为数据分析做准备的常见环节。

假设我们有一个小型数据集，包含产品信息，它可能来自不同来源或通过手动录入。通常情况下，这类数据会存在需要处理的不一致。

这是我们初始数据的简化视图：

| ID | 产品名称 | 产地 | 重量字符串 |
| --- | --- | --- | --- |
| 1 | ' Super Widget ' | 'usa' | '5 kg' |
| 2 | ' Power Gadget ' | 'USA' | '11 lbs' |
| 3 | 'Flexo Thing' | ' U.S.A. ' | '2 kg' |
| 4 | 'Super Widget' | 'Canada' | '7 kg' |
| 5 | 'Power Gadget' | 'usa' | '15 lbs' |
| 6 | 'Flexo Thing ' | 'United States' | '2.1 kg' |

我们能看到几个问题：

- `ProductName` 含有开头/结尾的空格。
- `Origin` 的大小写不一致（'usa'、'USA'、'U.S.A.'、'United States'），并且有周边空格。
- `WeightStr` 既有千克（'kg'）也有磅（'lbs'），而且值以字符串形式与单位组合存储。

我们的目标是清理这些数据，使其统一并适合分析。我们假设使用Python中常用的数据处理工具pandas库，但这些方法广泛适用。

### 第一步：规范文本大小写和去除空格

首先，我们来处理 `Origin` 和 `ProductName` 列。我们想要统一的大小写（选择小写）并且没有多余的空格。

**处理 `Origin`：**

1. 将所有条目转换为小写。
2. 去除开头和结尾的空格。

```python
# 使用pandas的例子
# 转换为小写
df['Origin'] = df['Origin'].str.lower()
# 去除开头/结尾空格
df['Origin'] = df['Origin'].str.strip()
```

**处理 `ProductName`：**

1. 去除开头和结尾的空格。

```python
# 使用pandas的例子
df['ProductName'] = df['ProductName'].str.strip()
```

完成这些步骤后，我们的 `Origin` 和 `ProductName` 列看起来是这样：

| ID | 产品名称 | 产地 | 重量字符串 |
| --- | --- | --- | --- |
| 1 | 'Super Widget' | 'usa' | '5 kg' |
| 2 | 'Power Gadget' | 'usa' | '11 lbs' |
| 3 | 'Flexo Thing' | 'u.s.a.' | '2 kg' |
| 4 | 'Super Widget' | 'canada' | '7 kg' |
| 5 | 'Power Gadget' | 'usa' | '15 lbs' |
| 6 | 'Flexo Thing' | 'united states' | '2.1 kg' |

### 第二步：简单字符串替换

`Origin` 列改善了，但我们仍然有 'usa'、'u.s.a.' 和 'united states' 等不同形式。我们应该将它们规范化为一种格式，例如 'usa'。

我们可以为此使用字符串替换。

```python
# 使用pandas的例子
replace_map = {
    'u.s.a.': 'usa',
    'united states': 'usa'
}
df['Origin'] = df['Origin'].replace(replace_map)
```

现在，`Origin` 列保持一致：

| ID | 产品名称 | 产地 | 重量字符串 |
| --- | --- | --- | --- |
| 1 | 'Super Widget' | 'usa' | '5 kg' |
| 2 | 'Power Gadget' | 'usa' | '11 lbs' |
| 3 | 'Flexo Thing' | 'usa' | '2 kg' |
| 4 | 'Super Widget' | 'canada' | '7 kg' |
| 5 | 'Power Gadget' | 'usa' | '15 lbs' |
| 6 | 'Flexo Thing' | 'usa' | '2.1 kg' |

### 第三步：基本单位转换

`WeightStr` 列混合了单位（'kg' 和 'lbs'），并且数据以文本形式存储。为了进行计算，我们需要一个带有统一单位的数值列。让我们将所有单位转换为千克（kg）。

转换系数大约是 $1 \text{ 磅} = 0.453592 \text{ 千克}$。

这通常涉及几个子步骤：

1. 从单位（'kg' 或 'lbs'）中分离出数值。
2. 将数值部分转换为浮点数。
3. 识别单位为 'lbs' 的行并应用转换系数。
4. 将结果存储在一个新的数值列中，例如命名为 `Weight_kg`。

```python
# 使用pandas的例子（步骤）

# 1. 提取数值和单位（例如，使用字符串分割或正则表达式）
# 例子：'5 kg' -> (5.0, 'kg'), '11 lbs' -> (11.0, 'lbs')
# 将这些存储在临时列中，例如 'WeightValue' 和 'WeightUnit'
df[['WeightValue', 'WeightUnit']] = df['WeightStr'].str.extract(r'(\d+\.?\d*)\s*(\w+)')
df['WeightValue'] = df['WeightValue'].astype(float)

# 2. 定义转换系数
lbs_to_kg = 0.453592

# 3. 在需要的地方应用转换
# 创建新列，最初复制千克值
df['Weight_kg'] = df['WeightValue']
# 找到单位是 'lbs' 的行并应用转换
lbs_mask = df['WeightUnit'].str.lower() == 'lbs'
df.loc[lbs_mask, 'Weight_kg'] = df.loc[lbs_mask, 'WeightValue'] * lbs_to_kg

# 4. (可选) 删除临时/原始列
# df = df.drop(columns=['WeightStr', 'WeightValue', 'WeightUnit'])
```

应用转换后，我们的数据集看起来整洁很多，拥有了一个专门的数值列来表示以千克为单位的重量：

| ID | 产品名称 | 产地 | 重量字符串 | WeightValue | WeightUnit | Weight\_kg |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 'Super Widget' | 'usa' | '5 kg' | 5.0 | 'kg' | 5.000000 |
| 2 | 'Power Gadget' | 'usa' | '11 lbs' | 11.0 | 'lbs' | 4.989512 |
| 3 | 'Flexo Thing' | 'usa' | '2 kg' | 2.0 | 'kg' | 2.000000 |
| 4 | 'Super Widget' | 'canada' | '7 kg' | 7.0 | 'kg' | 7.000000 |
| 5 | 'Power Gadget' | 'usa' | '15 lbs' | 15.0 | 'lbs' | 6.803880 |
| 6 | 'Flexo Thing' | 'usa' | '2.1 kg' | 2.1 | 'kg' | 2.100000 |

如果需要，我们可以将 `Weight_kg` 列四舍五入到合理的小数位数，然后如果不再需要，可以删除中间列（`WeightStr`、`WeightValue`、`WeightUnit`）。

### 实操总结

通过这些步骤，我们应用了本章介绍的格式化方法：

- 规范了 `Origin` 列的文本大小写。
- 去除了 `ProductName` 和 `Origin` 开头/结尾的空格。
- 使用字符串替换统一了 `Origin` 列中的不同形式。
- 对重量数据进行了基本单位转换，创建了一个统一的数值型 `Weight_kg` 列。

我们的数据集现在大幅更一致，更易于进行分析或与其他数据合并。本实操练习说明了这些看似简单的格式化步骤如何成为构建可靠数据集的重要组成部分。

## 参考资料

- [Working with text data](https://pandas.pydata.org/docs/user_guide/text.html) — The pandas development team (2024)
  pandas官方字符串操作方法指南，有助于文本数据清洗和标准化。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本使用pandas进行数据操作的基础书籍，包括实用的数据清洗技巧。
