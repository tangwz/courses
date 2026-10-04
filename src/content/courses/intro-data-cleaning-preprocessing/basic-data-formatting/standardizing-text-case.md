---
course: "intro-data-cleaning-preprocessing"
chapter: "basic-data-formatting"
lesson: "standardizing-text-case"
sourceId: 4028
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-5-basic-data-formatting/standardizing-text-case"
title: "统一文本大小写（大写/小写）"
description: "使用函数统一列中所有文本项的大小写。"
order: 2
plots: []
sourceHash: "42cd6daa1a9e9243a806a84de7e6bcf59c6c14f97282eb73d1ecd009d15e080c"
sourceCorrections: []
---

数据分析的可靠性离不开一致性。在文本数据中，大小写变化是最常见的不一致问题之一。计算机非常刻板；对程序而言，'New York'、'new york'和'NEW YORK'是三个完全不同的字符串。如果您尝试对包含此类变化的列进行筛选、分组或计数，将得到不准确的结果，因为计算机不会将它们识别为表示同一事物。

考虑一列包含城市名称的数据：

| 城市 |
| --- |
| London |
| london |
| New York |
| new york |
| LONDON |
| San Francisco |

如果您被要求统计“London”出现的次数，简单地搜索“London”只会找到第一个条目。按城市分组会把“London”、“london”和“LONDON”视为不同的类别。这显然错误地呈现了原始数据。

### 通过大小写转换实现统一

解决方法很简单：将列中所有文本项转换为单一、统一的大小写。您主要有两种选择：

1. **转换为小写：** 将每个字符串中的所有字符转换为其对应的小写形式。
2. **转换为大写：** 将每个字符串中的所有字符转换为其对应的大写形式。

#### 转换为小写

这需要应用一个将每个字母都转换为小写的函数。将此方法应用于我们的示例`City`列，结果如下：

| 城市 |
| --- |
| london |
| london |
| new york |
| new york |
| london |
| san francisco |

现在，如果您计数“london”的出现次数，就会正确地得到三个条目。按城市分组也能按预期进行。将文本转换为小写通常是处理一般文本数据的首选方法，因为它能很好地处理大多数情况，并且与文本的常见书写方式保持一致。

#### 转换为大写

或者，您也可以将所有内容转换为大写：

| 城市 |
| --- |
| LONDON |
| LONDON |
| NEW YORK |
| NEW YORK |
| LONDON |
| SAN FRANCISCO |

这也实现了统一。现在计数“LONDON”也能正确地找到三个条目。大写转换对于规范化代码（如国家代码 'US'、'GB'、'CA'）或当您希望条目在视觉上更醒目时很有用。

### 小写与大写之间的选择

您应该选择哪种方法？

- **小写：** 通常建议用于自由文本字段、类别、名称和描述。它在视觉上更柔和，对分析而言通常感觉更自然。大多数文本处理工作都能有效利用小写数据。
- **大写：** 适用于标识符、首字母缩写或通常采用大写约定的代码（例如，'SKU123'、'NY'）。

最重要的一点是，**选择一种方法并始终如一地应用于**整个列。两种方法都能解决不一致问题。

### 如何实现大小写规范化

大多数数据处理工具和编程库都提供简单的大小写转换函数。例如，如果您在使用pandas DataFrame（Python中常用的数据分析结构）处理数据，可以直接在列上使用字符串方法（通常称为Series）。

假设您的数据在一个名为`df`的DataFrame中，且列名为“City”：

```python
# 确保已导入pandas，通常命名为pd
import pandas as pd

# 假设df是包含“City”列的DataFrame

# 将“City”列转换为小写：
df['City'] = df['City'].str.lower()

# 或者，将“City”列转换为大写：
# df['City'] = df['City'].str.upper()

# 显示更新后列的前几行
print(df.head())
```

在这个使用pandas的Python例子中：

- `df['City']` 选择了您想要修改的列。
- `.str` 访问了该列的特殊字符串处理方法。
- `.lower()` 是将列中每个条目转换为小写的函数。
- `.upper()` 则会将每个条目转换为大写。
- 结果会替换原始的“City”列，确保更改保存回您的DataFrame。

即使您使用不同的工具（如电子表格或SQL数据库），通常也有类似的函数（`LOWER()`、`UPPER()`）来执行这些大小写转换。

统一文本大小写是数据清洗中一个基础的步骤。这是一个快速简便的方法，可以消除常见的错误源，并确保您后续的分析、分组或合并操作正确执行。

## 参考资料

- [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用Python中pandas库进行数据操作和清洗的实用指南。
- [Working with Text Data](https://pandas.pydata.org/docs/user_guide/text.html) — pandas development team (2024)
  pandas官方文档，展示了包括大小写转换函数在内的字符串处理方法。
- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, and Jian Pei (2011)
  Publisher: Elsevier
  一本涵盖数据预处理，包括数据清洗和转换技术的综合性教材。
- [String Functions and Operators](https://www.postgresql.org/docs/current/functions-string.html) — PostgreSQL Global Development Group (2024)
  PostgreSQL官方文档，详细介绍了SQL字符串函数，例如用于大小写标准化的LOWER()和UPPER()。
