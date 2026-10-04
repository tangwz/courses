---
course: "intro-data-cleaning-preprocessing"
chapter: "correcting-data-types"
lesson: "identifying-incorrect-data-types"
sourceId: 4007
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-4-correcting-data-types/identifying-incorrect-data-types"
title: "识别不正确的数据类型"
description: "使用编程工具检查分配给列的当前数据类型。"
order: 3
plots: []
sourceHash: "836cb10c04d0cecbb4dc7e01b5a3840a37d599a272304e7ce418c290ff52e18d"
sourceCorrections: []
---

正如我们在章节简介中讨论的，确保您的数据列具有正确的*数据类型*对于可靠的分析来说非常重要。对存储为文本的数字执行数学运算，或者尝试对仅仅是字符串的日期进行按时间顺序排序，通常会导致错误或不正确的结果。想象一下，尝试计算 $5 + 10$；如果将'5'视为文本字符'5'而不是数值5，结果会非常不同。在解决这些问题之前，我们首先需要准确地确定数据集中每个列当前分配的数据类型。

幸运的是，大多数数据分析工具都提供了简单的方法来检查数据类型。如果您正在使用Python和pandas库（这在数据处理中非常常见），您可以在加载数据后轻松检查数据类型。

考虑一个包含产品信息的小型数据集。我们可以将其加载到pandas DataFrame中，然后使用`.dtypes`属性检查数据类型。

```python
import pandas as pd
import io

# 模拟从文件加载的示例数据
data = """ProductID,ProductName,Price,StockCount,LastUpdated,IsActive
101,Widget A,"50.99",500,2023-10-26,True
102,Gadget B,"120.50",150,2023-10-25,False
103,Thingamajig,"15.00"," N/A",2023-10-26,True
104,Doohickey,"75.25",200,2023-10-27,True
"""

# 使用 io.StringIO 模拟读取 CSV 文件
df = pd.read_csv(io.StringIO(data))

# 显示每列的数据类型
print(df.dtypes)
```

执行这段代码通常会打印出类似如下的输出：

```
ProductID        int64
ProductName     object
Price           object
StockCount      object
LastUpdated     object
IsActive          bool
dtype: object
```

让我们解释一下这段输出，以了解我们数据当前的状况：

- `int64`: 这表示64位整数，即整数。在我们的示例中，`ProductID`已被正确识别为整数列。
- `object`: 这是pandas中常见的数据类型，通常表示该列包含字符串（文本）。它也可能意味着该列包含不同数据类型的混合。请注意，`ProductName`、`Price`、`StockCount`和`LastUpdated`都被归类为`object`。
  - `ProductName`预期是文本，因此`object`在这里是合适的。
  - `Price`包含数值，但因为在示例数据中它们被括在引号中（例如，`"50.99"`），pandas将其解释为文本字符串。这对于计算是不正确的。
  - `StockCount`包含数字，但也包括一个非数字字符串`" N/A"`。这种不一致性迫使pandas对整个列使用通用的`object`类型，从而阻止了数值运算。
  - `LastUpdated`包含日期，但它们目前存储为简单的文本字符串。
- `bool`: 这代表布尔值，可以是`True`或`False`。`IsActive`列被正确地推断为布尔类型。

您可能还会遇到其他数据类型，例如：

- `float64`: 用于64位浮点数（带小数的数字，如 $3.14$ 或 $98.6$）。
- `datetime64[ns]`: Pandas中日期和时间值的专用类型（`[ns]`表示纳秒精度）。列需要显式转换才能达到此类型。
- `category`: 一种特殊类型，用于具有有限且固定唯一值集合的列（例如“低”、“中”、“高”）。

`object`数据类型通常表明某列需要关注。虽然它适合真正的文本字段，但出现在预期为数值、日期/时间或布尔类型的列中时，通常表明存在潜在问题。这可能由于以下原因：

- 数字格式为文本（例如，`'123'`、`'$5.00'`）。
- 日期以各种字符串格式存储（例如，`'10/26/2023'`、`'26-Oct-2023'`）。
- 布尔值以不一致的字符串形式表示（例如，`'True'`、`'false'`、`'Y'`、`'N'`）。
- 数值数据中混有非数字字符（例如，`'5 apples'`、`'10 kg'`）。
- 缺失数据表示不一致（例如，`'N/A'`、`'missing'`、`'?'`）。

这里有一张汇总表，说明了推断数据类型可能不正确的常见情况：

| 示例列内容 | 可能推断的类型 | 期望/正确类型 | 常见不匹配原因 |
| --- | --- | --- | --- |
| `'101'`、`'102'`、`'103'` | `object` | `int64` | 数字被引号括起或作为文本读取。 |
| `'99.9'`、`'15.5'`、`'NA'` | `object` | `float64` | 混合了类似数字的字符串和非数字文本。 |
| `'$10.50'`、`'€20.00'` | `object` | `float64` | 货币符号阻碍了直接的数值读取。 |
| `'2023-11-01'`、`'01/11/2023'` | `object` | `datetime64[ns]` | 日期以文本字符串形式存储。 |
| `'Yes'`、`'No'`、`'TRUE'` | `object` | `bool` | 布尔值以各种文本字符串形式存储。 |
| `1`、`2`、`3` (邮政编码) | `int64` | `object` / `category` | 不应进行数学运算的数字编码。 |

使用像`.dtypes`这样的函数或属性是诊断数据类型问题的主要第一步。通过检查输出，您可以确定哪些列未以其适当的格式存储。一旦识别出来，您就可以应用后续章节中涵盖的转换技术，以确保您的数据结构正确，从而进行有意义的分析。

## 参考资料

- [Dtype introduction](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#dtype-introduction) — pandas development team (2023)
  Pandas数据类型的官方指南，包含如何检查和解释`dtypes`属性。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781491957653/) — Wes McKinney (2017)
  Publisher: O'Reilly Media
  使用pandas进行数据操作的基础资源，包含理解和清洗数据类型的部分。
- [Practical Data Cleaning: The Essential Guide to Tidy Your Data](https://www.oreilly.com/library/view/practical-data-cleaning/9781098135832/) — Andreas Kretz (2023)
  Publisher: O'Reilly Media
  提供了识别和纠正数据类型问题的实用策略和示例，作为核心数据清洗任务。
