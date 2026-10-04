---
course: "essential-numpy-pandas"
chapter: "data-selection-indexing-pandas"
lesson: "mixing-loc-iloc"
sourceId: 1786
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas/mixing-loc-iloc"
title: "混合标签和基于位置的索引"
description: "了解标签和位置选择结合使用时的区别与最佳实践。"
order: 4
plots: []
sourceHash: "8c8605acae8f7febdd1b926bb62d823904461c2fc1405f7bec74df0696263942"
sourceCorrections: []
---

在 Pandas 中，可以使用 `.loc` 通过标签选择数据，并使用 `.iloc` 通过整数位置选择数据。这些工具非常有用，但了解它们各自的作用以及为什么 Pandas 中通常避免在单个访问器（如 `.loc` 或 `.iloc`）中直接混合基于标签和基于位置的索引，这一点很重要。

尽管同时通过行标签和列的整数位置（或反之）来指定数据可能看似直观，但 Pandas 有意将这些方法分开，以保持清晰度和可预测性。

### 为什么不建议直接混合使用

主要原因是模糊不清。让我们看看为什么尝试混合使用不会如预期那样起作用：

1. **`.loc` 期望标签：** `.loc` 访问器是*专门*为基于标签的索引而设计的。如果你提供一个整数，例如 `df.loc['row_label', 0]`，`.loc` 会将 `0` 解释为*列标签*，而不是整数位置。只有当你的列实际命名为 `0` 时，这才会起作用。否则，它通常会引发 `KeyError` 错误。唯一的例外是当*索引本身*由整数组成时；在这种情况下，`df.loc[0]` 将选择标签为 `0` 的行。但即便如此，解释也是基于*标签*，而不是位置。
2. **`.iloc` 期望整数：** 相反地，`.iloc` 是*专门*为基于整数位置的索引而设计的。它根据行和列的零起始位置进行操作，无论它们的标签是什么。如果你提供一个标签，例如 `df.iloc[0, 'column_label']`，`.iloc` 将无法理解字符串标签，并将引发 `TypeError` 错误。

### 历史背景：已弃用的 `.ix`

在 Pandas 的旧版本中，有一个名为 `.ix` 的索引器，它试图同时处理基于标签和基于整数的索引，允许混合类型。然而，这导致了一些难以察觉的错误和令人困惑的代码，因为它的行为可能会根据 DataFrame 的索引是否包含整数而改变。例如，如果索引包含整数标签，`df.ix[0]` 可能会通过标签选择；如果索引不包含整数，则会通过位置选择。这种模糊性使得代码更难阅读和调试。

由于这些问题，`.ix` 被弃用。`.loc`（仅限标签）和 `.iloc`（仅限整数）之间的明确分离提供了一种更清晰、更易于维护的数据选择方式。

### 安全地实现“混合”选择结果

那么，当你已知一个轴（行）的标签和另一个轴（列）的整数位置，或者反之，该如何选择数据呢？你不应在一次 `.loc` 或 `.iloc` 调用中混合使用它们。相反，你应该一致地使用一个访问器，并可能转换另一个轴的标识符，或者通过链式操作实现。

让我们使用一个示例 DataFrame：

```python
import pandas as pd
import numpy as np

data = {'Score': [85, 92, 78, 88, 95],
        'Attempts': [1, 3, 2, 3, 1],
        'Grade': ['B', 'A', 'C', 'B', 'A']}
index_labels = ['Student1', 'Student2', 'Student3', 'Student4', 'Student5']
df = pd.DataFrame(data, index=index_labels)

print(df)
```

Output:

```
          Score  Attempts Grade
Student1     85         1     B
Student2     92         3     A
Student3     78         2     C
Student4     88         3     B
Student5     95         1     A
```

#### 场景 1：按行标签和列位置选择

假设你想要获取 `Student3`（标签）在*第二列*（位置 1，即 'Attempts'）的数据。

- **选项 A：使用 `.loc`（将位置转换为标签）**
  首先找到位置 1 处列的*标签*。

  ```python
  # 获取整数位置 1 处的列标签
  col_label = df.columns[1]
  print(f"位置 1 处的列标签: {col_label}")

  # 使用 .loc 和两个标签
  value = df.loc['Student3', col_label]
  print(f"Student3，列位置 1 处的值: {value}")
  ```

  Output:

  ```
  位置 1 处的列标签: Attempts
  Student3，列位置 1 处的值: 2
  ```
- **选项 B：使用 `.iloc`（将标签转换为位置）**
  首先找到行标签 `Student3` 的*整数位置*。

  ```python
  # 获取行标签 'Student3' 的整数位置
  row_pos = df.index.get_loc('Student3')
  print(f"Student3 的行位置: {row_pos}")

  # 使用 .iloc 和两个整数位置
  value = df.iloc[row_pos, 1]
  print(f"Student3，列位置 1 处的值: {value}")
  ```

  Output:

  ```
  Student3 的行位置: 2
  Student3，列位置 1 处的值: 2
  ```
- **选项 C：链式选择**
  首先按标签选择行，然后从结果 Series 中按位置选择元素。对于大型 DataFrame，这有时效率较低，但通常易于阅读。

  ```python
  value = df.loc['Student3'].iloc[1]
  print(f"使用链式选择的值: {value}")
  ```

  Output:

  ```
  使用链式选择的值: 2
  ```

#### 场景 2：按行位置和列标签选择

现在，假设你想要获取*第四行*（位置 3）和标签为 `'Grade'` 的列的数据。

- **选项 A：使用 `.iloc`（将标签转换为位置）**
  找到列标签 `'Grade'` 的*整数位置*。

  ```python
  # 获取列标签 'Grade' 的整数位置
  col_pos = df.columns.get_loc('Grade')
  print(f"Grade 的列位置: {col_pos}")

  # 使用 .iloc 和两个整数位置
  value = df.iloc[3, col_pos]
  print(f"行位置 3，列 Grade 处的值: {value}")
  ```

  Output:

  ```
  Grade 的列位置: 2
  行位置 3，列 Grade 处的值: B
  ```
- **选项 B：使用 `.loc`（将位置转换为标签）**
  找到位置 3 处行的*标签*。

  ```python
  # 获取整数位置 3 处的行标签
  row_label = df.index[3]
  print(f"位置 3 处的行标签: {row_label}")

  # 使用 .loc 和两个标签
  value = df.loc[row_label, 'Grade']
  print(f"行位置 3，列 Grade 处的值: {value}")
  ```

  Output:

  ```
  位置 3 处的行标签: Student4
  行位置 3，列 Grade 处的值: B
  ```
- **选项 C：链式选择**
  首先按标签选择列，然后从结果 Series 中按位置选择元素。

  ```python
  value = df['Grade'].iloc[3]
  print(f"使用链式选择的值: {value}")
  ```

  Output:

  ```
  使用链式选择的值: B
  ```

### 总结

尽管混合使用标签和整数索引的想法看起来很方便，但 Pandas 通过 `.loc` 和 `.iloc` 强制分离是有充分理由的，主要为了清晰度和可预测性。当你需要使用标签和位置信息的组合来选择数据时，选择与你的主要标识符匹配的访问器（`.loc` 或 `.iloc`），并使用辅助方法如 `df.columns.get_loc()`、`df.index.get_loc()`、`df.columns[]` 或 `df.index[]` 根据需要转换另一个标识符，或者使用链式选择。这种明确的方法使你的代码更容易理解，并且不易出错。

## 参考资料

- [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html) — The pandas development team (2024)
  全面介绍了基于标签（.loc）和基于位置（.iloc）的索引，包括它们分离的原因以及.ix的弃用。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media; Pages: Chapter 3: Data Manipulation with Pandas, "Data Indexing and Selection" section
  一个被广泛引用的资源，清晰地解释了Pandas数据索引和选择，对比了基于标签（.loc）和基于整数位置（.iloc）的方法。
