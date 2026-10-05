# 将数据写入 CSV 文件

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas/writing-csv-files)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 Pandas DataFrame 中处理数据时，保存处理结果或整理后的数据集是常见需求，以便将来使用、分享或作为其他过程的输入。正如 `pd.read_csv()` 是读取 CSV 数据的标准方式一样，将 DataFrame 数据写回到 CSV 文件的相应方法是 `.to_csv()`。

CSV（逗号分隔值）格式因其简单性以及与众多应用程序（包括 Microsoft Excel 或 Google Sheets 等电子表格软件、数据库和其他编程环境）的兼容性，仍然是交换表格数据被广泛使用的标准。

### `to_csv()` 的基本用法

`.to_csv()` 最基本的用法是在您的 DataFrame 对象上调用它，并提供您希望保存数据的文件路径。

假设我们有以下 DataFrame，它可能是在之前的步骤中创建或修改的：

```python
import pandas as pd
import numpy as np

data = {'col_A': [1, 2, 3, 4, 5],
        'col_B': ['apple', 'banana', 'orange', 'grape', 'kiwi'],
        'col_C': [0.1, 0.2, np.nan, 0.4, 0.5]}
df_to_save = pd.DataFrame(data)

print(df_to_save)
# 预期输出:
#    col_A   col_B  col_C
# 0      1   apple    0.1
# 1      2  banana    0.2
# 2      3  orange    NaN
# 3      4   grape    0.4
# 4      5    kiwi    0.5
```

要将此 DataFrame 保存到当前工作目录中名为 `output_data.csv` 的文件，您只需运行：

```python
df_to_save.to_csv('output_data.csv')
```

如果您用文本编辑器打开 `output_data.csv`，您会看到类似以下内容：

```csv
,col_A,col_B,col_C
0,1,apple,0.1
1,2,banana,0.2
2,3,orange,
3,4,grape,0.4
4,5,kiwi,0.5
```

请注意以下几点：

1. 所有值都由逗号分隔。
2. 列标题（`col_A`、`col_B`、`col_C`）作为第一行包含在内。
3. DataFrame 的索引（0、1、2、3、4）已作为第一列写入，并且它没有列头名称。

通常，您可能不想在输出文件中包含 DataFrame 索引，特别是如果它只是默认的整数索引（0、1、2...），不表示有意义的数据。

### 控制输出：常用参数 (parameter)

`.to_csv()` 方法提供多种参数来自定义输出文件。以下是一些最常用的参数：

#### `index` 参数

正如所见，默认行为是写入 DataFrame 索引。为了避免这种情况，请将 `index` 参数设置为 `False`。这是一个非常普遍的需求。

```python
# 保存时不包含 DataFrame 索引
df_to_save.to_csv('output_data_no_index.csv', index=False)
```

现在，`output_data_no_index.csv` 将会是这样：

```csv
col_A,col_B,col_C
1,apple,0.1
2,banana,0.2
3,orange,
4,grape,0.4
5,kiwi,0.5
```

这种格式通常更整洁，更适合导入到可能生成自己行标识符的其他系统中。

#### `header` 参数

同样地，您有时可能希望省略标题行（即列名）。这可以通过将 `header` 参数设置为 `False` 来完成。

```python
# 保存时不包含标题行（也不包含索引）
df_to_save.to_csv('output_data_no_header.csv', index=False, header=False)
```

`output_data_no_header.csv` 的内容将是：

```csv
1,apple,0.1
2,banana,0.2
3,orange,
4,grape,0.4
5,kiwi,0.5
```

相比于省略索引，这不太常见，但在特定情况下会有用，例如向已包含标题的现有文件追加数据。

#### `sep` 参数

尽管 CSV 代表逗号分隔值，但数据文件有时会使用不同的字符（分隔符）来分隔字段。常见的替代选项包括制表符（`\t`）、分号（`;`）或竖线（`|`）。`sep` 参数允许您指定分隔符。例如，要创建一个制表符分隔值（TSV）文件：

```python
# 保存为 TSV 文件（制表符分隔），不包含索引
df_to_save.to_csv('output_data.tsv', index=False, sep='\t')
```

打开 `output_data.tsv` 会显示字段由制表符而不是逗号分隔。

#### `na_rep` 参数

请注意，在默认输出中，`col_C` 中缺失的 `NaN` 值导致 CSV 中出现一个空字段（`3,orange,,`）。您可以使用 `na_rep` 参数为缺失值指定一个自定义的字符串表示。

```python
# 保存时用 'MISSING' 表示 NaN 值，不包含索引
df_to_save.to_csv('output_data_na_rep.csv', index=False, na_rep='MISSING')
```

文件 `output_data_na_rep.csv` 将包含：

```csv
col_A,col_B,col_C
1,apple,0.1
2,banana,0.2
3,orange,MISSING
4,grape,0.4
5,kiwi,0.5
```

#### `columns` 参数

如果您只想保存 DataFrame 中一部分列，可以向 `columns` 参数提供一个列名列表。

```python
# 只保存 'col_A' 和 'col_B' 列，不包含索引
df_to_save.to_csv('output_data_subset.csv', index=False, columns=['col_A', 'col_B'])
```

文件 `output_data_subset.csv` 将只包含这两列的数据：

```csv
col_A,col_B
1,apple
2,banana
3,orange
4,grape
5,kiwi
```

#### `encoding` 参数

文本文件使用特定的字符编码存储。虽然通常默认处理正确（Pandas 通常默认使用 'utf-8'，它具有广泛的兼容性），但有时如果接收系统有要求，或者您的数据包含需要特定编码（如 'latin1' 或 'cp1252'）的特殊字符，您可能需要指定不同的编码。

```python
# 使用特定编码保存（例如 latin1）
# df_to_save.to_csv('output_data_encoding.csv', index=False, encoding='latin1')
# 注意：通常首选 utf-8（默认），除非您有特殊需求。
```

有效保存数据与加载数据同样重要。`.to_csv()` 方法提供灵活的选项，以确保您的 DataFrame 以所需格式导出，准备好用于存储、分享或数据工作流程的后续步骤。请记住，将 `index=False` 通常是一个好的做法，除非您的 DataFrame 索引包含需要明确在输出文件中保留的重要信息。

## 参考资料

- [pandas.DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html) — The Pandas Development Team (2024)
  Pandas DataFrame.to_csv 方法的官方文档，列出了所有参数和使用示例。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Python和Pandas进行数据操作的经典书籍，涵盖数据加载、保存和一般的DataFrame操作。（第三版）
- [Hands-On Data Analysis with Pandas: A Practical Guide to Building Powerful Data Analysis Pipelines](https://www.packtpub.com/product/hands-on-data-analysis-with-pandas-second-edition/9781800563456) — Stefanie Molin (2021)
  Publisher: Packt Publishing
  一份使用Pandas进行数据分析的实用指南，包括数据导出功能，例如保存到CSV。（第二版）

---

[上一节](03-%E4%BB%8E%E5%85%B6%E4%BB%96%E6%A0%BC%E5%BC%8F%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md) · [下一节](05-%E5%86%99%E5%85%A5%E6%95%B0%E6%8D%AE%E5%88%B0Excel%E6%96%87%E4%BB%B6.md)
