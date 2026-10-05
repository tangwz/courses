# 从CSV文件读取数据

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas/reading-csv-files)

[返回章节目录](README.md) · [返回课程目录](../README.md)

逗号分隔值（CSV）文件可能是存储和交换表格数据最普遍的格式。可以把它们看作简单的文本文件，每行代表一条数据记录，而行内值则用逗号分隔。由于它们是纯文本格式，人类易于阅读，并广泛兼容于不同的软件应用。

Pandas提供了一个功能强大且灵活的函数`pd.read_csv()`，专门用于将CSV文件数据直接读取到DataFrame中。此函数能自动处理许多复杂情况，同时也提供了多种选项来定制数据加载方式。

### 基本读取

在最简单的情况下，如果你的脚本或Notebook所在目录下有一个名为`my_data.csv`的CSV文件，你可以这样加载它：

```python
import pandas as pd

# 假设 'my_data.csv' 存在于当前目录
df = pd.read_csv('my_data.csv')

# 显示前几行以查看
print(df.head())
```

默认情况下，`pd.read_csv()`假定：

1. 值由逗号（`,`）分隔。
2. 文件的第一行包含列标题。

### 指定文件路径

`pd.read_csv()`的第一个参数 (parameter)是文件路径。这可以是：

- **本地文件路径**：`'data/sales_records.csv'`或`'C:\\Users\\YourName\\Documents\\data.csv'`。
- **URL**：`'https://raw.githubusercontent.com/...'`，直接指向在线CSV文件。

```python
# 从URL读取的例子
url = 'https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv'
titanic_df = pd.read_csv(url)
print(titanic_df.head())
```

### 处理不同分隔符

尽管逗号是标准的，但数据有时可能使用其他字符作为分隔符（也称为定界符），例如制表符（`\t`）、分号（`;`）或空格。你可以使用`sep`（或`delimiter`）参数 (parameter)指定分隔符。例如，要读取一个以制表符分隔的文件（通常以`.tsv`结尾）：

```python
# 假设 'data.tsv' 使用制表符作为分隔符
df_tsv = pd.read_csv('data.tsv', sep='\t')
# 或者等效地：
# df_tsv = pd.read_csv('data.tsv', delimiter='\t')
```

### 处理标题行

Pandas默认假定第一行为标题行（`header=0`）。

- 如果你的文件**没有标题行**，Pandas会错误地将第一行数据用作标题。为避免这种情况，请使用`header=None`。Pandas随后会自动分配默认的整数列名（0, 1, 2, ...）。

  ```python
  # 对于没有标题行的CSV文件
  df_no_header = pd.read_csv('no_header_data.csv', header=None)
  ```
- 如果标题在**不同行**（例如，第二行，索引为1），你可以指定为：`header=1`。

### 分配列名

如果你的文件缺少标题行（`header=None`）或你想覆盖现有标题，你可以使用`names`参数 (parameter)提供自己的列名。这应该是一个字符串列表。

```python
# 对于 'no_header_data.csv'，分配有意义的列名
column_names = ['ID', 'Measurement', 'Timestamp', 'Status']
df_named = pd.read_csv('no_header_data.csv', header=None, names=column_names)
```

注意：如果你提供`names`而*没有*设置`header=None`，并且文件*确实*有标题行，你提供的`names`将覆盖文件中找到的标题（假定`header=0`）。如果你同时设置`header=0`（或保持默认）并提供`names`，文件中的原始标题行将被丢弃并替换为你提供的`names`。如果你想使用`names`并*跳过*原始标题，你可以使用`header=0`和`names=...`，或者如果文件一开始就没有标题行，可以使用`skiprows=1`和`names=...`。

### 指定索引列

通常，你的CSV文件中的某一列包含唯一标识符，你可能想将其用作DataFrame的索引，而不是默认的整数索引（0, 1, 2, ...）。为此，请使用`index_col`参数 (parameter)。你可以通过列名（如果存在标题）或其整数位置（从0开始计数）来指定列。

```python
# 读取泰坦尼克号数据时，使用 'PassengerId' 列作为索引
titanic_df_indexed = pd.read_csv(url, index_col='PassengerId')
print(titanic_df_indexed.head())

# 如果 'no_header_data.csv' 的第一列（位置0）有ID
# df_indexed = pd.read_csv('no_header_data.csv', header=None, index_col=0)
```

### 只读取特定列

对于列数较多的大型数据集，你可能只需要一部分。只读取必要的列可以节省内存并加快加载速度。使用`usecols`参数 (parameter)，提供列名或整数位置的列表。

```python
# 只读取泰坦尼克号数据集中的 'Survived'、'Pclass' 和 'Age' 列
titanic_subset = pd.read_csv(url, usecols=['Survived', 'Pclass', 'Age'])
print(titanic_subset.head())

# 按位置读取列（例如，第一列和第三列）
# subset_by_pos = pd.read_csv('my_data.csv', usecols=[0, 2])
```

### 限制行数

处理非常大的文件时，你可能只想加载前几行来检查数据结构或测试代码，而不将整个文件加载到内存中。`nrows`参数 (parameter)允许你指定要读取的精确行数（不包括标题行）。

```python
# 只读取泰坦尼克号数据集的前10行
titanic_sample = pd.read_csv(url, nrows=10)
print(titanic_sample)
```

这些参数涵盖了使用Pandas读取CSV文件时最常见的场景。`pd.read_csv()`函数还有更多选项用于处理日期、缺失值、引用规则、注释和编码，使其成为一个非常多功能的工具，用于将数据导入DataFrame。目前，掌握这些基本选项将使你能够有效地加载各种CSV数据。

## 参考资料

- [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) — pandas development team (2023)
  `pd.read_csv` 函数的官方文档，详细说明了所有参数和使用示例，用于可靠的数据摄取。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  第三版。由 Pandas 创建者撰写，关于使用 Python 和 Pandas 进行数据分析的奠基性著作，广泛涵盖数据加载。

---

[上一节](../05-Pandas%20%E4%BB%8B%E7%BB%8D/07-%E5%AE%9E%E6%93%8D%EF%BC%9A%E5%88%9B%E5%BB%BA%E5%92%8C%E6%9F%A5%E7%9C%8B%20Series-DataFrame.md) · [下一节](02-%E4%BB%8EExcel%E6%96%87%E4%BB%B6%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md)
