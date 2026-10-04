---
course: "data-visualization-matplotlib-seaborn"
chapter: "plotting-pandas-dataframes"
lesson: "loading-data-pandas-dataframe"
sourceId: 1217
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/loading-data-pandas-dataframe"
title: "将数据加载到DataFrame"
description: "了解如何从CSV等常见文件格式中读取数据到Pandas DataFrame。"
order: 2
plots: []
sourceHash: "32f03739158525b0d0c33946bd027fd43f2eae35884f105bce7d508cd87f2de7"
sourceCorrections: []
---

数据很少直接在Python脚本中生成。多数情况下，你会处理存储在外部文件中的数据。电子表格、数据库和CSV（逗号分隔值）等文本文件是常见的数据来源。Pandas库擅长将这些结构化数据文件读取为Python、Matplotlib和Seaborn易于理解的格式。

Pandas提供的用于保存表格数据（像电子表格一样按行和列组织的数据）的核心数据结构是**DataFrame**。可以把DataFrame看作是一个功能强大、灵活的表格，其中列有名称，行有标签（索引）。这种结构非常适合数据分析和可视化工作。

### 导入Pandas

在使用Pandas之前，你需要先导入它。标准做法（你应该始终遵循）是：

```python
import pandas as pd
```

这一行导入了Pandas库并给它起了别名`pd`，这样可以缩短并简化Pandas命令的输入（例如，使用`pd.DataFrame()`而不是`pandas.DataFrame()`）。

### 使用`read_csv`读取数据

从CSV文件加载数据是最常见的任务之一。CSV文件以纯文本形式存储表格数据，每行代表一条记录，行中的值通常由逗号分隔。Pandas为此提供了功能多样的`pd.read_csv()`函数。

最基本的用法是将文件路径传递给函数：

```python
# 假设 'my_data.csv' 在与脚本相同的目录中
df = pd.read_csv('my_data.csv')

# 如果文件在其他位置，请提供完整路径
# Windows示例: df = pd.read_csv('C:\\Users\\YourUser\\Documents\\data\\my_data.csv')
# macOS/Linux示例: df = pd.read_csv('/Users/youruser/documents/data/my_data.csv')
```

运行此代码时，Pandas会读取指定的CSV文件并创建一个DataFrame对象，我们将其赋值给变量`df`（这是DataFrame变量的常见命名习惯）。

**关于文件路径的说明：** 提供正确的数据文件路径很重要。

- **相对路径：** 如果CSV文件与你的Python脚本在同一目录中，你可以直接使用文件名（例如，`'my_data.csv'`）。
- **绝对路径：** 如果文件位于计算机上的其他位置，你需要提供从根目录开始的完整路径。请注意根据你的操作系统使用正斜杠（`/`）或双反斜杠（`\\`）。

### 定制数据加载

`pd.read_csv()`函数有许多可选参数 (parameter)，可以处理不同的CSV格式和加载要求。以下是一些常用参数：

- `sep`（或`delimiter`）：指定文件中用于分隔值的字符。虽然逗号是标准分隔符（`sep=','`），但数据有时也可能由制表符（`sep='\t'`）或分号（`sep=';'`）分隔。

  ```python
  # 制表符分隔文件的示例
  df_tsv = pd.read_csv('data.tsv', sep='\t')
  ```
- `header`：告诉Pandas哪一行包含列名。默认情况下，`header=0`表示第一行是标题行。如果你的文件没有标题行，请使用`header=None`，Pandas将分配默认的整数名称（0, 1, 2...）。如果标题不在第一行，你也可以指定不同的行号。

  ```python
  # 无标题行的文件
  df_no_header = pd.read_csv('data_no_header.csv', header=None)

  # 标题行在第三行（索引为2）的文件
  df_header_row3 = pd.read_csv('data_header_late.csv', header=2)
  ```
- `index_col`：你可以指定CSV文件中的一列作为DataFrame的索引（行标签）。传入列名或其数字索引（0代表第一列）。

  ```python
  # 使用第一列（'ID'）作为索引
  df_indexed = pd.read_csv('data_with_id.csv', index_col=0)
  # 或者按名称指定：
  # df_indexed = pd.read_csv('data_with_id.csv', index_col='ID')
  ```
- `usecols`：如果你的CSV文件包含很多列但你只需要其中几列，你可以使用列名或索引列表来指定要加载的列。这可以为大型文件节省内存并加快加载速度。

  ```python
  # 只加载 'Date' 和 'Temperature' 列
  df_subset = pd.read_csv('weather_data.csv', usecols=['Date', 'Temperature'])
  ```
- `nrows`：要仅加载大型文件的前几行（对于快速查看而不加载全部内容很有用），请使用`nrows`参数。

  ```python
  # 只加载前100行
  df_preview = pd.read_csv('very_large_data.csv', nrows=100)
  ```

### 检查已加载的DataFrame

加载数据后，检查其是否正确读取是必不可少的操作。Pandas DataFrames为此提供了几种有用的方法：

- `df.head(n)`：显示前`n`行（默认值为5）。用于快速查看结构和一些初始数据。
- `df.tail(n)`：显示最后`n`行（默认值为5）。适合检查文件末尾。
- `df.shape`：返回一个元组，表示DataFrame的维度（行数，列数）。
- `df.columns`：显示所有列的名称。
- `df.info()`：提供DataFrame的简洁摘要，包括索引类型、列名、每列的数据类型、非空值的数量以及内存使用情况。这对于发现潜在问题（例如列以错误的数据类型读取或存在意外的缺失值）非常有用。

### 示例：加载和检查简单的CSV

我们来创建一个小型、独立的示例。我们将使用Python的`io.StringIO`来模拟从文件读取，而无需实际的外部文件。设想这个字符串是一个名为`sensor_log.csv`的文件内容：

```csv
Timestamp,SensorID,Temperature,Humidity
2023-10-26 10:00:00,A1,22.5,45.1
2023-10-26 10:00:00,B2,21.8,46.5
2023-10-26 10:01:00,A1,22.6,45.0
2023-10-26 10:01:00,B2,,46.6
2023-10-26 10:02:00,A1,22.7,44.9
2023-10-26 10:02:00,B2,21.9,46.7
```

现在，让我们使用Pandas加载并检查这些数据：

```python
import pandas as pd
from io import StringIO # 模拟文件所需

# 模拟CSV文件内容
csv_data = """Timestamp,SensorID,Temperature,Humidity
2023-10-26 10:00:00,A1,22.5,45.1
2023-10-26 10:00:00,B2,21.8,46.5
2023-10-26 10:01:00,A1,22.6,45.0
2023-10-26 10:01:00,B2,,46.6
2023-10-26 10:02:00,A1,22.7,44.9
2023-10-26 10:02:00,B2,21.9,46.7
"""

# 读取模拟的CSV数据
# StringIO(csv_data) 作用类似于一个打开的文件句柄
df_sensors = pd.read_csv(StringIO(csv_data))

# 检查已加载的DataFrame
print("--- 前3行 ---")
print(df_sensors.head(3))
print("\n--- DataFrame 信息 ---")
df_sensors.info()
print("\n--- DataFrame 形状 ---")
print(df_sensors.shape)
print("\n--- 列名 ---")
print(df_sensors.columns)
```

运行这段代码将输出：

```
--- First 3 Rows ---
             Timestamp SensorID  Temperature  Humidity
0  2023-10-26 10:00:00       A1         22.5      45.1
1  2023-10-26 10:00:00       B2         21.8      46.5
2  2023-10-26 10:01:00       A1         22.6      45.0

--- DataFrame Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Timestamp    6 non-null      object
 1   SensorID     6 non-null      object
 2   Temperature  5 non-null      float64
 3   Humidity     6 non-null      float64
dtypes: float64(2), object(2)
memory usage: 320.0+ bytes

--- DataFrame Shape ---
(6, 4)

--- Column Names ---
Index(['Timestamp', 'SensorID', 'Temperature', 'Humidity'], dtype='object')
```

注意`df.info()`如何正确识别出`Temperature`和`Humidity`列为`float64`（浮点数），以及`Timestamp`和`SensorID`列为`object`（在Pandas中通常表示字符串）。它还指出`Temperature`列有一个缺失值（6个条目中有`5 non-null count`）。这种初步检查在进行可视化之前非常宝贵。

### 其他数据源

尽管`pd.read_csv`非常常用，但Pandas也提供了读取许多其他格式的函数，包括：

- `pd.read_excel()`：用于读取Microsoft Excel文件（`.xls`、`.xlsx`）中的数据。
- `pd.read_json()`：用于读取JSON文件或字符串中的数据。
- `pd.read_sql()`：用于读取SQL数据库中的数据（需要数据库连接）。
- `pd.read_html()`：用于直接从网页读取表格。
- `pd.read_parquet()`：用于读取高效的Parquet列式存储格式的数据。

基本原理保持不变：使用对应的`pd.read_*`函数，提供路径或来源，并根据需要通过参数 (parameter)进行定制。

随着数据成功加载到Pandas DataFrame中，你现在拥有了一个强大的结构，为接下来的步骤做好了准备：数据查看和使用DataFrame自带的绘图方法或将其列传递给Matplotlib和Seaborn函数来创建可视化图表。

## 参考资料

- [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) — The pandas development team (2024)
  加载CSV文件的官方文档，详细介绍了数据导入的所有参数和选项。
- [pandas User Guide: Intro to Data Structures - DataFrame](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) — The pandas development team (2024)
  Pandas中DataFrame基本结构的介绍，说明了其组织方式和作用。
