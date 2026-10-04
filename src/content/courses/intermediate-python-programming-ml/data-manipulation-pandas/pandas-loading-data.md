---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "pandas-loading-data"
sourceId: 2142
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-loading-data"
title: "从多种来源加载数据"
description: "学习如何将数据从CSV、Excel和SQL数据库等常见文件格式读取到Pandas DataFrames中。"
order: 2
plots: []
sourceHash: "5f0e9e24dbf4b1b8c98969826cfd54d28f30ac53c2573f21c75f5194487f138d"
sourceCorrections: []
---

数据载入DataFrame是使用Pandas处理数据的重要前提。幸运的是，Pandas提供了一系列函数，用于读取多种常见存储格式的数据。将文件或数据库中的数据加载到内存中作为DataFrame，通常是任何数据分析或机器学习 (machine learning)流程中的第一步。

我们来看看如何从一些最常遇到的来源加载数据。

### 从CSV文件读取数据

逗号分隔值（CSV）文件可能是存储表格数据最常见的文本文件格式。每行通常代表一个记录，行内的值由分隔符（通常是逗号）分隔。Pandas为此提供了`read_csv()`函数。

其最简单的用法是提供文件路径：

```python
import pandas as pd

# 假设 'data.csv' 在同一目录下
df_csv = pd.read_csv('data/sales_data.csv')

# 显示前5行
print(df_csv.head())
```

`read_csv()`函数功能非常多样，接受众多可选参数 (parameter)，以处理不同的文件结构和潜在问题：

- `sep`或`delimiter`：指定用于分隔值的字符。虽然默认是逗号（`,`），您可能会遇到使用制表符（`\t`）、分号（`;`）或其他字符的文件。

  ```python
  # 制表符分隔文件的示例
  df_tsv = pd.read_csv('data/product_data.tsv', sep='\t')
  ```
- `header`：指示哪一行（从0开始计数）包含列名。如果文件没有标题行，请设置`header=None`，Pandas将分配默认的数字列名（0, 1, 2...）。您还可以使用`names`参数提供一个名称列表。

  ```python
  # 没有标题的文件，提供列名
  df_no_header = pd.read_csv('data/log_data.csv', header=None, names=['timestamp', 'user_id', 'action'])
  ```
- `index_col`：指定哪一列应用作DataFrame的索引（行标签）。这可以是列号或列名。

  ```python
  # 使用 'transaction_id' 列作为索引
  df_indexed = pd.read_csv('data/transactions.csv', index_col='transaction_id')
  ```
- `usecols`：允许您通过提供列名或索引列表来只读取特定的列子集。这对于只需要某些特征的大文件很有用。

  ```python
  # 只读取 'customer_id' 和 'purchase_amount' 列
  df_subset = pd.read_csv('data/orders.csv', usecols=['customer_id', 'purchase_amount'])
  ```
- `nrows`：只读取文件中指定数量的行，从文件开头开始。这有助于快速查看大文件，而无需将整个数据集加载到内存中。

  ```python
  # 只读取前1000行
  df_sample = pd.read_csv('data/large_dataset.csv', nrows=1000)
  ```
- `encoding`：指定文件编码，如果它不是标准的UTF-8（例如，'latin1'，'iso-8859-1'）。不正确的编码可能导致错误或乱码。

“查阅Pandas关于`pd.read_csv`的文档，以查看完整的参数列表；掌握这些选项对于处理各类数据文件是必不可少的。”

### 从Excel文件读取数据

Excel电子表格（`.xls`或`.xlsx`）是另一种常见的格式，尤其在商业环境中。Pandas使用`read_excel()`函数从这些文件加载数据。如果您尚未安装，可能需要安装`openpyxl`（用于`.xlsx`文件）或`xlrd`（用于较旧的`.xls`文件）等额外库（`pip install openpyxl xlrd`）。

基本用法与`read_csv()`相似：

```python
import pandas as pd

# 从Excel文件的第一个工作表加载数据
df_excel = pd.read_excel('data/inventory.xlsx')

print(df_excel.head())
```

Excel文件的一个显著不同是它们可以包含多个工作表。`read_excel()`函数通过`sheet_name`参数 (parameter)处理这一点：

- `sheet_name=None`：将所有工作表读取到DataFrames字典中，其中键是工作表名称，值是对应的DataFrame。
- `sheet_name='SheetName'`：按名称读取指定的工作表。
- `sheet_name=0`：读取第一个工作表（索引从0开始）。
- `sheet_name=[0, 1, 'Summary']`：将多个特定工作表（按索引或名称）读取到DataFrames字典中。

```python
# 从 'Q1_Sales' 工作表加载数据
df_q1 = pd.read_excel('data/regional_sales.xlsx', sheet_name='Q1_Sales')

# 将所有工作表加载到字典中
all_sheets_dict = pd.read_excel('data/regional_sales.xlsx', sheet_name=None)
df_q2 = all_sheets_dict['Q2_Sales'] # 访问 'Q2_Sales' 工作表的DataFrame
```

`read_excel()`与`read_csv()`共享许多参数，例如`header`、`index_col`和`usecols`，从而可以精细控制电子表格数据的导入方式。

### 从SQL数据库读取数据

机器学习 (machine learning)数据常存在于关系型数据库（如PostgreSQL、MySQL、SQLite、SQL Server）中。Pandas可以使用`read_sql()`、`read_sql_query()`或`read_sql_table()`直接查询数据库并将结果加载到DataFrame中。

这需要建立与数据库的连接。在Python中，执行此操作的标准方法是使用SQLAlchemy库（`pip install sqlalchemy`），该库为不同的数据库系统提供一致的接口。您还需要一个数据库特定的驱动程序（例如，PostgreSQL的`psycopg2`，MySQL的`mysql-connector-python`）。

以下是使用SQLAlchemy和`read_sql_query()`的典型流程：

1. **创建数据库引擎：** 使用SQLAlchemy的`create_engine`函数和数据库连接字符串。
2. **编写SQL查询：** 定义SQL查询以获取所需数据。
3. **使用Pandas执行：** 将查询和引擎（或活动的连接）传递给`pd.read_sql_query()`。

```python
import pandas as pd
from sqlalchemy import create_engine

# SQLite示例（请替换为您的实际数据库连接字符串）
# 格式：'dialect+driver://username:password@host:port/database'
# SQLite连接字符串更简单：'sqlite:///path/to/your_database.db'
db_path = 'data/mydatabase.db'
engine = create_engine(f'sqlite:///{db_path}')

# 定义SQL查询
query = "SELECT user_id, age, registration_date FROM users WHERE country = 'USA';"

# 使用查询和引擎加载数据
try:
    with engine.connect() as connection:
        df_sql = pd.read_sql_query(sql=query, con=connection)
    print(df_sql.head())
except Exception as e:
    print(f"连接数据库或执行查询出错：{e}")

# 另外，pd.read_sql通常可以判断是读取表还是执行查询
# 读取整个 'products' 表
# try:
#     with engine.connect() as connection:
#         df_table = pd.read_sql(sql='products', con=connection, index_col='product_id')
#     print(df_table.head())
# except Exception as e:
#     print(f"读取表出错：{e}")
```

- `pd.read_sql_query(sql, con)`：使用提供的数据库连接（`con`）执行特定的SQL查询（`sql`）。
- `pd.read_sql_table(table_name, con)`：读取由`table_name`指定的整个数据库表。
- `pd.read_sql(sql, con)`：一个更通用的函数，可以委托给`read_sql_query`或`read_sql_table`。

使用`pd.read_sql_*`函数将数据库数据直接集成到您的Pandas工作流程中非常高效。

> 常见数据源（CSV、Excel、SQL数据库）通过特定的Pandas函数（`pd.read_csv`、`pd.read_excel`、`pd.read_sql`）引入，以创建统一的DataFrame结构供分析使用。

虽然CSV、Excel和SQL非常常见，Pandas也支持从许多其他格式读取数据，包括JSON（`pd.read_json`）、HTML表格（`pd.read_html`）、HDF5（`pd.read_hdf`）、Parquet（`pd.read_parquet`）等。基本原则保持不变：使用适当的`pd.read_*`函数，并配置其参数 (parameter)以正确解析源数据结构。熟练地从各种来源加载数据是数据处理和准备的一项基本技能。

## 参考资料

- [Input/Output Tools (IO)](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) — The pandas development team (2024)
  Pandas 关于读写数据的官方用户指南部分，详细介绍了 read_csv、read_excel 和 read_sql 等函数及其众多参数。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas 创始人撰写的权威指南，深入解释了数据结构和基本操作，包括详细介绍如何从各种格式加载数据。
- [SQLAlchemy Engine Configuration](https://docs.sqlalchemy.org/en/20/core/engines.html) — The SQLAlchemy project team (2025)
  SQLAlchemy 中配置数据库引擎的官方文档，这对于建立 Pandas 用于从 SQL 数据库读取数据的连接至关重要。
