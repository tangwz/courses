---
course: "essential-numpy-pandas"
chapter: "loading-saving-data-pandas"
lesson: "reading-excel-files"
sourceId: 1765
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas/reading-excel-files"
title: "从Excel文件读取数据"
description: "使用`pd.read_excel()`从Excel电子表格（.xls, .xlsx）加载数据，并可指定工作表。"
order: 2
plots: []
sourceHash: "39cebda2166ec508e98eabc3256e05def64120a951b528e8e6523c4411b8b583"
sourceCorrections: []
---

Microsoft Excel电子表格（旧版本为`.xls`，新版本为`.xlsx`）常用于存储表格数据，尤其是在商业环境中，因此非常普遍。Pandas提供了一个方便的函数`pd.read_excel()`，专门用于直接从这些文件读取数据到DataFrame。

### Excel文件基本读取

最简单地，读取Excel文件与读取CSV文件非常相似。你只需提供文件路径：

```python
import pandas as pd

# 假设 'data.xlsx' 在同一目录下
try:
    df_excel = pd.read_excel('data.xlsx')
    print("Excel文件加载成功：")
    print(df_excel.head())
except FileNotFoundError:
    print("错误：未找到data.xlsx。请确保文件存在。")
# 如果需要，可添加其他可能的异常，例如权限错误
```

默认情况下，`pd.read_excel()`会读取Excel工作簿中的*第一个*工作表。它还假定该工作表的第一行包含列标题。

### 处理多个工作表

Excel工作簿通常包含多个工作表，每个工作表可能存储不同的表格或相关信息。`pd.read_excel()`函数允许你使用`sheet_name`参数 (parameter)指定要加载哪个工作表。

- **按工作表名称加载：** 如果你知道工作表的名称（例如，“Sales Data”，“Inventory”），你可以将其作为字符串提供：

  ```python
  # 加载名为 'Sales Data' 的工作表
  sales_df = pd.read_excel('multi_sheet_data.xlsx', sheet_name='Sales Data')
  print("\n已加载 'Sales Data' 工作表：")
  print(sales_df.head())
  ```
- **按工作表索引加载：** 工作表也可以按其位置（索引）引用，第一个工作表从0开始，第二个从1开始，依此类推。

  ```python
  # 加载第二个工作表（索引1）
  inventory_df = pd.read_excel('multi_sheet_data.xlsx', sheet_name=1)
  print("\n已加载第二个工作表（索引1）：")
  print(inventory_df.head())
  ```
- **加载所有工作表：** 要将所有工作表加载到一个字典中，其中键是工作表名称，值是DataFrames，请设置`sheet_name=None`：

  ```python
  all_sheets = pd.read_excel('multi_sheet_data.xlsx', sheet_name=None)

  print("\n已加载所有工作表：")
  for sheet_name, df_sheet in all_sheets.items():
      print(f"\n--- 工作表：{sheet_name} ---")
      print(df_sheet.head())
  ```

  这会返回一个字典，当你在同一工作簿中需要处理来自多个工作表的数据时，这会很有用。

### 引擎要求

读取Excel文件需要额外的Python库来处理文件格式本身。Pandas为此使用不同的“引擎”。

- 对于较新的`.xlsx`文件（Excel 2007+），通常使用`openpyxl`库。
- 对于较旧的`.xls`文件（Excel 97-2003），传统上使用`xlrd`库，尽管出于安全原因，其在最近版本中已移除对`.xlsx`的支持。

如果你尝试读取Excel文件但未安装所需的引擎，Pandas通常会给出一条提示你安装的信息性错误消息。你通常可以使用pip安装所需的引擎：

```bash
# 对于 .xlsx 文件
pip install openpyxl

# 如果你需要处理旧的 .xls 文件（如果需要，请安装特定版本）
pip install xlrd
```

如果你打算处理现代Excel文件，通常建议安装`openpyxl`。

### 其他常用参数 (parameter)

与`pd.read_csv()`类似，`pd.read_excel()`提供多个参数来调整数据加载方式：

- `header=`: 指定用作列名称的行号（0索引）。默认为0。
- `index_col=`: 指定用作DataFrame索引的列（按名称或0索引位置）。
- `usecols=`: 允许你指定要读取的列，可以通过名称或位置。这可以在你只需要部分数据时节省内存和时间。
- `skiprows=`: 跳过工作表开头指定数量的行。
- `nrows=`: 仅读取工作表中指定数量的行。

```python
# 示例：从“Inventory”工作表读取指定列，
# 并使用“ProductID”列作为索引
inventory_subset = pd.read_excel(
    'multi_sheet_data.xlsx',
    sheet_name='Inventory',
    usecols=['ProductID', 'ProductName', 'Quantity'], # 仅读取这些列
    index_col='ProductID' # 将 ProductID 设置为索引
)
print("\n已加载“Inventory”工作表的子集，并以 ProductID 为索引：")
print(inventory_subset.head())
```

掌握`pd.read_excel()`对于处理存储在电子表格中数据的工作流程来说是必不可少的。它在处理不同工作表和自定义导入过程方面的灵活性，使其成为Pandas中数据获取的一个有价值的工具。请记住，在开始读取Excel文件之前，要安装必要的引擎（`openpyxl`通常是你需要的）。

## 参考资料

- [pandas.read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html) — pandas development team (2023)
  pd.read_excel 函数的全面指南，详细说明了将 Excel 文件读取到 DataFrame 的所有参数和用法。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广受认可的书籍（第 3 版），涵盖使用 Pandas 进行数据处理，包括从 Excel 等多种文件格式加载数据的实用方法（参见第 6 章：数据加载、存储和文件格式）。
- [openpyxl: A Python library to read/write Excel 2010 xlsx/xlsm files](https://openpyxl.readthedocs.io/en/stable/) — Eric Gazoni, Charlie Clark (2024)
  openpyxl 库的官方文档，Pandas 使用此库作为读写 .xlsx 文件的引擎。
