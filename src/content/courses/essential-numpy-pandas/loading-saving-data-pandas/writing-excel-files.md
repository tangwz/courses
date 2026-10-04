---
course: "essential-numpy-pandas"
chapter: "loading-saving-data-pandas"
lesson: "writing-excel-files"
sourceId: 1773
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas/writing-excel-files"
title: "写入数据到Excel文件"
description: "使用 .to_excel() 方法将 Pandas DataFrame 保存到 Excel 文件，并管理工作表。"
order: 5
plots: []
sourceHash: "ffb87ba01068ef7df9bf70bc9c6c4c099952fbf94ce7debe52474e451adc87b3"
sourceCorrections: []
---

Pandas 提供了简单的方法将您的 DataFrame 保存为不同格式。虽然 CSV 文件因其简洁性和系统间的互操作性而表现出色，Microsoft Excel 文件（`.xlsx` 或旧版 `.xls`）在商业环境中非常常见，并支持在单个文件中包含多个工作表等功能。

Pandas 使用 `to_excel()` 方法可以简单地将 DataFrame 写入 Excel 文件，该方法与我们之前看到的 `to_csv()` 方法类似。

### 基本用法：将 DataFrame 保存到 Excel 文件

我们从一个基本示例开始。假设我们已经处理了一些数据并将其存储在一个 DataFrame 中：

```python
import pandas as pd

# 示例 DataFrame
data = {'Product': ['Widget A', 'Widget B', 'Gadget C', 'Widget A'],
        'Region': ['North', 'South', 'North', 'West'],
        'Sales': [150, 200, 120, 180],
        'Profit': [30, 45, 25, 38]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

# 将 DataFrame 保存到 Excel 文件
# 这会在您的脚本/笔记本所在的目录中创建 'output_data.xlsx'
df.to_excel('output_data.xlsx')

print("\nDataFrame 已成功保存到 output_data.xlsx")
```

如果您运行此代码，然后使用 Microsoft Excel、LibreOffice Calc 或 Google Sheets 等电子表格软件打开生成的 `output_data.xlsx` 文件，您会看到您的 DataFrame 以表格形式呈现。

### 使用参数 (parameter)控制输出

`to_excel()` 方法提供了多个参数用于自定义输出文件。以下是一些常用的参数：

1. **`excel_writer`（文件路径）：** 这是第一个参数，用于指定输出 Excel 文件的文件路径和名称（例如，`'data/processed_sales.xlsx'`，`'report.xlsx'`）。
2. **`sheet_name`：** 默认情况下，DataFrame 会写入名为 'Sheet1' 的工作表。您可以使用此参数指定不同的名称。例如，`df.to_excel('report.xlsx', sheet_name='SalesData')`。
3. **`index`：** 与 `to_csv()` 类似，`to_excel()` 默认将 DataFrame 的索引作为 Excel 工作表的第一列写入。通常，此索引只是默认的顺序整数索引（0, 1, 2...），并不是您希望在最终电子表格中显示的有效数据。为避免写入索引，请设置 `index=False`：

   ```python
   # 不带 DataFrame 索引保存
   df.to_excel('output_data_no_index.xlsx', index=False)
   print("DataFrame 已保存到 output_data_no_index.xlsx，不含索引列。")
   ```

   将 `index=False` 设置为常见做法，尤其是在为不熟悉 Pandas 索引的人导出数据时。
4. **`columns`：** 如果您只想保存 DataFrame 中的特定列，可以向 `columns` 参数提供列名列表：

   ```python
   # 仅保存 'Product' 和 'Sales' 列
   df.to_excel('output_subset.xlsx', columns=['Product', 'Sales'], index=False)
   print("已将 Product 和 Sales 列保存到 output_subset.xlsx")
   ```
5. **`header`：** 此参数控制是否将列名（标题行）写入文件。它默认为 `True`。如果您不想包含标题行，请设置 `header=False`。
6. **`startrow` 和 `startcol`：** 这些参数允许您指定 DataFrame 在工作表中写入的左上角单元格（0-索引）。例如，`startrow=1, startcol=2` 将从 C2 单元格开始写入 DataFrame（如果适用，包括其标题）。这在向现有工作表添加数据或创建更复杂布局时会很有用。

### 将多个 DataFrame 写入单个 Excel 文件

Excel 文件相对于简单 CSV 的一个重要优点是它们能够包含多个工作表。Pandas 允许您使用 `pd.ExcelWriter` 对象将多个 DataFrame 写入同一个 `.xlsx` 文件中的不同工作表。这通常使用 `with` 语句完成，它能确保文件正确保存和关闭。

您可以这样做：

```python
import pandas as pd

# 创建两个示例 DataFrame
df_sales = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Sales': [1000, 1500, 1200, 1800]
})

df_inventory = pd.DataFrame({
    'Product ID': ['P101', 'P102', 'P103', 'P104'],
    'Stock': [50, 75, 30, 90],
    'Warehouse': ['WH-A', 'WH-B', 'WH-A', 'WH-C']
})

# 使用 ExcelWriter 将两个 DataFrame 保存到同一个文件中的不同工作表
output_filename = 'multi_sheet_report.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    df_sales.to_excel(writer, sheet_name='销售汇总', index=False)
    df_inventory.to_excel(writer, sheet_name='库存水平', index=False)

print(f"两个 DataFrame 已保存到 {output_filename}")
```

如果您打开 `multi_sheet_report.xlsx`，您会发现两个名为“销售汇总”和“库存水平”的工作表，每个工作表都包含相应 DataFrame 的数据。

### 所需库（引擎）

为写入 Excel 文件，Pandas 依赖于称为“引擎”的外部库。

- 对于现代 `.xlsx` 文件（推荐），您通常需要 `openpyxl` 库。
- 对于旧版 `.xls` 文件，曾使用 `xlwt` 库。

如果您尝试使用 `to_excel()` 但未安装所需引擎，Pandas 通常会引发 `ImportError`，并附带安装说明。您可以使用 pip 安装 `openpyxl`：

```bash
pip install openpyxl
```

或使用 conda：

```bash
conda install openpyxl
```

### 总结

使用 `df.to_excel()` 将 DataFrame 保存到 Excel 文件是一项有用的技能，可用于分享您的分析结果或处理过的数据，尤其是在电子表格是数据查看和操作标准工具的环境中。请记住 `index=False` 选项可以使输出更整洁，并在需要将相关数据整理到单个文件中的多个工作表时，考虑使用 `pd.ExcelWriter`。虽然 Excel 很方便，但请注意，对于非常大的数据集，其写入速度可能比 CSV 或更专业的二进制格式慢。

## 参考资料

- [pandas.DataFrame.to_excel](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html) — Pandas Development Team (2023)
  将DataFrame保存到Excel文件的官方文档，详细说明了可用参数和使用示例。
- [pandas.ExcelWriter](https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html) — Pandas Development Team (2024)
  管理Excel文件写入的官方文档，对于将多个DataFrame放置在不同工作表中很有用。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于Pandas的综合性书籍，由其创建者编写，包含数据加载和保存（例如到Excel文件）的部分。
