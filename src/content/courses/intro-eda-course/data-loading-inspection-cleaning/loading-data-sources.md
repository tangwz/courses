---
course: "intro-eda-course"
chapter: "data-loading-inspection-cleaning"
lesson: "loading-data-sources"
sourceId: 1135
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/loading-data-sources"
title: "从多种来源加载数据（CSV、Excel、JSON）"
description: "将数据从CSV、Excel和JSON等常见文件格式加载到Pandas DataFrame中的方法。"
order: 1
plots: []
sourceHash: "daca61d581f282b3af702fba53b930c95a55a4c84bced9dbffca922e53c6e254"
sourceCorrections: []
---

在分析数据之前，你需要将其导入到你的分析环境中。正如前一章所述，Python，特别是Pandas库，为此提供了工具。本节着重于从你将遇到的一些最常见文件格式中加载数据的实际操作步骤：逗号分隔值（CSV）、Microsoft Excel电子表格以及JavaScript对象表示（JSON）。掌握这些初始步骤对于任何数据分析项目都非常重要。

我们假设你已按照第1章的说明，设置好Python环境并安装了Pandas。

### 加载CSV文件

CSV文件或许是存储表格数据最常见的格式。它们是纯文本文件，值通常用逗号分隔。Pandas提供了`read_csv()`函数，这是一个将这些文件读取到DataFrame中的多功能工具。

最基本的用法只需提供文件路径：

```python
import pandas as pd

# 假设 'data.csv' 与你的脚本或Jupyter Notebook在同一目录下
try:
    df_csv = pd.read_csv('data.csv')
    print("CSV文件加载成功。")
    # 继续检查 df_csv（将在下一节介绍）
except FileNotFoundError:
    print("错误：未找到 data.csv。请检查文件路径。")
except Exception as e:
    print(f"发生错误：{e}")
```

尽管这适用于标准、逗号分隔的文件，但`read_csv()`提供了许多参数 (parameter)来处理各种情况：

- `filepath_or_buffer`：这是主要参数，接受文件路径（字符串）、URL或任何具有`read()`方法的对象。
- `sep`（或 `delimiter`）：指定用于分隔值的字符。如果你的文件使用制表符（`\t`）、分号（`;`）或其他分隔符，你必须相应地设置此参数。例如：`pd.read_csv('data.tsv', sep='\t')`。Pandas通常会推断分隔符，但明确定义更安全。
- `header`：指示哪一行包含列名。默认情况下，它是`header=0`（第一行）。如果你的文件没有标题行，请使用`header=None`。Pandas将自动分配默认的整数列名（0, 1, 2...）。如果标题行不在第一行，你也可以指定不同的行索引。
- `index_col`：你可以指定CSV文件中的一个或多个列作为DataFrame加载时的索引。例如，`index_col='ID'`将使用名为“ID”的列作为索引。
- `usecols`：如果你只需要部分列，请向此参数提供列名或索引列表，例如`usecols=['Name', 'Age', 'City']`。这可以节省宽数据集的内存和加载时间。
- `dtype`：允许你在加载时使用字典指定特定列或所有列的数据类型，例如`dtype={'CustomerID': str, 'Amount': float}`。这可以防止不正确的类型推断（例如将数字ID读取为整数，而它们应该是字符串）。
- `nrows`：对于非常大的文件很有用。设置`nrows=1000`可以只读取前1000行（不包括标题行）。这有助于在加载整个数据集之前进行快速查看。
- `encoding`：如果文件编码不是默认的（通常是UTF-8），则指定其编码。常见的替代方案包括`'latin1'`、`'iso-8859-1'`或`'cp1252'`，特别是在处理旧文件或来自不同操作系统的文件时。如果你遇到`UnicodeDecodeError`，请尝试指定编码。

```python
# 示例：加载一个没有标题、使用特定列的分号分隔文件
try:
    columns_to_load = ['Timestamp', 'SensorValue']
    df_sensor = pd.read_csv(
        'sensor_data.log',
        sep=';',
        header=None,
        names=columns_to_load, # 如果 header=None，则提供名称
        usecols=[0, 2],        # 只加载第1列和第3列
        encoding='latin1'
    )
    print("传感器日志加载成功。")
except FileNotFoundError:
    print("错误：未找到 sensor_data.log。")
except Exception as e:
    print(f"发生错误：{e}")
```

### 加载Excel文件

Excel文件（`.xls`或`.xlsx`）也经常用于存储数据，通常包含多个工作表。Pandas使用`read_excel()`函数来处理此任务。请注意，`read_excel()`需要额外的库：对于`.xlsx`文件需要`openpyxl`，对于较旧的`.xls`文件需要`xlrd`。如果你尚未安装它们，可以通过pip安装：

```bash
pip install openpyxl xlrd
```

基本用法与`read_csv()`类似：

```python
import pandas as pd

try:
    # 默认加载第一个工作表
    df_excel = pd.read_excel('spreadsheet.xlsx')
    print("Excel文件加载成功（第一个工作表）。")
except FileNotFoundError:
    print("错误：未找到 spreadsheet.xlsx。")
except Exception as e:
    print(f"发生错误：{e}")
```

`read_excel()`的重要参数 (parameter)包括：

- `io`：文件路径、URL或类似文件的对象。
- `sheet_name`：这对Excel文件很重要。你可以指定：
  - 工作表名称的字符串（例如，`sheet_name='Sales Data'`）。
  - 工作表索引的整数（从0开始，例如，`sheet_name=0`表示第一个工作表）。
  - 字符串/整数列表，用于将特定的多个工作表加载到DataFrames字典中（例如，`sheet_name=['Orders', 'Customers']`）。
  - `None`表示将所有工作表加载到DataFrames字典中。
- `header`、`index_col`、`usecols`、`dtype`、`nrows`：这些参数的功能与`read_csv()`中对应的参数类似。

```python
# 示例：加载特定工作表（'Inventory'）并指定标题行
try:
    df_inventory = pd.read_excel(
        'company_data.xlsx',
        sheet_name='Inventory',
        header=2 # 标题行在第3行（索引为2）
    )
    print("库存工作表加载成功。")

    # 示例：加载所有工作表
    all_sheets = pd.read_excel('company_data.xlsx', sheet_name=None)
    print(f"已加载工作表：{list(all_sheets.keys())}")
    # all_sheets 现在是一个字典，其键是工作表名称
    # 值是对应的DataFrame。
    # df_orders = all_sheets['Orders']

except FileNotFoundError:
    print("错误：未找到 company_data.xlsx。")
except ImportError:
    print("错误：请安装 'openpyxl' 或 'xlrd' 来读取Excel文件。")
    print("运行：pip install openpyxl xlrd")
except Exception as e:
    print(f"发生错误：{e}")
```

### 加载JSON文件

JSON（JavaScript对象表示法）是一种轻量级数据交换格式，在Web API和配置文件中很常见。其结构可能比扁平的CSV或Excel文件更复杂，常包含嵌套对象和数组。Pandas提供了`read_json()`来处理这种格式。

`read_json()`的有效性很大程度上取决于JSON文件的结构。一个常见的问题是JSON不强制单一的表格结构。

```python
import pandas as pd

# 示例假设一个简单的JSON结构（记录列表）
# [{'col1': 'a', 'col2': 1}, {'col1': 'b', 'col2': 2}]
try:
    df_json = pd.read_json('data.json')
    print("JSON文件加载成功。")
except FileNotFoundError:
    print("错误：未找到 data.json。")
except ValueError as ve:
    # 如果JSON结构不是直接的表格形式，通常会发生这种情况
    print(f"加载JSON时发生ValueError：{ve}")
    print("检查JSON结构并考虑 'orient' 参数。")
except Exception as e:
    print(f"发生错误：{e}")
```

`read_json()`最重要的参数 (parameter)是`orient`，它告诉Pandas如何解读JSON结构：

- `'records'`：（如果`lines=False`则为默认值）JSON字符串是一个JSON对象数组。示例：`[{'col A': 1, 'col B': 0.5}, {'col A': 2, 'col B': 0.7}]`。每个对象成为一行。
- `'columns'`：JSON字符串是一个字典，其中键是列名，值是将索引标签映射到单元格值的字典。示例：`{'col A': {'row1': 1, 'row2': 2}, 'col B': {'row1': 0.5, 'row2': 0.7}}`。
- `'index'`：与`'columns'`类似，但内部字典将列名映射到单元格值。示例：`{'row1': {'col A': 1, 'col B': 0.5}, 'row2': {'col A': 2, 'col B': 0.7}}`。
- `'split'`：JSON字符串是一个字典，包含键`'index'`、`'columns'`和`'data'`。示例：`{'columns': ['col A', 'col B'], 'index': ['row1', 'row2'], 'data': [[1, 0.5], [2, 0.7]]}`。
- `'values'`：JSON字符串只是一个包含数据值的数组的数组。示例：`[[1, 0.5], [2, 0.7]]`。列名将是数字（0, 1, ...）。

你通常需要先检查JSON文件的结构，以确定正确的`orient`值。

```python
# 示例：以 'columns' 方向加载JSON
# {"col1": {"0": "a", "1": "b"}, "col2": {"0": 1, "1": 2}}
try:
    df_json_cols = pd.read_json('data_columns.json', orient='columns')
    print("JSON（'columns' 方向）加载成功。")
except FileNotFoundError:
    print("错误：未找到 data_columns.json。")
except Exception as e:
    print(f"发生错误：{e}")

# 示例：加载行分隔的JSON（每行是一个有效的JSON对象）
# {"col1": "a", "col2": 1}
# {"col1": "b", "col2": 2}
try:
    # 对于行分隔的JSON，使用 lines=True（orient 默认为 'records'）
    df_json_lines = pd.read_json('data_lines.jsonl', lines=True)
    print("行分隔JSON加载成功。")
except FileNotFoundError:
    print("错误：未找到 data_lines.jsonl。")
except Exception as e:
    print(f"发生错误：{e}")
```

对于深度嵌套的JSON数据，`read_json()`可能会生成包含字典或列表的列。虽然Pandas可以处理这些，但你可能需要更高级的技术，例如`pandas.json_normalize()`或自定义解析逻辑，以将结构展平为适合分析的标准表格格式。这些是值得了解的有用工具。

### 常见加载注意事项

- **文件路径：** 确保你的文件路径正确。相对路径（如`'data.csv'`）是相对于你的脚本或Jupyter Notebook的当前工作目录。绝对路径（如`'/Users/you/Documents/data.csv'`或`'C:\\Users\\you\\Documents\\data.csv'`）也有效。使用`os`或`pathlib`等库可以帮助在不同操作系统中可靠地管理路径。
- **错误处理：** 始终将文件加载操作包装在`try...except`块中，以妥善处理`FileNotFoundError`或其他问题，例如格式不正确或权限问题。

成功加载数据是重要的第一步。一旦数据进入Pandas DataFrame，你就可以进入下一个阶段：检查其基本属性、结构和类型，这是即将到来的章节的主要内容。

## 参考资料

- [User Guide: Input/Output tools (Text, CSV, HDF5, ...) - pandas 2.2.0 documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) — The Pandas Development Team (2024)
  Pandas官方用户指南，全面介绍如何从CSV、Excel、JSON等多种格式加载数据。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者撰写的权威书籍，详细讲解使用Pandas加载和处理数据的实用示例和说明。
- [pandas.read_json - pandas 2.2.0 documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html) — The Pandas Development Team (2024)
  `pandas.read_json`的官方API文档，详细说明其参数和行为，对于理解JSON结构解释特别有用。
