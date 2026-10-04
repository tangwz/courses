---
course: "intro-data-cleaning-preprocessing"
chapter: "correcting-data-types"
lesson: "converting-to-datetime"
sourceId: 4016
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-4-correcting-data-types/converting-to-datetime"
title: "转换为日期时间类型"
description: "学习如何将日期和时间的字符串表示解析为日期时间对象。"
order: 6
plots: ["plots/4016-0.json"]
sourceHash: "b9460fda6bec04903ab6ec5184cd76d5a3ade11740457d0be84e2b2461ca62cf"
sourceCorrections: []
---

日期和时间是许多数据集的基本组成部分，它们记录事件发生、交易进行或测量的时间。执行操作要求数据格式正确，日期和时间更是如此。如果日期以纯文本字符串形式存储（如“2023-10-26”或“Nov 5, 2022”），可能会遇到以下问题：

- **排序不当：** 按字母顺序排序文本无法得到时间顺序。例如，如果按第一个字符作为字符串排序，“02/15/2023”会排在“10/01/2022”之前。
- **无法计算：** 如果数据只是文本，无法轻松计算持续时间（例如，两个日期之间的天数）或进行基于时间的聚合。
- **提取组件：** 从字符串中提取年份、月份或星期几需要复杂的文本处理，而专门的日期/时间对象则使这变得直接简单。

大多数数据分析工具都提供专门的 `datetime` 对象（或类似结构），它们专门用于存储和处理日期和时间。我们的目标是将包含日期或时间信息的列（通常最初读取为字符串）转换为这些专门的类型。

### 多样格式的难题

一个难点是日期和时间可以用多种不同方式书写：

- `YYYY-MM-DD` (例如，`2023-11-21`) - 通用国际标准。
- `MM/DD/YYYY` (例如，`11/21/2023`) - 在美国常见。
- `DD/MM/YYYY` (例如，`21/11/2023`) - 在欧洲及其他地区常见。
- `DD-Mon-YYYY` (例如，`21-Nov-2023`)
- `Month DD, YYYY` (例如，`November 21, 2023`)
- 包含时间：`YYYY-MM-DD HH:MM:SS` (例如，`2023-11-21 14:30:00`)

转换过程需要处理这种多样性。

### 将字符串解析为日期时间对象

Python 中的数据分析库，例如 pandas，提供强大的函数来将这些字符串解析为日期时间对象。pandas 中最常用的函数是 `pd.to_datetime()`。

假设我们数据集中的 `order_date` 列如下所示：

```
['2023-01-15', '2023-02-10', '2023-03-05']
```

这些目前是字符串。应用 `pd.to_datetime()` 通常可以自动检测格式：

```python
# 假设 'df' 是你的 DataFrame，'order_date' 是要处理的列
import pandas as pd

df['order_date'] = pd.to_datetime(df['order_date'])
```

转换后，`order_date` 列将不再包含字符串值，而是专门的日期时间对象。

### 指定格式

有时，自动检测可能会失败或产生歧义，特别是对于像 `DD/MM/YYYY` 与 `MM/DD/YYYY` 这样的格式。在这种情况下，你可以使用格式代码明确告诉函数预期的格式。常用代码包括：

- `%Y`: 4 位年份（例如，2023）
- `%y`: 2 位年份（例如，23）
- `%m`: 2 位月份（01-12）
- `%d`: 2 位日期（01-31）
- `%b`: 缩写月份名（Jan, Feb）
- `%B`: 完整月份名（January, February）
- `%H`: 小时（24 小时制，00-23）
- `%M`: 分钟（00-59）
- `%S`: 秒（00-59）

如果你的日期是 `MM/DD/YY` 格式（例如，`11/21/23`），你将这样指定格式：

```python
df['event_date'] = pd.to_datetime(df['event_date'], format='%m/%d/%y')
```

使用正确的格式字符串可以确保准确解析，即使默认检测可能不可靠。

### 处理解析错误

如果列中的某个值无法转换怎么办？例如，该列可能包含“'unknown'”、“'2023-13-01'”（无效月份）或简单拼写错误等条目。默认情况下，`pd.to_datetime()` 将引发错误并停止转换。

通常，更实际的做法是将有问题的数据转换为表示缺失或无效时间的特殊值，即 `NaT`（非时间）。这类似于数值类型中的 `NaN`。可以使用 `errors` 参数 (parameter)实现此目的：

```python
# 转换有效日期，将错误转换为 NaT
df['measurement_time'] = pd.to_datetime(df['measurement_time'], errors='coerce')
```

将 `errors='coerce'` 设置为告诉函数尝试转换，如果某个特定值转换失败，则用 `NaT` 替换该值，而不是停止整个过程。然后，你可以使用前面讨论的缺失数据处理方法（例如，插补或删除）来处理这些 `NaT` 值。另一个选项 `errors='ignore'`，如果无法解析，将简单地返回原始输入，将有问题的字符串保留在原位，这通常不太理想。

### 转换后的好处

一旦你的日期/时间列转换为正确的日期时间格式，你将获得重要的分析能力：

1. **时间顺序排序：** 现在对列进行排序可以正确地排列日期/时间。
2. **组件提取：** 轻松获取日期或时间的部分。例如，在 pandas 中：
   - `df['order_date'].dt.year` 提取年份。
   - `df['order_date'].dt.month` 提取月份数字。
   - `df['order_date'].dt.day` 提取日期。
   - `df['order_date'].dt.dayofweek` 提取星期几（周一=0，周日=6）。
   - `df['order_date'].dt.hour` 提取小时（如果存在时间）。
3. **基于时间的计算：** 计算日期之间的差异以求持续时间。

   ```python
   # 示例：计算两个日期列之间的天数
   df['processing_days'] = (df['ship_date'] - df['order_date']).dt.days
   ```
4. **筛选和分组：** 在特定日期范围内选择数据，或按月份、年份或星期几对数据进行分组以进行分析。

让我们看看转换后如何通过提取月份轻松进行聚合。假设我们已转换 `event_timestamp` 列并想查看每月事件的数量：

```python
# 假设 df['event_timestamp'] 现在是日期时间列
monthly_counts = df['event_timestamp'].dt.month.value_counts().sort_index()
# 这将给出月份 1（一月）、2（二月）等的计数。
```

然后我们可以绘制这些计数：



![每月事件计数](plots/4016-0.json)



> 一个简单的条形图，显示在将时间戳列转换为日期时间格式后，按月份聚合的已记录事件数量。

将日期和时间的字符串表示转换为正确的日期时间对象是必要的一步，以实现准确的排序、计算和基于时间的分析。使用像 `pd.to_datetime` 这样具有适当格式指定和错误处理功能的工具，确保你的时间序列数据可以进行有意义的解读。

## 参考资料

- [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) — pandas development team (2024)
  主要功能的官方文档，涵盖所有参数、格式代码和错误处理。
- [Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) — pandas development team (2023)
  pandas日期时间功能的详细用户指南，包括to_datetime、dt访问器和基于时间的操作。
- [datetime - Basic date and time types](https://docs.python.org/3/library/datetime.html) — Python Software Foundation (2023)
  解释datetime模块、其对象以及strftime和strptime格式代码的Python官方文档。
