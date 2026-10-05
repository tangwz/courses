# 使用 Matplotlib 配合 DataFrame 列

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/matplotlib-with-pandas-columns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 Pandas DataFrame 提供便捷的内置绘图方法，但通过直接使用 Matplotlib 函数处理 DataFrame 数据，您通常会获得更多控制和灵活性。这种方式让您能够使用 Matplotlib 库的所有功能进行自定义，并处理更复杂的图表结构，例如子图。

核心思路很简单：不是在 DataFrame 本身调用方法（例如 `df.plot()`），而是调用 Matplotlib 函数（例如 `plt.scatter()` 或 `plt.plot()`），并传递 DataFrame 中的特定列作为数据输入。

### 获取 DataFrame 列进行绘图

回想一下，一个 Pandas DataFrame 就像一个表格，其中每列都有一个名称（标签）。您可以使用方括号表示法 `df['column_name']` 或点表示法 `df.column_name`（如果列名是有效的 Python 标识符且不与 DataFrame 方法冲突）来选择单个列。以这种方式选择列会返回一个 Pandas Series 对象，Matplotlib 能够将其识别为输入数据。

### 使用 Matplotlib 函数创建图表

假设您已将数据加载到名为 `sales_df` 的 DataFrame 中，其结构大致如下：

```python
# 示例 DataFrame 创建（在实际情况中，您会从文件中加载数据）
import pandas as pd
import matplotlib.pyplot as plt

data = {'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'units_sold': [150, 165, 180, 175, 190, 210],
        'revenue': [3000, 3300, 3600, 3500, 3800, 4200]}
sales_df = pd.DataFrame(data)

print(sales_df)
# 输出：
#   month  units_sold  revenue
# 0   Jan         150     3000
# 1   Feb         165     3300
# 2   Mar         180     3600
# 3   Apr         175     3500
# 4   May         190     3800
# 5   Jun         210     4200
```

现在，您可以使用 Matplotlib 函数来显示数据内的关系。

#### 示例 1：按月份销量折线图

要绘制 `units_sold` 在 `month` 上的趋势，您可以将相应列传递给 `plt.plot()`：

```python
# 选择列并将其传递给 Matplotlib 的绘图函数
plt.plot(sales_df['month'], sales_df['units_sold'])

# 添加标签和标题以增强清晰度
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.title('Monthly Units Sold')

# 显示图表
plt.grid(True, linestyle='--', alpha=0.6) # 添加一个不显眼的网格
plt.show()
```

这段代码直接指示 Matplotlib 使用 'month' 列作为 x 轴值，使用 'units\_sold' 列作为 y 轴值。然后，我们使用 `plt.xlabel()`、`plt.ylabel()` 和 `plt.title()` 等标准 Matplotlib 函数为图表添加注释。



[交互图表：月销量](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/matplotlib-with-pandas-columns#plot-1a5nm68)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "x": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun"
      ],
      "y": [
        150,
        165,
        180,
        175,
        190,
        210
      ],
      "marker": {
        "color": "#228be6"
      },
      "name": "销量"
    }
  ],
  "layout": {
    "title": "月销量",
    "xaxis": {
      "title": "月份"
    },
    "yaxis": {
      "title": "销量"
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "height": 300,
    "template": "simple_white"
  }
}
```

</details>



> 月销量趋势图，使用 Matplotlib 并从特定的 DataFrame 列获取数据生成。

#### 示例 2：收入与销量的散点图

要查看 `units_sold` 与 `revenue` 之间的关系，您可以使用 `plt.scatter()` 创建一个散点图：

```python
# 将列传递给 Matplotlib 的散点图函数
plt.scatter(sales_df['units_sold'], sales_df['revenue'], color='#fd7e14', marker='o') # 使用橙色圆点

# 添加标签和标题
plt.xlabel('Units Sold')
plt.ylabel('Revenue ($)')
plt.title('Revenue vs. Units Sold')

# 显示图表
plt.grid(True, linestyle=':', alpha=0.5) # 添加一个虚线网格
plt.show()
```

这里，`sales_df['units_sold']` 提供散点图中每个点的 x 坐标，`sales_df['revenue']` 提供 y 坐标。我们还在 Matplotlib 函数调用中直接指定了颜色和标记 (token)样式。



[交互图表：收入与销量](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/matplotlib-with-pandas-columns#plot-11e6zqi)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        150,
        165,
        180,
        175,
        190,
        210
      ],
      "y": [
        3000,
        3300,
        3600,
        3500,
        3800,
        4200
      ],
      "marker": {
        "color": "#fd7e14",
        "size": 8
      },
      "name": "销售数据"
    }
  ],
  "layout": {
    "title": "收入与销量",
    "xaxis": {
      "title": "销量"
    },
    "yaxis": {
      "title": "收入 ($)"
    },
    "margin": {
      "l": 60,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "height": 300,
    "template": "simple_white"
  }
}
```

</details>



> 销量与收入的关系图，使用 Matplotlib 散点图并以 DataFrame 列作为输入显示。

### 为何采用这种方式？

1. **对 Matplotlib 的全面控制：** 您可以直接访问整个 Matplotlib API。这包括对图表每个元素的精细控制，例如特定的标记 (token)样式、复杂的注释、误差棒等，可以直接在函数调用中实现，或使用 Figure 和 Axes 对象（后续会介绍）。
2. **一致性：** 如果您已大量使用 Matplotlib 处理各种绘图任务，那么直接配合 DataFrame 使用它能够保持一致的工作流程。
3. **复杂布局：** 当创建包含多个子图的图表时，这种方法能够自然地集成。您可以轻松地将不同的 DataFrame 列或数据子集引导到同一 Figure 中的不同 Axes 对象。

### 总结

直接使用 Matplotlib 函数配合 Pandas DataFrame 列，提供了一种强大而灵活的方式来生成可视化内容。您可以从 DataFrame 中选择所需的列（Pandas Series），并将其作为参数 (parameter)传递给 Matplotlib 绘图函数，例如 `plt.plot()`、`plt.scatter()`、`plt.bar()` 等。这种方式让您能够对图表的外观和结构进行全面控制，当内置的 Pandas 绘图方法不足，或者当您需要将绘图紧密集成到更大的 Matplotlib 图表结构中时，它是一项重要技巧。这强调了这些库的作用：Pandas 用于数据管理，Matplotlib 用于制作可视化内容。

## 参考资料

- [\`pyplot\` tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) — Matplotlib Developers (2024)
  Publisher: Matplotlib Development Team
  解释了`matplotlib.pyplot`模块，该模块提供类似MATLAB的绘图接口，包括创建图形、绘图区域和自定义绘图元素的函数。
- [User Guide: Indexing and selecting data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) — Pandas Development Team (2024)
  详细介绍了从Pandas DataFrame和Series中选择数据的各种方法，这对于将数据传递给绘图函数至关重要。
- [Python for Data Analysis](https://oreil.ly/python-data-analysis-3e) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  Pandas创建者撰写的权威参考书，全面涵盖了使用Pandas进行数据操作及其与Matplotlib的数据可视化集成。

---

[上一节](03-%E7%9B%B4%E6%8E%A5%E4%BB%8EDataFrame%E8%BF%9B%E8%A1%8C%E5%9F%BA%E7%A1%80%E7%BB%98%E5%9B%BE.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20Seaborn%20%E5%92%8C%20DataFrames.md)
