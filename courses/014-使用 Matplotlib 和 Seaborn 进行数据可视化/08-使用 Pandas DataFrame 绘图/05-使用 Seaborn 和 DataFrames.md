# 使用 Seaborn 和 DataFrames

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/seaborn-with-pandas-dataframes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Matplotlib 确实可以绘制 Pandas DataFrames 中包含的数据，通过访问特定列（通常是 Series 对象），但 Seaborn 库提供了一种更直接且通常更便捷的方法。Seaborn 在设计时就考虑到了 Pandas DataFrames，这使得从结构化数据创建富有信息的统计图变得非常简单。

Seaborn 与 DataFrames 结合的主要方式是通过其大多数绘图函数中可用的 `data` 参数 (parameter)。您通常会将整个 DataFrame 传递给 `data` 参数，而不是为 x 和 y 值传递单独的 NumPy 数组或 Pandas Series。然后，通过将列*名称*（字符串形式）传递给 `x`、`y`、`hue`、`size` 等参数，指定 DataFrame 中的哪些*列*应用于不同的绘图轴或属性（如颜色、大小或样式）。

我们通过一个例子来说明这一点。假设您已将数据加载到一个名为 `df` 的 DataFrame 中：

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个示例 DataFrame（在实际情况中，您会从文件中加载）
data = {
    'Experiment_ID': ['Exp1', 'Exp1', 'Exp1', 'Exp2', 'Exp2', 'Exp2', 'Exp3', 'Exp3', 'Exp3'],
    'Temperature': [20, 25, 30, 20, 25, 30, 20, 25, 30],
    'Yield': [75, 82, 88, 78, 85, 91, 72, 79, 85],
    'Replicate': [1, 2, 3, 1, 2, 3, 1, 2, 3]
}
df = pd.DataFrame(data)

# 显示 DataFrame
print(df)
```

这可能会输出：

```
  Experiment_ID  Temperature  Yield  Replicate
0          Exp1           20     75          1
1          Exp1           25     82          2
2          Exp1           30     88          3
3          Exp2           20     78          1
4          Exp2           25     85          2
5          Exp2           30     91          3
6          Exp3           20     72          1
7          Exp3           25     79          2
8          Exp3           30     85          3
```

现在，要创建一个散点图，显示 `Temperature` 和 `Yield` 之间的关系，并根据 `Experiment_ID` 对点进行着色，您可以使用 Seaborn 如下操作：

```python
# 使用 Seaborn 创建散点图
sns.scatterplot(data=df, x='Temperature', y='Yield', hue='Experiment_ID')

# 添加标题（使用 Matplotlib 的函数）
plt.title('Effect of Temperature on Yield by Experiment')

# 显示图表
plt.show()
```



[交互图表：温度对不同实验产量的影响](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/seaborn-with-pandas-dataframes#plot-1tj938g)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        20,
        25,
        30
      ],
      "y": [
        75,
        82,
        88
      ],
      "marker": {
        "color": "#4263eb",
        "symbol": "circle"
      },
      "name": "实验1"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        20,
        25,
        30
      ],
      "y": [
        78,
        85,
        91
      ],
      "marker": {
        "color": "#f76707",
        "symbol": "circle"
      },
      "name": "实验2"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        20,
        25,
        30
      ],
      "y": [
        72,
        79,
        85
      ],
      "marker": {
        "color": "#12b886",
        "symbol": "circle"
      },
      "name": "实验3"
    }
  ],
  "layout": {
    "xaxis": {
      "title": {
        "text": "温度"
      }
    },
    "yaxis": {
      "title": {
        "text": "产量"
      }
    },
    "title": {
      "text": "温度对不同实验产量的影响"
    },
    "legend": {
      "title": {
        "text": "实验ID"
      }
    },
    "template": "simple_white",
    "autosize": true
  }
}
```

</details>



> 散点图显示产量与温度的关系，其中点根据 DataFrame 中的 Experiment\_ID 列着色。

请注意我们如何将整个 `df` 传递给 `data`。然后，我们只需使用字符串 `'Temperature'`、`'Yield'` 和 `'Experiment_ID'` 来告诉 `scatterplot` 分别将哪些列映射到 x 轴、y 轴和点的颜色 (`hue`)。

### Seaborn 方法的优点

使用 `data` 参数 (parameter)有以下几个优点：

1. **易读性：** 代码清楚地说明了数据源 (`data=df`) 和正在绘制的变量 (`x='ColumnA'`, `y='ColumnB'`)。这使得代码更易于理解和维护。
2. **简洁性：** 您无需在将列传递给绘图函数之前手动提取它们（例如 `df['Temperature']`、`df['Yield']`）。
3. **自动标注：** Seaborn 经常使用提供的列名自动标注轴和图例，减少了样板代码。如果需要，您仍然可以使用 Matplotlib 函数进一步自定义这些标注。
4. **与整洁数据配合使用：** Seaborn 函数通常设计为最适合‘整洁’数据，其中每行是一个观测值，每列是一个变量。这是 Pandas DataFrames 的标准格式，使得结合很自然。

### 更多示例

这种模式适用于大多数 Seaborn 绘图函数。例如，要创建箱线图以比较每个 `Experiment_ID` 的 `Yield` 分布：

```python
sns.boxplot(data=df, x='Experiment_ID', y='Yield')
plt.title('各实验的产量分布')
plt.show()
```

或者，要计算每个 `Temperature` 值的出现次数（尽管对于这个特定的小型数据集意义不大，但它说明了这种模式）：

```python
sns.countplot(data=df, x='Temperature')
plt.title('温度读数的频率')
plt.show()
```

### 结合 Seaborn 和 Matplotlib

请记住，Seaborn 图是在 Matplotlib 轴上绘制的。这意味着您可以在创建 Seaborn 图后，始终使用 Matplotlib 函数自定义其外观。如上例所示，`plt.title()`、`plt.xlabel()`、`plt.ylabel()`、`plt.xlim()` 等函数在 Seaborn 生成的图上运行良好。您获得了 Seaborn 的高级统计绘图功能以及 Matplotlib 的精细控制能力。

总而言之，当您的数据位于 Pandas DataFrame 中时，借助 Seaborn 的 `data` 参数 (parameter)以及 `x`、`y`、`hue` 和其他参数的列名字符串，通常是创建可视化的最有效和符合 Python 习惯的方式。它简化了代码，并与数据分析中使用的标准数据结构顺畅结合。

## 参考资料

- [Introduction to Seaborn](https://seaborn.pydata.org/tutorial/introduction.html) — Michael Waskom and the Seaborn development team (2024)
  官方入门教程，演示如何使用 `data` 参数将 Pandas DataFrames 传递给 Seaborn 函数，并通过列名指定绘图数据。
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本全面的资源，涵盖 Pandas DataFrames、Matplotlib 和 Seaborn，详细介绍了它们各自的使用以及它们在数据分析和可视化中的相互作用。
- [Python for Data Analysis](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG6ecv0iyGQwnZbMPLNM5z2y9nEn9uB2RPyL-wRdySU4v99qxCWqoq8sKOD6TXwdi4NYFgCgDeRMaWSJxFRcI3cLTKiPOewYnEPdBNnNRc52Ye2sNbylojPLalEPmyBu5PG5su5drqyEL_6kVHQBW8dQPh-5mG56RgXzQ3Z6GElK79vp9l4FwMgiokdR__PqmyN9laMLeg=) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  这本关于 Pandas 的权威书籍提供了 DataFrames 的重要背景知识，DataFrames 是 Seaborn 用于统计绘图的基础数据结构。第三版。

---

[上一节](04-%E4%BD%BF%E7%94%A8%20Matplotlib%20%E9%85%8D%E5%90%88%20DataFrame%20%E5%88%97.md) · [下一节](06-%E6%95%B0%E6%8D%AE%E7%AD%9B%E9%80%89%E4%B8%8E%E7%BB%98%E5%9B%BE%E5%87%86%E5%A4%87.md)
