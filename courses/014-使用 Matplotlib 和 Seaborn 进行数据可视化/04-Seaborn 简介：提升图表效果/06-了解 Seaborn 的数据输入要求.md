# 了解 Seaborn 的数据输入要求

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-data-input)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Seaborn 简化了创建美观且统计信息丰富的图表的流程。这种简化很大程度上源于 Seaborn 对其绘图函数数据提供方式的预期。Matplotlib 通常能很好地处理表示单个变量的简单列表或 NumPy 数组，而 Seaborn 主要设计用于处理结构化数据集，最常见的是 Pandas DataFrame。

### 整洁数据的重要性

许多 Seaborn 函数在您的数据采用“整洁”格式（也称为长格式数据）时表现最佳。整洁数据是一种组织表格数据的约定，它与统计分析和可视化非常吻合。主要原则是：

1. **每个变量形成一列。** 您测量或观察到的每个不同信息（例如温度、身高、类别、时间）都应有其自己的列。
2. **每个观测值形成一行。** 您测量的每个不同实例或单位都构成一行。
3. **每种观测单位形成一个表。** 如果您测量不同类型的事物，它们最好位于单独的表（或 DataFrame）中。

设想一个简单的电子表格，其中列代表测量值（例如“日期”、“温度”、“降雨量”），每行代表一个具体的观测值（例如周一的数据，周二的数据）。这种结构使得告知绘图函数将哪个变量映射到哪个视觉属性（如 x 轴、y 轴或颜色）变得简单明了。

### 使用 Pandas DataFrame

向 Seaborn 提供数据最常用且推荐的方式是通过 Pandas DataFrame。这与整洁数据的观念自然契合。Seaborn 函数通常包含一个 `data` 参数 (parameter)，您在此处传递整个 DataFrame。然后，您可以使用 `x`、`y`、`hue`、`size` 或 `style` 等其他参数来指定 DataFrame 中的*哪些列*应用于图表的不同方面。您以字符串形式提供列的*名称*。

我们来看一个例子。假设您有一个名为 `weather_df` 的 Pandas DataFrame，其中包含“Day”、“Temperature”、“Humidity”和“City”列。

```python
# 假设 pandas 已导入为 pd，seaborn 已导入为 sns
# 假设 weather_df 是一个如下所示的 DataFrame：
#     Day  Temperature  Humidity     City
# 0   Mon           25        60   London
# 1   Tue           28        65   London
# 2   Wed           22        55   London
# 3   Mon           30        70    Paris
# 4   Tue           32        75    Paris
# 5   Wed           28        68    Paris

# 使用列名创建散点图
sns.scatterplot(data=weather_df, x='Temperature', y='Humidity', hue='City')

# 创建一个折线图，显示每个城市每天的温度变化
sns.lineplot(data=weather_df, x='Day', y='Temperature', hue='City')

# 显示图表（假设您使用的是 Matplotlib 后端）
# import matplotlib.pyplot as plt
# plt.show()
```

请注意我们如何将整个 `weather_df` 传递给 `data` 参数。然后，我们使用列的*字符串名称*（'Temperature'、'Humidity'、'City'、'Day'）作为 `x`、`y` 和 `hue` 参数。Seaborn 使用这些名称在 DataFrame 中找到相应的数据。这种方法使代码可读，并将可视化参数直接与数据结构关联起来。它还允许 Seaborn 自动使用列名作为轴标签和图例，从而省去了使用基本 Matplotlib 时通常需要的手动配置步骤。

### 其他输入格式

虽然 DataFrame 是首选，但 Seaborn 有时也接受其他格式，例如：

- **NumPy 数组或列表：** 您可以直接将 NumPy 数组或 Python 列表传递给 `x` 和 `y` 等参数 (parameter)。在这种情况下，您通常不会使用 `data` 参数。

  ```python
  temperatures = [25, 28, 22, 30, 32, 28]
  humidity = [60, 65, 55, 70, 75, 68]
  # 注意：在没有 DataFrame 结构的情况下，颜色映射通常不够直接
  sns.scatterplot(x=temperatures, y=humidity)
  # plt.show()
  ```

  然而，使用单独的数组或列表通常需要您手动设置标签，并且与使用 DataFrame 相比，基于第三个变量应用像 `hue` 这样的分组美学变得不那么直接。
- **宽格式数据：** 一些 Seaborn 函数可以解析“宽格式”数据，其中不同的列可能代表变量的水平（例如列“Temp\_London”、“Temp\_Paris”）。虽然这有可能，但它通常不如整洁/长格式方法灵活，特别是对于涉及多个分类分组的复杂图表。长格式数据通常提供一种更一致、更有效的方法，将变量映射到 Seaborn 中的视觉美学。

### 总结

为了充分发挥 Seaborn 轻松创建信息丰富的统计图表的能力，强烈建议您使用整洁数据的观念，将数据构建为 Pandas DataFrame。通过将 DataFrame 传递给 `data` 参数 (parameter)，并使用列名字符串来引用 `x`、`y`、`hue` 等，您可以有效使用 Seaborn 的高级界面，从而使代码更清晰，并且通常以更少的精力获得更具洞察力的可视化效果。在您学习本课程的过程中，大多数示例都会假设您的数据处于或可以轻松转换为整洁的 Pandas DataFrame。

## 参考资料

- [Tidy Data](https://www.jstatsoft.org/article/view/v059i10) — Wickham, H. (2014)
  Journal: Journal of Statistical Software; Volume: 59; Pages: 1-23; DOI: [10.18637/jss.v059.i10](https://doi.org/10.18637/jss.v059.i10)
  这篇基础性论文介绍了整洁数据的概念，阐述了其原则及其在统计分析和可视化方面的优势。
- [Data structures supported by seaborn](https://seaborn.pydata.org/tutorial/data_input.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn官方用户指南中关于数据结构的部分，着重介绍了Seaborn如何处理数据，主要强调Pandas DataFrame和长格式（整洁）数据。
- [User Guide: Data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) — The Pandas development team (2024)
  Pandas官方文档中关于DataFrame数据结构的介绍，解释了其结构和基本操作，这对于理解Seaborn的常用输入格式很有帮助。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media; Pages: Chapter 4.14: Visualization With Seaborn
  一本备受推崇的书籍，提供了Python数据科学栈的详细介绍，其中有一个章节专门讨论Seaborn，内容涉及数据可视化中其数据输入规范。

---

[上一节](05-%E4%BD%BF%E7%94%A8%20Seaborn%20%E7%BB%98%E5%88%B6%E6%8A%98%E7%BA%BF%E5%9B%BE%20%28lineplot%29.md) · [下一节](07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E4%BD%BF%E7%94%A8Seaborn%E9%87%8D%E5%88%B6%E5%9B%BE%E8%A1%A8.md)
