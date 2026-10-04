---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-pie-charts"
sourceId: 1175
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-pie-charts"
title: "用于表示比例的饼图"
description: "学习如何使用 Matplotlib 的 `pie` 函数创建饼图，并了解其常见局限性。"
order: 5
plots: ["plots/1175-0.json"]
sourceHash: "39bf27ec2c63397b5185b53eba59dd5a3e4e9e1bed9dc0a58970a627ac19c638"
sourceCorrections: []
---

条形图和直方图有助于比较不同数值或展示分布，但有时，您需要将不同部分如何构成整体的情况可视化。这时，饼图就派上用场了。饼图是一种圆形统计图，通过切片来表示数值比例。每个切片的弧长（以及其中心角和面积）与它所代表的数量成比例。

想象一下在朋友之间分一个真实的派；饼图对您的数据也如此，显示每个类别在总数中所占的百分比。当您拥有的类别数量较少（通常少于6个）并希望突出单个实体如何划分为不同部分时，它们最有效。

### 使用 `plt.pie()` 创建基本饼图

Matplotlib 提供了 `plt.pie()` 函数来创建饼图。其最基本的用法需要一个参数 (parameter)：一个表示每个切片大小的数值数组或列表。

假设我们对100人进行了一项关于他们最喜欢的原色的调查，结果是：45人选择蓝色，30人选择红色，15人选择绿色，10人选择黄色。

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据：每种颜色的票数
votes = np.array([45, 30, 15, 10])
colors = ['Blue', 'Red', 'Green', 'Yellow']

plt.figure(figsize=(6, 6)) # 使图形呈圆形
plt.pie(votes)
plt.title('最喜欢原色的调查结果（100票）')
plt.show()
```

这段简单的代码生成了一个饼图，但它缺少一些重要信息，比如哪个切片对应哪种颜色。

### 自定义饼图

`plt.pie()` 函数提供了几个参数 (parameter)，让您的图表更具信息量和视觉吸引力：

1. **`labels`**：一个字符串列表，为每个切片提供标签。顺序应与数据数组中值的顺序一致。
2. **`autopct`**：一个字符串或函数，用于格式化显示在每个切片上的数值。一个常用的格式字符串是 `'%1.1f%%'`，它显示百分比，保留一位小数，后面跟着一个 '%' 符号。末尾的 '%%' 用于转义百分号，使其原样显示。
3. **`colors`**：一个颜色名称或十六进制代码列表，用于手动设置每个切片的颜色。如果未提供，Matplotlib 会使用默认的颜色循环。
4. **`startangle`**：将第一个切片的起始位置从X轴逆时针旋转（以度为单位指定）。默认值为0。设置 `startangle=90` 通常看起来不错，它将第一个切片置于顶部（12点钟位置）。
5. **`explode`**：允许您将一个或多个切片从中心“拉出”以进行突出。它接受一个与您的数据长度相同的元组或列表，其中每个值指定切片偏移半径的比例。值为0表示切片保持原位。

让我们改进之前的例子：

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
votes = np.array([45, 30, 15, 10])
colors_labels = ['Blue', 'Red', 'Green', 'Yellow']
# 使用建议调色板中的十六进制代码定义自定义颜色
custom_colors = ['#339af0', '#fa5252', '#51cf66', '#fcc419']
# 略微“拉出”蓝色切片
explode_values = (0.1, 0, 0, 0) 

plt.figure(figsize=(7, 7)) 
plt.pie(votes, 
        labels=colors_labels, 
        colors=custom_colors,
        autopct='%1.1f%%', # 显示百分比
        startangle=90,      # 将第一个切片置于顶部
        explode=explode_values) # 突出第一个切片
plt.title('最喜欢原色的调查结果（100票）')
plt.show() 
```

这个版本清晰许多。我们可以看到标签、每种颜色的百分比贡献，以及蓝色切片为了突出而略微偏移。



![最喜欢原色的比例](plots/1175-0.json)



> 最喜欢原色的投票比例。蓝色切片略微拉出以作突出。

### 局限性以及何时避免使用饼图

虽然饼图在显示整体各部分时直观易懂，但它们存在明显局限：

- **难以比较相似的切片**：人类不擅长精确判断角度和面积。如果切片大小相似（例如，23%、25%、27%），仅凭饼图很难判断哪个最大或精确估计差异。
- **类别过多时表现不佳**：随着切片数量的增加，图表会变得杂乱且难以阅读。标签可能会重叠，小切片变得难以区分。一般建议是避免将饼图用于超过5-6个类别的情况。
- **零值或负值**：饼图无法表示零值或负值，因为它们处理的是正总数的比例。

**替代方案：** 为了比较类别之间的特定值，特别是当您有多个类别时，**条形图**（`plt.bar()` 或 `plt.barh()`）几乎总是更好的选择。条形图使用公共基线上的长度，这比角度或面积更容易让我们的眼睛进行精确比较。堆叠条形图也能有效地显示部分与整体的关系。

**谨慎使用饼图。** 它们在以下情况最适用：

- 您想显示整体的大致比例。
- 类别总数较少（理想情况下为5个或更少）。
- 各部分总和构成有意义的100%。
- 您想突出各部分对总体的相对贡献，而不是精确比较各部分之间的数据。

总结来说，`plt.pie()` 允许您创建表示比例的图表。虽然生成简单，但请记住它们在精确比较和适用于多类别方面的局限性。请始终考虑条形图是否能更有效地传达您的数据。

## 参考资料

- [matplotlib.pyplot.pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html) — Matplotlib Development Team (2024)
  Matplotlib饼图函数的官方文档，提供了详细的API参考和使用示例。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.05-histograms-and-binnings.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  Python数据科学工具的综合指南，其中包含有关Matplotlib的详细部分，涵盖各种图表类型和自定义设置。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/amounts.html) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供了一种数据可视化的原则性方法，讨论了何时使用以及何时避免使用饼图等特定图表类型。
