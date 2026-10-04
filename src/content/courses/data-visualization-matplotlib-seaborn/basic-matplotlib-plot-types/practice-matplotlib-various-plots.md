---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "practice-matplotlib-various-plots"
sourceId: 1178
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/practice-matplotlib-various-plots"
title: "动手实践：绘制不同数据类型图表"
description: "应用所学知识，通过使用 Matplotlib 创建条形图、直方图和饼图。"
order: 7
plots: []
sourceHash: "10ba06114f5da3a30a2534d6dd0a8b353f7b52a6765b2b79117d26a846718c05"
sourceCorrections: []
---

通过示例数据制作条形图、直方图和饼图。这种动手操作的体验将帮助你巩固对每种 Matplotlib 图表类型使用时机和方法的理解。

首先，请确保已导入 Matplotlib，通常使用以下标准约定：

```python
import matplotlib.pyplot as plt
import numpy as np # 我们将使用 NumPy 生成一些示例数据
```

### 实践 1：创建条形图

条形图非常适合比较不同类别之间的数量。假设我们想可视化一个小杂货摊中不同类型水果的销售数量。

**数据：**
假设我们有以下销售数据：

- 苹果：50 个单位
- 橙子：35 个单位
- 香蕉：65 个单位
- 葡萄：42 个单位

**步骤：**

1. **准备数据：** 为水果名称（类别）及其对应的销售数据（值）创建列表。
2. **创建条形图：** 使用 `plt.bar()` 函数，传入 x 轴的类别和 y 轴的值。
3. **添加标签和标题：** 使用 `plt.xlabel()`、`plt.ylabel()` 和 `plt.title()` 使图表易于理解。
4. **显示图表：** 使用 `plt.show()`。

**代码：**

```python
import matplotlib.pyplot as plt

# 1. 准备数据
fruit_names = ['Apples', 'Oranges', 'Bananas', 'Grapes']
units_sold = [50, 35, 65, 42]

# 2. 创建条形图
plt.figure(figsize=(6, 4)) # 可选：调整图表大小
plt.bar(fruit_names, units_sold, color='#339af0') # 使用蓝色系的颜色

# 3. 添加标签和标题
plt.xlabel('水果类型')
plt.ylabel('销售数量')
plt.title('水果销售比较')
plt.xticks(rotation=45) # 如果 x 轴标签重叠，则旋转
plt.tight_layout() # 调整布局以防止标签重叠

# 4. 显示图表
plt.show()
```

运行此代码将生成一个条形图，显示每种水果的销售情况，从而轻松看出香蕉的销量最高。

**挑战：** 尝试修改上述代码，使用 `plt.barh()` 创建一个水平条形图。请记住相应地切换参数 (parameter)并调整标签（`plt.xlabel` 变为“销售数量”，`plt.ylabel` 变为“水果类型”）。

### 实践 2：生成直方图

直方图通过将数据分组到“箱”（bins）中并显示每个箱中观测值的频率，帮助我们理解单个数值变量的分布情况。让我们可视化一个班级的考试分数分布。

**数据：**
我们将使用 NumPy 生成一些随机考试分数来模拟分布。假设分数为 100 分制，并且大致服从均值为 75 的正态分布。

**步骤：**

1. **生成示例数据：** 使用 `np.random.normal()` 创建一个分数数组。
2. **创建直方图：** 使用 `plt.hist()`，传入数据并可选地指定箱的数量，或者让 Matplotlib 自行决定。
3. **添加标签和标题：** 为坐标轴添加标签，并给图表一个描述性标题。
4. **显示图表：** 使用 `plt.show()`。

**代码：**

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成示例数据 (例如，100 个考试分数)
np.random.seed(42) # 为确保结果可重现
exam_scores = np.random.normal(loc=75, scale=10, size=100)
# 确保分数在实际范围内 (例如，0-100)
exam_scores = np.clip(exam_scores, 0, 100)

# 2. 创建直方图
plt.figure(figsize=(7, 5))
plt.hist(exam_scores, bins=10, color='#51cf66', edgecolor='black') # 使用 10 个箱，绿色

# 3. 添加标签和标题
plt.xlabel('考试分数')
plt.ylabel('学生数量')
plt.title('考试分数分布')

# 4. 显示图表
plt.show()
```

这段代码将生成一个直方图，显示在特定分数范围（即箱）内有多少学生得分。你可以通过改变 `bins` 参数 (parameter)（例如 `bins=5`，`bins=20`）来试着调整，看看它如何影响分布的呈现。

### 实践 3：制作饼图

饼图用于显示整体的比例。让我们可视化一个学生每月开支的百分比构成。

**数据：**
假设开支分类如下：

- 租金：40%
- 食物：25%
- 交通：15%
- 水电费：10%
- 娱乐：10%

**步骤：**

1. **准备数据：** 为类别标签及其对应的百分比值创建列表。
2. **创建饼图：** 使用 `plt.pie()`，传入值和标签。使用 `autopct` 在扇区上显示百分比。
3. **添加标题：** 使用 `plt.title()`。添加 `plt.axis('equal')` 以确保饼图为圆形。
4. **显示图表：** 使用 `plt.show()`。

**代码：**

```python
import matplotlib.pyplot as plt

# 1. 准备数据
expense_categories = ['Rent', 'Food', 'Transport', 'Utilities', 'Entertainment']
percentages = [40, 25, 15, 10, 10]
colors = ['#748ffc', '#ff922b', '#51cf66', '#fcc419', '#f06595'] # 靛蓝色, 橙色, 绿色, 黄色, 粉色

# 2. 创建饼图
plt.figure(figsize=(6, 6))
plt.pie(percentages, labels=expense_categories, colors=colors,\n        autopct='%1.1f%%', # 将百分比格式化为一位小数
        startangle=90)     # 从顶部（90度）开始绘制第一个扇区

# 3. 添加标题并确保为圆形
plt.title('每月开支细分')
plt.axis('equal') # 相等纵横比确保饼图为圆形

# 4. 显示图表
plt.show()
```

这会生成一个饼图，说明学生预算是如何分配的。虽然它在显示简单比例时很有用，但请记住，当类别过多或比例非常相似时，饼图会变得难以阅读。对于比较而言，条形图通常是更好的选择。

你现在已经练习了本章涵盖的基本图表类型：用于比较的条形图、用于分布的直方图和用于比例的饼图。你可以随意修改这些示例中的样本数据或参数 (parameter)，看看图表如何变化。这种尝试是培养直觉的好方法，能帮助你判断哪种图表最适合不同类型的数据和分析问题。

## 参考资料

- [Matplotlib Documentation](https://matplotlib.org/stable/) — The Matplotlib Development Team (2024)
  Matplotlib 库的官方全面文档，涵盖本节讨论的用于创建各种图表类型的所有函数、模块和示例。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  Python 科学计算和数据分析的基础指南，其中专门的 Matplotlib 章提供基本图表类型的实用实现。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media; DOI: [10.12987/9780691193950](https://doi.org/10.12987/9780691193950)
  一份备受推崇的资源，阐明了有效数据可视化的原则和最佳实践，为针对不同数据和目的选择适当的图表类型提供指导。
