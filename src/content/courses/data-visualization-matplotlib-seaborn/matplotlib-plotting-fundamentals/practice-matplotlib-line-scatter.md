---
course: "data-visualization-matplotlib-seaborn"
chapter: "matplotlib-plotting-fundamentals"
lesson: "practice-matplotlib-line-scatter"
sourceId: 1163
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-2-matplotlib-plotting-fundamentals/practice-matplotlib-line-scatter"
title: "实践：创建简单的折线图和散点图"
description: "动手实践使用 Matplotlib 创建和自定义基本折线图和散点图。"
order: 8
plots: ["plots/1163-0.json", "plots/1163-1.json"]
sourceHash: "ad776832aaa2d270557ed618d239a1f0add0f6e9ef2e533e3f21a601c6d1cfd0"
sourceCorrections: []
---

创建简单的折线图和散点图。这涉及应用各种自定义方法以有效地可视化数据。动手实践这些练习将有助于增强对 Matplotlib 绘图基本流程的理解。

首先，请确保您已导入 Matplotlib 和 NumPy。我们将使用 NumPy 生成一些用于绘图的示例数据。

```python
import matplotlib.pyplot as plt
import numpy as np
```

### 练习一：创建和自定义折线图

假设我们想在一个特定范围内展示一个简单二次函数（例如 $y = x^2$）的变化趋势。

1. **生成数据**：我们来创建一些 $x$ 值并计算相应的 $y$ 值。

   ```python
   # 生成从 -5 到 5 的 x 值
   x_line = np.linspace(-5, 5, 100)
   # 计算 y 值 (y = x^2)
   y_line = x_line**2
   ```
2. **绘制图形**：使用 `plt.plot()` 创建折线图。

   ```python
   # 创建一个 Figure 和 Axes 对象
   fig, ax = plt.subplots()

   # 绘制数据
   ax.plot(x_line, y_line)

   # 显示图形
   plt.show()
   ```

   运行这段代码将显示一个基本的抛物线折线图。
3. **添加标签和标题**：通过添加标题和坐标轴标签，使图形更具说明性。

   ```python
   fig, ax = plt.subplots()
   ax.plot(x_line, y_line)

   # 添加标题和标签
   ax.set_title("二次函数：y = x^2")
   ax.set_xlabel("X 值")
   ax.set_ylabel("Y 值 (x^2)")

   plt.show()
   ```
4. **自定义外观**：让我们将线条改为红色虚线，并使其稍微粗一些。

   ```python
   fig, ax = plt.subplots()

   # 绘制并自定义
   ax.plot(x_line, y_line, color='#f03e3e', linestyle='--', linewidth=2) # 红色虚线

   ax.set_title("自定义二次函数：y = x^2")
   ax.set_xlabel("X 值")
   ax.set_ylabel("Y 值 (x^2)")

   plt.show()
   ```
5. **设置坐标轴范围**：将视图集中在 $x$ 值介于 0 和 4 之间的正 x 轴部分。

   ```python
   fig, ax = plt.subplots()
   ax.plot(x_line, y_line, color='#f03e3e', linestyle='--', linewidth=2)
   ax.set_title("聚焦二次函数：y = x^2")
   ax.set_xlabel("X 值")
   ax.set_ylabel("Y 值 (x^2)")

   # 设置 x 轴和 y 轴的范围
   ax.set_xlim(0, 4)
   ax.set_ylim(0, 16) # y 最高到 4^2 = 16

   plt.show()
   ```

以下是整合了所有步骤的最终代码：

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成数据
x_line = np.linspace(-5, 5, 100)
y_line = x_line**2

# 2. 创建 Figure 和 Axes
fig, ax = plt.subplots()

# 3. 绘制并自定义
ax.plot(x_line, y_line, color='#f03e3e', linestyle='--', linewidth=2)

# 4. 添加标题和标签
ax.set_title("聚焦二次函数：y = x^2")
ax.set_xlabel("X 值")
ax.set_ylabel("Y 值 (x^2)")

# 5. 设置坐标轴范围
ax.set_xlim(0, 4)
ax.set_ylim(0, 16)

# 6. 显示图形
plt.show()
```

该图清晰地显示了 $x$ 在 0 到 4 之间时 $y = x^2$ 的关系，并带有说明性标签和独特样式。



![聚焦二次函数：y = x^2](plots/1163-0.json)



> 折线图显示了 $x$ 在 0 到 4 之间时 $y=x^2$ 的关系。线条为红色虚线。

### 练习二：创建和自定义散点图

现在，我们来创建一个散点图，以展示两组随机但有些关联的数据点之间的关系。

1. **生成数据**：我们将创建两个数组 `x_scatter` 和 `y_scatter`。让 `y_scatter` 稍微依赖于 `x_scatter`，并加入一些随机性。

   ```python
   # 设置随机种子以保证结果可重现
   np.random.seed(42)

   # 生成 50 个随机 x 值
   x_scatter = np.random.rand(50) * 10 # 值在 0 到 10 之间

   # 生成具有一定关联性和噪声的 y 值
   y_scatter = x_scatter + np.random.randn(50) * 2 # y = x + 随机噪声
   ```
2. **绘制图形**：使用 `plt.scatter()` 绘制这些点。

   ```python
   fig, ax = plt.subplots()

   # 创建散点图
   ax.scatter(x_scatter, y_scatter)

   plt.show()
   ```

   这将显示原始散点图。
3. **添加标签和标题**：为图形添加背景信息。

   ```python
   fig, ax = plt.subplots()
   ax.scatter(x_scatter, y_scatter)

   # 添加标题和标签
   ax.set_title("两个变量之间的关系")
   ax.set_xlabel("自变量 (X)")
   ax.set_ylabel("因变量 (Y)")

   plt.show()
   ```
4. **自定义外观**：让我们将标记 (token)颜色改为橙色，增大其尺寸，并使用 `alpha` 参数 (parameter)使其稍微透明。

   ```python
   fig, ax = plt.subplots()

   # 绘制并自定义
   ax.scatter(x_scatter, y_scatter, color='#fd7e14', s=50, alpha=0.7) # 橙色，大小 50，略透明

   ax.set_title("自定义的两个变量之间的关系")
   ax.set_xlabel("自变量 (X)")
   ax.set_ylabel("因变量 (Y)")

   plt.show()
   ```

以下是散点图的最终代码：

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成数据
np.random.seed(42) # 用于结果可重现
x_scatter = np.random.rand(50) * 10
y_scatter = x_scatter + np.random.randn(50) * 2

# 2. 创建 Figure 和 Axes
fig, ax = plt.subplots()

# 3. 绘制并自定义
ax.scatter(x_scatter, y_scatter, color='#fd7e14', s=50, alpha=0.7) # 橙色，大小 50，透明度 0.7

# 4. 添加标题和标签
ax.set_title("自定义的两个变量之间的关系")
ax.set_xlabel("自变量 (X)")
ax.set_ylabel("因变量 (Y)")

# 5. 显示图形
plt.show()
```

该散点图显示了代表 X 和 Y 之间关系的独立数据点，并进行了样式处理以提高可见性。



![自定义的两个变量之间的关系](plots/1163-1.json)



> 散点图显示了两个变量之间的关系。标记是橙色的半透明圆圈。

这些练习涵盖了本章的核心技能：创建折线图和散点图，添加必要的标签和标题，以及自定义颜色、线条样式和标记属性等视觉元素。通过为折线图尝试不同的函数、调整散点图数据中的噪声水平，或查看 Matplotlib 中可用的其他颜色代码和线条样式，进一步尝试。您练习得越多，就会越熟悉这些基本工具。

## 参考资料

- [\`matplotlib.pyplot.plot\` documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) — Matplotlib Development Team (2024)
  Matplotlib 中创建带多种样式选项的线图的官方文档。
- [\`matplotlib.pyplot.scatter\` documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) — Matplotlib Development Team (2023)
  详细介绍如何在 Matplotlib 中创建散点图并自定义标记属性的官方文档。
- [\`numpy.linspace\` documentation](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) — NumPy Development Team (2023)
  NumPy 中用于生成等间距数值序列的 `linspace` 函数的官方文档。
- [NumPy Legacy Random Generation documentation](https://numpy.org/doc/stable/reference/random/legacy.html) — NumPy Development Team (2023)
  NumPy 旧版随机数生成函数（包括 `rand` 和 `randn`）的官方文档，用于创建样本数据。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本涵盖 NumPy 数值运算和 Matplotlib 各种绘图及外观自定义的参考资料。
