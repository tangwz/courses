---
course: "data-visualization-matplotlib-seaborn"
chapter: "customizing-saving-visualizations"
lesson: "plot-text-annotations"
sourceId: 1208
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-7-customizing-saving-visualizations/plot-text-annotations"
title: "为图表添加文本和注释"
description: "学习如何在 Matplotlib 图表的特定点或区域添加文本标签和注释。"
order: 1
plots: []
sourceHash: "eb2bdc54711a5840ae4f39aa23ec6bab96bbac0b0a5b07ee22eb59a972afa988"
sourceCorrections: []
---

虽然标题、轴标签和图例为您的图表提供了背景信息，但有时您需要直接在可视化图表上添加特定文本。这可能是为了标记 (token)某个数据点，突出显示感兴趣的区域，解释异常情况，或添加其他不适合标准图表元素的上下文 (context)信息。Matplotlib 提供了灵活的工具，可以精确地在您需要的位置添加文本和注释。

### 使用 `text()` 放置常规文本

向 Matplotlib 图表添加文本最直接的方法是使用 `plt.text()` 函数（如果您使用面向对象的方法处理 `Axes` 对象，通常称为 `ax`，则可以使用等效的 `ax.text()` 方法）。此函数会将文本放置在图表数据空间内的指定坐标处。

基本语法如下：

```python
plt.text(x, y, 'Your desired text string')
```

此处，`x` 和 `y` 是文本开始处的坐标，基于您坐标轴的数据刻度。

我们来看一个例子：

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 创建图表
plt.figure(figsize=(8, 5)) # 创建一个图表
plt.plot(x, y_sin, label='sin(x)', color='#1c7ed6') # 蓝色
plt.plot(x, y_cos, label='cos(x)', color='#fd7e14') # 橙色
plt.title('正弦波和余弦波')
plt.xlabel('角度 (弧度)')
plt.ylabel('值')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 使用 plt.text() 添加文本
plt.text(np.pi / 2, 1.05, '正弦波峰值', ha='center', va='bottom', color='#1c7ed6')
plt.text(np.pi, -1.05, '余弦波最小值', ha='center', va='top', color='#fd7e14')
plt.text(4, 0, '交点', ha='left', va='center', fontsize=9, color='#495057') # 灰色

plt.ylim(-1.5, 1.5) # 调整 y 轴限制以留出文本空间
plt.show()
```

在这个例子中：

- 我们绘制了正弦波和余弦波。
- 我们使用了三次 `plt.text()` 来在特定点附近添加标签。
- 请注意可选参数 (parameter)的使用：
  - `ha`（或 `horizontalalignment`）：控制水平对齐 (alignment)方式（'center' 居中, 'left' 靠左, 'right' 靠右）。
  - `va`（或 `verticalalignment`）：控制垂直对齐方式（'center' 居中, 'top' 靠上, 'bottom' 靠下, 'baseline' 基线）。
  - `color`：设置文本颜色。
  - `fontsize`：调整文本大小。

使用 `plt.text()` 很适合一般性标签，但要将文本精确地相对于特定数据点对齐，尤其是在需要指示器的情况下，通常需要使用注释功能。

### 使用 `annotate()` 指出特征

当您想引起图表中某个点或特征的注意并添加描述性文本时，`plt.annotate()`（或面向对象方法中的 `ax.annotate()`）更合适。它允许您指定被注释点的位置和文本标签的位置，通常用箭头将它们连接起来。

核心语法涉及：

```python
plt.annotate('Annotation text',
             xy=(x_point, y_point),          # 要注释的点
             xytext=(x_text, y_text),        # 文本应该在的位置
             arrowprops=dict(...)           # 箭头的属性
             )
```

- `'Annotation text'` 是您想要显示的字符串。
- `xy` 是一个元组 `(x, y)`，表示您要注释点的坐标（箭头指向此处）。这些通常是数据坐标。
- `xytext` 是一个元组 `(x, y)`，表示文本标签应放置的坐标。默认情况下，这些也是数据坐标。
- `arrowprops` 是一个字典，定义连接 `xytext` 到 `xy` 的箭头的样式。如果您省略 `arrowprops` 或将其设置为 `None`，则不会绘制箭头。

让我们注释上一个例子中正弦波的第一个峰值：

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)

# 找到第一个峰值 (近似值)
peak_x = np.pi / 2
peak_y = np.sin(peak_x) # 这是 1.0

# 创建图表
fig, ax = plt.subplots(figsize=(8, 5)) # 现在使用面向对象方法
ax.plot(x, y_sin, color='#1c7ed6') # 蓝色
ax.set_title('注释正弦波峰值')
ax.set_xlabel('角度 (弧度)')
ax.set_ylabel('值')
ax.grid(True, linestyle=':', alpha=0.6)

# 使用 ax.annotate() 添加注释
ax.annotate('第一个峰值 (x=π/2)',
            xy=(peak_x, peak_y),                      # 要注释的点 (数据坐标)
            xytext=(peak_x + 1, peak_y + 0.3),        # 文本位置 (数据坐标)
            arrowprops=dict(facecolor='#495057',      # 箭头颜色 (灰色)
                            edgecolor='#495057',
                            shrink=0.05,              # 箭头和点之间的间隙
                            arrowstyle='->',          # 箭头样式
                            connectionstyle='arc3,rad=0.2' # 弧形箭头
                           ),
            fontsize=10,
            color='#495057' # 灰色文本
           )

# 为了清晰，标记点本身
ax.plot(peak_x, peak_y, 'o', color='#f03e3e') # 峰值处的红色标记

ax.set_ylim(-1.2, 1.5) # 确保有足够的注释空间
plt.show()
```

在这段代码中：

- 我们使用 `plt.subplots()` 明确创建了一个 `Figure` 和 `Axes` 对象。这允许我们使用 `ax.annotate()`。
- 我们确定了正弦波第一个峰值的坐标 `(peak_x, peak_y)`。
- `xy` 参数 (parameter)将箭头直接指向此峰值。
- `xytext` 参数将文本标签定位在峰值上方偏右的位置。
- `arrowprops` 字典定义了箭头的样式：
  - `facecolor` 和 `edgecolor` 设置箭头颜色。
  - `shrink` 在箭头末端和它们连接的点之间创建了一个小间隙。
  - `arrowstyle` 定义了箭头的类型（例如，'->'，'-|>'，'<|-|>'，'fancy'）。
  - `connectionstyle` 控制箭头线的路径（例如，'arc3,rad=0.2' 创建一个弧形箭头；'angle3' 创建一个弯曲的箭头）。通常需要尝试这些值才能获得所需的外观。
- 我们使用 `ax.plot()` 在被注释点添加了一个标记 (token)，使其在视觉上更清晰。

虽然 `xy` 和 `xytext` 默认使用数据坐标，但 `annotate` 通过使用 `xycoords` 和 `textcoords` 参数（例如，将文本放置在相对于图表或坐标轴边界的位置）提供了对坐标系的精细控制。然而，对于许多常见用途，依赖数据坐标就足够且更直观。

### 选择 `text` 还是 `annotate`

- 当您需要在特定数据坐标处放置一个简单的文本标签，而不需要明确的指示器时，请使用 `text()`。
- 当您想特别突出显示一个数据点或特征，并在该特征及其描述之间绘制视觉连接（箭头）时，请使用 `annotate()`。`annotate` 提供了对文本相对于关注点定位的更多控制。

添加文本和注释需要仔细考虑。请确保您的添加物能阐明图表，而不是使其混乱。有目的地使用它们，以引导观看者注意数据可视化中最重要的方面。

## 参考资料

- [matplotlib.pyplot.text](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html) — Matplotlib Developers (2024)
  Matplotlib 官方文档，介绍如何在图表上放置通用文本，并详细说明定位、对齐和样式参数。
- [matplotlib.pyplot.annotate](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html) — Matplotlib Developers (2024)
  Matplotlib 官方文档，介绍如何在图表上创建带箭头的注释，并涵盖各种连接样式、箭头属性和坐标系。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本备受推崇的指南，其中包含专门的章节，介绍如何自定义 Matplotlib 图表，并提供 `plt.text()` 和 `plt.annotate()` 函数的实用示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本关于有效数据可视化原则的基础书籍，提供关于如何适当地添加文本元素和注释以增强图表清晰度和可解释性的指导。
