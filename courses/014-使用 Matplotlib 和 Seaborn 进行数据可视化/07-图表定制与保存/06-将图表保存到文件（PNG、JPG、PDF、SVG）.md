# 将图表保存到文件（PNG、JPG、PDF、SVG）

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-7-customizing-saving-visualizations/saving-plots-files)

[返回章节目录](README.md) · [返回课程目录](../README.md)

保存可视化内容到文件是为报告、演示文稿、网页或电子邮件等用途准备它们的重要一步。Matplotlib 提供了一种直接的方法，可以将图表存储为各种格式的文件。

### 使用 `savefig` 保存图表

Matplotlib 中保存图表的主要函数是 `savefig()`。无论您是使用基于状态的接口（`pyplot` 或 `plt`）还是面向对象的方法（直接使用 `Figure` 和 `Axes` 对象），其方法都类似。

如果您一直在使用 `pyplot` 命令，例如 `plt.plot()` 和 `plt.title()`，则可以使用 `plt.savefig()` 保存当前图表：

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图表
plt.plot(x, y, label='正弦波')
plt.title('简单正弦波')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()

# 将图表保存为PNG文件
plt.savefig('sine_wave_plot.png')

# 可选：显示图表（通常在保存后进行）
# plt.show()
```

如果您明确创建了 `Figure` 对象（通常命名为 `fig`），则可以使用该图表自身的 `savefig()` 方法：

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 使用面向对象方法创建图表和轴
fig, ax = plt.subplots()

# 在轴上绘图
ax.plot(x, y, label='正弦波')
ax.set_title('简单正弦波（面向对象）')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.legend()

# 使用图表对象的方法保存图表
fig.savefig('sine_wave_plot_oo.png')

# 可选：显示图表
# plt.show()
```

在这两种情况下，`savefig()` 的主要参数 (parameter)都是所需文件名，包括文件扩展名。Matplotlib 会智能地使用此扩展名来确定输出格式。

### 常见文件格式

Matplotlib 支持多种文件格式。以下是一些最常用的格式：

1. **PNG（便携式网络图形）：** 一种**栅格**格式，意味着图像存储为像素网格。PNG 使用无损压缩，完美保留图像质量。它还支持透明度，非常适合网页图形和叠加。文件扩展名：`.png`。
2. **JPG/JPEG（联合图像专家组）：** 另一种**栅格**格式，常用于照片。JPG 使用有损压缩，这意味着为实现更小的文件大小，一些图像数据会被丢弃。这可能导致伪影，尤其是在处理锐利的线条或文本时。它不支持透明度。文件扩展名：`.jpg` 或 `.jpeg`。
3. **PDF（便携式文档格式）：** 一种（主要）**矢量**格式。矢量图形使用数学路径、线条和形状定义图像，而非像素。这意味着它们可以无限缩放而不会损失质量或出现像素化。PDF 非常适合用于打印或高分辨率显示的文档。文件扩展名：`.pdf`。
4. **SVG（可伸缩矢量图形）：** 一种基于 XML 的**矢量**格式。与 PDF 类似，SVG 图像可伸缩且不损失质量。它们特别适合网页使用，因为可以通过代码（CSS、JavaScript）进行操作，并且通常比用于图表的高分辨率栅格图像文件更小。文件扩展名：`.svg`。

栅格（PNG、JPG）和矢量（PDF、SVG）格式之间的选择取决于您打算如何使用该图表。我们将在下一节中更仔细地查看这个选择。目前，您只需知道，保存为这些格式中的任何一种，都像更改 `savefig()` 命令中的文件扩展名一样简单。

```python
# 保存为不同格式
fig.savefig('sine_wave_plot.jpg')
fig.savefig('sine_wave_plot.pdf')
fig.savefig('sine_wave_plot.svg')
```

### 自定义保存输出

`savefig()` 函数提供多个参数 (parameter)来控制输出：

- `dpi` (每英寸点数)：此参数与栅格格式（如 PNG 和 JPG）相关。它决定了保存图像的分辨率。DPI 值越高，图像质量越高，但文件大小也越大。常见值为 100、300 或 600。默认值通常由 Matplotlib 的配置（例如 `figure.dpi`）控制。

  ```python
  # 保存更高分辨率的PNG
  fig.savefig('sine_wave_plot_high_res.png', dpi=300)
  ```
- `bbox_inches='tight'`: 这是一个非常有用的选项。它尝试自动调整保存图表的边界框，以最小化图表元素（标题、标签、绘图区域）周围的空白区域。

  ```python
  # 保存为最小化空白区域的图表
  fig.savefig('sine_wave_plot_tight.png', bbox_inches='tight')
  ```
- `transparent=True`: 如果保存到支持透明度的格式（如 PNG 或 SVG），将其设置为 `True` 将使图表背景透明，而不是默认的白色（或设置为其他前景色）。这在将图表叠加到其他背景上时很有用。

  ```python
  # 保存为透明背景的图表
  fig.savefig('sine_wave_plot_transparent.png', transparent=True)
  ```
- `facecolor` 和 `edgecolor`：您可以在保存时指定图表的背景颜色。

  ```python
  # 保存为浅灰色背景的图表
  fig.savefig('sine_wave_plot_gray_bg.png', facecolor='#dee2e6')
  ```

### 重要提示：`savefig()` 与 `show()`

通常建议在调用 `plt.show()` **之前**调用 `savefig()`。在某些环境或后端中，`plt.show()` 可能会显示图表然后清除图表对象，这意味着随后的 `savefig()` 调用可能导致空白图像。典型的工作流程是：

1. 创建图表元素。
2. 自定义图表（标题、标签等）。
3. 调用 `savefig()` 将图表保存到文件。
4. 调用 `plt.show()` 以交互方式显示图表（如果需要）。

通过熟练掌握 `savefig()`，您能够将精心制作的 Matplotlib 和 Seaborn 可视化内容从开发环境中取出，并有效地整合到最终报告、论文或演示文稿中。

## 参考资料

- [matplotlib.pyplot.savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) — Matplotlib Development Team (2024)
  Matplotlib `savefig` 函数的官方文档，包括其参数和用法。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Python 数据处理和可视化指南，包含 Matplotlib 保存图表的部分。

---

[上一节](05-%E5%BA%94%E7%94%A8%20Seaborn%20%E4%B8%BB%E9%A2%98.md) · [下一节](07-%E9%80%89%E6%8B%A9%E6%AD%A3%E7%A1%AE%E7%9A%84%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F.md)
