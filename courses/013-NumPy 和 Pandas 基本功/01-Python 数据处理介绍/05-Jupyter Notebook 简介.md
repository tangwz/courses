# Jupyter Notebook 简介

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-1-intro-data-handling-python/intro-jupyter-notebooks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Jupyter Notebook 是您在交互式数据分析中可能会最常使用的工具。可以将其视为一个数字实验室笔记本，您可以在其中编写和运行 Python 代码，添加解释性文本，并显示表格和图表等结果，所有这些都可以在您的网页浏览器中完成。

### 什么是 Jupyter Notebook？

A Jupyter Notebook 是一个开源的 Web 应用程序，允许您创建和分享包含实时代码、公式、可视化内容和说明性文本的文档。“Jupyter”这个名字来源于它支持的核心编程语言：Julia、Python 和 R。笔记本在数据科学中被广泛使用，因为它们让代码试验变得简单，可以立即查看结果，并在整个过程中记录您的思考过程。

笔记本文档本身（`.ipynb` 文件）会存储所有内容：您的代码、生成的输出，以及您添加的任何文本或图像。这使得它们独立且易于分享。

### 笔记本界面：单元格和内核

Jupyter Notebook 的主要工作组件是**单元格**和**内核**。

- **单元格：** 它们是笔记本的构建块。主要有两种类型：

  - **代码单元格：** 这些包含您打算执行的 Python 代码。当您运行一个代码单元格时，代码会发送到内核执行，并且输出（如计算结果、表格或图表）会直接显示在该单元格下方。
  - **Markdown 单元格：** 这些包含使用 Markdown（一种简单的标记 (token)语言）格式化的文本。您使用这些单元格来编写说明、添加标题、创建列表、插入图像，并通常用于构建您的笔记本，使其易于理解。
- **内核：** 这是在后台运行的计算引擎。当您运行一个代码单元格时，内核会执行该代码。它维护您的计算状态，这意味着在一个单元格中定义的变量在同一会话的后续单元格中都可用。在本课程中，您将使用 Python 内核。

> Jupyter Notebook 中的基本交互流程。单元格中的代码发送到内核执行，结果显示回笔记本中。

### 单元格操作

当您打开 Jupyter Notebook 时，您会看到一个主要由这些单元格组成的界面。以下是基本操作：

1. **选择单元格：** 只需点击它。选中的单元格会有一个边框（通常是蓝色或绿色，取决于模式）。
2. **运行单元格：** 最常见的方法是按下 `Shift + Enter`。这会运行当前单元格中的代码并自动选择其下方的下一个单元格。`Ctrl + Enter` 会运行选定的单元格但保持焦点在该单元格上。您也可以使用工具栏中的“运行”按钮。运行代码单元格的顺序很重要，因为后面的单元格可能依赖于前面定义的变量或导入。
3. **更改单元格类型：** 您可以使用工具栏中的下拉菜单或键盘快捷键（通常是 `Esc`，然后按 `M` 转换为 Markdown，或按 `Y` 转换为 Code）在代码和 Markdown 类型之间切换。
4. **添加单元格：** 使用工具栏中的“+”按钮或键盘快捷键（例如 `Esc`，然后按 `A` 在上方插入，或按 `B` 在下方插入）。
5. **编辑单元格：** 双击 Markdown 单元格以编辑其文本。对于代码单元格，只需点击内部即可开始输入。
6. **删除单元格：** 选择单元格，然后使用工具栏中的“剪切”（剪刀）图标，或者按下 `Esc` 键，然后连续按两次 `D` 键。

### 代码单元格的实际应用

让我们看一个简单的例子。在一个代码单元格中，您可以输入：

```python
import numpy as np
import pandas as pd

print("NumPy and Pandas imported successfully!")
```

按下 `Shift + Enter` 会执行此操作。如果一切设置正确，您将看到输出：

```
NumPy and Pandas imported successfully!
```

现在，在*下一个*代码单元格中，您可以使用 `np` 和 `pd`，因为内核记住了之前执行中定义的它们：

```python
my_array = np.array([1, 2, 3, 4])
my_series = pd.Series(['a', 'b', 'c'])

print(my_array)
print(my_series)
```

运行此单元格会输出：

```
[1 2 3 4]
0    a
1    b
2    c
dtype: object
```

请注意，输出是如何直接出现在生成它的单元格下方的。

### 用于文档的 Markdown

Markdown 单元格允许您添加格式化文本。您可以创建结构并解释您的代码。以下是一些基本示例：

```markdown
# 这是一个一级标题

## 这是一个二级标题

这里有一些普通文本。您可以解释下一个代码单元格的作用。

使用星号或破折号创建项目符号列表：
* 项目 1
* 项目 2
  * 子项目

或者使用数字创建有序列表：
1. 第一步
2. 第二步

您可以使用反引号添加 `行内代码`。
```

当您运行 Markdown 单元格（`Shift + Enter`）时，文本将以指定的格式呈现。

### 为什么在数据分析中使用 Jupyter？

Jupyter Notebooks 特别适合您在本课程中将要进行的工作类型：

- **交互式分析：** 运行小段代码，即时查看结果，并根据输出调整方法。这对于理解数据非常有益。
- **集成环境：** 将您的代码、其输出（表格、图表）和您的笔记保存在一个文档中。
- **易于分享：** 与他人分享您的 `.ipynb` 文件，他们可以看到您的工作流程、结果和说明（他们需要安装 Jupyter 才能自己运行代码）。
- **可视化：** 像 Matplotlib 和 Seaborn（它们通常基于 NumPy 和 Pandas）这样的库可以直接与 Jupyter 集成，以便内联显示图表。

您通常会从终端（导航到您的工作目录后）通过输入 `jupyter notebook` 或 `jupyter lab` 来启动 Jupyter Notebook。Anaconda Navigator 也提供了图形界面启动方式。随着本课程的进展，您将获得更多使用笔记本处理 NumPy 和 Pandas 的实践经验。

## 参考资料

- [The Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/) — Project Jupyter Development Team (2024)
  Publisher: Project Jupyter
  提供Jupyter Notebook的全面细节，涵盖安装、界面和功能。
- [IPython: A System for Interactive Scientific Computing](https://doi.org/10.1109/MCSE.2007.53) — Fernando Pérez and Brian E. Granger (2007)
  Journal: Computing in Science & Engineering; Publisher: IEEE; Volume: 9; Pages: 21-29; DOI: [10.1109/MCSE.2007.53](https://doi.org/10.1109/MCSE.2007.53)
  描述了IPython，一个作为Jupyter项目及其笔记本界面基础的交互式计算环境。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本广泛使用的参考书，展示了Jupyter Notebook在NumPy和Pandas数据分析中的实际应用。

---

[上一节](04-%E8%BF%90%E8%A1%8C%E4%BD%A0%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E4%BB%A3%E7%A0%81%E7%89%87%E6%AE%B5.md) · [下一节](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BE%E7%BD%AE%E4%B8%8E%E9%AA%8C%E8%AF%81.md)
