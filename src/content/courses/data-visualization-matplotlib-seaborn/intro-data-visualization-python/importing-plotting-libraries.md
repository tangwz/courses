---
course: "data-visualization-matplotlib-seaborn"
chapter: "intro-data-visualization-python"
lesson: "importing-plotting-libraries"
sourceId: 1144
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/importing-plotting-libraries"
title: "导入库"
description: "学习在Python脚本中导入Matplotlib和Seaborn的标准惯例。"
order: 7
plots: []
sourceHash: "f3ddfda9802149f6b499edfc761c0f1e9cdaf68d821bc086be1d1154f520a3c5"
sourceCorrections: []
---

在我们开始创建图表之前，我们需要将必要的工具引入我们的Python环境。就像你在烹饪前准备食材一样，我们需要`导入`计划使用的库。在Python中，`import`语句使得其他模块（或库）中的代码可以在你当前的脚本或交互式会话中使用。

对于数据可视化，我们讨论过的两个主要库是Matplotlib和Seaborn。导入它们有既定的惯例，这使得Python代码对于他人（以及将来的你！）来说更易读、更易理解。

### 导入Matplotlib

Matplotlib中用于绘图的主要接口是它的`pyplot`模块。它包含的功能让你可以生成图、创建绘图区域、绘制线、添加标签等等。导入`pyplot`的标准做法是为其指定一个更短的别名，即`plt`。这个别名在Python数据科学界被广泛使用。

```python
import matplotlib.pyplot as plt
```

通过使用`as plt`，我们现在可以通过输入`plt.`后跟函数名（例如，`plt.plot()`）来调用`pyplot`的功能，而不是更长的`matplotlib.pyplot.plot()`。这省去了打字，并使代码更整洁。

### 导入Seaborn

Seaborn通常使用别名`sns`导入。虽然这个特定别名的起源没有明确说法，但这是随处可见的标准惯例。Seaborn的功能通常直接与Pandas DataFrame（我们稍后会讲到）配合使用，并能自动处理许多绘图细节。

```python
import seaborn as sns
```

有了这个导入，你可以使用Seaborn的功能，例如`sns.lineplot()`或`sns.set_theme()`。

### 导入其他辅助库

虽然Matplotlib和Seaborn是我们绘图的重点，但数据通常以其他库管理的格式出现。NumPy对于数值运算非常重要，常用于创建示例数据，而Pandas对于数据处理和分析来说必不可少，尤其是其DataFrame结构。导入这些库也是常见做法，尤其是在处理数据文件时。

```python
import numpy as np
import pandas as pd
```

同样，NumPy的`np`和Pandas的`pd`是标准别名。

### 整合起来

在大多数数据可视化脚本或Notebook中，你会在一开始就导入这些库：

```python
# 核心可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 数据处理库（常一起使用）
import numpy as np
import pandas as pd

# (可选) 设置Seaborn样式以获得更好的图 - 我们稍后会讨论这个
sns.set_theme()

print("库导入成功！")
```

运行这些导入语句，使得`matplotlib.pyplot`、`seaborn`、`numpy`和`pandas`中的功能和对象可在你代码的其余部分使用，使用它们各自的别名（`plt`、`sns`、`np`、`pd`）。

既然我们知道如何将工具引入工作区，我们已准备好在下一节中创建第一个可视化图。

## 参考资料

- [The Python Standard Library: Modules](https://docs.python.org/3/tutorial/modules.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  解释了 Python 中模块的运作方式，包括 `import` 语句和别名使用。
- [Matplotlib: Pyplot tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) — Matplotlib Development Team (2024)
  介绍了 `pyplot` 模块、其常用的 `plt` 别名以及基本绘图技术。
- [Seaborn: User guide and tutorial](https://seaborn.pydata.org/tutorial.html) — Michael Waskom and the Seaborn development team (2024)
  概述了 Seaborn 及其功能，以及数据可视化中标准的导入实践。
- [Pandas: 10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) — The Pandas Development Team (2024)
  展示了 Pandas 的核心功能，包括使用 `pd` 别名的常见导入和 DataFrame 的创建。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  这本广为使用的书籍涵盖了 Pandas、NumPy 和相关工具，包括数据分析工作流中标准的导入约定。
