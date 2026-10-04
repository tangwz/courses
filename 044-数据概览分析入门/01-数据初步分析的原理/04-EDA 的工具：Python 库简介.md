# EDA 的工具：Python 库简介

来源：[原文](https://apxml.com/zh/courses/intro-eda-course/chapter-1-eda-foundations/eda-python-libraries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效进行数据初步分析，特别是在处理较大规模数据集时，我们需要专门的软件工具。虽然手动查看可能适用于少量记录，但现代数据的规模和复杂性要求采用程序化方法。Python 生态系统提供了一系列强大的开源库，它们已成为数据分析任务（包括数据初步分析）的标准。这些库提供了高效的数据结构、全面的数据处理与计算功能，以及多样的可视化功能。

下面我们介绍您将在本课程中主要使用的库：

### NumPy：数值运算的根本

NumPy（数值 Python）是 Python 中进行数值运算的根本库。尽管在进行数据初步分析时您可能不总是*直接*与 NumPy 交互，但它支撑着许多其他数据分析库（特别是 Pandas）中的操作。

- **主要作用：** NumPy 提供了 `ndarray` 对象，它是一种高效的多维数组。这些数组支持快速的向量 (vector)化数学和逻辑运算，其速度远超使用标准 Python 列表进行计算。
- **对数据初步分析的意义：** 尽管 Pandas 经常封装 NumPy 的功能，但了解 NumPy 是有益的。许多 Pandas 函数会返回 NumPy 数组，并且 NumPy 用于生成随机数、线性代数和傅里叶变换的函数在更高级的数据分析和准备阶段会派上用场。在处理大型数据集时，其效率十分要紧。

### Pandas：数据整理与分析的主力

Pandas 可以说是 Python 中进行日常数据分析和数据初步分析最重要的库。它提供了高性能、易于使用的数据结构和数据分析工具。

- **主要结构：** Pandas 中的两个主要数据结构是：
  - **`Series`：** 一维带标签的数组，可以容纳任何数据类型（整数、字符串、浮点数、Python 对象等）。可以把它想象成电子表格或数据库表中的一列。
  - **`DataFrame`：** 二维的、大小可变且可能异构的表格数据结构，带有标签轴（行和列）。这是您在分析表格数据时会用到的主要对象，类似于电子表格、SQL 表或 Series 对象的字典。
- **对数据初步分析的意义：** Pandas 擅长：
  - **数据加载：** 将各种格式（CSV、Excel、JSON、SQL 数据库等）的数据读入 DataFrame。
  - **检查：** 快速查看数据（`head()`、`tail()`）、获取维度（`shape`）、知晓数据类型（`dtypes`）以及获取描述性统计信息（`describe()`）。
  - **清洗：** 处理缺失值（`isnull()`、`fillna()`、`dropna()`）、查找并删除重复项（`duplicated()`、`drop_duplicates()`）。
  - **转换：** 选择数据子集（切片、索引）、筛选行、添加或修改列、数据分组（`groupby`）、合并和连接数据集。
  - **基本分析：** 计算频率（`value_counts()`）、相关性（`corr()`），并进行各种聚合。

本质上，Pandas 提供了使数据成形并进行初步汇总和处理的工具，它构成了数据初步分析工作流程的支柱。

### Matplotlib：可视化构建库

数据加载并清洗后，对其进行可视化对于知晓模式、分布和关系非常重要。Matplotlib 是 Python 中最成熟的绘图库，它提供了一个底层接口，用于创建各种静态、动画和交互式可视化。

- **主要作用：** Matplotlib 可以对图表的几乎每个方面（图、轴、线、标签、标题等）进行细致的控制。它能够以各种打印格式和交互式环境生成出版质量的图。
- **对数据初步分析的意义：** 它是许多其他绘图库（如 Seaborn）的驱动。您将直接或间接使用 Matplotlib 来制作基本的数据初步分析图表：
  - 直方图（用于查看分布）
  - 条形图（用于比较分类计数）
  - 线图（常用于时间序列）
  - 散点图（用于查看数值变量之间的关系）
  - 箱线图（用于汇总分布和发现异常值）

虽然功能强大，但 Matplotlib 的语法有时对于创建数据初步分析中常见的复杂统计图表来说可能过于繁琐。

### Seaborn：统计数据可视化

Seaborn 构建于 Matplotlib 之上，并与 Pandas 数据结构紧密结合。它提供了一个更高级别的接口，专门用于制作美观且富有信息量的统计图。

- **主要作用：** Seaborn 简化了与数据初步分析相关的复杂可视化的制作。它自带美观的默认样式和调色板，旨在展现数据中的模式。
- **对数据初步分析的意义：** 与 Matplotlib 相比，Seaborn 可以用更少的代码生成常见的数据初步分析图表，同时通常能提供更多统计背景信息：
  - **增强型分布：** 改进的直方图（`histplot`）、核密度估计图（`kdeplot`）和组合图（`displot`）。
  - **分类比较：** 精密的箱线图（`boxplot`）、小提琴图（`violinplot`）、散布点图（`stripplot`）和条形图（`barplot`），它们能方便地展示数值和分类数据之间的关系。
  - **关系可视化：** 高级散点图（`scatterplot`）、回归图（`regplot`）、相关矩阵热力图（`heatmap`）和成对关系图（`pairplot`）。

Seaborn 擅长快速制作数据中富有见解的视图，使其成为在数据初步分析期间考察关系和分布的非常有用的辅助。

这四个库构成了在 Python 中进行数据初步分析的主要工具集。Pandas 负责数据整理，NumPy 提供数值运算引擎，Matplotlib 提供绘图功能，Seaborn 则提供专门的高级统计可视化。熟练掌握它们的联系，对有效分析和知晓您的数据集帮助很大。接下来的章节将考察如何使用这些工具进行特定的数据初步分析任务。

## 参考资料

- [Pandas documentation](https://pandas.pydata.org/docs/) — The Pandas Development Team (2025)
  Pandas 库的官方和全面资料。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; DOI: [10.1007/978-1-4842-7098-1](https://doi.org/10.1007/978-1-4842-7098-1)
  一本关于使用 Pandas 和 NumPy 进行数据操作的标准著作，由 Pandas 的创建者撰写。
- [Seaborn: statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom (2023)
  使用 Python 进行统计数据可视化的官方指南，包含示例。
- [Matplotlib documentation](https://matplotlib.org/stable/contents.html) — The Matplotlib Development Team (2023)
  创建静态和交互式绘图的权威参考。

---

[上一节](03-%E6%95%B0%E6%8D%AE%E5%88%9D%E6%AD%A5%E5%88%86%E6%9E%90%E6%B5%81%E7%A8%8B.md) · [下一节](05-%E8%AE%BE%E7%BD%AE%E6%82%A8%E7%9A%84%E7%8E%AF%E5%A2%83.md)
