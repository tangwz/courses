---
course: "essential-numpy-pandas"
chapter: "intro-data-handling-python"
lesson: "what-are-numpy-pandas"
sourceId: 1674
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-1-intro-data-handling-python/what-are-numpy-pandas"
title: "什么是 NumPy 和 Pandas？"
description: "了解 NumPy 和 Pandas Python 库的用途和基本原理。"
order: 1
plots: []
sourceHash: "21d37bb1c69e42b0234177de07dc53ad8752bacc51ca7e32659942fc3a97ac8e"
sourceCorrections: []
---

NumPy 和 Pandas 是 Python 生态中用于处理数值数据和执行数据分析任务的基本库。它们是专门的工具集，在数据科学和人工智能的应用中，当需要处理数字集合、数据表格或时间序列信息时，它们会非常常用。

### NumPy：数值计算的核心

NumPy，是 Numerical Python 的简称，它是 Python 中许多科学计算构建的底层部分。它的主要贡献是 `ndarray` 对象，一个强大的 N 维数组。

NumPy 的数组有何不同之处？

1. **效率高：** NumPy 数组使用 C 语言实现，并针对数值运算进行了优化。在 NumPy 数组上逐元素执行的计算比在标准 Python 列表中执行的等效操作快很多。处理大型数据集时，这种速度很重要。
2. **内存占用：** NumPy 数组比 Python 列表更节省内存，特别是处理大量数值数据时，因为它们在内存中连续存储相同数据类型的元素。
3. **功能性：** NumPy 提供一整套高级数学函数，可直接对这些数组进行操作。这包括线性代数运算、傅里叶变换、随机数生成以及用于统计分析的工具。

本质上，如果您需要在数值数据块（如向量 (vector)、矩阵或更高维张量）上执行数学运算，NumPy 提供基本对象和函数来高效完成。它构建了许多其他数据分析和机器学习 (machine learning)库（包括 Pandas）赖以存在的支撑。

### Pandas：让数据分析和处理更便捷

NumPy 提供底层的数值支撑，而 Pandas 则提供更高级的数据结构和分析工具，这些工具旨在提高实用性和易用性，特别是处理 *表格* 数据，例如电子表格或 SQL 表。

Pandas 中两个主要的数据结构是：

1. **Series：** 一维带标签数组，类似于电子表格中的一列，或带有相关索引的单个数据向量 (vector)。它可以存储任何 NumPy 数据类型的数据。
2. **DataFrame：** 二维带标签数据结构，其列可以包含不同数据类型，非常类似于电子表格、SQL 表或 Series 对象的字典。这是最常用的 Pandas 对象。

Pandas 擅长处理以下方面：

- **数据处理：** 轻松读取各种文件格式（如 CSV、Excel、JSON、SQL 数据库）的数据，并将数据写回。
- **数据清洗：** 提供处理缺失数据（查找、填充或删除 `NaN` 值）、筛选行和转换数据的工具。
- **数据分析：** 提供选择数据子集（切片和切块）、根据条件对数据分组、执行计算和聚合（如求和、平均值、计数）以及合并或连接不同数据集的方法。
- **时间序列：** 包含处理时间戳数据的专门工具。

> 这是 Pandas 和 NumPy 如何融入典型数据工作流程的简化视图。Pandas 处理更高级的数据结构和输入/输出，通常在内部依赖 NumPy 进行高效的数值计算。

"总而言之，NumPy 提供优化的数组对象和数学运算机制，而 Pandas 则在此基础上提供灵活、易用的数据结构（Series 和 DataFrame）以及一套丰富的功能，用于加载、清洗、转换、合并和分析数据。在 Python 中进行以数据为中心的项目时，您几乎总会同时使用它们。本课程将指导您掌握有效使用这两者的实用技能。"

## 参考资料

- [NumPy documentation](https://numpy.org/doc/stable/) — NumPy Developers (2024)
  NumPy库的官方文档，详细介绍了其核心功能、N维数组和数学函数。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  pandas创建者撰写的一本书，提供了使用pandas和NumPy进行数据操作和分析的实用示例和说明。（第3版）
- [Data Structures for Statistical Computing in Python](https://doi.org/10.25080/Majora-92bf1922-00a) — Wes McKinney (2010)
  Journal: Proceedings of the 9th Python in Science Conference (SciPy 2010); Publisher: SciPy Proceedings; Pages: 56-61; DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)
  这篇基础性论文介绍了pandas库在Python中进行统计计算的设计原则和最初的数据结构。
