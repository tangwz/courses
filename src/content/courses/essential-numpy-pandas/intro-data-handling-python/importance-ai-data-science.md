---
course: "essential-numpy-pandas"
chapter: "intro-data-handling-python"
lesson: "importance-ai-data-science"
sourceId: 1676
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-1-intro-data-handling-python/importance-ai-data-science"
title: "对人工智能和数据科学的重要性"
description: "了解Numpy和Pandas为何是人工智能项目中数据准备和分析的重要工具。"
order: 2
plots: []
sourceHash: "bfc88a85de15a9e18ccea610d4d5fb3c8e0472a129dd56a99f4b122b5095b391"
sourceCorrections: []
---

人工智能和数据科学应用本质上由数据驱动。无论是训练机器学习 (machine learning)模型识别图像、分析客户行为，还是处理传感器读数，高效处理和操作数据的能力都十分必要。然而，原始数据很少以可直接用于分析或模型训练的格式出现。它通常需要大量的清理、转换和结构化工作。

Python及其丰富的库生态系统已成为人工智能和数据科学应用中的主导语言。NumPy和Pandas在Python数据处理中是实现高效数据操作和分析的必要工具。

### NumPy的高效数值计算

从本质上讲，数据科学很大一部分涉及数值计算，通常处理大型数据集。标准的Python列表和循环虽然灵活，但在处理大量数字进行数学运算时效率不高。

NumPy（Numerical Python）直接解决了这一难题。它提供：

1. **`ndarray` 对象：** 一种强大的N维数组对象，比标准Python序列在内存效率上更高，且数值运算速度快得多。操作通过优化过的预编译C代码实现。
2. **向量 (vector)化操作：** NumPy允许您对整个数组进行一次性操作，无需显式Python循环。例如，两个NumPy数组相加会进行逐元素相加。这种“向量化 (quantization)”极大地加快了计算速度。
3. **数学函数：** 它提供一个广泛的库，包含作用于这些数组的高级数学函数，涵盖线性代数、傅里叶变换、随机数生成等。

许多机器学习 (machine learning)算法要求输入数据为数值数组形式。NumPy提供了标准格式以及有效操作这些数组的工具，使其成为Python中科学计算和人工智能不可或缺的组成部分。

### Pandas的灵活数据结构与数据分析

“尽管NumPy擅长处理原始数值数组，但数据通常具有更多结构。我们可能拥有包含数字、文本、日期和类别信息的混合数据集，它们通常以带有有意义的行和列标签的表格形式组织（例如电子表格或数据库中的数据）。”

Pandas基于NumPy构建，并提供专门为此类表格和异构数据设计的更高级数据结构和分析工具：

1. **DataFrame和Series：** Pandas引入了两种主要数据结构：`DataFrame`（一个二维带标签数据结构，如同一个表格）和`Series`（一维带标签数组，如同一个单列）。这些结构允许您通过行（索引）和列的标签直观地处理数据。
2. **数据处理能力：** Pandas使常见数据任务变得简单明了。这包括从各种文件格式（CSV、Excel、SQL数据库）读取数据、处理缺失值、根据条件筛选和选择数据、重塑数据、合并多个数据集以及执行分组分析。
3. **集成性：** Pandas与NumPy以及其他数据科学库，如Matplotlib（用于绘图）和Scikit-learn（用于机器学习 (machine learning)），顺利集成。使用Pandas准备和清理的数据通常可以直接输入机器学习模型。

### 典型工作流程

在典型的人工智能或数据科学项目中，NumPy和Pandas贯穿于初期阶段的使用：

1. **加载：** Pandas用于将来自各种来源的数据加载到DataFrames中。
2. **检查与清理：** 这两个库都有助于检查数据。Pandas尤其适用于查找和处理缺失值、修正数据类型以及发现不一致之处。
3. **转换与特征工程：** 数据通常需要重塑、聚合或组合。可能需要从现有特征中创建新特征。Pandas为这些任务提供了强大的工具（如`groupby`、`merge`、`apply`）。NumPy函数可用于数值转换。
4. **模型准备：** 对于许多机器学习 (machine learning)模型，Pandas DataFrames中的结构化数据会被转换为NumPy数值数组，这是Scikit-learn等库所期望的格式。

如果没有NumPy高效的数值数组和Pandas灵活的数据结构及操作工具，在Python中为人工智能和数据科学任务准备数据将会显著更复杂且性能更低。因此，掌握这些库是使用Python进入这些领域的重要一步。它们提供了整理数据所需的必备工具包。

## 参考资料

- [NumPy documentation](https://numpy.org/doc/stable/) — NumPy community (2023)
  提供 NumPy 的 ndarray 对象、向量化操作和数学函数的全面细节，对高效数值计算至关重要。
- [pandas documentation](https://pandas.pydata.org/docs/) — pandas development team (2025)
  提供有关 Pandas DataFrame、Series 及其在表格数据中进行数据操作、清洗和分析能力的广泛信息。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas 创建者撰写的权威书籍，为数据科学工作流程中 NumPy 和 Pandas 的数据处理、清洗和转换提供了实用指导。(第4版)
