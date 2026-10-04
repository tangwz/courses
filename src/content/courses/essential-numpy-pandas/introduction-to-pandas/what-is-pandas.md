---
course: "essential-numpy-pandas"
chapter: "introduction-to-pandas"
lesson: "what-is-pandas"
sourceId: 1747
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/what-is-pandas"
title: "Pandas 是什么？"
description: "了解 Pandas 库的目的及其在 Python 数据分析中的作用。"
order: 1
plots: []
sourceHash: "24a9087a8637611cdee9d922e74c175dd0ca8d3b6195a095080d896b3d519d0b"
sourceCorrections: []
---

强大的数组对象是 Python 中数值计算的基本构成要素。然而，数据分析通常不仅仅涉及原始数字。我们经常遇到以表格形式组织的数据，类似于电子表格或数据库表，它们带有描述性的行和列标签，不同列中可能包含不同数据类型，并且存在常见的缺失值问题。

**Pandas** 在此发挥作用。Pandas 是一个基于 NumPy 构建的开源 Python 库，专门用于**数据处理和分析**。它提供数据结构和操作，以便高效且直观地处理结构化数据。可以将其视为提供了您在电子表格软件或关系型数据库中可能找到的数据分析功能，但直接集成到您的 Python 环境中。

那么，是什么使得 Pandas 在数据任务中如此有用呢？

- **专门的数据结构：** Pandas 引入了两种主要的数据结构：`Series`（一维带标签数组）和 `DataFrame`（二维带标签数据结构，本质上是一个表格）。这些结构允许您将标签（行的索引，列的名称）与数据关联起来，使得操作比使用普通 NumPy 数组处理表格数据更加直观。我们很快会详细介绍这些结构。
  "\* **处理异构数据：** 与通常要求所有元素具有相同数据类型的 NumPy 数组不同，Pandas `DataFrame` 可以轻松处理不同数据类型的列（例如整数、浮点数、字符串、Python 对象）。这种灵活性符合大多数数据集的特点。"
- **缺失数据管理：** 数据很少是完美的。Pandas 提供方便的函数来检测、移除或替换数据集中的缺失数据点（通常表示为 `NaN`，非数字）。
- **带标签数据操作：** Pandas 的一个显著优势是它能够基于标签而非仅仅位置进行操作。在执行算术运算或组合数据集时，Pandas 会根据其行和列标签自动对齐 (alignment)数据，减少使用无标签数组时常见的错误风险。
- **高效输入/输出：** Pandas 包含工具，可轻松从各种文件格式（例如逗号分隔值 (CSV)、Excel 电子表格、JSON 文件、SQL 数据库等）读取数据，并将处理过的数据写回这些格式。
- **强大的数据处理工具：** 除了基本结构之外，Pandas 提供丰富的功能集，用于选择数据子集、根据条件筛选行、重塑表格、合并和连接多个数据集、对数据进行分组以进行聚合计算，以及处理时间序列数据。

本质上，Pandas 提供加载、清洗、转换、合并和分析结构化数据所需的高层工具。它在底层采用 NumPy 的计算效率，同时提供更具表现力且用户友好的界面，专为数据分析工作流程定制。在您后续学习中，您将看到 `Series` 和 `DataFrames` 如何成为处理数据的主力，然后数据可能会被输入到机器学习 (machine learning)模型或用于生成洞察。

## 参考资料

- [Pandas Documentation](https://pandas.pydata.org/docs/) — The Pandas Development Team (2025)
  提供关于所有 Pandas 功能、数据结构（Series、DataFrame）、函数和使用示例的全面且最新的信息。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas 创建者所著的权威书籍，解释了其设计、数据结构、操作技术以及与 Python 数据科学环境的集成。（第三版）
- [Data Structures for Statistical Computing in Python](https://doi.org/10.25080/Majora-92bf1922-00a) — Wes McKinney (2010)
  Journal: Proceedings of the 9th Python in Science Conference (SciPy 2010); Publisher: SciPy Proceedings; Pages: 56-61; DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)
  介绍 Pandas 库设计和动机的论文，概述了其核心数据结构及其在数据分析任务中的用途。
- [NumPy Documentation](https://numpy.org/doc/stable/) — The NumPy Developers (2024)
  提供 NumPy 基本数组对象及其高效数值计算能力的详细信息，这些能力是 Pandas 的基础。
