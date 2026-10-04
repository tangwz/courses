---
course: "intro-data-cleaning-preprocessing"
chapter: "importance-clean-data"
lesson: "what-is-data-cleaning"
sourceId: 3941
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-1-importance-clean-data/what-is-data-cleaning"
title: "什么是数据清洗？"
description: "了解数据准备背景下数据清洗的定义和目的。"
order: 1
plots: []
sourceHash: "f36e9ba81b9a25dbbfbed5187577a5bdecd4e2dbbba663a4c77b3559fed29ce0"
sourceCorrections: []
---

数据清洗是指识别、纠正或移除数据集中错误、不一致和不准确之处的过程。可以将其视为使原始数据可用的必要第一步。如前所述，数据很少以完美状态出现。它通常包含可能扭曲分析或导致机器学习 (machine learning)模型表现不佳的问题。

数据清洗的主要目标是提高数据质量，确保您使用的信息准确、一致且可靠。当数据干净时，您可以对获得的数据洞察、生成的报告以及模型做出的预测更有信心。

我们正在寻找哪些问题？数据清洗过程中常见的问题包括：

- **缺失值：** 数据应存在但为空的单元格或特定占位符（例如 `NULL`、`NA` 或 `?`）。
- **不正确的数据类型：** 数字存储为文本、日期存储为字符串，这会使计算或比较变得困难。
- **错误和拼写错误：** 简单的拼写错误（例如将“New York”写成“New Yourk”）、数据录入错误或不可能的值（例如年龄为 -5）。
- **不一致的格式：** 大小写差异（“usa”、“USA”、“U.S.A.”）、不同的日期格式（“10/05/2023”与“May 10, 2023”）或条目周围多余的空格。
- **重复记录：** 数据集中多次出现的整行或记录，可能导致计数或平均值偏差。

清洗过程包括检测这些问题（通常使用编程工具和目视检查），然后决定处理它们的最佳方式。这可能包括：

- 如果可能，直接纠正错误。
- 移除有问题的记录或列。
- 使用适当的策略（填充）来填补缺失值。
- 标准化格式以确保一致性。

您可能会听到数据清洗被讨论为一个更大的范畴的一部分，即**数据预处理**。数据清洗确实是预处理的重要组成部分。预处理包含一套更广泛的任务，旨在为分析或建模准备数据，这些任务涵盖清洗，同时也可以涉及数据转换（例如数值缩放）或特征工程。在这个初始阶段，我们的重点完全放在清洗方面：修正原始数据本身固有的错误和不一致。

另外值得注意的是，数据清洗通常是一个迭代过程。您可能会根据初步检查来清洗数据，然后进行一些分析，之后发现需要重新审视清洗步骤的新不一致或问题。让数据真正准备就绪很少是一条完全线性的路径。

有效地清洗数据是非常重要的。没有这一步，任何随后的分析或建模都将建立在不稳定的基础上。

## 参考资料

- [Data Cleaning: A Practical Guide](https://link.springer.com/book/10.1007/978-3-030-80249-1) — Ihab F. Schopf, Christian Müller, and Hermann J. P. W. Schöberl (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-80249-1](https://doi.org/10.1007/978-3-030-80249-1)
  本书提供了数据清洗技术的最新全面概述，讨论了常见的数据质量问题及实用的解决方法。
- [Data Quality: The Accuracy Dimension](https://www.elsevier.com/books/data-quality/olson/978-1-55860-891-7) — Jack E. Olson (2003)
  Publisher: Morgan Kaufmann; Pages: xviii, 293
  一本关于数据质量的奠基性著作，详细阐述了数据准确性的定义、衡量和提升方法，这是数据清洗的核心目标。
- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Morgan Kaufmann
  一本被广泛使用的教科书，其专门章节详细阐述了数据预处理，全面介绍了数据清洗技术、常见问题以及处理各类数据的方法。
