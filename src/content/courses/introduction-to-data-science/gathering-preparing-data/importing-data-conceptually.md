---
course: "introduction-to-data-science"
chapter: "gathering-preparing-data"
lesson: "importing-data-conceptually"
sourceId: 2125
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-4-gathering-preparing-data/importing-data-conceptually"
title: "导入数据"
description: "了解将数据从来源加载到分析环境的一般过程。"
order: 2
plots: []
sourceHash: "0c1466dbf2a262a5c01e98572bf47b70263ef883b05c1baca47964843bbdaafb"
sourceCorrections: []
---

数据分析的来源可能多种多样，例如您电脑上的文件、数据库中的表格，或通过网络API获取的信息。将这些数据导入分析工具并使其可用是一个重要的初始步骤。这一过程通常被称为数据*导入*或*加载*。

可以将数据导入看作是将信息从其原始存储位置带入您用于分析的软件或编程环境的内存或工作空间的过程。无论您是使用电子表格程序、统计软件包，还是用Python或R等语言编写代码，数据都需要从其来源读取，并以工具能够理解的方式进行组织。

### 导入为何必要

分析工具通常无法直接操作存储在外部文件或数据库中的原始格式数据，除非先将其带入自己的操作环境中。数据导入有几个重要作用：

1. **可访问性：** 它使数据能够随时供您所选分析工具中的函数和操作使用。
2. **标准化：** 导入过程通常将数据转换为工具设计用于高效处理的标准内部格式（如数据帧或表格）。
3. **性能：** 处理已加载到内存中的数据通常比反复从硬盘或网络位置等外部来源访问数据快得多。

### 导入的步骤

尽管不同工具的具体命令或菜单选项有所不同，但数据导入的基本过程通常涉及以下步骤：

1. **定位来源：** 工具需要知道数据*在哪里*。这可能是您电脑上的文件路径（例如，`C:\Users\YourName\Documents\data.csv`）、网址（URL）或数据库的连接详情。
2. **打开并读取：** 工具打开与来源的连接并读取原始数据内容。这可能是字符序列、字节或记录。
3. **解析格式：** 这是重要的一步。工具需要根据原始数据的格式*理解*其组织方式。
   - 对于 **CSV（逗号分隔值）** 文件，它需要识别逗号通常分隔不同的数据值，而换行符通常表示新行。
   - 对于 **JSON（JavaScript 对象表示法）** 文件，它需要理解键值对、数组和对象的嵌套组织方式。
   - 对于 **Excel 电子表格**，它需要识别不同的工作表、列和行。
   - 对于 **数据库表**，它使用特定协议（如 SQL）请求和接收组织好的数据。
4. **组织数据：** 最后，工具将解析后的数据组织成适合分析的内部组织方式。通常，这是一个二维的表格状组织方式：行代表单个观察值或记录，列代表不同的变量或属性。

> 数据导入过程的视图，将数据从来源移至分析环境。

### 工具和格式

大多数数据分析环境都提供内置函数或库，专门用于导入各种数据格式。例如，您可以使用名为 `read_csv` 的函数来导入 CSV 文件，或使用 `read_json` 来导入 JSON 文件。特定函数会告知工具如何正确执行解析步骤。您需要选择与数据来源格式匹配的导入机制。

理解这个过程很重要。尽管具体的实现细节会有所不同，但基本思路保持不变：定位数据，读取数据，理解其组织方式，并将其加载到您的环境中。一旦数据成功导入，您就可以继续本章中讨论的后续步骤：检查、清理和准备数据以进行分析。

## 参考资料

- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython, 3rd Edition](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由 pandas 创始人撰写的实用指南，涵盖数据导入和处理。
- [R for Data Science (2nd Edition, in progress)](https://r4ds.hadley.nz/) — Hadley Wickham, Garrett Grolemund, and Mine Çetinkaya-Rundel (2023)
  一个在线资源，解释了如何使用 R 中的 Tidyverse 进行数据导入和转换。
- [I/O tools (text, CSV, HDF5, ...)](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) — The pandas development team (2023)
  pandas 用于读取和写入各种格式数据的官方指南。
- [readr: Read Rectangular Text Data](https://readr.tidyverse.org/) — Hadley Wickham, Jim Hester, and Lionel Henry (2023)
  `readr` 包的文档，它简化了将数据导入 R 的过程。
