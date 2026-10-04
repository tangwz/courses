---
course: "introduction-to-data-science"
chapter: "understanding-data-fundamentals"
lesson: "introduction-to-metadata"
sourceId: 2104
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-2-understanding-data-fundamentals/introduction-to-metadata"
title: "元数据简介"
description: "了解元数据的定义，即关于数据的数据，及其作用。"
order: 6
plots: []
sourceHash: "d0d705688238dcfc28992a78d29ad3e0a0d4321c8205525e1d7338f80a6e2fe9"
sourceCorrections: []
---

数据有多种类型，例如结构化表格与非结构化文本，以及数值型与描述型数值。但通常，数据本身并不能说明全部情况。设想一下，您在一个共享盘中找到一个名为 `report_final_v3.csv` 的电子表格文件。里面是什么？它是什么时候创建的？谁制作的？`val1`、`val2` 和 `cat_A` 这些列究竟是什么意思？要回答这些问题，需要*关于*数据的信息。这就是元数据的用处。

### 什么是元数据？

元数据通常被简单定义为**关于数据的数据**。它是描述性信息，提供关于数据集或数据元素的背景、结构和管理细节。可以把它想象成食品包装上的标签：食品本身是数据，而配料表、营养成分、保质期和生产商信息则是元数据。没有标签，您将不知道自己究竟在食用什么，或者是否安全。

同样，没有元数据，原始数据将难以（甚至无法）可靠地解释和使用。它帮助我们了解数据的何物、何因、何时、何地、何人以及如何。

### 元数据为何重要？

元数据在数据科学和数据管理中起着多种作用：

1. **了解和解释：** 它明确了数据元素的含义。名为 `QTY` 的列代表什么？它是单位、公斤还是箱？元数据，比如解释列名和单位的数据字典，提供这种必要的背景信息。
2. **数据查找：** 查找相关数据集时，元数据就像目录一样。您可以根据关键字、创建日期、数据所有者或元数据中描述的主题进行搜索，以找到所需的数据。
3. **数据质量评估：** 元数据可以包含关于数据血缘（数据来源）、处理步骤和已知问题的信息，帮助您判断数据的可靠性及其是否适合您的分析。
4. **数据管理和治理：** 它有助于组织数据资源、追踪使用情况、管理访问权限，并确保符合法规。

### 元数据示例

元数据无处不在，通常不被察觉。以下是一些常见示例：

- **电子表格文件：**
  - **数据：** 单元格中的数字和文本。
  - **元数据：** 列标题（例如，“客户ID”、“购买日期”）、文件名、文件大小、创建日期、作者、最后修改日期、工作表名称。
- **数码照片：**
  - **数据：** 构成图像的像素。
  - **元数据（EXIF数据）：** 相机型号、拍摄日期和时间、GPS坐标（位置）、分辨率、光圈设置、快门速度。
- **数据库表：**
  - **数据：** 表中存储的记录行。
  - **元数据：** 表名、列名（例如，`user_id`、`email_address`）、每列的数据类型（例如，INTEGER、VARCHAR）、约束（例如，主键、非空）、索引定义、数据库模式名称。
- **网页：**
  - **数据：** 页面上可见的文本和图像。
  - **元数据（HTML标签）：** 页面标题（`<title>`）、描述（`<meta name="description">`）、关键词（`<meta name="keywords">`）、使用的字符集。

> 一个显示数据集的视图包含实际数据值以及描述这些值和数据集本身的元数据。

### 数据科学工作流程中的元数据

在数据科学过程的早期，了解元数据尤为重要：

- **问题定义：** 元数据有助于确认数据集是否包含解决问题所需的必要信息。
- **数据获取：** 获取数据时，相关的元数据（如源文档或API说明）对于了解如何访问和解释数据是必不可少的。
- **数据准备和分析前查看：** 元数据指导数据清洗（例如，了解预期数据类型有助于查找错误）和分析前查看（例如，分析前有必要了解变量定义）。

本质上，元数据将原始数据点转换为可用信息。在您处理不同数据集时，请务必查找随附的元数据。如果缺失，您最初的任务之一可能是调查并创建它，以确保您的分析可靠且结果有意义。

## 参考资料

- [Fundamentals of Database Systems](https://www.pearson.com/us/higher-education/program/Elmasri-Fundamentals-of-Database-Systems-7th-edition/PGM285098.html) — Ramez Elmasri, Shamkant B. Navathe (2021)
  Publisher: Pearson
  一本广泛使用的教材，涵盖了数据库模式、数据模型和数据字典，作为结构化数据管理中元数据的基本形式。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  解释了现代数据系统中模式、数据模型和数据演变的作用，强调了元数据如何确保数据的一致性和互操作性。
- [Metadata](https://www.alastore.ala.org/content/metadata-third-edition) — Marcia Lei Zeng, Jian Qin (2022)
  Publisher: ALA Neal-Schuman; Pages: 640
  一本基础书籍，全面介绍了元数据的原则、类型、标准及其在信息组织和数据发现中的应用。
