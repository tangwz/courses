---
course: "intro-data-engineering"
chapter: "foundational-concepts"
lesson: "understanding-data-types"
sourceId: 5500
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-2-foundational-concepts/understanding-data-types"
title: "认识数据类型"
description: "了解结构化、半结构化和非结构化数据格式。"
order: 1
plots: []
sourceHash: "ace318f751a343fbb77c9a31d9a780abc4f04e01ea920688e2540edbbcb209b2"
sourceCorrections: []
---

数据，以其原始形式，是数据工程师工作的基本材料。正如建筑师需要了解木材、砖块和钢材的特性一样，数据工程师也必须了解数据呈现的不同形式。数据并非都一样；它在组织方式和格式上差异很大。识别这些差异是有效收集、存储和处理数据的第一步。接下来，我们看看数据通常归属的主要类别。

### 结构化数据

将结构化数据想象成可以整齐地放入预设容器的信息，就像将物品整理到贴有标签的箱子里。它遵循固定的模式，这意味着它具有特定且预期的格式，通常组织成行和列。这种组织方式使其易于使用标准工具进行查询和分析。

- **特性：** 遵循严格的数据模型，常呈表格形式，可通过可预测的模式轻松搜索。
- **常见示例：**
  - 关系型数据库表（如包含ID、姓名、电子邮件、地址等列的客户信息）。
  - 电子表格（例如，包含交易行和日期、金额、类别列的财务报表）。
  - 以一致格式记录的传感器读数（时间戳、传感器ID、温度读数）。

由于其可预测的结构，系统可以高效存储、检索和处理结构化数据。大多数传统数据分析严重依赖此类型数据。

### 非结构化数据

非结构化数据则相反；它缺乏预定义的数据模型或组织架构。它就像一个巨大的图书馆，里面堆满了书籍、文章、音频和视频，但都没有以统一方式编目。尽管它包含海量信息，但提取特定信息需要更高级的方法。

- **特性：** 无固有结构，格式差异大，不易放入行和列。
- **常见示例：**
  - 文本文件（电子邮件、文章、社交媒体帖子、聊天记录）。
  - 媒体文件（如JPEG图片、MP3音频文件、MP4视频文件）。
  - 演示文稿（如PowerPoint文件）。

分析非结构化数据通常涉及自然语言处理（NLP）技术（针对文本）或计算机视觉技术（针对图像），以首先施加一些结构或提取有意义的特征，然后才能进行分析。尽管存在挑战，它是一种增长迅速的数据类型，包含有价值的信息。

### 半结构化数据

半结构化数据介于结构化数据的高度组织化结构和非结构化数据的自由形式特性之间。它不符合关系型数据库中表的严格结构，但包含标签、标记 (token)或其他组织形式，使其元素可识别且呈层级结构。

- **特性：** 包含组织标记（如标签），但不符合严格的表格模型，常使用层级或嵌套结构。
- **常见示例：**
  - **JSON（JavaScript对象表示法）：** 常用于Web API，使用键值对和嵌套结构。示例：`{"name": "Alice", "email": "alice@example.com", "orders": [{"order_id": 123, "amount": 50}, {"order_id": 456, "amount": 75}]}`
  - **XML（可扩展标记语言）：** 使用标签定义元素，常见于配置文件和较旧的Web服务。示例：`<customer><name>Bob</name><email>bob@example.com</email></customer>`
  - Web服务器日志。
  - 电子邮件（包含“收件人”、“发件人”、“主题”等结构化标题，但正文为非结构化文本）。

半结构化数据比结构化数据提供更大的灵活性，同时比纯粹的非结构化数据更容易自动解析和处理。

> 比较三种主要数据类型及其典型示例的图表。

理解这些区别对数据工程师非常重要。你正在处理的数据类型直接影响有关以下方面的决策：

1. **存储：** 关系型数据库擅长处理结构化数据，而数据湖常用于非结构化或半结构化数据。NoSQL数据库可能被选择用于特定类型的半结构化数据。
2. **处理：** 转换结构化数据（如SQL）的工具和技术与处理非结构化数据（如NLP库）所需的工具和技术有所不同。
3. **管道设计：** 提取、转换和加载过程（ETL/ELT）必须根据源数据的特定格式和特性进行调整。

随着课程推进，你将看到这些基本数据类型如何在关于数据库、数据仓库、数据湖以及连接它们的管道的讨论中反复出现。识别它们是掌握数据管理的第一步。

## 参考资料

- [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media; Pages: 447
  本书提供了数据工程的通用指南，其中包含解释不同数据类型及其如何影响系统设计的部分。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本备受推崇的图书，帮助读者理解各种数据系统、其模型、存储和针对不同数据类型的处理方法。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  本书涵盖数据库管理系统原理，特别是关系模型和结构化数据，为数据组织提供了背景知识。
