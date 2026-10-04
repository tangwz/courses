---
course: "introduction-to-data-science"
chapter: "understanding-data-fundamentals"
lesson: "what-constitutes-data"
sourceId: 2091
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-2-understanding-data-fundamentals/what-constitutes-data"
title: "什么是数据？"
description: "了解数据的含义以及信息如何表示以便进行分析。"
order: 1
plots: []
sourceHash: "1731b6bd8d2259b01d18ac6e8e448026ae711f5bec026a8367df1b338eb108ea"
sourceCorrections: []
---

数据是数据科学的原始材料。可以把它看作是为分析而收集的事实、数字、观察结果、符号或描述的集合。在我们获取见解或构建模型之前，首先需要了解我们正在处理的是什么。

从根本上讲，数据代表着信息片段。但在数据科学的背景下，我们通常认为数据是任何可以被数字化记录、存储和处理的事物。它不仅仅是电子表格中的数字，尽管那是一种常见形式。数据可以多种多样：

- **数字：** 气象站记录的温度（$25.5^\circ C$），股票价格（$175.30$），网页链接的点击次数（1,204）。
- **文本：** 客户评论（“很棒的产品！”）、电子邮件、文章、社交媒体帖子。
- **图像：** 照片、医学扫描（如X射线）、卫星图像。每张图像都由像素组成，每个像素的颜色和强度都可以视为数据。
- **音频：** 录音中的口语、音乐、产生声音模式的传感器读数。
- **视频：** 图像序列与音频的结合。

考虑一个简单例子：追踪一家小型网店的销售情况。数据可能包括：

- 客户姓名（文本）
- 购买商品（文本/分类）
- 购买金额（数字，特指货币）
- 购买日期和时间（日期/时间）
- 客户地点（文本/分类）

每条信息，例如“Alice Smith”、“笔记本电脑包”、“$49.99$”、“2023-10-26 14:30:05”、“New York”，都是一个*数据点*或*观察值*。当我们收集许多此类相关观察值时，通常会将它们组织成一个**数据集**。通常，这个数据集采用表格形式，其中行代表单个记录（如单次购买），列代表不同的属性或特征（如客户姓名、商品、价格）。

```text
客户ID      | 姓名  | 商品  | 价格   | 购买日期        
-----------|-------------|--------------|-------|--------------------
101        | Alice Smith | Laptop Bag   | 49.99 | 2023-10-26 14:30:05
102        | Bob Johnson | USB Cable    | 12.50 | 2023-10-26 15:01:22
101        | Alice Smith | Mouse        | 25.00 | 2023-10-27 09:15:10
...        | ...         | ...          | ...   | ...
```

> 一个表示客户购买情况的简单表格数据集。

区分原始数据和**信息**很重要。原始数据，比如数字 `101`，独立来看可能意义不大。只有当我们赋予它背景时，它才成为信息。知道 `101` 代表“Alice Smith”的 `客户ID` 使其具有意义。数据科学通常涉及将原始数据转换为有用的信息，并最终转化为可操作的见解。

了解什么是数据是重要的第一步。识别其各种形式使你能够思考如何组织、清理和分析它，这些是我们接下来会介绍的主题。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in Python](http://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38914-1](https://doi.org/10.1007/978-3-031-38914-1)
  一本广泛认可的教材，介绍数据类型、数据表示以及统计学习和数据科学中使用的基本数据概念。
- [Database System Concepts](https://www.db-book.com/) — Abraham Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill Education
  一本标准教材，为数据、信息以及将数据组织成表格等结构化数据集的原则提供了严谨的概念基础。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本实践指南，展示了如何使用Python的数据结构（特别是DataFrame）来表示和操作不同形式的数据（数字、文本等）。
- [Introduction to Computational Thinking and Data Science (Course 6.0002)](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/) — Prof. Eric Grimson, Prof. John Guttag, Dr. Ana Bell (2016)
  Publisher: MIT OpenCourseWare
  本课程提供了基础讲义和作业，有助于定义数据以及如何在数据科学背景下通过计算方法处理数据。
