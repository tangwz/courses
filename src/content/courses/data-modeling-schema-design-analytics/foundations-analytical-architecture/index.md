---
course: "data-modeling-schema-design-analytics"
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-1-foundations-analytical-architecture"
sourceId: 1389
chapter: "foundations-analytical-architecture"
title: "分析架构原理"
order: 1
description: "了解 OLTP 和 OLAP 系统之间的核心差异，以及列式存储在分析中的运行机制。"
hasQuiz: false
---

大多数应用数据库都旨在高效处理单个事务。当用户创建订单或更新个人资料时，系统会优先考虑数据完整性和写入速度。这些被称为在线事务处理（OLTP）环境的系统，当目标转变为大规模数据聚合和报告时，通常会遇到问题。

分析架构通过优化数据组织以提高检索速度而非事务吞吐量，来解决这一局限。在本章中，我们将阐明操作型工作负载与分析型工作负载（OLAP）之间的结构差异。一个主要研究点是物理存储层。我们将对比适用于事务系统的行式存储，与支持大规模数据集更快聚合的列式存储。

例如，对数据集 $D$ 执行一个标准求和操作：

$$S = \sum_{x \in D} x$$

在行式系统中，计算 $S$ 通常需要读取整个行才能获取变量 $x$。在列式系统中，数据库引擎只读取包含 $x$ 的特定列，这大大减少了输入/输出（I/O）开销。

我们还将考察反范式化的作用。尽管范式化消除了操作型数据库中的冗余，但分析模型常有意引入冗余，以减少查询执行中的复杂连接。到本章结束时，你将能够区分这些架构模式，并解读查询执行计划以验证性能差异。
