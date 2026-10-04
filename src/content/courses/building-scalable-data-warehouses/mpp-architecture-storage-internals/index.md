---
course: "building-scalable-data-warehouses"
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-1-mpp-architecture-storage-internals"
sourceId: 1399
chapter: "mpp-architecture-storage-internals"
title: "MPP 架构与存储原理"
order: 1
description: "理解 MPP 架构、列式存储和微分区在 Snowflake 和 Redshift 等现代数据仓库中的应用。"
hasQuiz: false
---

高容量分析设计的数据仓库，其运作原理与传统事务型数据库有所不同。当数据集超出单个服务器的承载能力时，性能表现有赖于系统分发工作的效率。本章考察大规模并行处理（MPP）的运作方式，其中无共享架构使多个计算节点能够同时处理数据片段。

如果大小为 $D$ 的数据集分布在 $n$ 个节点上，则单个节点的理想处理负载接近：

$$ L = \frac{D}{n} $$

在实际生产中实现这种理论效率，需要对存储原理有扎实的了解。我们将分析现代平台如何将计算与存储分离，使工程师能够根据工作负载需求而非存储容量独立调整资源。

本内容包括数据的物理组织，比较行式存储与BigQuery、Redshift和Snowflake中使用的列式格式。您将回顾压缩算法如何减少I/O开销，以及元数据如何通过微分区使查询引擎忽略不相关的数据块。学完本模块后，您将能够检查存储配置文件，并评估架构选择如何直接影响查询延迟和成本。
