---
course: "building-scalable-data-warehouses"
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale"
sourceId: 1400
chapter: "advanced-data-modeling-scale"
title: "大规模数据建模进阶"
order: 2
description: "实践高级数据建模技术，包括 Data Vault 2.0 和复杂的 SCD 模式。"
hasQuiz: false
---

大规模并行处理 (MPP) 系统中的数据建模，需要将重心从存储资源节约转向计算优化和灵活性。像第三范式 (3NF) 这样的传统方法，常因分布式节点间大量数据混洗而产生明显的性能开销。即使标准的星型模式对展现层有效，但在PB级规模环境中，处理频繁结构变动或高速数据摄入时，也可能变得僵化。

本章将审视旨在支持可扩展性和迭代开发的架构模式。您将首先评估维度建模在分布式存储环境中的具体局限。接着，内容会转到 Data Vault 2.0，您将学习构建 Hubs、Links 和 Satellites，以实现业务键与其描述性属性的分离。我们还会介绍摄入和查询 JSON 和 Parquet 等半结构化数据格式的技术方法，这里将采用原生SQL扩展，而不是外部ETL流程。本节最后将介绍管理模式演进的技术，使您的数据仓库能够适应源系统变化，同时不破坏现有数据约定。
