---
course: "vector-databases-semantic-search"
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-2-introducing-vector-databases"
sourceId: 733
chapter: "introducing-vector-databases"
title: "向量数据库介绍"
order: 2
description: "学习向量数据库的核心组成部分、架构、数据模型、CRUD操作及扩展性考量。"
hasQuiz: true
---

在上一章中，我们已了解数据如何转换为有意义的向量表示。现在我们要解决一个实际问题：如何存储、管理并高效查找这些高维向量？传统的结构化（关系型）或非结构化（NoSQL）数据库通常不适合在此所需的主要操作：使用相似性度量（例如 $cosine\_similarity$ 或欧氏距离 ($$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$)）在高维空间中查找与查询向量最接近的向量。

本章将介绍向量数据库，它们是专门为处理这些特殊需求而设计的系统。我们将了解它们的核心架构组成部分，并理解它们与传统数据库的区别。您将学习用于向量及其相关元数据的常用数据模型，在此场景下，基本的创建、读取、更新和删除（CRUD）操作如何工作，以及将向量相似性搜索与基于元数据属性的过滤相结合的能力。我们还将讨论关于这些数据库扩展性以处理大量向量数据的基本考量。到本章结束时，您将对向量数据库的相关知识及其运行原理有初步的认识。
