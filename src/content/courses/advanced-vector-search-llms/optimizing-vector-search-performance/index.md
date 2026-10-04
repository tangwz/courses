---
course: "advanced-vector-search-llms"
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance"
sourceId: 781
chapter: "optimizing-vector-search-performance"
title: "向量搜索性能与效率的优化"
order: 2
description: "学习优化向量搜索速度、内存使用和效率的技术，包括量化和过滤。"
hasQuiz: false
---

基于前一章的近似最近邻算法，本章将集中讨论优化向量搜索性能和资源使用的实际需求。低搜索延迟、高查询吞吐量和高效的内存管理，对于部署高效的LLM应用来说非常重要。

本章将介绍实现这些目标的方法。我们将讨论向量压缩技术，例如标量量化（$SQ$）和乘积量化（$PQ$），以及其优化型（$OPQ$）变体。你将学习实现高效元数据过滤的策略，并比较前置过滤和后置过滤机制。此外，本章还将讨论硬件加速选项，如CPU SIMD指令和GPU，以及内存管理和缓存方法。完成本章学习将让你掌握实用技术，以调整向量搜索操作，提升速度和效率。
