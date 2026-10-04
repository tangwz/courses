---
course: "advanced-vector-search-llms"
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-1-ann-algorithms"
sourceId: 779
chapter: "ann-algorithms"
title: "近似最近邻算法"
order: 1
description: "HNSW、IVF、PQ及其他进阶近似最近邻（ANN）算法的详尽解读，应用于向量搜索。"
hasQuiz: false
---

在高维空间中查找查询向量的精确最近邻点，这在使用大型语言模型（LLM）的嵌入时很常见，但其扩展性较差。采用精确 $k$-NN 在数百万甚至数十亿向量中匹配最接近点，这对于诸多应用而言，计算耗费巨大，难以执行。

近似最近邻（ANN）算法通过舍弃完全精确度，换取搜索速度的极大提高，从而提供了实用方案。

本章将主要讲解几种广泛应用的ANN技术的核心运作方式，它们对于创建高效的向量搜索系统非常必要。我们将考察分层可导航小世界（HNSW）、倒排文件索引（IVF）的不同形式以及乘积量化（PQ）的构建与查询过程。我们还将介绍其他图结构方法，并探讨在选择合适算法时，如何根据构建时间、内存占用和查询表现等因素进行主要考量。本章最后会安排一次关于HNSW实现和调整的实际操作环节。
