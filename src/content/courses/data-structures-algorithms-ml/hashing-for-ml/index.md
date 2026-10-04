---
course: "data-structures-algorithms-ml"
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-3-hashing-for-ml"
sourceId: 977
chapter: "hashing-for-ml"
title: "散列在特征工程和相似搜索中的应用"
order: 3
description: "了解散列表、冲突处理、稀疏数据的特征散列以及局部敏感散列 (LSH)。"
hasQuiz: true
---

处理大型数据集和高维特征向量是机器学习中普遍存在的问题。散列提供多种方法来高效管理这些数据，在理想条件下，查找和插入操作通常能达到接近常数时间$O(1)$的效率。

本章将介绍散列函数和散列表的基本原理，包括当多个键映射到同一索引时处理冲突的方法。接下来，我们将这些思路应用于具体的机器学习场景：

*   您将了解到**特征散列**（“散列技巧”）如何紧凑地表示稀疏、高维数据，减少内存需求，而无需预先构建字典。
*   我们将介绍**局部敏感散列 (LSH)**，这是一种用于在大型数据集中快速查找近似最近邻的方法，对相似性搜索任务尤其有用。

在本章中，我们将考虑使用 Python 的实际实现细节，并分析这些散列方法在机器学习流程中的性能特点和权衡。
