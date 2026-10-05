# 第 5 章：无监督学习：聚类

来源：[原章节](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering)

[返回课程目录](../README.md)

前面的章节主要讲解了监督学习，模型从包含预设答案或标签的数据中进行学习。本章将转而讲解无监督学习，这是一类不同的机器学习方法，其数据缺少这些明确的标签。这里的目标是发现数据本身固有的结构或规律。

具体来说，我们将介绍聚类，这是一种常见的无监督任务，旨在将相似的数据点归为一类。你将学到：

*   无监督学习的定义和用途。
*   聚类的方法原理及其应用。
*   K-Means 算法如何将数据划分成指定数量的簇，通常用 $K$ 表示。
*   选择簇数量 ($K$) 时需要考虑的事项。
*   K-Means 过程中的步骤及其一些局限性。
*   如何在实际练习中将 K-Means 应用到数据集。

本章介绍了使用 K-Means 聚类技术在无标签数据中发现规律的基础知识。

## 小节

- 1. [什么是无监督学习？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%97%A0%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%EF%BC%9F.md)
- 2. [聚类简介](02-%E8%81%9A%E7%B1%BB%E7%AE%80%E4%BB%8B.md)
- 3. [K-Means 算法](03-K-Means%20%E7%AE%97%E6%B3%95.md)
- 4. [选择聚类数量 (K)](04-%E9%80%89%E6%8B%A9%E8%81%9A%E7%B1%BB%E6%95%B0%E9%87%8F%20%28K%29.md)
- 5. [K-Means 如何找到簇群](05-K-Means%20%E5%A6%82%E4%BD%95%E6%89%BE%E5%88%B0%E7%B0%87%E7%BE%A4.md)
- 6. [K-均值算法的局限性](06-K-%E5%9D%87%E5%80%BC%E7%AE%97%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 7. [动手操作：K-Means 在简单数据上的应用](07-%E5%8A%A8%E6%89%8B%E6%93%8D%E4%BD%9C%EF%BC%9AK-Means%20%E5%9C%A8%E7%AE%80%E5%8D%95%E6%95%B0%E6%8D%AE%E4%B8%8A%E7%9A%84%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/quiz)
