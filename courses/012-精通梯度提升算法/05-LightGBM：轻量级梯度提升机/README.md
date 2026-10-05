# 第 5 章：LightGBM：轻量级梯度提升机

来源：[原章节](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting)

[返回课程目录](../README.md)

尽管XGBoost等算法比标准梯度提升机在性能上有大幅提升，但它们在处理超大数据集和高维特征空间时，仍可能面临计算瓶颈。本章将介绍LightGBM，这是一个专门为解决这些挑战而设计的框架，它优先考虑训练速度和内存效率，同时在准确性方面没有重大牺牲。

你将学习有助于LightGBM高效运行的核心技术。我们将涵盖：

*   **基于梯度的单侧采样（GOSS）：** 一种有选择地关注梯度较大数据实例的方法，旨在减少计算量同时保持模型准确性。
*   **互斥特征捆绑（EFB）：** 一种将互斥特征打包的方式，可有效减少训练时需考虑的特征数量。
*   **基于直方图的算法：** LightGBM如何使用离散化特征值（直方图）来加快寻找最优分割点的过程。
*   **逐叶生长树策略：** 将LightGBM逐叶生长树的策略与更常见的逐层生长方法进行对比，并理解其对性能和过拟合可能性的影响。
*   **优化后的类别特征处理：** 查看LightGBM对类别变量的原生支持。

本章还将指导你学习LightGBM Python API的主要参数，并最终通过一个实践练习，让你实现并训练一个LightGBM模型。

## 小节

- 1. [动机：应对XGBoost的局限性](01-%E5%8A%A8%E6%9C%BA%EF%BC%9A%E5%BA%94%E5%AF%B9XGBoost%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [基于梯度的单侧采样 (GOSS)](02-%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E7%9A%84%E5%8D%95%E4%BE%A7%E9%87%87%E6%A0%B7%20%28GOSS%29.md)
- 3. [独占特征捆绑 (EFB)](03-%E7%8B%AC%E5%8D%A0%E7%89%B9%E5%BE%81%E6%8D%86%E7%BB%91%20%28EFB%29.md)
- 4. [基于直方图的分裂点查找](04-%E5%9F%BA%E4%BA%8E%E7%9B%B4%E6%96%B9%E5%9B%BE%E7%9A%84%E5%88%86%E8%A3%82%E7%82%B9%E6%9F%A5%E6%89%BE.md)
- 5. [逐叶式树生长](05-%E9%80%90%E5%8F%B6%E5%BC%8F%E6%A0%91%E7%94%9F%E9%95%BF.md)
- 6. [优化的类别特征处理](06-%E4%BC%98%E5%8C%96%E7%9A%84%E7%B1%BB%E5%88%AB%E7%89%B9%E5%BE%81%E5%A4%84%E7%90%86.md)
- 7. [LightGBM API：参数与配置](07-LightGBM%20API%EF%BC%9A%E5%8F%82%E6%95%B0%E4%B8%8E%E9%85%8D%E7%BD%AE.md)
- 8. [动手实践：实现 LightGBM](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20LightGBM.md)
