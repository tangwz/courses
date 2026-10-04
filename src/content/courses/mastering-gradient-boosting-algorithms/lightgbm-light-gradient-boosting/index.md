---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting"
sourceId: 537
chapter: "lightgbm-light-gradient-boosting"
title: "LightGBM：轻量级梯度提升机"
order: 5
description: "学习LightGBM的新颖技术，如GOSS和EFB，以提高速度和效率，尤其是在大型数据集上。"
hasQuiz: false
---

尽管XGBoost等算法比标准梯度提升机在性能上有大幅提升，但它们在处理超大数据集和高维特征空间时，仍可能面临计算瓶颈。本章将介绍LightGBM，这是一个专门为解决这些挑战而设计的框架，它优先考虑训练速度和内存效率，同时在准确性方面没有重大牺牲。

你将学习有助于LightGBM高效运行的核心技术。我们将涵盖：

*   **基于梯度的单侧采样（GOSS）：** 一种有选择地关注梯度较大数据实例的方法，旨在减少计算量同时保持模型准确性。
*   **互斥特征捆绑（EFB）：** 一种将互斥特征打包的方式，可有效减少训练时需考虑的特征数量。
*   **基于直方图的算法：** LightGBM如何使用离散化特征值（直方图）来加快寻找最优分割点的过程。
*   **逐叶生长树策略：** 将LightGBM逐叶生长树的策略与更常见的逐层生长方法进行对比，并理解其对性能和过拟合可能性的影响。
*   **优化后的类别特征处理：** 查看LightGBM对类别变量的原生支持。

本章还将指导你学习LightGBM Python API的主要参数，并最终通过一个实践练习，让你实现并训练一个LightGBM模型。
