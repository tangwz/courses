---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-lightgbm-catboost"
lesson: "lightgbm-goss"
sourceId: 7598
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/lightgbm-goss"
title: "LightGBM 介绍：基于梯度的单边采样"
description: "了解 LightGBM 的基于梯度的单边采样 (GOSS) 技术，该技术通过关注有价值的数据点来提高训练效率。"
order: 1
plots: []
sourceHash: "793b53c4bd603f4ca6d1a97af91499b823761a5a9c307e031023c20caa2aa9d7"
sourceCorrections: []
---

XGBoost，一种广泛使用的梯度提升算法，表现出色。然而，在实例数量庞大的数据集上，其树构建过程可能成为计算瓶颈。主要问题在于，每次分裂时，算法都必须扫描每个数据点以评估潜在的增益。微软的 LightGBM (Light Gradient Boosting Machine) 经过专门设计，通过引入更高效的训练方法来应对这一难题。

其主要优化之一是一种新颖的采样技术，称为基于梯度的单边采样，简称 GOSS。此方法基于一个简单而有效的观察：并非所有数据实例对训练过程的贡献都相同。

### GOSS 的基本思路

在梯度提升中，每个实例损失函数 (loss function)的梯度代表当前模型对该实例的预测“错”了多少。大梯度表示该实例预测不佳，因此是一个“有价值的”例子，模型可以从中获得很多。反之，小梯度的实例已经被集成模型很好地预测了；模型从中学到的东西较少。

传统的随机梯度提升方法均匀地采样数据。GOSS 提出了一种更巧妙的替代方案。它不平等对待所有实例，而是将学习过程集中在那些较难拟合的实例上。核心思想是保留所有具有大梯度的实例，而只对具有小梯度的实例进行随机采样。

### GOSS 工作原理

名称中的“单边”指我们仅从数据的一侧进行降采样，即那些梯度小、信息量低的实例一侧。该过程可以分解为几个步骤：

1. **计算梯度：** 对于当前的提升迭代，计算所有训练实例的梯度。
2. **划分数据：** 根据实例梯度的绝对值降序排列实例。
3. **选择实例：**
   - 保留梯度最大的 `a * 100%` 实例。这些是最有价值的样本。
   - 从剩余的 `(1 - a) * 100%` 实例中，随机采样其中的 `b * 100%`。这些是信息量较低的样本。
4. **放大采样数据：** 为在计算信息增益时保持原始数据分布，随机采样的（小梯度）数据的梯度会乘以一个常数因子 `(1 - a) / b` 进行放大。这种重新加权确保了采样数据对梯度统计量的贡献与其原始大小成比例，从而避免模型偏向大梯度数据。

这个过程使 LightGBM 能够使用更小、更集中的数据集来为每棵新树找到最佳分裂点，大幅减少计算时间，同时没有明显降低准确性。

> 基于梯度的单边采样 (GOSS) 过程图。该算法保留所有具有大梯度的数据点，并对具有小梯度的部分数据点进行采样，然后对其进行重新加权以保持整体数据分布。

在 LightGBM 库中，通过将 `boosting_type` 参数 (parameter)设置为 `'goss'` 即可启用 GOSS。比例 `a` 和 `b` 分别由 `top_rate` 和 `other_rate` 超参数 (hyperparameter)控制。这种策略性采样方法是 LightGBM 训练速度通常比其他梯度提升实现快很多的原因之一，使其成为处理大规模数据集的有力选项。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154
  介绍LightGBM及其关键优化（包括梯度单侧采样GOSS）的原始论文。
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/) — Microsoft (2024)
  提供LightGBM功能、API、配置参数和实际使用（包括GOSS超参数）的详细信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE4ZoVvGt17GqAYXF_fPTs6_i1un8P_Nqv8Povj5SYpxxj2b-ADPjCXmfv_8R3gjGCMtnhYCeL0eqd60gMCtjVUuPWqGDPxdNjb-Bvf0tNuEknpyXR7dlGY_eTY9fTmLlbni3ENPOg7zfD3-ggRIrxMgmrlx7kTNuSfLw50K4aeJwO0XiVBeeo=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本涵盖机器学习算法的实用指南，包括对LightGBM等梯度提升方法的讨论及其在更广泛背景下的应用。
