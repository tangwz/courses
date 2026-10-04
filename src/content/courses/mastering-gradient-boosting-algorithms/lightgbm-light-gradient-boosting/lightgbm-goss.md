---
course: "mastering-gradient-boosting-algorithms"
chapter: "lightgbm-light-gradient-boosting"
lesson: "lightgbm-goss"
sourceId: 1998
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-goss"
title: "基于梯度的单侧采样 (GOSS)"
description: "详细解释GOSS数据子采样技术。"
order: 2
plots: ["plots/1998-0.json"]
sourceHash: "294979c35f0eb0ca1fdaef4e49de1ef71402f56b4a6d3413fedb365874c0de90"
sourceCorrections: []
---

在大数据集上训练梯度提升模型可能计算量很大。这部分成本很大程度上来源于在构建每棵树时，需要遍历所有数据实例来评估可能的划分点。基于梯度的单侧采样（GOSS）是LightGBM中的一项主要改进，旨在通过智能地减少用于寻找最佳划分点的数据实例数量来加快此过程。

GOSS背后的基本思想是，在每次提升迭代中，并非所有训练实例对学习过程的贡献都相同。回想一下，梯度提升的工作原理是顺序拟合弱学习器（通常是决策树）来拟合损失函数 (loss function)关于当前模型预测的负梯度。一个实例的梯度本质上代表了残差误差，或者说当前集成模型对该特定实例的“错误”程度。

梯度大的实例是当前模型错误较多的实例。这些实例在某种意义上是“训练不足”的，并提供了关于如何在下一次迭代中改进模型的重要信息。相反地，梯度小的实例已经被当前集成模型很好地预测；过度基于这些实例进行进一步优化可能导致对细微变化的过拟合 (overfitting)，并在整体模型改进方面产生边际效益递减，同时仍然产生计算成本。

GOSS利用了这一观察结果，提出了一种非均匀采样策略。GOSS不是平等对待所有实例或使用简单随机采样，而是优先处理梯度较大的实例，同时保持梯度较小实例的有代表性样本。这在不大幅改变有效学习所需数据分布的情况下，实现了计算节省。

### GOSS机制

在每棵树的训练过程中，GOSS步骤可以分解为以下几个部分：

1. **计算梯度：** 根据目前已构建集成模型的预测，计算所有训练实例的损失函数 (loss function)负梯度。
2. **按梯度绝对值排序：** 根据实例梯度的绝对值 $|g_i|$，按降序排列实例。
3. **选择高梯度实例：** 保留绝对梯度最大的前 $a \times 100\%$ 的实例。将此集合表示为 $A$。这些是模型当前最难以处理的实例。参数 (parameter) $a$ 在LightGBM中通常被称为 `top_rate`。
4. **采样低梯度实例：** 从剩余的 $(1-a) \times 100\%$ 的实例（梯度较小的实例）中，随机采样一部分 $b$。这意味着选择 $b \times (1-a) \times N$ 个实例，这里 $N$ 是实例总数。将此集合表示为 $B$。参数 $b$ 在LightGBM中被称为 `other_rate`。
5. **合并并放大：** 合并这两个实例集合：$A \cup B$。这个更小的组合数据集用于在当前树中寻找最优划分。
6. **放大小梯度：** 为了确保低梯度实例（集合 $B$）的贡献不会因降采样而被低估，它们的梯度被放大。具体来说，集合 $B$ 中实例 $i$ 的梯度 $g_i$ 乘以一个常数因子：$\frac{1-a}{b}$。这种放大有助于保持整体梯度分布的近似无偏估计。如果不进行放大，在计算划分增益时，那些经常被很好预测的实例的贡献将被系统性地低估。

### 为何放大很重要

决策树中寻找最佳划分的信息增益计算很大程度上依赖于潜在子节点中梯度和黑塞矩阵（或仅梯度，取决于具体目标）的总和。通过以概率 $b$ 对低梯度实例（集合 $B$）进行采样，我们减少了它们的数量。为了补偿这种减少，并确保集合 $B$ 中梯度之和的*期望值*近似于它所代表的原始低梯度群体的总和，我们需要按比例放大采样梯度。因子 $\frac{1-a}{b}$ 精确地实现了这一点，它对集合 $B$ 中实例 $i$ 的采样梯度 $g_i$ 进行缩放，使其总和统计上代表从其采样的 $(1-a) \times N$ 个低梯度实例的整个池子的总和。

### GOSS可视化

想象一下数据集中绝对梯度值的分布。GOSS有效地保留了高于某个阈值（前 'a' 比例）的所有实例，然后从低于该阈值的实例中随机采样，并将它们的影响力按比例放大。



![GOSS实例选择 (a=1/3, b=0.4)](plots/1998-0.json)



> GOSS选择的图示。梯度高的实例（红色方块）总是被保留。梯度低的实例（灰色条形）被随机采样（蓝色圆圈），并且它们的梯度在增益计算时被放大。没有蓝色圆圈的灰色条形所代表的实例在此树的划分寻找中被丢弃。

### GOSS的优点

- **效率：** 减少了寻找划分时所需扫描的数据量，显著加快了训练时间，在大数据集上尤其明显。
- **准确性：** 通过专注于信息量大的实例并适当加权采样的低梯度实例，GOSS旨在找到与使用完整数据集找到的划分几乎一样好的划分，从而保持模型准确性。它通常优于朴素随机采样，后者可能会偶然丢弃过多重要的、高梯度的实例。

### GOSS与随机梯度提升

区分GOSS与传统随机梯度提升（如Scikit-learn或标准XGBoost中实现的那样）中使用的子采样技术很重要。随机梯度提升通常在构建每棵树之前，对实例执行*均匀随机采样*（行子采样）。相比之下，GOSS基于梯度大小执行*有偏采样*。它假设梯度较小的实例对于定义最优划分结构不那么关键，并且可以更稀疏地采样，前提是它们的影响被正确缩放。

### LightGBM中的实际使用

在使用LightGBM时，GOSS作为'boosting\_type'（或'boosting'）参数 (parameter)设置为'gbdt'的一部分，默认是启用的。其具体行为由以下参数控制：

- `top_rate` ($a$)：要保留的具有最大梯度的实例比例。默认值通常为0.2。增加此值会保留更多高梯度数据，但会降低速度增益。
- `other_rate` ($b$)：要采样的剩余（低梯度）实例的比例。默认值通常为0.1。增加此值会使用更多低梯度数据，可能会以牺牲速度为代价略微提高准确性。

这些参数通常不需要大量调优，但如果您怀疑GOSS对特定问题上的准确性产生负面影响，或者您需要进一步最大化训练速度，则可以进行调整。

总之，基于梯度的单侧采样是一种巧妙的技术，它通过减轻寻找划分的计算负担，并将计算精力集中在每一步改进模型最重要的数据点上，从而显著提升了LightGBM的效率。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154
  介绍 LightGBM 并详细阐述梯度单侧采样 (GOSS) 算法的原始学术论文。
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — LightGBM Contributors (2024)
  LightGBM 官方文档，提供了 GOSS 的当前实际使用细节和配置参数，包括 `top_rate` 和 `other_rate`。
- [Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473%2801%2900065-2) — Jerome H. Friedman (2002)
  Journal: Computational Statistics & Data Analysis; Publisher: Elsevier; Volume: 38; Pages: 367-378; DOI: [10.1016/S0167-9473(01)00065-2](https://doi.org/10.1016/S0167-9473%2801%2900065-2)
  这篇基础论文介绍了随机梯度提升的概念，利用子采样提高计算效率和泛化能力，为理解 GOSS 的特点提供了背景。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本内容全面的教科书，为梯度提升机和其他集成学习方法提供了理论基础，有助于理解 GOSS 背后的原理。
