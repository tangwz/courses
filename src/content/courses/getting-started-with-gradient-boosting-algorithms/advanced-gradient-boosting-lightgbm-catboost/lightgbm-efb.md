---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-lightgbm-catboost"
lesson: "lightgbm-efb"
sourceId: 7599
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/lightgbm-efb"
title: "LightGBM 的独占特征捆绑"
description: "了解 LightGBM 中独占特征捆绑 (EFB) 如何通过将互斥特征分组来减少特征数量，从而加快训练速度。"
order: 2
plots: []
sourceHash: "87cc886405c8af13a00a059cc3cafb6ad022a20513bbd99be45da02e40142bd7"
sourceCorrections: []
---

LightGBM 通过多种技术优化训练速度，其中包括基于梯度的单边采样 (GOSS)，该技术通过减少数据实例数量来加速训练。与此相辅相成，LightGBM 还采用了独占特征捆绑 (EFB) 这种强大方法，专门用于减少特征数量。这种方法在高维数据集中尤为有效，因为这类数据集中许多特征是稀疏的，这在处理独热编码的类别变量或来自文本的词频矩阵时很常见。

### 高维稀疏数据带来的难题

在稀疏特征空间中，大多数特征对于大多数观测值都为零。例如，如果你对一个包含 500 个不同城市的 `city` 列进行独热编码，那么每个数据点在这 500 个新特征中将有 499 个零和一个一。

当决策树算法寻找最佳分裂点时，它必须遍历所有特征及其值。在稀疏数据集中，这个过程涉及大量不必要的计算，因为要扫描那些对数据分裂没有提供任何信息的零值。EFB 通过将稀疏的、互斥的特征捆绑成一个单一的、更密集的特征，直接解决了这种低效问题。

### 互斥特征

如果两个特征对于同一个数据实例从不同时取非零值，它们就被认为是**互斥的**。独热编码的 `city` 特征就是一个很好的例子。一个房产不能同时位于“纽约”和“伦敦”，因此，如果 `city_is_new_york` 特征为 1，那么 `city_is_london` 特征就必须为 0。

LightGBM 智能地识别出这些互斥特征组，并将它们合并成一个新特征，即**特征捆绑**。这极大地减少了算法需要评估的特征数量，从而大幅提高速度，同时不牺牲准确性。

### 捆绑如何运作

这个过程包括两个主要步骤：识别哪些特征可以捆绑，然后将它们合并。

1. **识别捆绑：** LightGBM 将此建模为一个图问题。每个特征是一个节点，并在任何两个*非*互斥特征之间绘制一条边（即它们在同一行至少有一次同时为非零值）。算法随后采用贪心着色方法，将节点（特征）分组为捆绑。具有相同“颜色”的特征被捆绑在一起。为了使这个过程更实用，算法允许少量冲突，由 `max_conflict_rate` 参数 (parameter)控制。
2. **将特征合并到捆绑中：** 一旦识别出一组独占特征，它们就被合并成一个新特征。为了保留每个原始特征的信息，LightGBM 通过添加偏移量在新特征中创建不同的分箱。

假设我们有两个稀疏的、独占的特征：`特征 A`（唯一值为 {0, 1, 2}）和 `特征 B`（唯一值为 {0, 1}）。为了合并它们，我们可以将 `特征 B` 的值偏移 `特征 A` 的最大值。

新的捆绑特征将按以下方式构建：

- 如果 `特征 A` 有值，直接使用该值（例如，1 或 2）。
- 如果 `特征 B` 有值，添加一个等于 `特征 A` 最大值的偏移量（例如，`value_B + 2`）。因此，`B=1` 变为 `1 + 2 = 3`。
- 新特征中的 0 值表示两个原始特征都为 0。

这确保了 `特征 A` 和 `特征 B` 的值在新特征中占据不同的、不重叠的范围，使算法能够像以前一样对它们进行分裂。

> 此图描绘了两个互斥特征如何合并。`特征 A` 的值得以保留，而 `特征 B` 的值则通过偏移占据最终捆绑中的新范围。结果是一个单一的、更密集的特征，它保留了所有原始信息。

### 独占特征捆绑的作用

通过将许多稀疏特征转换为较少数量的密集特征，EFB 大幅提升了性能。主要优势来自于降低直方图构建的成本。算法无需为数百或数千个稀疏特征构建直方图，只需为几十个特征捆绑构建即可。

对于你这个实践者来说，EFB 是使 LightGBM 速度极快的“神奇”组成部分之一，尤其是在具有大量特征工程（如独热编码）的表格数据上。它默认启用，并且很少需要调整，但理解其机制有助于解释为什么 LightGBM 常常优于其他库，尤其是在特定类型数据集的训练速度和内存使用方面。与 GOSS 一起，EFB 构成了强大的优化组合，定义了 LightGBM 的效率。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 3146-3154; DOI: [10.5555/3295222.3295240](https://doi.org/10.5555/3295222.3295240)
  介绍 LightGBM 的基础论文，详细说明了独占特征捆绑 (EFB) 作为处理高维稀疏数据集的关键优化方法。
- [Features - LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Features.html) — LightGBM Contributors (2024)
  LightGBM 设计概述，其中包含独占特征捆绑如何提高训练效率和减少内存占用的内容。
