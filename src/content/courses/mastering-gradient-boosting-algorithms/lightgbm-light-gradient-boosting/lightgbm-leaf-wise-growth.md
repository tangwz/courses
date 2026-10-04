---
course: "mastering-gradient-boosting-algorithms"
chapter: "lightgbm-light-gradient-boosting"
lesson: "lightgbm-leaf-wise-growth"
sourceId: 2002
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-leaf-wise-growth"
title: "逐叶式树生长"
description: "对比逐叶式生长与层级式生长及其影响。"
order: 5
plots: []
sourceHash: "0de9f0d03974339566b381b482d24b903d3d47514d601da3d478c728e2f1f5ca"
sourceCorrections: []
---

大多数决策树算法，包括标准梯度提升实现和XGBoost中使用的算法，采用**层级式**（或深度优先）树生长方式，使树逐层扩展。在进入下一个更深层级之前，给定深度上的所有节点都会被分裂。这种方法保持了平衡的树结构，易于管理，并且不太容易立即出现过拟合 (overfitting)。

然而，LightGBM默认采用一种不同的策略：**逐叶式**（或最佳优先）树生长。它不是逐层扩展，逐叶式策略侧重于在当前树结构中的任意位置找到能使损失函数 (loss function)减少幅度最大的单个叶节点，然后分裂该特定叶节点。这个过程重复进行：找到潜在损失减少最大的叶节点并分裂它。

### 层级式与逐叶式生长比较

想象构建一棵树。

- **层级式：** 你找到根节点的最佳分裂。然后，你找到两个结果子节点的最佳分裂。接着，你找到下一层级所有四个节点的最佳分裂，依此类推。你横向构建树，每次完成一层。
- **逐叶式：** 你找到根节点的最佳分裂。现在你有两个叶子。你评估分裂其中任意一个叶子可能带来的损失减少。假设分裂左侧叶子比分裂右侧叶子能带来更大的损失减少。你分裂左侧叶子。现在你有三个叶子。你再次评估分裂这三个叶子中任意一个的潜在改进，并分裂能提供最佳改进的那个。树倾向于沿着一条路径更深地生长，然后才考虑其他路径。

> 层级式和逐叶式生长之间的分裂顺序差异示例。逐叶式优先分裂能带来最大损失减少的节点，这通常会使树在初期呈现不对称形态。

### 逐叶式生长的优点

逐叶式策略的主要优点是**收敛速度快**。通过总是分裂能带来最大即时改进（损失减少）的节点，模型可能以更少的分裂次数达到更低的损失值，相比层级式方法。这是因为它不会浪费计算资源去分裂那些只为完成一个层级而提供微小提升的节点。对于大型数据集，这可以转化为显著更快的训练时间，并且在相同叶子数量下，模型可能更精确，因为算法将精力集中在最重要的部分。

结合基于直方图的分裂查找（这种方法能快速评估潜在分裂），逐叶式生长使LightGBM能够快速构建有效的树。

### 缺点与正则化 (regularization)

逐叶式生长的主要缺点是它倾向于产生**更深、可能不平衡的树**，并且过拟合 (overfitting)的风险增加，尤其是在较小的数据集上。由于它贪婪地追求损失减少最大的路径，它可能会生成非常深的树枝，这些树枝基于完美拟合训练数据噪声的分裂，而不是先考虑特征空间的其他部分。

为了解决这个问题，LightGBM大量依赖特定的正则化参数 (parameter)：

- `num_leaves`：这是在使用逐叶式生长时，LightGBM中用于控制树模型复杂度的主要参数。它直接限制每棵树的叶节点总数。较小的`num_leaves`值会产生更简单的树，并有助于防止过拟合。在逐叶式策略中，它通常比`max_depth`在控制复杂度方面更有效。
- `max_depth`：虽然仍可使用，但`max_depth`起着保护作用。它对树的深度施加了明确的限制，防止单个分支变得过深，即使尚未达到`num_leaves`的限制。设置`max_depth`有时可以帮助防止仅靠`num_leaves`无法避免的过拟合，但调整`num_leaves`通常是第一步。
- `min_child_samples`（或`min_data_in_leaf`）：此参数指定叶节点所需的最小数据点数量。如果分裂会导致叶节点中的样本少于此阈值，则该分裂将被拒绝，从而有效防止模型对非常小的数据点组中的噪声进行拟合。

在使用LightGBM时，有效调整`num_leaves`非常重要。默认值（31）通常是一个不错的起点，但它经常需要根据数据集的大小和复杂性进行调整。

总而言之，逐叶式生长是LightGBM速度和效率的一个重要影响因素。它通过关注最有希望的分裂，使模型能够快速收敛。然而，这种积极的策略需要仔细的正则化，主要通过`num_leaves`参数，以避免过拟合并确保良好的泛化性能。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154; DOI: [10.55917/cb_volume30-2017-1090](https://doi.org/10.55917/cb_volume30-2017-1090)
  LightGBM 的基础论文，介绍了其高效的叶子生长策略决策树算法及其优势。
- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — Microsoft (LightGBM Contributors) (2024)
  LightGBM 官方文档，涵盖了 `num_leaves`、`max_depth` 和 `min_child_samples` 等参数，这些参数对于调节叶子生长策略的树至关重要。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了 XGBoost 算法，该算法主要采用层级生长策略的决策树，与 LightGBM 的叶子生长策略形成对比。
