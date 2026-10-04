---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-xgboost"
lesson: "why-xgboost"
sourceId: 7591
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/why-xgboost"
title: "为何选择XGBoost？速度与性能"
description: "了解XGBoost（极端梯度提升）受欢迎的原因，着重介绍其计算速度和出色的预测表现。"
order: 1
plots: ["plots/7591-0.json"]
sourceHash: "8ceb9ddfda6aa4ccc111facf936a4de11efb7e7750997d2e5aa6b4940121956a"
sourceCorrections: []
---

XGBoost，即极端梯度提升，以其卓越的计算速度和预测准确性，已成为数据科学界的主流算法选择。它被开发旨在优化提升算法的能力，显著改进了标准的梯度提升机（GBM）实现。这种先进算法在需要高效率和强大模型性能的场景中表现出色，使其成为生产环境和数据科学竞赛的首选方案。

### 速度优势：系统级优化

当您从Scikit-Learn的GBM转向XGBoost时，最明显的区别是训练时间大幅缩短，尤其是在大型数据集上。这种速度并非偶然，而是深思熟虑的工程决策的结果。

#### 并行与分布式计算

起初，对提升算法进行并行化似乎不可能。由于每棵新树都旨在纠正前一棵树的错误，因此这个过程本质上是顺序的。然而，训练单个树计算开销最大的部分是为每个特征寻找最佳划分点。XGBoost巧妙地并行化了这一内部循环。它能够同时在多个CPU核心上评估不同特征的潜在划分点。

对于更大的数据集，XGBoost可以在Apache Spark或Dask等分布式计算框架上运行，使其能够跨多台机器进行扩展。

> 一张简化图，比较了标准GBM中特征划分的顺序评估与XGBoost中构建单棵树时的并行评估。

#### 缓存感知优化

现代CPU并非每次操作都直接访问主内存。它们使用更小、更快的内存缓存来存储常用数据。XGBoost被设计为“缓存感知”，这意味着它以一种能最大化利用这些CPU缓存的方式组织内存中的数据。它将数据预取到缓冲区，从而实现更高效的梯度计算。尽管这是一种低级优化，但通过减少等待主内存数据造成的延迟，它对性能产生了显著影响。

#### 核外计算

XGBoost可以处理大到无法完全载入内存的数据集。它通过一项名为“核外”计算的功能实现这一点。数据被分成数据块并存储在磁盘上。在训练期间，XGBoost按需将这些数据块载入内存，处理后再丢弃。这使得您可以使用RAM少得多的机器训练TB级别数据集上的模型。

### 性能优势：算法改进

除了原始速度，XGBoost通常通过多项算法改进来生成更准确的模型。

#### 规范化正则项

正如章节概述中所述，XGBoost直接在其目标函数中包含正则项。Scikit-Learn中的标准GBM主要通过`max_depth`和`subsample`等超参数 (parameter) (hyperparameter)控制过拟合 (overfitting)。XGBoost也这样做，但它向其优化的损失函数 (loss function)添加了L1（Lasso）和L2（Ridge）正则项。

目标函数如下所示：
$目标函数 = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$

其中，项 $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda||\omega||^2$ 惩罚叶节点数量 ($T$) 和叶权重 (weight) ($\omega$) 的大小。这种更系统化的正则方法有助于防止过拟合，并常能在未见数据上带来更好的泛化能力。

#### 稀疏感知划分点查找

实际数据通常是稀疏的，包含许多缺失值或零条目（例如，在独热编码之后）。XGBoost拥有内置的缺失数据处理程序。XGBoost无需您事先填充缺失值，而是在训练期间为每个树节点上包含缺失值的样本学习一个默认方向。这种方法不仅更方便，还能通过从数据本身学习处理缺失信息的最佳方式，带来更准确的模型。

这些优化的综合效果是形成了一个比其前身更快且通常更准确的框架。下面的图表展示了在中型数据集上训练时的典型时间比较。



![训练时间比较](plots/7591-0.json)



> Scikit-Learn的`GradientBoostingClassifier`与`XGBClassifier`在包含100,000行和50个特征的示例数据集上的模型训练时间比较。

总之，XGBoost通过将系统优化与算法改进相结合而赢得声誉。它提供了一个高性能工具，能够有效扩展，并包含直接应对机器学习 (machine learning)中常见难题（例如过拟合和缺失数据）的功能。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost的原始论文，阐述了其系统优化（并行、分布式、缓存感知、核外计算）和算法改进（正则化、稀疏感知分裂查找）。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) — XGBoost Contributors (2024)
  XGBoost库的官方文档，提供关于其功能、API以及高级用法（包括分布式和核外训练）的最新信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本全面的机器学习指南，提供关于梯度提升算法的实践指导，包含XGBoost的详细比较和应用。
