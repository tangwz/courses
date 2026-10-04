---
course: "mastering-gradient-boosting-algorithms"
chapter: "lightgbm-light-gradient-boosting"
lesson: "lightgbm-motivation"
sourceId: 1995
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-motivation"
title: "动机：应对XGBoost的局限性"
description: "了解LightGBM旨在解决的效率瓶颈。"
order: 1
plots: []
sourceHash: "704425120dea85184562d9d6bd53e4700da087b5216a9911240ea0d155395257"
sourceCorrections: []
---

XGBoost 是一种强大且高效的梯度提升框架。它将正则化 (regularization)技术直接引入目标函数，并进行了稀疏感知和并行处理等系统级优化，这标志着相对于传统梯度提升机（GBM）的重大进步。对于许多问题，XGBoost在预测准确性和计算性能之间提供了出色的平衡。

然而，随着数据集在实例数量（$N$）和特征数量（$M$）两方面的持续增长，即使是像XGBoost这样经过优化的算法，也可能遇到显著的计算瓶颈。主要困难通常源于树构建过程中寻找最佳分裂点的过程。

考虑XGBoost默认的精确贪婪算法。为了在特定节点上为一个特征找到最优分裂，算法通常需要：

1. 根据特征值对实例进行排序。
2. 遍历该特征的所有可能分裂点（通常在每个唯一的排序值之间有一个）。
3. 计算每个潜在分裂的质量得分（例如，基于正则化目标函数的增益）。
4. 对*所有*特征重复此过程。

在每个节点上对所有实例和所有特征进行这种穷举搜索会变得计算量巨大。其成本大致与非缺失条目数量成比例，在密集情况下，每次分裂的成本可近似为$O(N \times M)$。尽管存在优化措施（如预排序和缓存，或使用直方图的近似算法），但扫描大量数据或特征值的基本需求依然存在。

这种计算成本主要体现在两个方面：

- **训练时间：** 对于拥有数百万或数十亿行，或数万个特征的数据集，为每棵树重复遍历数据点和特征会显著增加总训练时间。
- **内存占用：** 存储中间结果，例如每个实例的排序特征值或梯度统计信息，可能需要大量内存，甚至可能超出单台机器的可用资源。XGBoost的近似算法可以减少这一点，但构建和存储直方图仍然会带来内存开销。

这些局限性在现代机器学习 (machine learning)中常见的场景中变得尤为明显：网络规模数据集、高维基因组数据，或涉及大量工程化或稀疏特征的问题。对一种梯度提升算法的需求，这种算法既能保持高准确度，又能大幅提高训练速度和减少内存使用，促成了LightGBM的开发。

LightGBM从一开始就以高效率为首要目标进行设计。它引入了几种新颖技术，专门用于减轻在大数据集上训练时产生的计算和内存成本。这些技术包括：

- **基于梯度的单边采样（GOSS）：** 减少分裂计算中需要考虑的数据实例数量。
- **互斥特征捆绑（EFB）：** 通过捆绑稀疏的、互斥的特征来减少需要扫描的有效特征数量。
- **基于直方图的算法：** 使用离散化的特征值（直方图）来加速分裂点查找并减少内存使用，这类似于XGBoost的近似算法，但进行了进一步的优化。
- **叶子生长策略：** 一种不同的树构建策略，通常收敛速度更快，但需要注意控制复杂度。

理解以往算法面临的这些具体计算障碍，为理解LightGBM中实现的设计选择和优化提供了背景，我们将在本章中详细审视这些内容。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost的奠基性论文，详细阐述了其精确贪婪算法、正则化和系统优化。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154
  介绍了LightGBM及其创新，如GOSS、EFB和优化的直方图算法，旨在解决XGBoost的局限性。
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/) — Microsoft and LightGBM Contributors (2024)
  官方文档，提供了LightGBM算法、参数和用法的详细说明，包括GOSS、EFB和基于直方图的学习。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/) — XGBoost Contributors (2024)
  官方文档，涵盖了XGBoost的算法，包括精确和近似贪婪算法及其系统优化。
