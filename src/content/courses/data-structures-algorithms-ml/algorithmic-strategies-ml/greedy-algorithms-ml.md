---
course: "data-structures-algorithms-ml"
chapter: "algorithmic-strategies-ml"
lesson: "greedy-algorithms-ml"
sourceId: 5348
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-6-algorithmic-strategies-ml/greedy-algorithms-ml"
title: "优化中的贪心算法"
description: "介绍某些聚类算法或决策树分裂等算法如何做出贪心选择。"
order: 3
plots: []
sourceHash: "782107655f4a7a4e874ec76c534e471d18b37b7b763ea2d34a0f35b22888bd93"
sourceCorrections: []
---

贪心算法是解决问题，尤其是在优化任务中，一种直接且通常直观的办法。其基本规则很简单：在过程的每一步，都做出当时看来最好的选择，而不充分考虑未来的影响或回溯之前的决定。这就像走迷宫时，总是选择当前看来最有希望的路，希望它能通向出口。

这种策略旨在逐步构建一个解决方案，通过做出局部最优的选择，期望这些选择最终能导向一个全局最优，或者至少是一个相当不错的整体解决方案。其吸引力在于简单以及通常的速度。贪心算法通常比动态规划等更复杂的策略更容易设计和实现。

然而，“希望”很重要。做出局部最好的选择并不总是保证最好的最终结果。一个短期内看起来不错的选择，可能会导致后续无法达成绝对最佳解决方案的路径。这是主要权衡：速度和简单性与全局最优的保证。

### 贪心选择性质

对于一个贪心算法而言，要持续找到全局最优解，问题必须表现出有时被称为“贪心选择性质”的特性。这意味着通过从局部最优选择开始，总能达到全局最优解。如果这个性质成立，贪心方法不仅快，而且正确。一个成功的典型例子是使用标准面额（如美元的25美分、10美分、5美分、1美分）找零所需的最少硬币数量：总是选择小于或等于所需剩余金额的最大面额硬币，效果非常好。然而，如果硬币系统不同（例如{1, 3, 4}单位，你需要6单位），贪心选择（4单位）会留下2单位，需要两个1单位硬币（总计3个硬币：4, 1, 1），而最优解是两个3单位硬币（总计2个硬币：3, 3）。

### 机器学习 (machine learning)中的贪心算法

尽管在许多复杂情况下缺乏全局最优保证，贪心策略在机器学习算法中经常出现，通常因为它们提供了高效且实用的启发式方法。

#### 决策树构建

构建决策树（如CART或ID3）的算法是一个典型的例子。在决定如何分裂节点时，算法通常会评估所有可能的分裂（例如，基于不同的特征和阈值），并贪心地选择在该特定节点上产生最高信息增益或最大基尼不纯度降低的分裂。

> 决策树中贪心分裂决策的简化视图。算法选择对“特征 B”进行分裂，因为它提供了最高的即时不纯度降低（0.3 对 0.2），而不考虑沿每条路径的潜在后续分裂的质量。

这种贪心方法不向前看，以判断现在稍微差一点的分裂是否会在以后带来更好的分裂，潜在地导向更小或更准确的整体树。虽然剪枝技术可以在树构建完成后在一定程度上缓解这种情况，但核心构建是贪心的。尽管如此，贪心决策树归纳在实践中非常有效，并为随机森林和梯度提升等强大的集成方法奠定基础。

#### 特征选择

贪心策略常见的另一个方面是特征选择。当面临大量潜在特征时，你可能希望选择一个子集来提高模型性能或降低计算成本。

- **前向选择：** 从空特征集开始。在每一步中，贪心地添加当被添加到当前集合时能为模型性能带来最大提升（例如，准确率的最高提升或误差的降低）的特征。
- **后向消除：** 从所有特征开始。在每一步中，贪心地移除其移除导致模型性能下降最小（或提升最大）的特征。

这些方法在计算上比评估所有可能的特征子集（这通常不可行）成本低得多。然而，它们可能会错过最优子集。例如，前向选择可能永远不会选择两个单独使用时作用不大，但组合使用时非常有效的特征。

### 优点和缺点

**优点：**

- **简单：** 通常比其他方法更容易理解、实现和调试。
- **高效：** 通常比需要穷举搜索或复杂状态管理（如动态规划）的方法更快。它们做出决策并继续前进。
- **有效：** 对于找到绝对最优解成本过高或不可能的复杂问题，能产生良好的近似解。

**缺点：**

- **次优：** 主要缺点。贪心选择可能导致显著劣于全局最优的解决方案。
- **短视：** 决策仅基于局部信息，不预见更广泛的影响。

### 何时考虑贪心方法

贪心算法是算法工具箱中有价值的一部分，特别是在机器学习 (machine learning)优化场景中。它们在以下情况下特别有用：

1. 问题结构保证局部最优能导向全局最优（在复杂机器学习中较不常见）。
2. 计算效率是主要考虑因素，并且“足够好”的解决方案可接受。
3. 作为启发式方法或更大算法中的一个组件（如决策树分裂）。

了解贪心方法有助于你认识到为什么某些算法是这样设计的，并领会计算成本与找到绝对最佳解决方案的保证之间的固有权衡。这是解决机器学习中计算难题的几种强大策略之一。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  涵盖贪婪算法的理论基础、贪婪选择性质及经典应用的重要教材。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://doi.org/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习的全面参考书，详细介绍了决策树构建（CART）和各种特征选择技术中的贪婪算法。
- [Classification and Regression Trees](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone) — Leo Breiman, Jerome H. Friedman, Richard A. Olshen, Charles J. Stone (1984)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/9781315139470](https://doi.org/10.1201/9781315139470)
  引入CART算法的奠基性著作，阐述了树归纳的贪婪自上而下方法。
- [Decision Trees](https://cs229.stanford.edu/notes_archive/cs229-notes-dt.pdf) — R. J. L. Townshend (2018)
  Journal: Stanford CS229 Lecture Notes; Publisher: Stanford University; Volume: 1; Pages: 1-9
  提供决策树的教育性概述，解释了使用信息增益和基尼不纯度等不纯度度量进行贪婪构建的方法。
