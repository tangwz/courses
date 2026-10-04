---
course: "intro-feature-engineering"
chapter: "feature-creation"
lesson: "automated-feature-creation-intro"
sourceId: 1385
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation/automated-feature-creation-intro"
title: "自动化特征生成（引言）"
description: "简要介绍旨在自动化特征生成过程的工具和技术。"
order: 7
plots: []
sourceHash: "7164b0cd517ab533459c40fb86ebe13af5717843aeb9aac713a1df5284e829e2"
sourceCorrections: []
---

手动构建交互项、多项式特征或从日期中提取成分能带来显著改进，但这个过程通常高度依赖于专业知识、直觉和耗时的实验。随着数据集维度的增加，潜在特征组合的数量会变得非常庞大，使得详尽的手动处理不可行。这自然引出了一个问题：我们能否将这个创建过程的一部分自动化？

自动化特征生成指的是旨在系统地从现有特征中生成大量候选特征的算法和框架。您无需决定计算 $f_a \times f_b$ 或 $\log(f_c)$，这些工具会自动应用多种数学运算和转换。

其核心思想是定义一组基本构建块或操作（例如，加法、减法、乘法、除法、绝对值、对数、三角函数），并将它们应用于初始特征。这些新生成的特征随后可以作为后续操作的输入，通过多个步骤可能创建出高度复杂的特征。

例如，给定初始特征 $f_1$ 和 $f_2$，自动化系统可能会生成以下特征：

- $f_1 + f_2$
- $f_1 \times f_2$
- $\text{绝对值}(f_1 - f_2)$
- $\log(f_1)$ （如果 $f_1 > 0$）
- $f_1 / (f_2 + \epsilon)$ （其中 $\epsilon$ 是一个很小的常数，以避免除数为零）
- 生成特征的组合，例如 $\log(f_1 \times f_2)$

此方面中一个广为人知的方法是**深度特征合成（Deep Feature Synthesis, DFS）**，它在Python库`Featuretools`中得到了显著实现。DFS 在处理跨多个表的关系数据时表现出色。它会自动遍历这些表之间的关系，并应用预定义函数（称为“基元”）来聚合和转换数据，从而创建可能跨越数据集结构的有用特征。想象一下自动生成像“客户上次会话前一个月的平均购买金额”这样的特征。

其他技术，例如**遗传编程（Genetic Programming）**，使用进化算法来“演化”数学表达式，这些表达式代表了可能有用的特征，并根据它们对模型预测性能的提升程度来优化这些表达式。

尽管功能强大，自动化特征生成也伴随着一系列考量：

1. **计算成本：** 生成大量特征可能需要密集的计算，并占用大量内存和处理时间。
2. **过拟合 (overfitting)风险：** 创建庞大的特征空间会显著增加找到虚假关联的可能性，这些关联可能很好地拟合训练数据，但无法泛化到未见过的数据。因此，后续严格的特征选择（在第6章中讨论）变得非常必要。
3. **可解释性：** 自动生成的特征，例如 `MAX(transactions.amount) / SQRT(ABS(customer.age - MEAN(transactions.time_since_last)))`，与源自特定专业见解的特征相比，可能更难解释。
4. **特征数量激增：** 生成的特征数量很容易达到数千甚至数百万，这需要高效的管理和选择策略。

自动化特征生成通常不能替代深思熟虑的、专业驱动的特征工程。相反，它是一种补充方法。它可以迅速考察广泛的可能性，可能会发现不那么显而易见的复杂交互或转换。生成的候选特征仍需要仔细评估、选择和验证，以确保它们提供真实的预测价值，并生成性能可靠的、可泛化的模型。本节对这一方面做了简要介绍；专门的库和技术通常需要进一步研究才能在实践中有效使用。

## 参考资料

- [Deep Feature Synthesis: Towards Automating Feature Engineering](https://dl.acm.org/doi/10.1145/2783258.2783300) — Max Kanter, Kalyan Veeramachaneni (2015)
  Journal: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15); Publisher: ACM; Pages: 1227–1236; DOI: [10.1145/2783258.2783300](https://doi.org/10.1145/2783258.2783300)
  介绍了深度特征合成，这是一种从关系数据中自动创建特征的基础算法，也是`Featuretools`的基础。
- [Featuretools Documentation](https://featuretools.alteryx.com/en/stable/) — Alteryx, Inc. (2024)
  Publisher: Alteryx, Inc.
  `Featuretools` Python库的官方文档，为实现深度特征合成提供了实用指导和示例。
- [Genetic Programming for Feature Generation and Selection: A Survey](https://doi.org/10.1016/j.asoc.2020.106312) — Sherif E. D. M. El-Din, E. H. H. Soliman (2020)
  Journal: Applied Soft Computing; Publisher: Elsevier; Volume: 92; Pages: 106312; DOI: [10.1016/j.asoc.2020.106312](https://doi.org/10.1016/j.asoc.2020.106312)
  一篇关于遗传编程在自动创建和选择特征方面的应用的全面综述，与文中提到的进化方法相关。
- [Automated Machine Learning: Methods, Systems, Challenges](https://link.springer.com/book/10.1007/978-3-030-05318-5) — Frank Hutter, Lars Kotthoff, Joaquin Vanschoren (2019)
  Publisher: Springer; DOI: [10.1007/978-3-030-05318-5](https://doi.org/10.1007/978-3-030-05318-5)
  这本权威的合著书籍全面概述了自动化机器学习，其中包含关于自动化特征工程的专门章节。
