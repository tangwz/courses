---
course: "building-ml-recommendation-system"
chapter: "constructing-a-hybrid-recommendation-system"
lesson: "blending-content-and-collaborative-signals"
sourceId: 7490
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-6-constructing-a-hybrid-recommendation-system/blending-content-and-collaborative-signals"
title: "构建结合内容与协同信号的系统"
description: "一种结合物品属性和用户行为进行推荐的系统设计方案。"
order: 6
plots: []
sourceHash: "f0b48a6663904710cd7bfd31b68a8f3ec3577065010e84c1466ac2212bb7e8f4"
sourceCorrections: []
---

推荐系统通常使用两种主要的推荐方式，每种方式都有其自身的优缺点。基于内容的过滤在处理丰富的物品元数据方面表现出色，但可能会产生过滤气泡。协同过滤可以发现新颖的物品，但在交互数据稀疏时表现不佳。实际生产中最有效的系统很少只选其一；相反，它们旨在将两者结合起来，形成一个更强大的单一系统。

本节概述了一个能够巧妙结合基于内容和协同过滤模型信号的系统架构。目标是创建一个更准确、能应对冷启动场景，并能提供多样化且相关建议的推荐器。

### 系统架构概览

混合系统的核心是并行运行多个推荐算法，然后整合它们的输出。一个被称为“混合引擎”的中心组件负责这种综合处理。它接收来自每个底层模型的预测或排名列表，并应用特定的逻辑来生成最终向用户展示的推荐列表。

下图展示了此类系统中的数据和预测流向。

> 混合推荐系统的架构概览，展示了从数据源到最终整合推荐的流程。

让我们分析一下该系统的运作方式：

1. **用户请求**：当针对特定用户发起推荐请求时，流程开始。
2. **并行处理**：请求同时传递给协同过滤和基于内容的模型。
   - **协同过滤模型**使用用户-物品交互矩阵来寻找口味相似的用户，或基于隐因子预测评分（如 SVD）。它根据群体行为生成推荐列表。
   - **基于内容的模型**使用物品元数据（如类别、描述）和用户的历史交互来构建用户画像。然后它会寻找与该画像属性匹配的物品。
3. **混合引擎**：这是系统的决策中心。它接收来自两个模型的输出，这些输出可以是预测评分或物品的排名列表。其任务是应用所选策略来合并这些输入。我们将查看该引擎的两种主要策略。

### 策略 1：加权评分组合

混合引擎最直接的方法是使用加权平均值。为了使其生效，两个模型必须为每个候选物品输出一个数值预测评分，且最好归一化 (normalization)到统一标准（例如 0 到 1）。然后，引擎使用线性组合计算最终的混合评分：


$$
\text{评分}_{混合} = \alpha \cdot \text{评分}_{内容} + (1 - \alpha) \cdot \text{评分}_{协同}
$$


在此公式中，$\alpha$ 是介于 0 和 1 之间的超参数 (parameter) (hyperparameter)，用于控制每个模型的影响力。

- 如果 $\alpha = 1$，系统就变成了纯粹的基于内容的推荐器。
- 如果 $\alpha = 0$，它就变成了纯粹的协同推荐器。
- 像 $\alpha = 0.5$ 这样的值会给两者分配相等的权重 (weight)。

$\alpha$ 的最佳值通常通过测试不同的数值并测量其对 NDCG 或 Precision@k 等离线评估指标的影响，由实验来确定。

### 策略 2：动态切换逻辑

更先进的混合引擎可以使用切换策略。它不总是一味混合，而是应用规则来决定在特定情况下信任哪个模型的输出。这对于处理冷启动问题特别有效。

引擎内部的逻辑可能如下所示：

- **针对新物品**：如果一个物品有元数据但交互为零，其协同过滤评分将不存在或不可靠。在这种情况下，可以将引擎配置为完全依赖基于内容的评分。这使得新物品可以立即被推荐。
- **针对新用户**：如果用户没有交互历史，协同模型无法运作。引擎将切换到仅使用基于内容的模型，或者根据用户在注册期间提供的初始偏好来推荐热门物品。
- **针对已有数据的用户和物品**：如果物品有足够的交互且用户有丰富的历史记录，引擎可以使用加权公式得出的混合评分，因为此时两个信号都很强。

这种基于规则的方法使系统更具适应性。它会针对任何给定的用户-物品对，默认使用最适合当前数据量的模型。

通过设计一个结合这些信号的系统，我们可以创建一个整体性能优于各部分之和的推荐器。它可以通过协同过滤挖掘出新颖的物品，同时确保每个物品（无论新旧）都有机会通过其内容特征被推荐。结果是获得更全面且可靠的推荐体验。在下一节中，我们将实现一个加权混合系统，以观察这些原理的实际应用。

## 参考资料

- [Recommender Systems: The Textbook](https://link.springer.com/book/10.1007/978-3-319-29659-3) — Charu C. Aggarwal (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29659-3](https://doi.org/10.1007/978-3-319-29659-3)
  一本综合性的教科书，涵盖了各种推荐系统范式，包括对基于内容、协同过滤和混合过滤技术的详细解释。
- [Recommender Systems Handbook (2nd ed.)](https://link.springer.com/book/10.1007/978-1-4899-7637-6) — Francesco Ricci, Lior Rokach, Bracha Shapira (2015)
  Publisher: Springer; Pages: 173-207 (Chapter 6: Hybrid Recommender Systems); DOI: [10.1007/978-1-4899-7637-6](https://doi.org/10.1007/978-1-4899-7637-6)
  一本权威手册，其中有一章专门介绍混合推荐系统，深入解释了不同的架构方法和集成策略。
- [Hybrid Recommender Systems: A Survey](https://dl.acm.org/doi/10.1145/3291176.3291262) — S. P. Manousos, K. Katsifarakis, M. Georgiopoulos, Y. Manolopoulos (2018)
  Journal: Proceedings of the 22nd Pan-Hellenic Conference on Informatics (PCI 2018); Publisher: ACM; Pages: 104-109; DOI: [10.1145/3291176.3291262](https://doi.org/10.1145/3291176.3291262)
  一篇近期综述论文，分类并比较了各种混合推荐系统方法，包括模型组合策略和解决冷启动等挑战。
