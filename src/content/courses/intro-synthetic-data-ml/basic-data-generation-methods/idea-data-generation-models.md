---
course: "intro-synthetic-data-ml"
chapter: "basic-data-generation-methods"
lesson: "idea-data-generation-models"
sourceId: 5642
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-2-basic-data-generation-methods/idea-data-generation-models"
title: "数据生成模型的构想"
description: "初步认识使用模型或规则来创建新数据点的方式。"
order: 1
plots: []
sourceHash: "7748cf8f9399a4c14dc06bd992b8c8019725af4de477e124766033ad97fe6d08"
sourceCorrections: []
---

我们已经弄清了合成数据*为什么*如此有益——它可以填补空白、保护隐私并扩充有限的数据集。现在，让我们开始了解*如何*实际创建它。我们暂时不会直接涉足复杂的算法。相反，我们将侧重于许多生成技术的核心构思：使用明确的步骤或“模型”来产生人工数据点。

数据生成模型不必被看作是复杂的机器学习 (machine learning)模型（比如我们之后可能*用*合成数据训练的那种），而更像一份食谱或一套指令。这份食谱规定了如何构建新的人工数据点。目标是依照这些指令创建数据，使其与我们实际希望或需要的数据类型具有重要的共同特点，尽管它并非采集自真实情况。

从根本上说，数据生成模型提供了一种机制，能够根据一些指定的输入或规则系统地产生输出（即我们的合成数据）。对于本章涉及的简单方法，这些“模型”通常分为两大类：

1. **统计描述：** 我们可以分析现有真实数据（如果可用），或定义所需属性并用统计数据来描述它们。例如，我们可能希望生成与真实客户年龄模式相似的合成客户年龄数据。我们可能会观察到，真实客户年龄常常围绕一个平均值集中，并具有一定的离散度。那么，我们的“模型”就成了由该平均值（$\mu$）和离散度（$\sigma$，标准差）定义的统计分布（比如您可能记得的正态分布，常被称为钟形曲线）。生成数据意味着*从*这个定义的分布中抽取随机值。该分布本身就是指导创建合理年龄值的模型。
2. **明确规则：** 有时，我们知道数据必须遵循的特定约束或逻辑。例如，在一个关于在线订单的数据集中，规则可能是“如果`country`（国家）列是‘Canada’（加拿大），则`currency`（货币）列必须是‘CAD’”。或者，“用户`age`（年龄）必须始终是18岁或更大。”一个基于规则的系统使用这些预设条件来生成严格遵守此逻辑的数据点。在这种情况下，这套规则*就是*生成模型。

考虑这个简易流程：

> 数据生成的一种简明视图：一个明确定义的模型或一套规则引导一个过程，从而产生合成数据。

无论我们使用统计属性还是明确规则，其核心构思都是一样的：我们需要一个蓝图来引导人工数据的创建。这个“模型”或步骤是我们从需要合成数据到实际产生数据的工具。在接下来的章节中，我们将了解如何使用统计分布和简易的基于规则的方法来实现这些初步构思，从而生成初级数值和分类数据。

## 参考资料

- [Synthetic Data for Machine Learning: Principles and Practice](https://link.springer.com/book/10.1007/978-3-030-74697-3) — Maksim Khakimov, Altynbek Seitov, Aziza Alimzhanova, and Zhamilya Baizakova (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-74697-3](https://doi.org/10.1007/978-3-030-74697-3)
  对合成数据进行了基础介绍，定义了其目的并概述了包括统计方法在内的基本生成原理。
- [A Survey on Synthetic Data Generation](https://link.springer.com/article/10.1007/s42493-022-00078-0) — Jingge Li, Xiangyu Han, Yaliang Li, Chengyu Song, Jun Ma (2022)
  Journal: Journal of Big Data Analytics in Transportation; Publisher: Springer; Volume: 4; Pages: 173-199; DOI: [10.1007/s42493-022-00078-0](https://doi.org/10.1007/s42493-022-00078-0)
  全面概述了合成数据生成方法，涵盖从基本统计模型到更复杂的机器学习方法。
- [Monte Carlo Statistical Methods](https://doi.org/10.1007/978-0-387-21239-5) — Christian P. Robert, George Casella (2004)
  Publisher: Springer; DOI: [10.1007/978-0-387-21239-5](https://doi.org/10.1007/978-0-387-21239-5)
  经典著作，详细介绍了从各种统计分布生成随机变量的方法，是合成数据“统计描述”方法的基石。（第2版）
- [Privacy-Preserving Data Publishing](https://link.springer.com/book/10.1007/978-1-4419-5867-0) — Raymond Wong, Min Wu, and Chengqi Zhang (2011)
  Publisher: Springer; DOI: [10.1007/978-1-4419-5867-0](https://doi.org/10.1007/978-1-4419-5867-0)
  讨论了在保护隐私的同时发布数据的技术，包括基于统计属性生成合成或扰动数据的方法。
