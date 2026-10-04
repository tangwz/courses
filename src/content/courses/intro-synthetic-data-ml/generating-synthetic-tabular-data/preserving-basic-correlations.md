---
course: "intro-synthetic-data-ml"
chapter: "generating-synthetic-tabular-data"
lesson: "preserving-basic-correlations"
sourceId: 5666
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-3-generating-synthetic-tabular-data/preserving-basic-correlations"
title: "维护列的基本关联性"
description: "介绍维护列之间关系的挑战和简单方法。"
order: 4
plots: ["plots/5666-0.json"]
sourceHash: "fa85a2cb03767e6740b5a1eeb584fefe8de94a24123f2ba77627aabfffe33f7c"
sourceCorrections: []
---

上一节中，我们探讨了如何独立地为合成表中的每一列生成值。这种方法直接了当：我们查看真实列（如“年龄”）中值的分布，然后生成具有相似模式的新值，而不考虑其他任何列。我们对每一列（“薪水”、“受雇年限”等）都重复此操作。

"这种方法虽然简单，但独立生成列常常忽视了一个重要的数据点：列之间经常相互关联。例如，在关于人员的数据集中，“年龄”和“收入”通常是相关的；年龄较大的人通常收入较高，尽管这不是一个完美的关系。同样，“身高”和“体重”也倾向于一同增长。"

如果我们分别根据“年龄”和“收入”各自的分布独立随机生成它们的值，就会失去这种关联。我们的合成数据集可能包含60岁却拿着入门级薪水，或者20岁却收入很高的记录，其出现频率远比现实中高得多。*单个*列在统计上可能看起来与真实数据相似，但*列之间的关系*却被破坏了。

### 为什么维护关系很重要？

失去这些列间关系或关联，会使合成数据的效用降低，特别是对机器学习 (machine learning)而言。
"\* **真实性：** 合成数据将无法准确反映其应代表的系统。一个年龄和收入无关的数据集与大多数就业市场的情况不符。"

- **模型训练：** 许多机器学习模型通过识别不同特征（列）之间的模式和依赖关系来进行学习。如果合成训练数据中缺失或存在不正确的依赖关系，模型可能学到错误的模式，或未能学到重要的模式。在一个年龄和收入不相关的合成数据上训练出的模型，在使用真实数据根据年龄预测收入时，表现可能不佳。

考虑一个简单的散点图，比较真实身高体重数据与身高体重独立生成的合成数据。



![真实数据 vs. 独立生成的合成数据关联](plots/5666-0.json)



> 散点图比较了真实的身高/体重数据（蓝色，显示总体上升趋势）与身高和体重独立生成的合成数据（红色，没有明显的趋势）。

在上图中，代表真实数据的蓝点显示出一种普遍趋势：较高的人通常体重更重。而代表身高和体重独立生成的数据的红点，则形成了一团随机分布的点。这两个变量之间的关联丢失了。

### 维护基本关系的简单方法

完美捕捉真实数据中存在的复杂关联网络是一项困难的任务，通常需要高级技术。然而，作为入门，我们可以考虑一些维护*某些*基本关联的简单方法：

1. **基于规则的依赖：** 与其独立生成所有列，不如引入简单规则。例如，在生成“收入”值时，我们首先检查该合成记录中已生成的“年龄”。

   - 如果 `年龄 < 30`，则从适合年轻员工的收入分布中生成“收入”。
   - 如果 `年龄 >= 30`，则从适合更有经验员工的收入分布中生成“收入”。
     这是*条件生成*的一种基本形式，其中一列的生成过程依赖于另一列的值。这需要根据观察到的真实数据模式定义规则。
2. **采样相关值：** 另一个想法涉及采样。与其独立地为每一列采样单个值，不如我们可以从原始数据集中一同采样相关的值对或小群组。例如，在创建一条合成记录时，我们可能从一条真实记录中一同采样（“年龄”、“收入”）对，而独立生成“城市”等其他列。这有助于维护采样列之间的特定关系，但需要仔细考虑哪些列是相关的。

### 权衡

在生成合成表格数据时，一个常见挑战是确保相关列之间保持其关联性。一些基本方法可以帮助避免相关列之间最明显的断裂。然而，这些方法通常只捕捉成对关系（如A影响B），并可能错过更复杂的相互作用（如A和B共同影响C）。尽管这些技术比完全独立地生成列数据更进一步，但仍未能完全捕捉到真实数据依赖关系的全部内容。

理解列之间关系的重要性非常高。在生成合成表格数据时，我们需要认识到仅仅匹配单个列的统计数据可能不够。考虑列之间如何相互关联，对于创建更真实、更有用的合成数据集是必要的，这在考虑如何在匿名化数据同时维护其效用时也具有关系。

## 参考资料

- [Synthetic Data Generation for Tabular Data: A Survey](https://dl.acm.org/doi/10.1145/3630248) — Jing Sun, Shuyue Wang, Ze Shi, Guang-Lei Chen (2023)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 56; Pages: 1-39; DOI: [10.1145/3630248](https://doi.org/10.1145/3630248)
  这篇综述回顾了生成合成表格数据的方法，讨论了保留数据分布和列间关系对数据实用性的重要性。
- [Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf) — Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 32; Pages: 7335-7345; DOI: [10.48550/arXiv.1907.00503](https://doi.org/10.48550/arXiv.1907.00503)
  这篇论文介绍了CTGAN，强调了生成高保真合成表格数据以保持复杂、多模态和列间关系的困难。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  本书提供了为机器学习准备数据的实用建议，包括理解特征关系及其对模型性能的影响。
- [Synthetic Data: Generating and Evaluating](https://ieeexplore.ieee.org/document/9371089) — Bas Van Breugel, Marc J. D. van der Weide (2021)
  Journal: IEEE Security & Privacy; Publisher: IEEE; Volume: 19; Pages: 56-62; DOI: [10.1109/MSEC.2021.3059170](https://doi.org/10.1109/MSEC.2021.3059170)
  本文概述了合成数据的生成和评估，强调了统计保真度（特别是变量间关系）对合成数据价值的重要性。
