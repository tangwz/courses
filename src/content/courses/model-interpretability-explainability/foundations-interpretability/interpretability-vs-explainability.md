---
course: "model-interpretability-explainability"
chapter: "foundations-interpretability"
lesson: "interpretability-vs-explainability"
sourceId: 4049
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-1-foundations-interpretability/interpretability-vs-explainability"
title: "可解释性与解释性"
description: "阐明模型可解释性与解释性之间的区别和关系。"
order: 2
plots: []
sourceHash: "f0a714896aae8513faa1275b8fcf6fbad96b9f7f0b76b72fdc28aaae81a3e148"
sourceCorrections: []
---

尽管“可解释性”和“解释性”这两个术语在讨论理解机器学习 (machine learning)模型时常被互换使用，但它们拥有不同的含义。当我们研究查看“黑箱”内部的方法时，认识到这种区别会有所助益。

### 可解释性：理解其运作机制

可解释性指的是人类能够持续预测模型输出或理解其内部机制的程度。可将其视为模型本身的固有透明度。您能多容易地掌握模型将输入特征($X$)转化为预测($y$)的过程？

模型固有的可解释性差异很大：

- **高可解释性：** 像线性回归这样的简单模型具有高度可解释性。每个特征与输出之间的关系由一个系数明确定义。您可以直接说：“在其他条件不变的情况下，特征A增加一个单位，预测会改变B个单位。” 类似地，小型决策树也是可解释的；您可以追踪导致预测的特定分割路径。
- **低可解释性：** 像深度神经网络 (neural network)、梯度提升机（GBM）或大型随机森林等复杂模型通常可解释性较低。它们涉及数千甚至数百万参数 (parameter)间复杂的关联。对于人类观察者来说，要精确理解特定输入如何通过网络层或树的集成来得到最终预测是极其困难的。

因此，可解释性根本在于模型本身的*结构和参数*是否可理解。

### 解释性：说明决策的依据

解释性侧重于为模型的输出提供人类可理解的原因，无论模型内部多么复杂。它关乎在模型做出预测或习得普遍行为*之后*提取见解或依据。解释性通常依赖辅助技术，特别是对于缺乏固有可解释性的模型。

解释性的目标是回答以下问题：

- “为什么模型对*这个特定数据点*预测了*这个特定结果*？”（局部解释）
- “哪些特征通常对模型的整体预测影响最大？”（全局解释）
- “当我改变某个特定特征的值时，模型的预测如何变化？”

像LIME和SHAP这样的技术，我们将在稍后详细介绍，它们主要是实现*解释性*的工具。它们旨在为可能是复杂、不可解释的模型提供对其行为的见解。

### 区别

以下是一种阐述区别的方式：

- **可解释性：** 您能本质上理解*模型如何运作*吗？（侧重于模型的机制）
- **解释性：** 即使模型的内部机制复杂，您能为*模型为何做出特定决策或以某种方式表现*提供一个可理解的原因吗？（侧重于模型的输出和依据）

想象一位技艺精湛的厨师（即模型）。

- **可解释性** 就像理解厨师制作一道特定菜肴的详细食谱和烹饪技巧。对于一道简单的菜肴（简单模型），这可能很容易。对于一道含有秘密配料和技巧的复杂菜肴（复杂模型），这可能几乎不可能。
- **解释性** 就像厨师告诉您，即使他们不泄露完整的复杂食谱，某些配料（特征）为何对您刚吃过的菜肴的最终味道（预测）很重要。

尽管固有的可解释模型通常也具备解释性，但模型无需完全可解释即可具备解释性。我们通常正是*因为*模型缺乏固有的可解释性，才应用解释性技术。本课程的主要侧重点是那些能为复杂、强大的机器学习 (machine learning)模型提供解释性的技术。

## 参考资料

- [Towards a Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608) — Finale Doshi-Velez, Been Kim (2017)
  Journal: NeurIPS Workshop on Interpretable ML; DOI: [10.48550/arXiv.1702.08608](https://doi.org/10.48550/arXiv.1702.08608)
  界定并区分了可解释性和可解释性，提出了评估可解释机器学习的框架。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本广泛引用的书籍，涵盖了可解释机器学习的定义、方法和应用。
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME，这是一种广泛使用的模型无关技术，用于黑盒模型预测的局部解释。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632c7cd8e8311a2f6027-Paper.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774
  提出了SHAP（SHapley Additive exPlanations），一个使用Shapley值来解释预测的统一框架。
