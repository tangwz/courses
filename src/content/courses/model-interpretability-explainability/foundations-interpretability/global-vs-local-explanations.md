---
course: "model-interpretability-explainability"
chapter: "foundations-interpretability"
lesson: "global-vs-local-explanations"
sourceId: 4053
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-1-foundations-interpretability/global-vs-local-explanations"
title: "解释的范围：整体与个体"
description: "了解整体解释（模型整体行为）与局部解释（单个预测）之间的差异。"
order: 4
plots: []
sourceHash: "34c6b698369c224dcd8a08b904c7122c9b4e5efd2e58095ea5babc8fd6033b5a"
sourceCorrections: []
---

在尝试理解机器学习 (machine learning)模型时，您所需的理解类型取决于您提出的问题。您是想了解模型从所有可能输入中学到的普遍逻辑，还是专注于了解模型为何对特定实例做出具体判断？这些不同的需求引出了两种主要的解释层面：整体和局部。

### 整体解释：把握整体情况

整体解释旨在描述训练模型的整体行为。它们旨在回答诸如以下问题：

- 模型在所有预测中认为哪些特征最重要？
- 模型通常如何将特定特征的变化与输出的变化关联起来？
- 模型学到了哪些特征之间的关联？
- 模型在数据的不同部分表现是否一致？

可以将整体解释理解为尝试理解模型学到的普遍策略或规则。例如，如果您有一个预测房价的模型，整体解释可能显示 `square_footage` 和 `number_of_bedrooms` 始终是最具影响力的因素，并且通常，增加 `square_footage` 会导致更高的预测价格。

提供整体解释的方法通常涉及分析模型结构（如果可访问，例如在线性模型或决策树中）或总结特征在许多数据点上的影响。整体特征重要性分数（为整个模型的每个特征分配一个单一的重要性值）是一种常见的整体解释类型。偏依赖图（PDPs）或累积局部效应（ALE）图（显示作为一两个特征函数的平均预测结果）也属于这一类别。

整体理解对于以下方面很重要：

- **模型验证：** 确保模型依赖于合理的特征和关联。
- **高层理解：** 向利益相关者传达预测的主要驱动因素。
- **特征工程：** 识别潜在不相关或高影响力的特征。
- **科学发现：** 了解从数据中学到的普遍模式。

### 局部解释：聚焦单个预测

局部解释侧重于阐明模型为何对*单个*输入实例做出了具体的预测。它们回答诸如以下问题：

- 为什么这位特定客户的贷款申请被拒绝了？
- 哪些特征对将此特定图像分类为猫的贡献最大？
- 对这位患者而言，哪些因素导致模型预测其患病风险高？
- 如果我更改此输入的特定特征值，预测会如何变化？

再次考虑房价预测模型。尽管 `square_footage` 可能在整体上很重要，但对*特定*房屋的局部解释可能显示其高价主要由其理想的 `location` 和近期 `renovation_status` 驱动，即使其 `square_footage` 是平均水平。反之，另一栋房屋的低预测价格可能由其糟糕的 `condition` 和低的 `neighborhood_rating` 解释，即使它有大的 `square_footage`。

局部解释通常通过近似复杂模型在正在解释的特定实例附近的行为来工作。LIME（局部可解释模型无关解释）和 SHAP（SHapley 加性解释）值等方法主要旨在提供这些实例层面的理解。它们通常为每个特征*针对该特定预测*分配一个重要性或贡献分数。

局部理解对于以下方面很重要：

- **建立信任：** 向用户或受影响的个体解释具体结果。
- **调试：** 识别模型为何对特定案例做出意外或不正确的预测。
- **可操作性：** 了解哪些因素可以改变以影响模型对特定实例的输出（例如，客户可以改进什么以获得贷款批准）。
- **公平性审计：** 审查对敏感群体个体的预测是否基于适当的因素。

> 预测模型的整体解释与局部解释之间的差异。整体解释基于整体数据分析模型的普遍行为，而局部解释则侧重于单个特定输入的预测背后的原因。

### 互补的视角

整体解释和局部解释并非相互排斥；它们提供了互补的模型视角。良好的整体理解为解释局部解释提供了背景。了解 `square_footage` 普遍重要有助于理解它为何可能出现（或出人意料地不出现）在局部解释中。反之，审查许多局部解释有时有助于形成对模型整体行为的直观感受，尽管在没有专门整体方法的情况下，系统地进行可能具有挑战性。

SHAP 等方法尤其值得关注，因为该框架同时支持局部解释（针对单个预测的每个特征的独立 SHAP 值）和整体解释（例如，通过聚合许多预测的 SHAP 值来获得特征重要性或创建汇总图）。LIME，根据其设计，主要侧重于局部解释。

理解这些层面的差异非常重要。在进行模型解释时，始终考虑您是需要了解模型的整体趋势，还是特定结果背后的原因。这会指导您选择解释方法以及如何分析结果，这些话题我们将在后续章节中详细阐述。

## 参考资料

- [“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD); Publisher: ACM; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME，一种广泛用于局部模型无关解释的技术。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774
  介绍了SHAP，一种统一了多种解释技术并提供局部和聚合全局解释的方法。
- [Interpretable Machine Learning. A Guide for Making Black Box Models Explainable.](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2022)
  Publisher: Lulu.com
  一本全面的开源书籍，涵盖了多种模型可解释性方法，包括PDP、ALE等全局技术以及LIME和SHAP等局部技术。
- [Visualizing the Effects of Predictor Variables in Black Box Models](https://doi.org/10.1111/rssb.12377) — Daniel W. Apley, Jingyu Zhu (2020)
  Journal: Journal of the Royal Statistical Society Series B: Statistical Methodology; Publisher: Wiley; Volume: 82; Pages: 1059-1086; DOI: [10.1111/rssb.12377](https://doi.org/10.1111/rssb.12377)
  介绍了累积局部效应（ALE）图，一种用于全局特征效应可视化的有效且无偏的方法。
