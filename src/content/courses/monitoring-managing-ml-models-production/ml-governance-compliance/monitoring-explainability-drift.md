---
course: "monitoring-managing-ml-models-production"
chapter: "ml-governance-compliance"
lesson: "monitoring-explainability-drift"
sourceId: 3323
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-6-ml-governance-compliance/monitoring-explainability-drift"
title: "随时间监控可解释性和可理解性"
description: "随时间跟踪模型解释的变化，以发现模型行为或公平性方面的潜在偏移。"
order: 3
plots: ["plots/3323-0.json"]
sourceHash: "e399631740981aa031b1584d11ba52dd10a424b57bb589dde8a606fa49bdf4fe"
sourceCorrections: []
---

虽然跟踪诸如准确率或AUC等标准性能指标能让您了解模型*实现*了什么，但随时间监控模型的可解释性和可理解性则有助于理解模型*如何*实现这些结果，以及其推理 (inference)是否保持稳定和可接受。这不仅仅是学术活动；它是模型治理、风险管理以及确保符合可能要求了解自动化决策过程的法规的核心组成部分。

部署在生产环境中的模型会受到各种形式的漂移影响。数据分布会变化，特征与目标变量之间的潜在关系（概念漂移）会演变。这些变化不一定会立即导致整体性能指标的灾难性下降，但它们会不易察觉地改变模型的内部逻辑。之前不重要的特征可能突然变得主要，或者模型可能开始依赖由新数据模式引入的潜在虚假关联。监控解释有助于发现这些潜在问题变化，以避免它们对结果造成重大影响或导致公平性违规。

### 应监控可解释性的哪些方面？

有效监控包括跟踪模型解释的不同方面：

1. **全局特征重要性：** 跟踪模型在所有预测中赋予每个特征的整体重要性。诸如置换重要性或聚合SHAP值（例如，平均绝对SHAP）等方法，可提供模型对不同输入的总体依赖情况。特征重要性排名或数值随时间发生显著变化，是模型推理 (inference)发生偏移的有力标志。
2. **局部解释稳定性：** 分析单个预测或特定数据集群的解释。例如，高风险预测背后的原因是否随时间保持一致？针对受保护人口群体的解释是否与其它群体显著不同？监控局部解释，通常使用LIME或单个SHAP值等方法，提供详细的了解。
3. **解释分布漂移：** 除了查看平均重要性，还要监控特定特征在不同预测批次中的解释值（例如SHAP值）的统计分布。漂移检测方法，类似于用于输入数据漂移的方法（在第2章讨论），也可以应用于此处。例如，使用Kolmogorov-Smirnov检验或总体稳定性指标（PSI）比较上周预测与本周预测中`feature_X`的SHAP值分布。
4. **分段解释分析：** 评估模型推理在预定义数据切片或分段之间是否有显著差异。这对于公平性评估尤其重要。分别跟踪不同人口群体、地理区域或客户类型的平均特征重要性或解释分布，可以显示模型是否对不同子群体采用不同逻辑，这可能表明存在偏见或特定分段的性能问题。

### 监控解释偏移的方法

实施解释监控需要将解释生成和分析集成到您的MLOps流程中。

- **定期计算和记录：** 定期（例如，每天、每小时）为生产预测的代表性样本生成解释（例如SHAP值）。将这些解释或从中导出的聚合指标（如每个特征的平均绝对SHAP）存储在适合时间序列分析的系统中，例如专用指标存储或与性能指标并列的时间序列数据库。
- **解释指标的统计监控：** 将统计过程控制（SPC）或漂移检测算法应用于解释指标的时间序列。

  - 对于全局特征重要性（例如，平均`|SHAP|`）：随时间跟踪该值，如果它与基线显著偏离或超出预定义控制限，则发出警报。
  - 对于解释分布：计算当前窗口中特征的SHAP值分布与参考窗口（例如，训练数据或之前的生产窗口）之间的PSI或Wasserstein距离等指标。如果距离超过阈值，则发出警报。

digraph G { rankdir=LR; node [shape=box, style=rounded, fontname="sans-serif", margin=0.2]; edge [fontname="sans-serif"]; Input [label="生产预测（抽样）"]; Explain [label="生成解释\n（例如，SHAP）"]; Store [label="记录解释 / 指标\n（例如，平均|SHAP|，分布）"]; Monitor [label="分析偏移\n（例如，PSI，SPC，距离指标）"]; Alert [label="预警系统", shape=cylinder]; Input -> Explain; Explain -> Store; Store -> Monitor; Monitor -> Alert [label="检测到偏移"]; }
\`\`\`
> 生产环境中监控模型解释的简化流程。

- **可视化仪表板：** 创建可视化图表，随时间跟踪解释指标。绘制重要特征的全局特征重要性排名或分布漂移指标（例如PSI），可以为操作员提供直观的方式观察变化。



![全局特征重要性趋势（平均绝对SHAP）](plots/3323-0.json)



> 随时间跟踪不同特征对模型输出的平均影响。请注意，从第5周左右开始，特征A的重要性增加，特征B的重要性下降，这可能需要进行调查。

### 监控解释的挑战

实施解释监控面临实际挑战：

- **计算开销：** 生成解释，特别是像SHAP这样与模型无关的解释，可能比单纯进行预测的资源消耗大得多。这通常需要采样策略，或者在可能的情况下使用计算效率更高的解释方法，这可能会在解释准确性与速度之间进行权衡。
- **可伸缩性和存储：** 存储每个预测甚至大量样本的详细局部解释（例如，完整的SHAP向量 (vector)）可能会导致庞大的存储需求。可能需要聚合指标，或者将存储重点放在特定分段或异常解释上。
- **定义有意义的偏移：** 确定解释中何为“显著”偏移需要仔细设置阈值和领域知识。统计显著性并不总是等同于实际重要性或明确的问题。除非偏移明确表明存在严重故障或偏见，否则警报应触发调查，而非自动回滚。
- **工具成熟度：** 尽管用于生成解释的库已经发展完善（SHAP，LIME，Captum，InterpretML），但MLOps平台中专门用于*随时间监控*这些解释的集成工具仍是一个发展中的方面。通常，需要定制解决方案，将解释库与现有监控基础设施（时间序列数据库、仪表板工具）结合使用。

将可解释性和可理解性监控整合到您的MLOps实践中，为生产模型提供了更全面的了解和控制。它超越了简单的性能跟踪，主动验证模型的推理 (inference)是否保持一致、公平，并与业务预期和监管要求保持一致。这种主动方法是负责任AI治理和模型长期维护的重要组成部分。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg and Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765–4774
  介绍了Shapley加性解释（SHAP）的基础方法，这是一种通过归因特征重要性来解释模型预测的常用技术。
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了局部可解释模型无关解释（LIME），这是一种以可解释且忠实的方式解释任何分类器个体预测的方法。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready ML Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  为构建和运行生产环境中的机器学习系统提供了全面的指导，包括模型监控、治理和负责任的AI考量。
