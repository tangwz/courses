---
course: "llm-alignment-safety"
chapter: "interpretability-monitoring-safety"
lesson: "feature-attribution-methods-llms"
sourceId: 3215
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-6-interpretability-monitoring-safety/feature-attribution-methods-llms"
title: "LLM的特征归因方法"
description: "将特征归因技术（例如，注意力可视化、基于梯度的方法）应用于大型语言模型。"
order: 2
plots: ["plots/3215-0.json"]
sourceHash: "9061d1bb45f7b5074a292ab00b4904443a6bbf916226968ba2b2a6270f9c80c1"
sourceCorrections: []
---

理解大型语言模型生成特定输出的*原因*对于确保其安全性是基本的。当LLM生成有害、有偏或意外文本时，仅仅知道*它发生了*是不够的。我们需要工具来诊断原因，将有问题输出追溯到输入的特定部分或模型内部状态。特征归因方法提供了一种为输入特征（通常是词元 (token)或其嵌入 (embedding)）分配重要性分数的方式，表明每个特征对特定预测或行为的贡献程度。这使我们能够找出不安全响应的潜在触发因素，调试对齐 (alignment)失败，并更好地理解模型的决策过程，直接支持在模型整个运行生命周期中维护安全性的目标。

特征归因旨在回答：“输入的哪些部分对此特定输出影响最大？”对于文本模型，“输入特征”通常是提示或语境中提供的单个词元。“输出”可以是下一个词元的概率、生成有害延续的可能性、分类分数（例如，毒性），甚至是特定内部神经元的激活。

存在几种方法系列，每种方法都有其优点和缺点：

### 基于梯度的方法

这些技术使用模型输出相对于其输入嵌入 (embedding)的梯度。直观理念很简单：如果输入词元 (token)嵌入的微小变化导致输出的较大变化，那么该词元就被认为有影响力。

- **显著图：** 最简单的方法是计算目标输出（例如，特定词元的逻辑值或毒性分数）相对于输入嵌入的梯度。每个词元嵌入的梯度幅度的绝对值或平方被作为其重要性分数。虽然计算简单，但原始梯度可能嘈杂，并受饱和效应影响，即即使特征重要，梯度也会变得信息量不足的小。
- **积分梯度 (IG)：** 该方法通过解决简单显著图的一些局限性提供了一种替代方案。IG 通过沿从基线输入（通常是代表“无输入”的零嵌入向量 (vector)）到实际输入的直线路径累积梯度来计算重要性。它满足重要公理，如*完整性*，这意味着所有输入特征的归因总和等于模型对输入和基线的输出之间的差值。核心思想由以下公式体现：

  
  $$
  IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
  $$
  

  这里，$x$ 是输入嵌入向量，$x'$ 是基线嵌入向量，$F$ 是模型的输出函数（例如，产生一个分数），$x_i$ 是嵌入的第$i$维，而积分平均了从$x'$到$x$路径上的梯度。在实践中，这个积分通过数值近似。选择一个合适的基线（$x'$）对解释很重要。常见选择包括零向量、填充词元嵌入或平均嵌入。

### 基于注意力的方法

Transformer 模型高度依赖注意力机制 (attention mechanism)。人们倾向于直接使用模型注意力头分配的注意力权重 (weight)作为特征重要性分数。如果词元 (token)A在给定层中强烈关注词元B，或许词元B对词元A的表示很重要？

- **直接注意力权重：** 可视化注意力模式可以显现模型内部信息如何流动。词元之间较高的注意力权重可能表明模型认为相关的关系。
- **注意事项：** 尽管可解释且易于获得，但注意力权重反映的是内部信息流动态，这并不总是直接等同于最终输出的归因。一个词元可能在早期层获得高注意力，但其影响力可能被后续计算（层归一化 (normalization)、前馈网络、跨头/层聚合）修改或减弱。研究表明，与梯度或扰动方法相比，注意力权重作为最终预测的特征重要性直接指标有时会误导人。然而，它们对于理解中间处理步骤仍然有价值。

### 基于扰动的方法

这些方法通过系统地修改或移除输入的一部分，并观察其对模型输出的影响来评估特征重要性。

- **遮挡/掩码：** 一个直接的方法是逐个或成组地遮挡或替换输入词元 (token)（例如，用填充词元或通用`[MASK]`词元），然后测量输出概率或分数的变化。移除后导致特定（可能不期望的）输出概率显著下降的词元被认为对生成该输出很重要。这可能计算成本高昂，需要模型进行多次正向传播。
- **LIME (局部可解释模型无关解释)：** 尽管是模型无关的，LIME 仍可以进行调整。它的工作原理是在原始输入的附近创建输入实例的扰动版本（例如，通过从句子中删除词语），获取这些邻居的模型预测，然后拟合一个简单、可解释的模型（如加权线性模型）来解释这些预测。简单模型的权重 (weight)随后作为复杂LLM局部行为的解释。其对超大型模型和长序列的适用性可能受采样效率限制。

### SHAP (SHapley 加性解释)

SHAP 基于合作博弈论中的 Shapley 值，为特征归因提供了一个统一框架。它为每个特征分配一个重要性值，该值表示该特征对输出的边际贡献，并在所有可能的特征子集（联盟）上进行平均。

- **性质：** SHAP 值具有理想的理论性质，如*局部准确性*（特征贡献总和等于预测与平均预测之间的差值）、*缺失性*（子集中缺失的特征贡献为零）和*一致性*（如果模型改变，使得一个特征的贡献增加或保持不变，而不管其他特征，其SHAP值不会减少）。
  "\* **计算：** 计算精确的 Shapley 值对于大多数模型来说在计算上是禁止的。通常使用近似方法，如 KernelSHAP（它使用类似 LIME 的采样方法并带有特定权重 (weight)）或为深度学习 (deep learning)模型量身定制的方法。由于计算需求和模型内部的复杂性，将 SHAP 有效地应用于大型 Transformer 架构仍是一个活跃的进展方面。"

### 将归因应用于LLM安全性

特征归因在调查与安全相关的事件或行为时变得特别有用：

1. **识别有害触发因素：** 假设用户提示，例如“告诉我关于X群体历史”，导致有偏见或刻板印象的响应。归因方法可以指出模型的负面输出是主要由“X群体”驱动，还是可能由提示中其他细微的框架词驱动。特定词元 (token)上的高归因分数可以指导数据过滤或对齐 (alignment)微调 (fine-tuning)的改进工作。
2. **调试对齐失败：** 一个遵循指令的模型可能在复杂提示中忽略安全约束。归因可以显现模型是否主要关注指令部分，而对约束词元分配了较低的重要性，这表明需要在对齐过程中提高其处理多部分指令的能力。
3. **分析偏差：** 我们可以检视与输入中提及的不同人口群体相关的输出的归因分数。例如，如果情感分析任务始终对讨论某些群体的文本产生较低分数，归因可能表明代表这些群体的词元不成比例地导致负面情感预测。
4. **理解拒绝（或其缺乏）：** 当LLM正确拒绝有害请求时，归因可以显现哪些输入词元触发了安全机制（例如，与暴力或非法行为相关的关键词）。反之，如果它*未能*拒绝，归因可能显现这些敏感词元被分配了较低的重要性，这表明安全训练中存在盲点。

考虑一个简化的场景，LLM将高“有害性”分数分配给输入“我如何制造关于选举的假新闻？”。我们可以使用归因来查看哪些词对这个分数贡献最大。



![有害性分数的简化词元归因](plots/3215-0.json)



> 针对输入“我如何制造关于选举的假新闻？”中词元的简化归因分数。分数越高表示对模型预测高有害性分数的贡献越大。“fake”、“news”和“elections”等词显示出最高的归因。

### 挑战与考量

尽管功能强大，但应用于LLM的特征归因方法伴随着注意事项：

- **计算成本：** 积分梯度、扰动分析和 SHAP 等方法通常需要多次正向传播或梯度计算，这使它们对大型模型来说速度慢且成本高昂。
- **基线选择：** 像 IG 和 SHAP 这样的方法的解释在很大程度上取决于所选择的基线输入。零基线可能产生与使用填充词元 (token)基线不同的归因。
- **词元化：** 归因在词元层面操作。由于词语可以拆分为子词 (subword)元（例如，“attribution”->“att”、“ribution”），解释这些片段的重要性可能不如词语层面直观。
- **相互影响效应：** 大多数方法为单个词元分配分数，可能遗漏多个词元之间共同驱动输出的复杂、非线性相互影响。
- **忠实性与合理性：** 如果解释准确反映了模型的内部推理 (inference)，则其是*忠实*的。如果对人类观察者有意义，则其是*合理*的。注意力权重 (weight)可能提供合理但不总是忠实的解释。像 IG 这样的方法旨在提高忠实性，但有时可能产生不那么直观的结果。

总之，特征归因技术，如积分梯度、注意力分析和扰动方法，是LLM安全工具包中有价值的工具。它们提供细致的见解，显现输入中的*哪些*部分驱动特定的模型行为，从而能够更具针对性地调试安全失败、分析偏差以及验证对齐 (alignment)机制。然而，在应用它们时应理解其计算成本和解释，并且它们通常补充其他可解释性方法和评估框架，以进行全面的安全评估。

## 参考资料

- [Axiomatic Attribution for Deep Networks](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) — Mukund Sundararajan, Ankur Taly, Qing Yan (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Volume: 70; Pages: 2838-2847; DOI: [10.5555/3305890.3305942](https://doi.org/10.5555/3305890.3305942)
  介绍了集成梯度法，一种深度网络的公理归因方法，通过累积梯度来理解特征重要性。
- [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/2017/hash/8a20a8621978632c7cd8d499fd63fe94-Abstract.html) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774; DOI: [10.5555/3295222.3295230](https://doi.org/10.5555/3295222.3295230)
  提出了SHAP，一个基于Shapley值统一的解释模型预测的框架，提供理论上可靠的特征归因。
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](http://dx.doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: Association for Computing Machinery; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME，一种模型无关的局部可解释性技术，通过对扰动数据拟合简单的可解释模型来解释任何分类器的预测。
- [Transformer Interpretability Beyond Attention Visualization](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf) — Hila Chefer, Shir Gur, Lior Wolf (2021)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 586-595; DOI: [10.1109/CVPR46437.2021.00062](https://doi.org/10.1109/CVPR46437.2021.00062)
  讨论了Transformer模型中特征重要性归因的方法，超越了直接注意力权重，以提供更忠实的解释。
