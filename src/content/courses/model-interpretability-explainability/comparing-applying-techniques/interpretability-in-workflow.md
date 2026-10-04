---
course: "model-interpretability-explainability"
chapter: "comparing-applying-techniques"
lesson: "interpretability-in-workflow"
sourceId: 4101
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-4-comparing-applying-techniques/interpretability-in-workflow"
title: "将可解释性融入机器学习工作流程"
description: "讨论如何将模型可解释性步骤融入典型的机器学习项目生命周期。"
order: 6
plots: []
sourceHash: "7cc24227f2e76c47d406d2072b7974e6914b24946523783bb5bb7959065aad89"
sourceCorrections: []
---

模型可解释性并非仅是学术研究，也非模型构建后的事后附加。为获取最大效用，LIME和SHAP等解释技术应成为机器学习 (machine learning)开发过程中的标准组成部分，并贯穿于整个过程。将可解释性视为核心环节，而非可选附加项，有助于从一开始就构建出更好、更可靠、更值得信赖的模型。

思考一下机器学习项目的典型生命周期。解释型方法适用于哪些地方？几乎无处不在。

> 将可解释性方法结合到机器学习工作流程的各个阶段。

我们来细分这些结合点：

### 模型开发与调试期间

这或许是开始使用LIME和SHAP最直观之处。当您的模型对验证样本做出出乎意料的预测时，不要只查看损失函数 (loss function)。生成一个局部解释：

- **识别有问题的预测：** 对模型失败或行为异常的特定实例，使用LIME或SHAP力图。哪些特征将预测推向错误方向？模型是否依赖于噪声或数据中的无关干扰？
- **理解特征重要性：** 初步训练后，生成SHAP汇总图。全局特征重要性是否与您的专业知识一致？如果模型过度依赖您认为不重要的特征，这值得检查。或许存在数据泄露，或者模型习得了您不期望的关联。
- **比较候选模型：** 如果您在评估多种模型架构（例如，随机森林模型与梯度提升机），请比较它们的解释。它们是否以相似方式依赖特征？一个模型可能实现相似准确度，但依赖更直观的特征，使其更受青睐。

在此处结合解释型方法，可作为强大的调试工具，有助于理解模型内部的运作方式。

### 模型验证期间

在最终确认模型之前，将可解释性技术作为验证套件的一部分来使用：

- **验证模型逻辑：** 对一组有代表性的验证数据点，特别是那些接近决策边界的，生成解释。模型的推理 (inference)是否合理？特征如何影响跨不同数据分段的预测，是否存在模式？
- **检查公平性和偏差：** 分析不同人群或敏感属性（如果适用且可用）的解释。模型是否对特定群体依赖不同特征或赋予不同重要性？SHAP在此处对于比较不同群体间的平均影响特别有用。
- **建立信任：** 全局解释（如SHAP汇总图）提供了模型行为的整体情况，提升了模型已习得有意义模式而非利用虚假关联的信任度。

这一步骤确保模型行为符合预期并符合要求。

### 利益相关者沟通与报告

在与非技术性利益相关者、客户或监管机构沟通模型行为时，可解释性必不可少：

- **解释决策：** 使用简单的局部解释（例如简化的LIME输出或从SHAP值提取的说明）来解释为何做出特定预测（例如，为何贷款申请被拒绝）。
- **记录模型行为：** 在模型文档中包含全局重要性图和解释示例。这提供了透明度，并且通常是受监管行业的要求。
- **建立信任：** 证明您理解模型为何运作，而不仅仅是知道它能运作，能培养用户和决策者间的信任。

### 部署后监控

模型行为可能因数据漂移或底层模式变化而随时间改变。可解释性在监控中发挥作用：

- **解释异常实时预测：** 如果监控系统标记 (token)异常预测，生成解释以诊断原因。
- **检测模式漂移：** 定期抽样实时预测并生成SHAP汇总图。将这些与验证期间生成的基线图进行比较。特征重要性的显著变化可能指示漂移，表明模型可能需要重新训练。
- **审计：** 为了合规目的，能够为已部署模型做出的过往预测生成解释可能很有必要。

### 实际考量

结合这些技术需要一些规划：

- **计算成本：** 生成解释，特别是SHAP值（尤其是KernelSHAP），可能计算密集。请确定何处成本合理。例如，完整的SHAP分析可在验证期间进行，而更快的LIME解释则用于开发期间的抽查或实时解释（如果性能允许）。TreeSHAP快得多，但仅适用于基于树的模型。
- **自动化：** 将解释生成集成到您的MLOps流程中。在每次训练运行后自动生成SHAP汇总图，或设置触发器以解释生产中标记 (token)的预测。
- **工具：** 在您的Python环境中，使用`shap`和`lime`等库。将其输出（图表、数值）集成到您偏好的实验跟踪或可视化工具（如MLflow、Weights & Biases，或自定义仪表板）中。

通过审慎嵌入 (embedding)LIME和SHAP到您的工作流程中，您将从构建黑盒 (black box)转向开发透明、可调试、最终更具价值的机器学习 (machine learning)系统。这使得可解释性从被动的分析步骤转变为开发文化中主动的一部分。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774; DOI: [10.5555/3295222.3295326](https://doi.org/10.5555/3295222.3295326)
  介绍了SHAP（Shapley Additive exPlanations）值，用于对单个预测进行一致且局部准确的解释。
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://api.semanticscholar.org/CorpusID:13029170) — Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME（Local Interpretable Model-agnostic Explanations），这是一种模型无关的方法，通过近似局部模型行为来解释单个预测。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本全面的开源书籍，涵盖了包括LIME和SHAP在内的各种可解释性方法及其在ML生命周期中的实际应用。
