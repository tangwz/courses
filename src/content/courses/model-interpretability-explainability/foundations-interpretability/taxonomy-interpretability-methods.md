---
course: "model-interpretability-explainability"
chapter: "foundations-interpretability"
lesson: "taxonomy-interpretability-methods"
sourceId: 4051
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-1-foundations-interpretability/taxonomy-interpretability-methods"
title: "可解释性方法分类"
description: "探讨不同类别的可解释性方法：内在与事后，模型特定与模型无关。"
order: 3
plots: []
sourceHash: "beec21c9d2835c83c84c1fc4f88711d71da9b1977e03291f14e63ab26436569a"
sourceCorrections: []
---

当我们面对日益复杂的机器学习 (machine learning)模型时，了解*如何*解释它们与构建它们同样重要。并非所有解释技术都一样。它们在方法、适用性以及提供的见解类型上有所不同。为了有效地理解这个方面，将这些方法按几个重要维度进行分类会很有帮助。这种分类有助于根据您正在使用的模型和需要回答的问题来选择合适的工具。

可解释性方法的主要分类方式是：

1. **内在可解释性与事后可解释性**
2. **模型特定与模型无关**

接下来考察这些维度。

### 内在可解释性与事后可解释性

这种区分侧重于在模型构建过程中何时考虑可解释性。

#### 内在可解释性

有些模型因其结构较简单而被认为是本身就可解释的。它们的内部运作机制相对容易理解和说明，无需额外工具。实例包含：

- **线性模型（例如，线性回归、逻辑回归）：** 分配给每个特征的权重 (weight)（系数）直接表明了该特征对预测的影响，前提是特征已适当缩放。正权重意味着特征值增加会使输出（或预测概率）增加，而负权重则相反。权重的大小表明了影响的程度。
- **决策树：** 从根节点到叶节点的路径代表了基于特征值的一系列明确决策。这条路径为落入该叶节点的任何预测提供了清晰的规则解释。
- **其他基于规则的系统（例如，某些条件下的朴素贝叶斯）：** 这些模型通常依赖于易于理解的条件概率或简单规则。

内在可解释性的优点在于其直接性。解释*即为*模型结构本身。然而，这些较简单的模型可能无法在具有复杂非线性特征关系的数据集上达到最高的预测准确度。模型复杂性（和潜在准确度）与内置可解释性之间通常需要权衡。

#### 事后可解释性

大多数复杂、高性能模型，如深度神经网络 (neural network)、梯度提升机（GBM）或随机森林，通常作为“黑箱”运行。其内部决策过程人类难以直接跟进，甚至不可能。对于这些模型，我们依赖**事后**方法。

这些技术在模型训练*完成之后*应用。它们通过分析已训练模型的输入-输出行为来发挥作用，而不试图剖析其内部结构（尽管有些方法在可用时可能会运用特定方面）。事后方法旨在提供模型行为的近似或概括，无论是全局性的还是针对特定预测。

**LIME和SHAP**是事后可解释性技术的主要实例。它们为复杂模型提供了本不易理解的理解。

> 此图比较说明了内在可解释性直接源于简单模型，而事后方法则在训练复杂模型后应用。

### 模型特定可解释性与模型无关可解释性

这个维度根据方法是否可以应用于任何模型或仅特定类型来分类。

#### 模型特定方法

这些技术专为特定类别的模型设计，通常运用该模型家族特有的内部机制或属性。实例包含：

- **线性模型的系数分析：** 如前所述，直接解释学习到的权重 (weight)。
- **基于树的特征重要性：** 计算特征重要性是基于每个特征在决策树或树集成模型（如随机森林或梯度提升）的所有分裂中，对减少不纯度（如基尼不纯度或熵）的贡献程度。
- **神经网络 (neural network)的激活最大化：** 用于可视化哪些模式能最大程度地激活网络内的特定神经元或层。
- **TreeSHAP：** 尽管是SHAP框架的一部分，但TreeSHAP是一种高效算法，专为基于树的模型设计，并运用了它们的结构。

模型特定方法可以非常高效，并提供针对模型类型的具体理解。然而，它们的主要局限是缺乏通用性。您不能使用为决策树设计的方法来解释支持向量 (vector)机（SVM）或神经网络。这也使得使用这些方法比较不同模型类型的解释变得困难。

#### 模型无关方法

这些方法将模型视为黑箱。它们通过分析输入变化与输出变化之间的关系来工作，而无需访问模型的内部结构（如权重或决策规则）。它们通常只需要访问模型的预测函数（`predict()` 或 `predict_proba()`）。

实例包含：

- **LIME（局部可解释模型无关解释）：** 围绕单个预测构建简单、可解释的局部代理模型。
- **SHAP（Shapley加性解释）：** 使用博弈论中的原理（Shapley值）来公平地将预测结果分配给特征。KernelSHAP是一种模型无关的实现。
- **偏依赖图（PDP）和个体条件期望（ICE）图：** 可视化一个或两个特征对预测结果的平均或个体影响。
- **排列特征重要性：** 通过观察当某个特征的值被随机打乱时，模型性能下降的程度来衡量该特征的重要性。

模型无关方法的主要优点是其灵活性。您可以应用相同的技术来解释和比较不同类型的模型（例如，比较随机森林与神经网络在相同任务上的特征重要性）。这在模型选择过程中很有价值。潜在的缺点是它们可能比模型特定方法（如TreeSHAP与KernelSHAP）计算成本更高，并且作为近似方法，有时可能会遗漏借助模型内部机制的方法所捕获的细节。

> 模型无关方法可应用于各种模型类型，而模型特定方法则专为特定模型架构定制。

### 理解方法间的关系

需要注意的是，这些类别常常相互关联。例如：

- LIME和KernelSHAP既是**事后**方法，也是**模型无关**方法。
- 解释线性模型的系数是**内在**且**模型特定**的。
- TreeSHAP是**事后**方法，但**模型特定**（仅限于基于树的模型）。

了解不同技术在此分类中的位置对于选择合适的可解释性方法非常重要。如果您使用的是简单的线性模型，您可能可以依赖其内在可解释性。如果您训练了一个复杂的梯度提升模型，需要向利益相关者解释其预测，那么像LIME或SHAP这样的事后、模型无关方法（或模型特定的TreeSHAP）将是合适的。这个分类框架为在后续章节研究LIME和SHAP等具体算法之前，提供了一种结构化的方式来思考可用的选项。

## 参考资料

- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2023)
  一本全面的在线书籍，提供了机器学习可解释性方法的结构化概述，涵盖了各种分类维度、局部/全局解释以及模型特定/模型无关技术。
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: Association for Computing Machinery; Pages: 1135–1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME的原始论文，LIME是一种广泛采用的事后模型无关方法，可以为任何分类器的单个预测生成局部、可解释的解释。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43141543f340f-Paper.pdf) — Scott M Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: NeurIPS; Pages: 4765-4774
  提出了SHAP的开创性论文，SHAP是一种博弈论解释框架，统一了多种可解释性方法并提供了一致的特征归因，包括模型无关（KernelSHAP）和模型特定（TreeSHAP）的实现。
- [A Survey of Explainable Artificial Intelligence (XAI): Methods, Applications, Challenges, and Opportunities](https://www.mdpi.com/2571-5577/1/4/31) — Amina Adadi, Mohammed Berrado (2018)
  Journal: Applied System Innovation; Publisher: MDPI; Volume: 1; Pages: 31; DOI: [10.3390/asi1040031](https://doi.org/10.3390/asi1040031)
  一项全面的调查，根据多个维度（包括模型无关/特定和预模型/事后模型方法之间的区别）对可解释人工智能（XAI）方法进行分类，提供了该领域的广泛概述。
