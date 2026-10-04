# 选择LIME和SHAP

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-4-comparing-applying-techniques/choosing-lime-or-shap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LIME通过扰动生成更简单、易于理解的替代模型，以模拟模型的局部表现。SHAP则使用合作博弈论的原理（Shapley值），在特征间分配预测结果。每种方法都有其优势，并且适用于不同的情况。决定使用哪种方法通常取决于您具体的模型、数据、计算资源以及所需理解的性质。

以下是在LIME和SHAP之间进行选择时需要考虑的因素细分：

### 模型类型

- **基于树的模型（决策树、随机森林、梯度提升）：** SHAP提供`TreeSHAP`，这是一种为这类模型专门设计的效率高且准确的算法。它计算SHAP值比基于置换的方法（如KernelSHAP或LIME）快很多。如果您主要使用树集成模型，`TreeSHAP`通常是有效的选择，因为它针对此模型类提供了速度和准确性保证。LIME仍然可以使用，但它不会用到树的内部结构来提升效率。
- **深度学习 (deep learning)模型：** SHAP提供特定的解释器，如`DeepSHAP`和`GradientSHAP`，它们借助神经网络 (neural network)的结构来提升效率。然而，它们的适用性可能取决于具体的框架（TensorFlow、PyTorch）和网络架构。LIME是模型无关的，可以用于任何深度学习模型，但可能需要仔细调整扰动方法（例如，对于图像或文本）。`KernelSHAP`也是一个选择，但速度可能较慢。
- **任何黑盒 (black box)模型：** LIME和`KernelSHAP`都被设计为模型无关的。LIME通过拟合 (overfitting)一个局部线性模型来解释单一预测，而`KernelSHAP`则使用特殊加权的局部线性回归来估算Shapley值。这里的选择通常取决于计算成本和解释的预期特点等其他因素。

### 理论特性与一致性

- **一致性与准确性：** SHAP值具有源自Shapley值的理想理论特性，包括局部准确性（特征归因的总和等于预测值与预期基线预测值之差）和一致性（如果模型更多地依赖某个特征，则该特征的重要性值不应降低）。这些特性为SHAP解释提供了更坚实的理论依据。
- **稳定性：** LIME解释有时会受生成的随机扰动以及用于加权的核宽度选择的影响。在同一实例上多次运行LIME可能会产生略微不同的特征重要性。SHAP值（特别是来自`TreeSHAP`等精确方法的值）通常更稳定。

### 计算成本

- **解释单一预测：** LIME通常在为*单个*实例生成解释时速度更快。它需要扰动实例并查询模型对这些扰动的预测，以拟合局部替代模型。所需的样本数量通常可配置。
- **解释多个预测 / 全局理解：** `KernelSHAP`的计算成本会随特征数量和用于近似的背景样本数量的增加而明显增加。使用`KernelSHAP`计算许多实例的SHAP值可能非常耗时。然而，`TreeSHAP`在应用于树模型时，计算整个数据集的SHAP值速度非常快。
- **资源限制：** 如果计算资源有限，或者您需要实时为单个预测提供非常快速的解释，LIME可能是一个更实用的选择，尤其是在您的应用中SHAP的理论保证不那么重要的情况下。

### 所需解释的类型

- **局部解释：** 两种方法都提供局部解释，为特定预测归因特征重要性。LIME提供局部线性模型的系数，这可能直观易懂。SHAP提供SHAP值，表示每个特征相对于基线对预测的边际贡献，常通过力图进行可视化。
- **全局解释：** SHAP被设计为连接局部和全局解释。通过汇总多个实例的SHAP值，您可以生成有意义的汇总，如特征重要性图（每个特征的平均绝对SHAP值）和特征依赖图（显示特征值如何影响其SHAP值，可能显示交互作用）。LIME不直接提供全局解释的方法；汇总局部LIME解释是可行的，但理论依据不如SHAP的方法充分。

### 数据类型

- **表格数据：** LIME和SHAP都表现良好。`TreeSHAP`非常适合用于表格数据上的树模型。`KernelSHAP`和LIME是好的模型无关选择。
- **文本和图像数据：** 应用这两种方法都需要仔细考虑扰动策略。LIME通常有直观的扰动方法（例如，删除词语，遮蔽图像区域）。SHAP需要定义合适的背景数据和扰动技术，这在正确实现时可能更复杂，尽管库为常见用例提供了辅助工具（例如，使用背景颜色进行图像遮蔽，或文本的特殊标记 (token)）。

### 易用性

- **LIME：** 初始理解起来可能更简单（局部线性近似），并且对于基本用例可能需要较少的配置，特别是在扰动方法直观的文本/图像数据方面。
- **SHAP：** SHAP库提供统一的接口，但需要理解不同的解释器（`Kernel`、`Tree`、`Deep`等）并选择合适的。理解Shapley值可能需要多一些初步的理论依据。然而，SHAP库中的绘图工具功能非常强大且标准化。

### 总结指南

| 因素 | 如果...考虑LIME | 如果...考虑SHAP |
| --- | --- | --- |
| **模型类型** | 需要对*任何*模型进行快速解释 | 使用基于树的模型（`TreeSHAP`表现最佳） |
|  | `DeepSHAP`/`GradientSHAP`设置复杂 | 需要为支持的深度学习 (deep learning)模型提供效率高的解释 |
| **理论保证** | 速度比一致性/准确性更重要 | 一致性和局部准确性属性很重要 |
| **计算成本** | 需要为*单个*实例提供快速解释 | 需要对树模型进行解释（使用`TreeSHAP`） |
|  | `KernelSHAP`对您的需求而言太慢 | 可以承担`KernelSHAP`的计算成本 / 使用`TreeSHAP` |
| **解释范围** | 主要只需要*局部*解释 | 需要*局部*和可信的*全局*解释 |
| **数据类型（文本/图像）** | 偏好更简单的扰动设置 | 需要清晰的解释并愿意配置背景数据 |
| **易用性** | 偏好更简单的初始思想和设置 | 重视全面的库和绘图功能，理解解释器 |

最终，选择通常取决于您项目的具体情况。没有一种技术在所有情况下都是“最好”的。有时，同时使用LIME和SHAP可以为您的模型行为提供互补的视角。根据您的模型、数据、性能要求以及您需要达到的理解程度来评估权衡。

## 参考资料

- [Why Should I Trust You? Explaining the Predictions of Any Classifier](https://dl.acm.org/doi/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 1135-1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  介绍了LIME，这是一种模型无关的局部可解释性技术，通过使用更简单的替代模型来近似复杂模型的行为。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632c7cd199b51e0653f4-Paper.pdf) — Scott M. Lundberg and Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4766-4777; DOI: [10.5555/3295222.3295330](https://doi.org/10.5555/3295222.3295330)
  提出了SHAP，一个基于Shapley值的统一模型预测解释框架，并介绍了TreeSHAP算法。
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/) — Scott Lundberg and SHAP Contributors (2024)
  SHAP Python库的官方文档，提供了关于各种解释器（KernelSHAP、DeepSHAP、GradientSHAP）和使用方法的实用细节。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本权威、全面的在线书籍，涵盖了各种可解释性方法，包括LIME和SHAP的详细章节和比较。

---

[上一节](02-LIME%20%E4%B8%8E%20SHAP%EF%BC%9A%E4%BC%98%E7%82%B9%E4%B8%8E%E4%B8%8D%E8%B6%B3.md) · [下一节](04-%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B%E8%A7%A3%E9%87%8A%E7%9A%84%E7%90%86%E8%A7%A3.md)
