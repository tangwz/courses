---
course: "intro-feature-engineering"
chapter: "feature-engineering-foundations"
lesson: "feature-quality-impact"
sourceId: 1304
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-1-feature-engineering-foundations/feature-quality-impact"
title: "特征质量对模型表现的影响"
description: "分析特征的质量和相关性如何直接影响模型的准确性和泛化能力。"
order: 3
plots: ["plots/1304-0.json", "plots/1304-1.json"]
sourceHash: "f939fcc77ab0ef11bf72f9f487ba0adb5b6497d88e99ba4cd8b1be6468058ae3"
sourceCorrections: []
---

您提供给机器学习 (machine learning)模型的特征质量，可以说是决定其表现的主要因素。您可能听说过计算机方面的“垃圾进，垃圾出”（GIGO）原则；它在机器学习中尤其适用。即使是最先进的算法，如果输入数据嘈杂、不相关或结构不良，也将难以产生有意义的结果。反之，精心设计的特征甚至能让简单模型取得出色表现。

让我们考察特征质量如何具体影响模型：

### 预测准确性

基本来说，机器学习 (machine learning)模型试图将输入特征映射到输出目标。特征充当模型学习此映射所用的信号。

- **有信息的特征：** 与目标变量有强烈、明确关系的特征能为模型提供直接指导。例如，在预测房价时，`square_footage`（面积）很可能是一个极具信息的特征。模型可以很容易学到，面积越大通常对应更高的价格。
- **不相关的特征：** 与目标几乎没有关系或完全没关系的特征会增加噪声和复杂性，却不提升预测能力。在房价预测模型中包含`homeowner_favorite_color`（房主最喜欢的颜色）特征，可能会使算法混淆或被忽略，还可能增加计算时间而无益处。
- **冗余特征：** 包含重叠信息的特征（例如，`area_in_square_feet` 和 `area_in_square_meters`）不增加新的见解，但有时会给某些算法（如线性模型中的多重共线性）带来问题。
- **嘈杂特征：** 包含错误或随机波动的特征会误导模型，使其学到虚假模式，而这些模式在底层过程中并不实际存在。

高质量特征提供强大、清晰的信号，使模型能更有效地学到真实的底层模式，直接带来更高的预测准确性。

### 模型泛化能力

模型最终的目标不仅是在训练数据上表现良好，还要能很好地泛化到新的、未见过的数据。特征质量在此直接起作用。

- **过拟合 (overfitting)：** 差的特征，特别是嘈杂或不相关的特征，可能导致模型过拟合。模型可能会学到*仅*存在于训练数据中的特定于噪声或不相关方面的模式。当面对不共享这种确切噪声的新数据时，模型表现不佳。
- **捕捉底层结构：** 精心设计的特征旨在代表数据中与预测任务相关的基本结构和关系。用这类特征训练的模型更可能学到真实的底层过程，从而在未见过数据上获得更好的表现。

考虑预测客户流失。像 `last_login_timestamp`（上次登录时间戳）这样的原始特征可能还行。但像 `days_since_last_login`（上次登录天数）或 `average_session_length_last_month`（上月平均会话时长）这样的设计过的特征，很可能能更好地捕捉底层行为（客户参与度），帮助模型泛化以更准确地预测未来的流失。

### 模型简洁性和效率

有时，模型所需的复杂性直接与特征表示的复杂性相关。

- **使关系更清晰：** 巧妙的特征工程能将复杂、非线性的关系转换为更简单的关系。例如，如果目标变量与特征 $x$ 呈二次关系，创建一个新特征 $x^2$ 可能允许一个简单的线性模型有效地捕捉该关系，而只使用 $x$ 则可能难以处理。
- **减少维度：** 尽管后面的章节会明确涵盖特征选择，但创建*好*特征的初始过程通常涉及选择信息密集的表示。这有时能导致总体上所需特征更少，相比于向模型投入许多原始、信息较少的特征。特征数量越少，通常意味着更快的训练时间，以及可能更简单的模型。

想象一下，只使用原始 $(x, y)$ 坐标来分离形成同心圆的两类数据点。线性模型将失效。但是，如果您设计一个新特征 $r = \sqrt{x^2 + y^2}$（半径），那么即使对于简单模型，分离也变得微不足道。



![原始特征：非线性分离](plots/1304-0.json)



> 使用原始 x, y 坐标，分离蓝色（类别0）和橙色（类别1）点需要非线性决策边界。



![设计过的特征：可线性分离](plots/1304-1.json)



> 通过创建一个“半径”特征（到原点的距离），这两类数据点可以基于这个单一新特征上的简单阈值而容易分离。

### 可理解性

最后，您特征的性质影响着您能多容易地理解模型做出某些预测的*原因*。从特定业务知识导出或通过可理解的转换（如计算`time_since_last_purchase`（上次购买以来的时间））创建的特征，常能带来更易于理解的模型。如果模型对此类特征赋予高权重 (weight)，其预测背后的原因会变得更清楚。反之，依赖自动生成的复杂、抽象特征的模型可能获得高准确性，但难以理解，这会降低信任度并使调试更困难。

总而言之，在打造高质量特征上投入时间和精力，不仅仅是一个初步的数据清洗步骤。它是构建有效、可靠、易懂的机器学习 (machine learning)模型的根本方面。后续章节将为您提供技术，将原始数据转换为模型所需的强大特征。

## 参考资料

- [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://doi.org/10.1201/9781315108230) — Max Kuhn, Kjell Johnson (2019)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/9781315108230](https://doi.org/10.1201/9781315108230)
  这本全面的书籍系统地介绍了特征的创建和选择，直接阐述了它们对模型准确性、可解释性和效率的影响。(第二版)
- [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/) — Andrew Ng (2017)
  Publisher: DeepLearning.AI
  这本策略指南强调数据质量和特征工程对于提升AI应用中的模型性能和泛化能力非常重要。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125961/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的实用教材，其中有专门的章节介绍数据预处理和特征工程，展示了它们在实现良好模型性能和泛化能力方面的作用。
- [A Survey on Feature Engineering Methods for Machine Learning](https://doi.org/10.14569/IJACSA.2020.110231) — Navneet Singh, Jagdeep Singh Sohal Punia and Monika Partap Singh Bhatia (2020)
  Journal: International Journal of Advanced Computer Science and Applications; Publisher: Science and Information Organization; Volume: 11; Pages: 251-260; DOI: [10.14569/IJACSA.2020.110231](https://doi.org/10.14569/IJACSA.2020.110231)
  这篇综述文章概述了各种特征工程技术，并从讨论特征质量对模型有效性的重要性开始。
