---
course: "intro-synthetic-data-ml"
chapter: "understanding-synthetic-data"
lesson: "common-terminology"
sourceId: 5635
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-1-understanding-synthetic-data/common-terminology"
title: "常用术语"
description: "讨论合成数据生成与使用时的基本术语定义。"
order: 4
plots: []
sourceHash: "6823fdc9a92fbe69b0517adb7a00d1f280739b2bc0326e09e0a3fc391a9e4a73"
sourceCorrections: []
---

当您开始使用合成数据时，您会遇到一些专门的术语。理解这些词汇有助于讨论不同的方法并评估成果。让我们解释一些您会经常听到的常用术语。

### 核心术语

- **合成数据：** 正如我们之前讨论的，这是通过人工方式生成的数据，而非通过直接测量或实际事件观察所得。它由算法生成。
  "\* **真实数据（或基准数据）：** 这是指从真实来源收集的数据。例子包括来自物理设备的传感器读数、人们的调查反馈，或相机拍摄的真实图像。真实数据通常作为基准，或作为生成合成数据的灵感来源。"
- **数据生成模型：** 这是创建合成数据背后的“引擎”。它可以是一组数学方程式、统计分布、预设规则，甚至是一个复杂的机器学习 (machine learning)模型。它的目的是生成新的数据点，这些数据点模仿真实数据的某些特点。我们将在下一章审视简单的模型。
- **数据合成：** 这仅仅是使用数据生成模型来创建合成数据的*过程*。

### 评估合成数据

创建合成数据还不够；我们需要了解它是否表现良好。有两个术语对此评估非常主要：

- **保真度：** 这衡量了合成数据集与真实数据集中发现的统计属性和模式的匹配程度。高保真度意味着合成数据在统计上看起来和行为都与原始数据相似。例如，合成数据中列的平均值是否与真实数据中的平均值匹配？列之间的关系是否看起来相似？
- **效用：** 这衡量了合成数据对于特定目的（通常是训练机器学习 (machine learning)模型）的有效性。高效用性意味着仅使用合成数据训练的模型在真实数据上进行测试时表现良好。值得注意的是，高保真度并不总是保证高效用性，反之亦然，尽管它们通常相关联。其有用性很大程度上取决于具体的任务。

### 隐私考量

使用合成数据的一个主要驱动因素是隐私。

- **隐私保护：** 这是创建合成数据的目标，此类数据能捕捉真实数据集中的有用模式，而不泄露原始数据集中个人或实体的敏感信息。目标是使合成数据点难以或不可能与真实个人关联起来。
- **匿名化：** 这是一个相关的过程，侧重于修改真实数据，以去除或模糊个人身份信息。尽管合成数据生成有所不同（它创建新数据），但它通常与匿名化方法有相似的隐私目标。一些先进的合成数据方法结合了诸如*差分隐私*之类的原则，以提供关于隐私的数学保证，但这属于一个更进阶的议题。

### 数据类型

合成数据方法可应用于各种数据格式。在本课程中，我们将着重于：

- **表格数据：** 数据以表格形式组织，其中行（表示记录或观察）和列（表示特征或属性）。可以想象包含客户信息、销售记录或实验结果的电子表格或数据库表。
- **图像数据：** 表示视觉信息的数据，通常是像素网格形式，其中每个像素具有指示颜色或强度的数据值。例子包括照片、医学扫描（如X射线）或卫星图像。

其他类型如文本、音频和时间序列数据也可以合成，但其基本原理通常保持不变。

### 生成方法

我们很快会审视具体方法，但术语常常暗示其底层方法：

- **基于规则的生成：** 通过遵循明确定义的规则来创建数据点。例如，一条规则可能规定：“如果`region`是‘North’，则`temperature`必须在-10到+15之间。”
- **统计抽样：** 通过从统计分布中抽取随机样本来生成数据（例如用于身高的正态分布或用于随机ID的均匀分布）。这些分布的参数 (parameter)通常是从真实数据中估算的。
- **基于模型的生成：** 使用训练好的机器学习 (machine learning)模型来生成数据。这些模型从真实数据中学习底层模式，然后可以生成新的、相似的例子。生成模型，如GANs（生成对抗网络 (GAN)）和VAEs（变分自编码器），属于此类，代表了您以后可能会遇到的更复杂的方法。

理解这些术语为审视本课程中讨论的合成数据方法和应用提供了认识。它们有助于确立在为机器学习生成人工数据时所涉及的挑战、目标和评估标准。

## 参考资料

- [A Survey of Synthetic Data Generation Methods for Tabular Data](https://ieeexplore.ieee.org/document/9924403) — Viktoriia Borisov, Tobias Leemann, Jörg Huber, Marc Fischer, Tobias Böhm, Michele F. Sutter, and Stefan Feuerriegel (2022)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Pages: 1-18; DOI: [10.1109/TKDE.2022.3214876](https://doi.org/10.1109/TKDE.2022.3214876)
  这项全面的调查详细介绍了表格数据的合成数据生成方法，这是本课程的一个主要方面。它涵盖了各种方法并讨论了它们的优缺点。
- [What makes good synthetic data? Statistical and data utility metrics for generative models](https://academic.oup.com/jamia/article/29/5/936/6530654) — Kyle M. O’Donoghue, Ryan D. Hays, Andrew A. Kress, and Daniel E. Casey (2022)
  Journal: Journal of the American Medical Informatics Association; Publisher: Oxford University Press; Volume: 29; Pages: 936-946; DOI: [10.1093/jamia/ocac014](https://doi.org/10.1093/jamia/ocac014)
  本文专门研究并分类了评估合成数据质量的各种指标，侧重于统计保真度和机器学习任务的实际效用。
- [A Survey on Privacy-Preserving Synthetic Data Generation](https://dl.acm.org/doi/10.1145/3478546) — Sangmin Kim, Tuan Anh Nguyen, and Yansong Li (2021)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 54; Pages: 1-36; DOI: [10.1145/3478546](https://doi.org/10.1145/3478546)
  这项调查回顾了在保护隐私的同时生成合成数据的方法和考虑因素，包括与匿名化和高级隐私机制相关的技术讨论。
- [Synthetic Data Generation for Machine Learning: A Practical Guide](https://www.research.ibm.com/downloads/cas/2JGR58G5) — Alistair Jordon, Paul van der Putten, Jinsung Yoon, Mihaela van der Schaar, and Richard Schmarzo (2022)
  Publisher: IBM Research
  这份来自IBM研究的实践指南清晰地介绍了用于机器学习的合成数据生成，解释了基本概念及其应用。
