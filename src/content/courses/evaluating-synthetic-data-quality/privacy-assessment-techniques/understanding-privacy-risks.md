---
course: "evaluating-synthetic-data-quality"
chapter: "privacy-assessment-techniques"
lesson: "understanding-privacy-risks"
sourceId: 5680
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-4-privacy-assessment-techniques/understanding-privacy-risks"
title: "了解合成数据中的隐私风险"
description: "详细概述了生成数据集中潜在的隐私脆弱性。"
order: 1
plots: []
sourceHash: "0b85d0fdc7cbb6bce3e38d8883c6f0a6d08e73545be8990378734d8cee5ab238"
sourceCorrections: []
---

虽然生成合成数据与发布原始数据相比，能在隐私保护方面提供重要优势，但必须认识到，生成过程本身并不能自动保证隐私。合成数据集仍然可能无意中泄露其源自的原始敏感数据集的信息。识别这种潜在泄露的性质和途径，是有效开展隐私评估的第一步。

主要问题通常源于生成模型对原始数据学习得*过多*，有时甚至会记忆特定细节或敏感的统计模式。这可以通过几种方式表现出来：

### 主要隐私脆弱点

1. **记录复制和近似复制：**

   - **定义：** 生成模型产生的合成记录与原始训练数据集中存在的记录相同或高度相似。
   - **风险：** 如果攻击者拥有关于原始数据集中个体的一些背景知识，他们可能能够通过将合成记录与真实人员匹配来重新识别个体。即使是近似复制也可能显著增加识别风险，特别是对于具有独特特征（异常值）的个体。这直接破坏了匿名化的目标。
2. **属性泄露：**

   - **定义：** 合成数据泄露了个体敏感属性的信息，即使其特定记录未被复制。当模型准确学习到真实数据中非敏感属性和敏感属性之间的强关联性时，就会发生这种情况。
   - **风险：** 攻击者可以利用合成数据集，根据个体已知的非敏感属性（例如，年龄、邮政编码、职业）来推断其敏感属性（例如，医疗状况、收入区间）。例如，如果模型学习到 $P(\text{条件 X} | \text{年龄}=A, \text{邮编}=Z)$ 非常高，那么观察到一条包含年龄 $A$ 和邮编 $Z$ 的合成记录，可能使攻击者能够以高置信度推断出符合这些非敏感属性的真实个体的条件 X。
3. **成员推断：**

   - **定义：** 攻击者试图判断特定个体的数据记录是否属于用于训练生成模型的原始数据集。
   - **风险：** 仅仅确认某人存在于敏感数据集（例如，特定疾病患者数据集、政治捐赠者数据集）中，本身就可能侵犯隐私。过拟合 (overfitting)或记忆部分训练数据的模型更容易受到这些攻击，因为当用与训练时相似的输入或未见的输入进行查询时，它们的表现可能会有所不同。
4. **聚合属性泄露：**

   - **定义：** 合成数据泄露了敏感数据集的聚合属性或统计数据，这些数据之前并非公开信息且被认为是敏感的。
   - **风险：** 尽管通常不如个体记录泄露直接，但显示精确的统计关系（例如，特定亚组内收入与贷款违约率之间的精确关联）可能会泄露商业敏感信息或敏感的人群特征。

### 影响隐私风险水平的因素

隐私风险程度并非在所有合成数据集中都相同。与数据、模型和生成过程相关的几个因素发挥重要作用：

- **生成模型选择：** 不同的模型具有不同的记忆倾向。例如，在某些条件下，容量非常高的生成对抗网络 (GAN) 可能比变分自编码器 (VAE) 更容易记忆，而明确设计了差分隐私机制的模型旨在通过添加校准噪声来限制信息泄露。
- **模型复杂度和参数 (parameter)：** 相对于训练数据大小参数过多的模型，更容易过拟合 (overfitting)并记忆个体记录。训练超参数 (hyperparameter)，例如训练轮数、学习率和正则化 (regularization)技术，也直接影响模型与训练数据的拟合紧密程度。
- **训练数据特征：** 小型数据集，或包含独特的异常值或稀有特征组合的数据集，带来更高的风险。异常值更难从中进行泛化，模型也更容易记忆。
- **数据维度：** 高维数据有时会因“维度灾难”而增加隐私风险，可能使记录更具独特性且更容易区分。
- **后处理步骤：** 生成后应用的技巧，例如数据平滑或扰动，可能减轻一些风险，但如果设计不当，也可能引入其他风险。

以下图表直观地概述了这些风险如何体现：

> 该图表说明了源于在敏感真实数据上训练的生成模型所产生的潜在隐私风险，导致合成数据集中出现近似复制、成员推断的可能性和属性泄露等问题。

认识到这些潜在的脆弱性非常重要。本章后续部分将为您提供具体的评估方法，例如成员推断攻击 (MIA)、属性推断攻击和基于距离的度量指标，以定量评估您生成的数据集中的这些风险。这有助于就合成数据在特定用途中的适用性做出明智的决定，从而平衡数据效用需求与隐私保护的必要性。

## 参考资料

- [The Algorithmic Foundations of Differential Privacy](https://doi.org/10.1561/0400000042) — Cynthia Dwork and Aaron Roth (2014)
  Publisher: Now Publishers; DOI: [10.1561/0400000042](https://doi.org/10.1561/0400000042)
  该书提供了差分隐私的理论基础，差分隐私是一种用于量化和限制数据分析与合成数据生成中隐私风险的数学框架。
- [Membership Inference Attacks Against Machine Learning Models](https://ieeexplore.ieee.org/document/7946764) — Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov (2017)
  Journal: 2017 IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 371-385; DOI: [10.1109/SP.2017.37](https://doi.org/10.1109/SP.2017.37)
  这篇论文开创性地介绍了成员推断攻击的概念和方法，这是针对使用敏感数据训练的模型（包括生成模型）的一个重要隐私风险。
- [PATE-GAN: Generating Private Data with PATE](https://openreview.net/forum?id=S1gP_x05FK) — Jihye Yoon, Jinsung Yoon, Mihaela van der Schaar (2019)
  Journal: International Conference on Learning Representations (ICLR); Publisher: ICLR; DOI: [10.48550/arXiv.1811.02562](https://doi.org/10.48550/arXiv.1811.02562)
  这篇论文提出了一种使用生成对抗网络（GANs）和PATE框架生成差分隐私合成数据的方法，直接解决了生成模型设计中的隐私问题。
