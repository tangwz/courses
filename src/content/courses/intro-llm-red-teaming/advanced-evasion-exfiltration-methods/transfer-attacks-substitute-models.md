---
course: "intro-llm-red-teaming"
chapter: "advanced-evasion-exfiltration-methods"
lesson: "transfer-attacks-substitute-models"
sourceId: 6330
sourceUrl: "https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-4-advanced-evasion-exfiltration-methods/transfer-attacks-substitute-models"
title: "迁移攻击：使用替代模型"
description: "了解迁移攻击，其中对抗样本在替代模型上构造，并应用于目标LLM。"
order: 2
plots: []
sourceHash: "f2b31fe2f48fced6d26bc277025b8ce1e3fb17249e963b0a7058048284cebf22"
sourceCorrections: []
---

尽管基于梯度的攻击能有效生成对抗样本，但它们通常需要对目标模型进行白盒访问，这意味着你需要了解其架构和参数 (parameter)，$M(x; \theta)$。然而，当你测试的大型语言模型（LLM）是黑盒 (black box)，只提供API接口且无法看到其内部运作时会怎样？这时，使用替代模型的迁移攻击就成为你红队工具包中的一种实用技术。

迁移攻击利用了对抗样本的一个有趣特性：它们的迁移性。这意味着为欺骗一个模型而设计的对抗性输入，通常很有可能欺骗另一个模型，即使第二个模型具有不同的架构或是在不同（但可能相关）的数据上训练的。就好像这些对抗样本触及了机器学习 (machine learning)模型解释高维输入空间时，更基本、更共通的弱点。

迁移攻击的主要策略是首先创建或训练一个 *替代模型*（有时也称作代理模型），它能模仿目标黑盒LLM的行为。一旦你拥有这个替代模型（并且你 *确实* 可以对其进行白盒访问），你就可以应用基于梯度或其他白盒攻击技术来生成对抗样本。最后，你将这些对抗样本“迁移”到原始目标LLM，期望它们能引起预期的错误行为。

让我们分析一下典型的工作流程：

### 1. 构建替代模型

第一步是创建一个数据集来训练你的替代模型。这涉及到与目标黑盒 (black box)LLM的交互：

- **查询目标LLM：** 你会向目标LLM发送一系列多样化的输入提示 $X_q$。这些可能是看起来合法的查询、精心设计的探测，甚至是随机字符串，具体取决于你的策略。
- **收集输出：** 对于每个输入 $X_{q_i}$，你都会记录目标LLM的相应输出 $Y_{q_i} = M_{target}(X_{q_i})$。这个输入-输出对 $(X_{q_i}, Y_{q_i})$ 成为你训练替代模型的数据点。你收集的数据点越多，且它们越能代表目标的行为，你的替代模型可能就越好。
- **训练替代模型：** 通过这个收集到的数据集，你就可以训练你自己的模型 $M_{substitute}$。这个替代模型可以是另一个LLM（例如一个可以微调 (fine-tuning)的、更小的开源模型，如GPT-2或LLaMA的变体），如果目标LLM的任务足够具体，它甚至可以是一个更简单的分类/回归模型。目的是使 $M_{substitute}$ 能够近似 $M_{target}$ 的输入-输出行为。

这种查询过程可能需要大量资源，可能涉及多次API调用，这可能会产生费用或触发目标系统的速率限制。

### 2. 在替代模型上构造对抗样本

一旦你的替代模型 $M_{substitute}$ 训练完成，你实际上就拥有了一个本地的白盒副本，其行为（希望）与目标模型类似。现在，你可以运用标准的白盒攻击技术来生成对抗样本。例如，你可以使用像快速梯度符号方法（FGSM）或投影梯度下降 (gradient descent)（PGD）这样的方法来找到一个输入 $X_{adv}$，当将其输入 $M_{substitute}$ 时，会产生不期望或恶意的输出 $Y_{adv_s} = M_{substitute}(X_{adv})$。因为你对 $M_{substitute}$ 拥有完全访问权限，包括其梯度，所以构造 $X_{adv}$ 要直接得多。

### 3. 将攻击迁移到目标LLM

最后的主要一步是将使用你的替代模型生成的对抗样本 $X_{adv}$ 输入到原始黑盒 (black box)目标LLM $M_{target}$。如果对抗属性发生迁移，那么 $M_{target}(X_{adv})$ 也会产生一个不期望的输出 $Y_{adv_t}$，与 $Y_{adv_s}$ 类似，或者至少达到攻击者的目标（例如，绕过安全过滤器，提取信息）。

下图说明了此过程：

> 迁移攻击过程：查询目标、训练替代模型、在替代模型上构造对抗样本，并将它们应用于目标。

### 影响迁移性的因素

迁移攻击的成功并非确定，但有几个因素可以增加其可能性：

- **替代模型的相似性：** $M_{substitute}$ 的架构、训练数据分布或决策边界与 $M_{target}$ 越接近，迁移的可能性就越高。如果已知或怀疑目标LLM所属的系列，使用同一系列中的预训练 (pre-training)LLM会比较有益。
- **攻击方法：** 有些对抗攻击生成方法本身就会产生更易迁移的样本。那些能找到更有效对抗样本的迭代方法通常比单步方法迁移效果更好。
- **替代模型的数据质量：** 用于训练替代模型的查询-响应对的多样性和数量作用显著。在一个能更好地反映目标行为细节的数据集上训练的替代模型，更有可能促成可迁移的攻击。
- **模型容量：** 大型模型有时会表现出更好的攻击迁移性。反之，在非常小的替代模型上构造的攻击，可能无法很好地迁移到大型、复杂的目标模型。

### 这对红队的意义

迁移攻击是一个值得关注的问题，因为它们确实降低了攻击LLM的门槛。攻击者不需要特权访问或目标模型的内部知识。他们“只需”查询访问权限和训练替代模型所需的资源。

然而，创建高保真替代模型可能具有挑战性。它需要：

- 对目标模型进行大量查询，如果查询不寻常或数量巨大，这可能会很慢、昂贵或触发检测机制。
- 训练替代模型本身的计算资源。
- 仔细选择替代模型的架构和训练过程。

尽管存在这些挑战，迁移攻击的可能性意味着即使那些因其黑盒 (black box)性质而被认为安全的LLM也并非免疫。作为一名红队人员，了解如何模拟或执行此类攻击有助于评估LLM系统的真正弹性。如果你能成功构造出可迁移的对抗样本，这指出了一个需要处理的弱点，可能需要通过改进输入过滤、输出净化，或开发通过查询模式检测替代模型训练的方法来解决。这项技术突显了LLM安全采用多层次防御策略的重要性。

## 参考资料

- [Practical Black-Box Attacks against Machine Learning](https://doi.org/10.1145/3052973.3053009) — Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, Ananthram Swami (2017)
  Journal: ACM Asia Conference on Computer and Communications Security (ASIACCS); Publisher: Association for Computing Machinery; Pages: 506-519; DOI: [10.1145/3052973.3053009](https://doi.org/10.1145/3052973.3053009)
  这篇基础性论文介绍了可转移性概念，并展示了使用替代模型进行实用的黑盒对抗性攻击，这是本节描述的核心技术。
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) — Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson, Xuezhou Ren, Dan Hendrycks (2023)
  Journal: arXiv preprint arXiv:2307.15043
  这项近期工作展示了针对现代对齐大型语言模型的通用可转移对抗性攻击的有效性，与LLM红队测试直接相关。
- [A Survey on Adversarial Attacks and Defenses in Large Language Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHqgwRL-i3_RXSrqFCHB3BOjByuJFqHbQrqHNz8oa9Hhfj2klE5R3Updy6qwTIiwaNNnCH5VY6EzaZ3M6_J6HGRGg3cRjBH4MFEwZvUrtk-To4FCmPslLoE-yhmgNTjd9QKh8ck25Kf5SUfi68YZzuWJIQ2dn3lJy9FScOqSwjZ9etwoRPP_YeL4rNU6hUlcWwiV989b1C1QFg=) — Weilin Su, Xuanli He, Baolin Zhao, Wei Wang (2023)
  Journal: arXiv preprint arXiv:2306.07765
  这篇全面的综述概述了大型语言模型的各种对抗性攻击和防御策略，包括对可转移性和黑盒攻击方法的讨论。
