---
course: "llm-alignment-safety"
chapter: "adversarial-attacks-defenses-llms"
lesson: "membership-inference-privacy-attacks"
sourceId: 3199
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-5-adversarial-attacks-defenses-llms/membership-inference-privacy-attacks"
title: "成员推断与隐私攻击"
description: "讨论大型语言模型上的成员推断攻击等隐私风险。"
order: 5
plots: []
sourceHash: "1210beb7a7df515b0c39990e741f5f7eeefb1f3a5d6a1a11ef422c98cba57c3f"
sourceCorrections: []
---

大型语言模型从海量文本数据中学习模式、语法，甚至事实信息。虽然对齐 (alignment)技术旨在控制模型生成*什么*内容，但它们并不能从根本上阻止模型无意中泄露*有关*其训练数据的信息。这种信息泄露会带来严重的隐私风险，特别是当训练语料库包含敏感或专有信息时。成员推断攻击（MIAs）是这类威胁中的一个主要方面。

## 了解成员推断攻击

成员推断攻击的目标直观明了：给定一个特定的数据片段（如句子、段落或代码片段），且可访问一个已训练好的大型语言模型，攻击者试图确定该数据片段是否曾是模型训练集的一部分。

为什么这是个问题？设想一个大型语言模型，其训练数据混合了公开网页文本与私人公司邮件或用户聊天记录。攻击者可以利用成员推断攻击来测试某个特定的机密邮件或敏感用户消息是否被包含在训练数据中，从而确认其已被泄露。这会侵犯用户隐私，并可能泄露有价值的知识产权。

## 成员推断攻击如何针对大型语言模型运行

成员推断攻击通常利用模型对其在训练期间见过以及从未见过的输入的回应方式中细微的差异。模型，特别是当模型非常大或训练时间很长时，有时会“记住”或对部分训练数据变得过度熟悉。这不代表它们逐字存储了数据，而是其内部表示和输出概率会偏向已见过的示例。

可以采用几种方法来推断成员资格：

1. **似然或困惑度分析：** 模型通常会给它们训练过的序列赋予更高的概率（从而更低的困惑度），相比于类似但未见过的序列。攻击者可以使用目标数据点 $x$ 查询模型，并观察其困惑度 $PPL(x)$。如果 $PPL(x)$ 明显低于可比较的已知非成员数据点的困惑度，这表明 $x$ 可能曾是训练集的一部分。序列 $x = (x_1, ..., x_N)$ 的困惑度通常计算如下：

   
   $$
   PPL(x) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log p(x_i | x_{<i}; \theta)\right)
   $$
   

   其中 $p(x_i | x_{<i}; \theta)$ 是模型 $\theta$ 根据前置 token 给出的第 $i$ 个 token 的概率。较低的值表示模型认为该序列更具可预测性，这可能源于它在训练期间曾见过该序列。
2. **损失值比较：** 与困惑度类似，针对特定输入计算的训练损失，对于训练集中的示例通常低于未见过的示例。具有一定访问权限（例如，梯度信息或损失输出）的攻击者可能会利用这种差异。
3. **基于参考的攻击（例如，LiRA）：** 更复杂的攻击，如似然比攻击（LiRA），通常涉及在与目标模型训练数据类似的数据分布上训练多个“影子”模型。通过比较目标模型对数据点 $x$ 的输出概率与在包含 $x$ 和不包含 $x$ 的情况下训练的影子模型所产生的概率分布，攻击者可以更精确地推断成员资格。
4. **校准差异：** 模型对训练数据和非训练数据可能表现出不同的输出置信度或校准特性，这有时可以被利用。

考虑一个简化的攻击流程：

> 一个针对大型语言模型的基本成员推断攻击流程。攻击者利用模型对特定数据点的响应指标来推断其训练集成员资格。

## 影响脆弱性的因素

并非所有模型或数据点都同样容易受到成员推断攻击。因素包括：

- **过拟合 (overfitting)：** 过度拟合的模型更有可能记住训练示例，使它们更容易受到影响。
- **数据重复：** 在训练集中多次出现的数据点通常更容易被推断。独特、异常的数据点有时也可能更脆弱。
- **模型大小与训练：** 更大的模型和更长的训练时间可以增加记忆能力。
- **微调 (fine-tuning)数据：** 在较小、特定数据集上进行微调（特别是当它们包含敏感信息时）可以大大增加记住这些微调示例的风险。
- **数据类型：** 高度独特的序列（例如，个人身份号码、特定的错误消息、独特的用户生成内容）通常比通用文本更容易受到影响。

## 相关隐私问题

成员推断攻击侧重于*成员资格*，而相关攻击则旨在*提取*或*重建*训练数据。一些模型可能会无意中生成训练数据中的逐字序列，特别是在得到适当提示时。这是一种独特但相关的隐私风险，通常源于记忆等类似的根本原因。

## 缓解策略

防御成员推断攻击和相关隐私攻击是一个活跃的研究方向。一些常见方法包括：

- **差分隐私（DP）：** 像DP-SGD（差分隐私随机梯度下降 (gradient descent)）这样的技术在训练期间注入噪声，以提供数学保证，确保模型的输出不会过度受任何单个训练示例的影响。这直接阻碍了成员推断。
- **正则化 (regularization)：** 像dropout或权重 (weight)衰减这样的技术可以减少过拟合 (overfitting)，从而减少记忆。
- **数据整理与去重：** 仔细过滤训练数据以移除敏感信息并减少过度重复可以降低风险。
- **审计与测试：** 在部署前，使用已知的攻击技术主动测试模型的记忆能力和成员推断攻击脆弱性。

防范像成员推断这样的隐私攻击是构建可信赖大型语言模型系统的一个重要组成部分。这需要在数据准备、模型训练和部署后监控阶段进行仔细考虑，补充了本课程中讨论的对齐 (alignment)和安全措施。

## 参考资料

- [Membership Inference Attacks Against Machine Learning Models](https://ieeexplore.ieee.org/document/7946765) — Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov (2017)
  Journal: IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 3-18; DOI: [10.1109/SP.2017.37](https://doi.org/10.1109/SP.2017.37)
  这篇基础论文介绍了针对机器学习模型的成员推断攻击的概念和方法。
- [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646) — Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang (2022)
  Journal: arXiv preprint arXiv:2202.07646; DOI: [10.48550/arXiv.2202.07646](https://doi.org/10.48550/arXiv.2202.07646)
  这项研究调查了神经语言模型中记忆的程度和机制，这是成员推断脆弱性的一个关键因素。
- [Deep Learning with Differential Privacy](https://doi.org/10.1145/2976749.2978318) — Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang (2016)
  Journal: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security; Publisher: Association for Computing Machinery; Pages: 308-318; DOI: [10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318)
  本文介绍了DP-SGD，一种广泛采用的训练深度学习模型的方法，具有差分隐私保证，可防止成员推断。
