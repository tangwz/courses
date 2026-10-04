---
course: "llm-alignment-safety"
chapter: "foundations-llm-alignment"
lesson: "defining-alignment-llms"
sourceId: 3102
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-1-foundations-llm-alignment/defining-alignment-llms"
title: "大型语言模型中的对齐定义"
description: "建立大型语言模型的对齐技术定义。"
order: 1
plots: []
sourceHash: "a52f660c5c87a831525f34839b0056e2b73d7171b0d73aef34fa18005bba7f35"
sourceCorrections: []
---

对齐 (alignment)大型语言模型（LLM）涉及的不仅仅是困惑度或特定自然语言处理基准测试的准确率等传统指标。对齐指的是LLM的行为在各种情境下，与人类设计者和用户的意图保持一致的程度。它确保模型按预期运行，遵循设定的道德规范，并防止不必要的负面结果。

### 对齐 (alignment)的核心组成部分

尽管由于“意图”的复杂性，单一的、普遍接受的定义仍然难以明确，但对齐通常通过一系列期望的行为属性来理解。这些属性常被概括为：

1. **有用性：** 模型应准确有效地理解并执行用户指令。它应以有用的方式完成分配的任务。
2. **如实性：** 模型应根据其训练数据提供符合事实的信息。值得注意的是，它还应准确地表达其自身的知识局限，避免“幻觉 (hallucination)”或自信地陈述虚假信息。如果其不确定，理想情况是应表明这种不确定性。
3. **无害性：** 模型必须避免生成有毒、歧视性、促进非法活动或导致其他形式损害的输出。这包括遵循开发者设定的安全协议和内容限制。

这三个组成部分（有用性、如实性、无害性，简称“HHH”）为评估和引导LLM行为提供了一个实用的框架，尽管每个组件的具体解读和权重 (weight)可能因应用情境而异。

> 对齐的观点是人类意图与观测到的模型行为之间的一致性，这通常通过有用性、如实性和无害性等属性来实现。

### 对齐 (alignment)与能力

区分对齐与模型的原始*能力*是很重要的。能力是指模型在预训练 (pre-training)期间学到的固有能力，例如理解语法、存储事实知识、推理 (inference)或生成创意文本。一个模型可能能力很强但对齐不佳（例如，在被提示时巧妙地生成有害内容），或者能力较低但对齐合理（例如，拒绝有害请求但也难以处理复杂指令）。我们将在本课程中考察的对齐技术，旨在引导模型现有能力实现期望的结果。

### 形式化视角

我们可以将对齐 (alignment)视为最小化模型输出的概率分布 $P_{\text{model}}(y | x)$ 与代表意图行为的理想化分布 $P_{\text{intended}}(y | x)$ 之间的差异，给定输入上下文 (context) $x$。


$$
\text{最小化 } D(P_{\text{model}}(y | x) || P_{\text{intended}}(y | x))
$$


在此，$D$ 代表某种差异度量（例如KL散度）。主要的挑战，如下一节所述，在于 $P_{\text{intended}}$ 极其难以，甚至不可能完全准确地指定所有可能的输入 $x$。它包含了复杂的人类偏好、道德规范和情境差异。对齐工作很大程度上在于寻找有效的替代方法来逼近和优化此意图分布。

### 谁的意图？

此外，“人类意图”并非单一的。开发者可能优先考虑安全性和鲁棒性，部署者可能侧重于特定的应用目标和品牌声誉，而最终用户则有即时任务目标。整个社会对公平性、偏见和长期影响抱有期望。平衡这些可能冲突的意图是对齐 (alignment)挑战的核心组成部分。

理解这种以意图行为而非仅任务表现为中心的对齐定义，为理解所涉困难（即“对齐问题”）和采用诸如人类反馈强化学习 (reinforcement learning)（RLHF）等我们将介绍的其他专门技术的必要性奠定了基础。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  这项工作进一步展示了RLHF在对齐会话式AI方面的应用，明确关注所需行为的“有用性和无害性”组成部分。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  这篇论文介绍了“宪法式AI”，一种替代的对齐方法，它基于一组原则使用AI生成的反馈来实现无害性，减少了对大量人工标注的依赖。
- [Ethical and Social Risks of Harmful Language Models](https://arxiv.org/abs/2112.04359) — Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geoffrey Irving, Iason Gabriel (2021)
  Journal: arXiv preprint arXiv:2112.04359; DOI: [10.48550/arXiv.2112.04359](https://doi.org/10.48550/arXiv.2112.04359)
  这篇论文系统地识别并分类了大型语言模型相关的伦理和社会风险，为对齐中的“无害性”组成部分提供了充分的理由。
