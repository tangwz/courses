---
course: "llm-constitutional-ai-rlaif"
chapter: "scalable-alignment-problem"
lesson: "sft-alignment-limitations"
sourceId: 4430
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-1-scalable-alignment-problem/sft-alignment-limitations"
title: "监督微调在对齐方面的局限性"
description: "分析为什么仅监督微调不足以实现复杂对齐目标。"
order: 1
plots: ["plots/4430-0.json"]
sourceHash: "d4b0b1ebd9bee56d4e6dfef8eb56ff8d6ceb66d39632e0b26711de0f3e87f38d"
sourceCorrections: []
---

监督微调 (fine-tuning)（SFT）是对齐 (alignment)大型语言模型的基础技术。主要思路直接明了：向模型提供期望行为、正确回答、有益对话轮次、安全回应的例子（演示），并微调模型使其模仿这些例子。这种方法能有效提升模型遵循指令和遵守基本行为准则的能力。

然而，单独依靠SFT来实现复杂对齐很快会遇到重大阻碍，尤其当我们追求复杂的AI行为时。这些局限性促使开发出更先进的对齐方法，例如宪法AI（CAI）和基于AI反馈的强化学习 (reinforcement learning)（RLAIF）。

### 数据获取瓶颈

为对齐 (alignment)创建高质量SFT数据是劳动密集且成本高昂的。这需要人工标注者（或高度可靠的自动化系统，对于细致的对齐常常不存在）在广泛的潜在输入范围内生成理想回应。

- **覆盖范围：** 大型语言模型的输入空间非常庞大。制作演示以覆盖所有可想到的情境、边缘情况或细微变化（这些地方对齐非常重要）是不切实际的。SFT数据集，即使是大型的，也只代表了可能交互的一小部分。
- **复杂性：** 随着对齐目标变得更复杂，从简单的“不要说脏话”进步到详细伦理考量、谨慎的帮助性或依赖情境的安全性，制作完美演示变得越来越困难和主观。什么构成“理想”回应通常涉及仔细判断和权衡。
- **可扩展性：** 所需数据量常常与对齐任务的复杂性和模型的能力扩展性差。在不同方面实现对齐需要一个庞大的高质量演示语料库，这使得SFT对于尖端模型和有抱负的对齐目标来说成本过高。

考虑随着对齐目标变得更精细而不断增加所需的工作量：



![SFT数据创建工作量与对齐复杂性估计](plots/4430-0.json)



> 随着对齐目标需要更多判断和覆盖复杂情境，工作量明显增加。

### 难以推广原则

SFT主要教模型模仿所提供例子的表面形式。它擅长学习训练数据中存在的风格模式或特定输入-输出映射。然而，它常常难以灌输这些例子背后的潜在*原则*或*意图*。

- **对演示过拟合 (overfitting)：** 模型可能学会复制演示数据的风格、语调或特定措辞，而未能理解其所例示的普遍规则。例如，一个在礼貌拒绝有害请求的例子上训练的模型，当面对训练中未见过的新颖、措辞细致的有害请求时，可能会失败。
- **缺乏抽象推理 (inference)能力：** SFT本身不赋予模型将对齐 (alignment)原则应用于全新情境所需的抽象推理能力。它学习在训练中显示的特定情境下*说什么*，但不一定理解根据“无害”或“诚实”等更广泛的原则，该回应*为什么*是恰当的。

### 处理歧义和情境权衡

对齐 (alignment)常常涉及处理歧义和进行依赖情境的权衡。例如，一个请求可能有害，取决于用户意图或后续影响；或者完全诚实可能与提供帮助或简洁冲突。

SFT基于静态输入-输出对，不适合教模型如何权衡相互冲突的价值或动态解释模糊情境。单个演示通常代表潜在权衡的一种特定解决方案，在未见情境中如何处理不同平衡方面提供有限指导。模型学习的是固定回应，而非灵活决策过程。

### 隐性知识鸿沟

人类演示者在制作理想回应时，依赖大量隐性知识、常识和伦理理解。这种潜在原理很少在演示文本本身中被充分阐明。SFT允许模型模仿输出，但不能直接传递这种隐性推理 (inference)根基。

这种鸿沟导致脆弱性。模型可能正确处理与其训练数据相似的提示，但在略微不同的输入上意外失败，因为它缺乏支撑原始人类回应的更深理解。它尚未学到“是什么”背后的“为什么”。

### 脆弱性和易受攻击性

或许最重要的实际局限之一是所得模型的脆弱性。主要通过SFT对齐 (alignment)的模型，常常仍然容易受到对抗性攻击或“越狱”提示的影响。这些是仔细设计的输入，以规避学到的安全性或帮助性模式。由于模型学到的是表面关联而非原则，超出SFT数据分布的输入可以容易触发不良行为。实现韧性不仅仅需要模仿例子；它需要对齐原则更深的整合，而仅SFT难以提供这一点。

尽管SFT仍然是大型语言模型训练工具包中的重要组成部分，尤其对于初始指令调优和基本行为塑造，但在目标是全面可靠对齐时，其局限性变得明显。与数据可扩展性、原则推广、处理歧义、传递隐性知识以及确保鲁棒性相关的挑战，推动了对超越简单模仿的方法的需求，例如涉及明确原则（CAI）或基于偏好学习（RLHF/RLAIF）的方法。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  这篇来自Anthropic的论文指出监督微调在复杂对齐方面的局限性，并介绍了使用人类偏好进行强化学习的RLHF作为一种可扩展的方法。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  本文介绍了宪法AI，通过AI反馈将模型与原则对齐，以解决人类监督的可扩展性问题和监督微调的不足。
- [Red Teaming Language Models to Reduce Harms: Methods, Limitations, and Recommendations](https://doi.org/10.48550/arXiv.2209.07858) — Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, Jack Clark (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  本文讨论了识别和减轻模型有害行为的方法，说明了语言模型的脆弱性和漏洞。
