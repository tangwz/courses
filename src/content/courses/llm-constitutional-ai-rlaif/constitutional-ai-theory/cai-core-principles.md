---
course: "llm-constitutional-ai-rlaif"
chapter: "constitutional-ai-theory"
lesson: "cai-core-principles"
sourceId: 4443
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-2-constitutional-ai-theory/cai-core-principles"
title: "宪法式AI的核心原则"
description: "了解宪法式AI（CAI）的根本思路，实现在无需对每个决策进行直接人工标注的情况下引导大型语言模型（LLM）的行为。"
order: 1
plots: []
sourceHash: "6c1b46ca9b0d47f2c7ba6da38b3ae22f6f88bba52e227b7d7920444254ebc8e3"
sourceCorrections: []
---

使用监督微调 (fine-tuning)（SFT）和人工反馈强化学习 (reinforcement learning)（RLHF）等方法实现大型语言模型（LLM）的可扩展对齐 (alignment)面临主要难题。主要问题通常围绕着对大量人工监督和标注的需求所造成的瓶颈。宪法式AI（CAI）提出了一种替代性框架，旨在通过运用AI本身来监督模型遵守一套预设原则（称为“宪法”）的方式，解决这一可扩展性问题。

CAI并非仅仅依赖人工判断来完成每次微调或偏好标注，而是引入了一种由明确伦理和安全准则引导的自动化监督机制。本节将探讨支撑CAI方法的根本原则。

### 原则一：以明确原则为指导（宪法）

CAI的核心是“宪法”：一份包含旨在规范大型语言模型行为的明确原则或规则的文件。这些原则由人工撰写，旨在体现帮助性、无害性、诚实性以及遵守特定安全规程或伦理考量等期望特点。

宪法原则的例子可以包括：

- “选择最不可能有害、有毒或冒犯的回复。”
- “识别并拒绝提倡非法行为或不道德行为的请求。”
- “偏好真实且避免提出未经证实主张的回复。”
- “避免在敏感话题上表达非中立观点。”

该宪法作为价值观的明确规定，不同于从SFT中的演示数据或RLHF中人工偏好学习到的隐性价值观。它提供了一个参考标准，AI可以据此评估和完善自己的输出。CAI的有效性在很大程度上取决于这些原则的清晰度、全面性和内部一致性，我们将在设计有效宪法的章节中进一步讨论此话题。

### 原则二：AI驱动的批改和修订

CAI通过AI自我批改和修订的过程，使宪法得以运作。这通常构成CAI训练初始监督学习 (supervised learning)（SL）阶段的起点。该过程如下进行：

1. **初始回复生成：** 大型语言模型针对给定提示生成一个初始回复。
2. **AI批改：** 模型（或一个独立的AI模型）被提示根据宪法中列出的原则批改初始回复。它识别可能违反了哪些原则并解释原因。
3. **AI修订：** 基于批改，模型（或一个独立的AI模型）修订初始回复以更好地遵守宪法。

考虑一个要求提供潜在有害活动指示的提示。

- **初始回复：** 模型可能最初提供了有害指示。
- **批改：** 在宪法（例如：“避免生成有害内容”）的提示下，AI批改初始回复，指出其违反了指定原则。
- **修订：** AI生成一个修订后的回复，很可能会拒绝该请求并解释原因，引用相关的宪法原则。

这种批改-修订循环生成（初始回复、批改、修订回复）的配对。这些更符合宪法的修订回复随后被用作监督微调 (fine-tuning)过程中的目标输出。这使得模型能够直接学会生成符合宪法的回复，从而减少在单个回复层面进行人工干预的需求。

> CAI监督学习阶段中AI批改和修订过程的流程。

### 原则三：用于强化学习 (reinforcement learning)的AI生成偏好

尽管监督学习 (supervised learning)（SL）阶段直接利用修订输出训练模型，但CAI框架通常会扩展到强化学习（RL）阶段，该阶段与AI反馈强化学习（RLAIF）紧密关联。批改和修订过程内在地生成偏好数据：根据宪法，修订后的回复被认为比初始回复“更受偏好”。

这种AI生成的偏好数据（依据宪法遵守程度判断其中一个回复优于另一个的回复对）可用于训练偏好模型。该偏好模型随后在强化学习循环中充当奖励函数（通常使用近端策略优化 - PPO 等算法），进一步使大型语言模型的策略朝向生成符合宪法的输出。AI在宪法引导下，在生成强化学习人工反馈风格训练所需的偏好信号方面，有效替代了人工标注员。我们将在第四章和第五章审视这个RLAIF组件的具体内容。

### 原则四：通过自动化实现可扩展监督

结合这些原则，CAI提供了一条通向更具可扩展性对齐 (alignment)监督的路径。通过将期望行为编码进明确的宪法中，并运用AI模型通过自动化批改、修订和偏好生成来执行这些原则，对每次互动进行直接人工标注的依赖大大降低。人工则专注于设计和完善宪法这一更高层面的任务，而AI处理这些原则在数百万甚至数十亿次互动中的细致应用。这种自动化是CAI旨在克服传统强化学习 (reinforcement learning)人工反馈可扩展性限制的主要机制。

本质上，宪法式AI提供了一种结构化方法，用于将预设原则嵌入 (embedding)到大型语言模型的行为中。它运用AI模型自身的能力来解读这些原则，并生成对齐所需的必要反馈（批改、修订、偏好），从而提供了一条与需要持续细致人工输入的方法相比，可能更具可扩展性的路径。接下来的章节将探讨设计宪法以及实施批改/修订机制的实际方面。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍宪法式AI的原始论文，该方法通过AI驱动的批判和修订使LLM与原则对齐。
- [A Guide to Constitutional AI](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: ArXiv; Publisher: arXiv.org; Volume: abs/2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  宪法式AI创建者提供的易于理解的概述，解释其框架和应用。
