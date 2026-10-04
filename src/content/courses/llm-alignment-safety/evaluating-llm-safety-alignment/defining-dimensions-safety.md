---
course: "llm-alignment-safety"
chapter: "evaluating-llm-safety-alignment"
lesson: "defining-dimensions-safety"
sourceId: 3165
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-4-evaluating-llm-safety-alignment/defining-dimensions-safety"
title: "定义安全方面：无害性、诚实性、有用性"
description: "定义并实施大型语言模型安全的各个方面，例如无害性、诚实性和有用性。"
order: 1
plots: []
sourceHash: "7ed292f3369b2aa818f98be2d4535b85113f1e41e5aeea46cafbf8493b988c75"
sourceCorrections: []
---

从构建对齐 (alignment)模型转向评估它们时，我们需要具体的标准来判断它们的行为。仅仅声称一个模型是“对齐的”或“安全的”是不够的，除非明确这些术语在实践中意味着什么。目标是将这些高层面的想法转化为可衡量的方面，以反映期望的表现和不理想的失败模式。

一个普遍采用的评估框架，特别受到Anthropic强调并在业界具有影响力的，围绕着三个主要原则：有用性、诚实性和无害性（通常简称HHH）。尽管存在其他表述方式，这三个方面为分析和评估大型语言模型的复杂行为提供了一个起点。它们既是训练期间对齐技术的指导，也是部署后评估规程的指引。

让我们详细审视每个方面：

### 有用性

有用性直接关系到模型理解并成功满足用户请求或隐含意图的能力。一个理想的有用模型应：

- **遵守指令：** 在其能力和安全限制范围内，准确执行提示中给出的明确指令。
- **理解意图：** 不止于字面指令，而要理解用户查询的潜在目标，并提供相关、有用的信息或有效完成预期任务。
- **保持相关和简洁：** 提供与请求相关的信息，避免不必要的冗长或无关的旁枝末节。
- **易用且格式良好：** 以清晰、连贯、易懂的方式呈现信息。这可能包括适当地使用列表或代码块等格式。

实现有用性并非总是简单直接。它要求模型解释歧义，做出合理假设，有时甚至澄清用户意图。过分优先考虑有用性而不顾及其他方面可能导致问题；例如，模型可能试图通过回答其实际上不知道答案的问题（违反诚实性）或满足有害请求（违反无害性）来表现出有用。

### 诚实性

大型语言模型背景下的诚实性主要涉及事实准确性和真实呈现。它无关道德意图，而是关于所提供信息的可靠性。重要的方面包括：

- **事实准确性：** 提供正确并与既有知识相符的信息。由于大型语言模型训练数据的特性以及生成听似合理但实为虚假信息（常被称为“幻觉 (hallucination)”）的可能，这是一个很大的难题。
- **避免编造：** 在不确定时，避免编造信息、来源或细节。
- **表示不确定性：** 在缺乏足够信息或信心提供明确答案时，适当地表明。这比简单地拒绝回答更具体；它涉及校准后的不确定性表达。
- **避免误导：** 不将意见或潜在偏见的观点呈现为客观事实。

评估诚实性通常涉及将模型输出与真实数据源进行比较，或使用专门设计用于测试事实回忆和抵抗生成错误信息的基准（如TruthfulQA，我们将在后面讨论）。一个重要难题是区分真正的知识空白和融入到其他正确陈述中的细微编造。

### 无害性

无害性侧重于防止模型生成可能对个人或群体造成损害的输出。这是一个广泛的分类，包含各种不理想的内容类型：

- **避免有害或辱骂性语言：** 防止仇恨言论、骚扰、威胁或歧视性言论。
- **拒绝危险内容：** 不提供非法行为的指导、不宣扬暴力，或不给出不安全的建议（例如，有害的医疗或金融指导）。
- **减轻偏见：** 避免延续有害刻板印象或对受保护群体进行不公平歧视。
- **保护隐私：** 不泄露在训练或交互过程中遇到的敏感个人信息。

准确定义无害性可能很复杂且依赖于语境。社会规范不同，表达意见和造成损害之间的界限可能模糊。模型通常采用安全过滤器和拒绝机制来阻止有害请求，但攻击者不断寻找绕过这些防御的方法（如第5章所述）。过于严格的无害性过滤器也可能通过拒绝被误解为有害的良性请求而妨碍有用性（即“斯肯索普问题”或相关情况）。

### 方面间的关系

了解这三个方面并非完全独立非常重要。通常存在张力和权衡：

- **有用性与无害性：** 如果一个潜在有用的请求触及敏感话题，模型可能会拒绝，从而将无害性置于有用性之上。反之，过于追求有用性可能会导致模型回应有问题的内容。
- **有用性与诚实性：** 一个力求最大程度有用的模型可能会倾向于猜测或编造答案，而非承认不确定，从而牺牲诚实性。
- **诚实性与无害性：** 在某些情况下，说出事实但可能有害的真相可能与无害性目标相冲突（例如，泄露敏感信息，尽管这与隐私问题有重叠）。校准拒绝机制需要在提供关于局限性的准确信息与避免生成有害内容之间取得平衡。

因此，评估大型语言模型的安全性和对齐 (alignment)度并非是孤立地最大化每个方面，而是要实现一个与特定应用目标和伦理考量相符的、可接受的平衡。本章后续部分将探讨用于衡量这些方面表现并处理其固有权衡的自动化基准、人工评估和红队技术。理解这些定义为有效应用这些评估方法提供了依据。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  提出了帮助性、诚实性和无害性 (HHH) 框架作为大型语言模型的主要对齐目标，并详细介绍了使用人类反馈强化学习 (RLHF) 的训练方法。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了一种通过根据一系列指导原则（包括 HHH 维度）进行自我批评来训练模型使其无害和有帮助的方法。
- [TruthfulQA: Measuring Models' Honesty on Hard Questions](https://arxiv.org/abs/2109.07958) — Stephanie Lin, Jacob Hilton, Owain Evans (2021)
  Journal: ACL 2022 (main conference); DOI: [10.48550/arXiv.2109.07958](https://doi.org/10.48550/arXiv.2109.07958)
  提出了一个基准数据集和评估方法，旨在衡量语言模型的事实准确性和诚实性，直接解决了“诚实性”维度。
