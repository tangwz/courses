---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rlhf-foundations-alignment"
lesson: "llm-alignment-problem"
sourceId: 5082
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-1-rlhf-foundations-alignment/llm-alignment-problem"
title: "大型语言模型中的人工智能对齐问题"
description: "理解为何需要将大型语言模型与人类意图和价值观对齐。"
order: 1
plots: []
sourceHash: "73782a5c227ecf1d3de535c9d16055bdfcf710fb1e36d5151401e40411b55645"
sourceCorrections: []
---

大型语言模型（LLMs）在处理和生成类人文本方面表现出卓越的能力。它们通过在海量互联网级别数据集上训练，学习了语法、事实、推理 (inference)能力，甚至编程技能，通过优化一个相对简单的目标来实现：预测序列中的下一个词元 (token)。然而，这种预训练 (pre-training)目标虽然对于构建通用能力非常有效，但其本身并不能保证模型的行为与人类偏好或预期用途保持一致。这一差距正是大型语言模型人工智能对齐 (alignment)问题的核心所在。

在大型语言模型的语境中，“对齐”指的是确保这些模型以“有用、诚实、无害”（常缩写为HHH）的方式行事的难题。

- **有益：** 模型应准确遵循指令，提供信息丰富的回答，并协助用户实现其目标。
- **诚实：** 模型应提供事实准确的信息，避免生成误导性或虚假内容（常称作“幻觉 (hallucination)”）。它还应准确呈现自身能力和局限。
- **无害：** 模型应拒绝生成有毒、偏见、歧视性或其它有害内容。它应避免宣传非法行为或产生不安全输出。

预测下一个词元的预训练目标与这些HHH标准无关。模型可能会熟练地预测文本序列，而这些序列不幸地也可能包含偏见、不真实或有害内容，因为训练数据中存在此类模式。模型学习模仿其训练语料库的统计特性，包括所有不足之处。

### 为何预训练 (pre-training)不足以解决问题

核心问题源于预训练过程中使用的*代理目标*（预测下一个词元 (token)）与我们所期望的*真正目标*（对齐 (alignment)行为）之间的差异。

> 预训练过程根据海量文本语料库中的模式来优化大型语言模型。这与部署时产生对齐的（有益、诚实、无害）输出的预期目标有显著差异。

这种目标不匹配体现在几个常见的未对齐问题上：

1. **生成不良内容：** 模型可能会复现训练数据中存在的偏见（种族、性别、政治），生成有害语言，或者在接收到适当（有时甚至不适当）提示时提供有害活动的指示。
2. **幻觉 (hallucination)：** 模型会自信地陈述虚假信息或捏造听起来合理但不以其训练数据或现实为基础的信息。发生这种情况是因为模型优先生成连贯文本而非事实准确性。
3. **未能遵循指令：** 尽管预训练模型理解语言，但它们可能无法精确遵循复杂的多步骤指令，或未能遵守指定的限制（例如，角色、格式、长度）。
4. **奉承：** 模型可能会学习生成仅仅附和用户陈述的意见或信仰的回复，即使这些意见或信仰不正确，而不是提供客观信息。如果此类回复在训练数据的某些部分统计上常见，则这种行为可能得到强化。
5. **脆弱性和被利用：** 未对齐可能产生漏洞。对抗性提示可能会诱骗模型绕过其安全协议或生成意外输出。

举一个简单的例子：如果用户问“我怎样才能让邻居的狗停止吠叫？”，一个仅基于下一个词元预测训练的未对齐模型可能会检索并生成其大量训练数据中发现的有害或非法建议，仅仅因为这些序列在统计上是合理的。然而，一个对齐的模型应该识别出潜在危害，并拒绝请求，或只提供安全、合法的替代方案（例如，“与你的邻居交谈”，“使用降噪耳机”）。

因此，仅仅通过在更多数据上训练更大的模型并不能自动解决对齐问题。实际上，如果不仔细引导，能力的增强有时反而会放大未对齐问题。我们需要特定的技术来引导模型行为，使其在初步预训练阶段后符合预期的人类价值观和意图。这为监督微调 (fine-tuning)（SFT）等方法，以及更强效的、直接将人类偏好纳入训练循环的“基于人类反馈的强化学习 (reinforcement learning)”（RLHF）提供了前提。理解这个基本对齐挑战是理解为何RLHF在开发更安全、更有用的大型语言模型中成为一项重要技术的第一步。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  这项工作提出了一种方法，通过使用人工智能模型根据一系列原则评估响应，实现LLM的无害性对齐，为某些对齐目标提供了直接人类反馈的替代方案。
- [Artificial Intelligence, Safety and Existential Risk: A Systematic Review](https://link.springer.com/article/10.1007/s11229-019-02201-6) — Iyad Gabriel (2020)
  Journal: Synthese; Publisher: Springer Netherlands; Volume: 197; Pages: 1261-1282; DOI: [10.1007/s11229-019-02201-6](https://doi.org/10.1007/s11229-019-02201-6)
  这篇系统综述提供了关于人工智能安全和生存风险的广泛学术视角，为大型语言模型对齐所处理的通用人工智能对齐问题奠定了概念基础。
- [Ethical and Social Risks of Harmful Language Models](https://arxiv.org/abs/2112.04359) — Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geoffrey Irving, Iason Gabriel (2021)
  Journal: arXiv preprint arXiv:2112.04359; DOI: [10.48550/arXiv.2112.04359](https://doi.org/10.48550/arXiv.2112.04359)
  这篇论文全面讨论了与大型语言模型相关的伦理和社会风险，详细阐述了偏见、毒性和错误信息等各种形式的失调。
