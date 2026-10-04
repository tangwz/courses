---
course: "how-to-build-a-large-language-model"
chapter: "supervised-fine-tuning-alignment"
lesson: "goals-llm-alignment"
sourceId: 6077
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/goals-llm-alignment"
title: "大型语言模型对齐的目标"
description: "了解对齐大型语言模型超越预训练表现的目的。"
order: 1
plots: []
sourceHash: "c6f0c22405317dc982aa24a821d70fe87c1e28da324ca8336f58994f3f3af5e8"
sourceCorrections: []
---

虽然预训练 (pre-training)的大型语言模型 (LLM) 展现出色的能力，能够基于从大量数据集中学习到的模式来理解和生成人类语言，但它们并不会自动地以对特定应用始终有用、真实或安全的方式行事。它们的训练目标通常侧重于预测序列中的下一个标记 (token)，即 $P(\text{标记}_{i+1} | \text{标记}_1, ..., \text{标记}_i)$，这最大化了预训练语料库上的似然度，但并不会直接优化以遵循用户指令或符合人类价值观。

对齐 (alignment)是调整预训练大型语言模型以更好地匹配人类意图和偏好的过程。它旨在引导模型的强大生成能力朝着期望的行为。监督式微调 (fine-tuning) (SFT)，作为本章的侧重点，是此过程中的重要一步。对齐的主要目标，也是SFT开始处理的，常分为三个大类，有时被称为“HHH”标准：有用性、诚实性、无害性。

### 有用性

这或许是SFT处理的最直接的目标。一个有用的模型应该理解并准确遵循提示中呈现的用户指令。它应该有效地执行所请求的任务，无论是回答问题、总结文本、编写代码、翻译语言，还是以特定的对话风格交流。

考虑一个被问及“解释梯度下降 (gradient descent)的原理”的预训练 (pre-training)模型。

- **无用（但预训练模型可能有的行为）：** 模型可能会生成笼统地定义微积分中的梯度，或者列出提及梯度下降的论文，而没有清晰地解释算法本身。
- **有用（对齐 (alignment)后的行为）：** 模型提供清晰、循序渐进的解释，针对可能的语境（例如，机器学习 (machine learning)）进行调整，可能包含类比或简单的例子。

SFT 通过使模型接触大量由高质量、有用响应配对的提示示例来做到这一点。微调 (fine-tuning)过程调整模型的参数 (parameter)，以增加为类似提示生成此类有用响应的概率。这涉及最小化模型生成响应与SFT数据集中目标有用响应之间的损失（例如，交叉熵）。

> 对齐通过SFT等技术，将通用预训练模型转变为表现出期望行为的模型。

### 诚实性 (真实性)

一个对齐 (alignment)的模型应该力求准确，并避免生成虚假信息，通常被称为“幻觉 (hallucination)”。虽然预训练 (pre-training)使模型接触到事实知识，但其生成性质意味着它很容易构建听起来合理但不正确的陈述。诚实性意味着：

- **事实准确性：** 在可能时提供正确的信息。
- **避免捏造：** 不虚构事实、来源或细节。
- **表达不确定性：** 当模型不知道答案或信息不明确时，应表明出来，而不是自信地猜测。

SFT 有助于提高诚实性，通过包含模型正确回答事实问题或明确说明其局限性的示例。然而，确保深层的事实性和校准的不确定性通常需要更先进的技术，例如整合检索机制或使用强化学习 (reinforcement learning)（如RLHF，在第26章中讨论）以惩罚人类反馈识别出的不真实输出。

### 无害性

此目标侧重于阻止模型生成有害、不道德、带有偏见、有毒或促进非法活动的输出。预训练 (pre-training)数据不可避免地包含互联网和数字化文本中存在的偏见和有害内容。一个未对齐 (alignment)的模型可能会轻易地再现或放大这些问题。无害性要求模型：

- **拒绝不当请求：** 拒绝生成属于有害类别的任何内容。
- **避免偏见：** 减少生成刻板或偏见的内容。
- **保持安全：** 不提供危险活动的指令。

SFT 通过包含模型拒绝有害请求或提供安全、中性响应的示例，在此方面发挥作用。精心策划的SFT数据集过滤掉不理想的示例，并明确呈现安全拒绝。类似于诚实性，在多样化和对抗性输入下实现无害性具有挑战性，并且通常会从后续的RLHF中获得显著益处，在RLHF中，模型根据人类判断被训练以偏好安全输出。

总而言之，对齐旨在使大型语言模型不仅有能力，而且在各种应用中成为有益且安全的伙伴。SFT 作为一个起点，主要增强了有用性和指令遵循能力，同时通过提供期望模型输出的具体示例，也开始了灌输诚实性和无害性的过程。这些目标指导SFT数据集的创建和对齐模型的评估，确保它们从仅仅预测文本转向生成真正有用和负责任的响应。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  介绍了通过人类反馈监督微调（SFT）和强化学习（RLHF）将大型语言模型与人类意图对齐的方法，侧重于有用性和无害性。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  探讨了大型语言模型无害性对齐的方法，展示了除人类反馈之外但仍能定义和追求对齐目标的替代方法。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个全面的语言模型评估框架，包括真实性、安全性和公平性指标，这些指标与大型语言模型对齐的目标直接相关。
