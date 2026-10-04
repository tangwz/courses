---
course: "synthetic-data-llm-pretrain-finetune"
chapter: "foundations-synthetic-data-llm"
lesson: "synthetic-data-pretraining-finetuning"
sourceId: 6117
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-1-foundations-synthetic-data-llm/synthetic-data-pretraining-finetuning"
title: "合成数据在预训练和微调中的作用"
description: "了解合成数据如何专门用于LLM预训练和微调阶段。"
order: 5
plots: []
sourceHash: "c4359925d23772e6617338c6be4fe6fcc857fcfa4e9bb9f925f38223541a58d7"
sourceCorrections: []
---

大型语言模型（LLM）会经历不同的训练阶段来形成自身能力。合成数据因其适应性和可控性，在最初的预训练 (pre-training)阶段和随后的微调 (fine-tuning)阶段都带来显著益处。弄清它在各个阶段的具体贡献，能够促进人工生成数据集的有效使用。

### 合成数据在LLM预训练 (pre-training)中

预训练是LLM学习通用语言理解、语法、常识推理 (inference)和知识的基础阶段。此阶段通常需要接触海量文本数据，规模常达到数TB。

**合成数据的作用：**

1. **增加数据量：** 合成数据在预训练中的主要用途之一是扩充现有语料库，或在某些情况下创建全新的语料库。对于许多语言或专业范围，大型、高质量的真实数据集根本不存在。合成数据生成技术可以生成大量文本，有助于满足预训练的“数据需求”。例如，如果你正在为一个低资源语言预训练模型，基于现有语言规则合成生成的文本，或从高资源语言翻译而来的文本，可以构成初始训练语料库的很大一部分。
2. **提高数据多样性：** 数据集，即使是大型的，也可能存在固有的偏见或覆盖不足。合成数据可以设计来引入可能未被充分体现的特定语言结构、写作风格或知识范围。这有助于预训练模型形成更全面的理解，并避免形成狭窄真实数据集中存在的强烈偏见。设想一个模型需要理解古体文本风格；如果这种风格在现代网络抓取中很少见，就可以生成合成数据来模仿它们。
3. **注入受控知识：** 预训练可以通过合成生成内容来引导，这些内容系统地教授模型特定事实、推理模式，甚至初步的编码能力。虽然大部分预训练依赖于从非结构化文本中偶然习得，但合成数据允许采用更直接的方式来灌输某些基础技能。例如，生成简单的事实陈述或逻辑三段论，可以帮助模型从零开始培养这些能力。
4. **启动特定领域模型：** 当目标是为医学或法律等专业范围预训练LLM时，足够的特定领域真实数据可能稀缺或受隐私法规保护。合成数据，可能由专家生成，或由在较小领域数据集上微调 (fine-tuning)过的现有模型生成，可以创建一个更大但人造的语料库，以便在该领域开始预训练，然后再用有限的真实数据进行进一步完善。

尽管它功能强大，但预训练中使用合成数据需要谨慎。生成的数据必须具备足够的质量、复杂性和多样性才能发挥作用。低质量或过于简单的合成数据可能无法对模型的学习产生有意义的帮助，甚至可能引入不想要的特征。

### 合成数据在LLM微调 (fine-tuning)中

微调是获取预训练 (pre-training)的基础模型，并使其适应特定任务、风格或行为。与预训练相比，此阶段通常使用更小、更精选的数据集。合成数据在各种微调方法中已变得尤其突出和有效。

**合成数据的作用：**

1. **指令遵循：** 一项重要的应用是为“指令微调”（IFT）生成指令-响应对。这个过程教授模型理解和遵循人类指令。以合成方式创建多样化且高质量的`(instruction, output)`对，通常比手动标注更具扩展性且成本更低。例如，要使模型擅长总结，可以生成数千个示例，例如 `{"instruction": "将这篇文章总结成三句话。", "input": "<文章文本>", "output": "<总结>"}`。
2. **任务专用适应：** 对于许多专业任务，例如用一种新编程语言生成代码、回答有关小众产品的问题，或扮演非常特定的人物角色，真实训练数据通常很少或根本没有。合成数据可以精心制作，为这些狭窄任务提供大量示例，使模型能够在原本无法表现良好的地方做得好。“自指令”等方法涉及使用LLM，根据少量种子示例生成新的指令和相应的输出。
3. **对齐 (alignment)与安全：** 确保LLM行为安全、符合道德，并与人类偏好保持一致，是一个主要关注点。合成数据在此处发挥重要作用。对于人类反馈强化学习 (reinforcement learning)（RLHF）或其变体（如AI反馈强化学习（RLAIF））等方法，合成数据可用于生成提示、多个可能的响应，甚至偏好标签（例如，“对于此提示，响应A优于响应B”）。这有助于引导模型远离有害、有偏见或不真实的输出。
4. **提高少样本或零样本表现：** LLM通常被期望通过极少量示例（少样本）或完全没有示例（零样本）来完成任务。合成数据可以生成以覆盖更广泛的潜在任务变体或表达方式，隐含地训练模型，使其在数据量有限的环境中，即使面对新颖的提示也能更好地泛化。
5. **控制风格与人物角色：** 如果你需要LLM持续采用特定写作风格（例如，正式、随意、幽默）或体现特定人物角色（例如，乐于助人的助教、机智的领域专家），合成数据可以提供所需示例。通过在持续展现所需特征的数据上进行微调，模型学会模仿它们。

合成数据的质量和相关性在微调中比在预训练中更具决定性。因为微调数据集更小，每个示例都产生相对更大的影响。设计不当的指令、错误的输出，或合成微调数据缺乏多样性，可能导致模型表现不佳、产生幻觉 (hallucination)或泛化失败。

下图展示了合成数据如何融入LLM开发生命周期的预训练和微调阶段，补充真实数据源。

> 合成数据融入LLM预训练和微调阶段，突出其与真实数据并存的独特作用。

本质上，合成数据是一种多功能工具。在预训练中，它常解决了对庞大数据量和广度的需求。在微调中，它转向提供有针对性、高质量的示例，以塑造特定模型行为和能力。随着本课程的进展，我们将考察生成这些不同类型合成数据的方法，以及如何在实践中有效应用它们。

## 参考资料

- [Self-Instruct: Aligning LLMs with Your Own Instructions](https://arxiv.org/abs/2212.10560) — Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi (2022)
  Journal: ACL 2023; DOI: [10.48550/arXiv.2212.10560](https://doi.org/10.48550/arXiv.2212.10560)
  描述了一种通过让LLM自身生成高质量指令遵循数据，从而对模型进行指令微调的方法。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  提出了一种通过AI反馈（RLAIF）生成偏好数据，使LLM与人类安全和无害偏好对齐的方法。
