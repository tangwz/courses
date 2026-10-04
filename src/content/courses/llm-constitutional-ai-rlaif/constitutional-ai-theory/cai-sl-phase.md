---
course: "llm-constitutional-ai-rlaif"
chapter: "constitutional-ai-theory"
lesson: "cai-sl-phase"
sourceId: 4449
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-2-constitutional-ai-theory/cai-sl-phase"
title: "监督学习阶段（批评与修改）"
description: "基于章程的AI自我批评与修改过程的详细解析。"
order: 3
plots: []
sourceHash: "c29ffecc4a6c5f485fe92fb42aba75fe09b02a362dbd857441eb9eff6d904c9b"
sourceCorrections: []
---

监督学习 (supervised learning)（SL）阶段是宪法AI的初始训练环节，其中设计有效章程的准则得到实际应用。此阶段通过运用AI本身生成期望行为的实例，从而将章程付诸实践，为微调 (fine-tuning)语言模型以达成对齐 (alignment)提供数据集，在任何强化学习 (reinforcement learning)步骤之前完成此项工作。

这里的中心目标是将章程的抽象规则转化为具体的训练信号。CAI不再仅仅依靠人工标注者来标记 (token)偏好输出，而是使用AI系统，依据所提供的章程准则，批评并修改模型自身的回复。这个过程构成了一种可扩展的机制，用于生成对齐数据。

这个SL阶段通常分为两个主要步骤，并常以迭代方式执行：

1. **批评生成：** 在初始模型回复中辨识出违反章程准则之处。
2. **修改生成：** 重写初始回复，以修正已辨识的违规之处，并更好地遵从章程。

让我们详细分析每个步骤。

### 批评生成

这一过程始于提示初始的、预先对齐 (alignment)的大语言模型（LLM）为一个给定输入提示生成回复。这个初始回复可能有所助益，但可能违反章程中列出的一项或多项准则（例如，规避回答、生成有害内容、表达不当偏见）。

- **输入：** 批评步骤的主要输入为：
  - LLM生成的初始回复。
  - 章程中回复可能违反的相关准则。
- **机制：** 一个AI系统，通常称为“批评器”，会专门从所提供的章程准则的角度分析初始回复。这个批评器可以是用于生成回复的同一个LLM，但通过精心设计的元提示（meta-prompt）调用；也可以是为这项任务微调 (fine-tuning)的独立模型。元提示通常会指示AI进行以下操作：
  1. 扮演一个AI助手的角色，依据一套特定规则评估回复。
  2. 辨识出给定回复违反了哪些规则（如果有的话）。
  3. 解释回复*为何*违反了已辨识的规则。
     此处少样本提示（Few-shot prompting）通常有效，提示中包含回复、相关准则和相应期望批评的例子。
- **输出：** 输出是一个文本批评。此批评明确指出已辨识的章程违规之处并提供理由。例如，如果某项准则规定AI应拒绝生成非法指令，而初始回复提供了此类指令，批评可能会是：“该回复提供了非法行为的指令，违反了关于安全和合法性的准则7。”

### 修改生成

批评生成后，下一步是修改初始回复以处理已辨识的问题。

- **输入：** 修改步骤的输入为：
  - 原始的、有问题的回复。
  - 上一步生成的批评。
- **机制：** 一个AI系统，“修改器”（同样，可能是以不同方式提示的同一LLM，或一个独立模型），负责重写原始回复。此步骤的元提示指示AI专门修改初始回复，以处理批评中提出的问题，确保输出与章程对齐 (alignment)，同时理想情况下保持对原始用户提示的助益性和相关性。
- **输出：** 结果是修改后的回复。这个修改后的回复代表了在章程上优于初始输出的替代方案。例如，在上述批评之后，修改后的回复可能是：“我无法提供非法活动的指示。但是，我可以讨论其法律影响或提供适用情况下更安全的替代方案的信息。”

### 创建监督微调 (fine-tuning)数据集

这种批评与修改循环的优势在于它能够自动生成用于监督微调（SFT）的训练数据。每次循环会得到一个配对：*初始提示*和*修改后的、与章程对齐 (alignment)的回复*。


$$
\text{SFT训练配对} = (\text{输入提示}, \text{修改后的回复})
$$


通过在多样化的提示集上生成许多此类配对，可以构建一个可观的数据集。该数据集以输入-输出实例的形式体现了章程准则。原始LLM随后会基于此数据集进行微调。SFT的目标是教导模型在收到相应提示时，直接产生与*修改后*输出类似的回复，从而有效地将批评与修改过程中展现的章程限制内化。

> CAI监督学习 (supervised learning)阶段的流程。初始回复会依据章程进行批评，然后根据批评进行修改，从而生成用于监督微调的训练配对。

这种由明确章程引导的自我修正机制，使得对齐过程的规模化能够超越单靠人工标注所能实现的范围。在这些AI生成数据上微调得到的模型，为构建更符合章程的系统提供了重要依据，并可能已为通过RLAIF等强化学习 (reinforcement learning)技术进行进一步完善做好准备，我们将在后续章节讨论这些技术。这个阶段的质量在很大程度上取决于章程的清晰度，以及用于有效引导批评和修改模型的提示策略的成熟程度。

## 参考资料

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark (2023)
  Journal: arXiv preprint arXiv:2303.17651; DOI: [10.48550/arXiv.2303.17651](https://doi.org/10.48550/arXiv.2303.17651)
  本文提出了一种LLM迭代自我修正的通用框架，为宪法式AI中使用的批改和修订循环提供了更广泛的概念性理解。
