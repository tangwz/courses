---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "advanced-rlhf-techniques"
lesson: "reinforcement-learning-ai-feedback-rlaif"
sourceId: 5188
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-6-advanced-rlhf-techniques/reinforcement-learning-ai-feedback-rlaif"
title: "基于AI反馈的强化学习 (RLAIF)"
description: "使用AI模型而非人类提供偏好（宪法AI）。"
order: 2
plots: []
sourceHash: "28f56c3c8d0a618ecfda672ddacacba2d87f8769e2258bf97bcee31e0f37b20f"
sourceCorrections: []
---

尽管收集人类偏好数据是标准RLHF的根本，但这一过程可能消耗大量资源、耗时且难以扩展。生成数百万高质量的人类比较数据需要对标注平台、质量控制和人力时间投入大量资金。基于AI反馈的强化学习 (reinforcement learning)（RLAIF）提供了一种替代方法，它用另一个AI模型（通常是能力很强的大型语言模型LLM）来代替人类标注员，以生成偏好标签。

其核心思想是运用一个现有、可能非常强大的语言模型来评估正在训练的模型的输出。我们不再询问人类“哪个回应更好？”，而是指派一个AI模型根据预定义的一套规则、准则或“宪法”来做出判断。

### RLAIF 工作流程

RLAIF过程修改了标准RLHF流程的数据收集阶段：

1. **定义指导原则：** 确立一套规则或原则（“宪法”），明确目标模型行为的期望特征（例如，“乐于助人”、“避免有害内容”、“清晰解释推理 (inference)”）。这些原则指导AI标注员。
2. **生成回应：** 如同标准RLHF，为给定的提示$x$从当前策略模型生成回应对$(y_1, y_2)$。
3. **AI偏好标注：** 将提示$x$和回应对$(y_1, y_2)$输入到一个独立且有能力的AI模型（“AI标注员”或“偏好模型”）中。这个模型通常会根据预定义的原则，被提示来判断哪个回应（$y_1$或$y_2$）根据宪法更好。输出是一个AI生成的偏好标签，指示$y_w$（胜者）和$y_l$（败者）。
4. **奖励模型训练：** 使用上一步生成的AI偏好数据集 $\{(x, y_w, y_l)\}$ 训练一个奖励模型（RM）。训练目标（例如，使用Bradley-Terry模型）与标准RLHF保持一致，旨在为偏好回应分配更高的分数：$r_\theta(x, y_w) > r_\theta(x, y_l)$。
5. **强化学习 (reinforcement learning)微调 (fine-tuning)：** 使用PPO等强化学习算法对目标大型语言模型进行微调，根据AI训练的奖励模型$r_\theta$和KL散度约束进行优化，如同在标准RLHF流程中一样。

> RLAIF工作流程的比较，展示了AI驱动的偏好生成如何输入到标准奖励模型训练和强化学习微调阶段。“宪法”指导“AI标注员”。

### 宪法AI：一个主要示例

Anthropic公司的宪法AI是RLAIF理念的一个显著实现。它使用一套书面原则（宪法）来指导AI驱动的反馈以实现模型对齐 (alignment)，明确目的是减少对人类直接标注有害内容方面的依赖。

该过程通常包括：

1. **宪法起草：** 定义最终模型应遵循的原则列表（例如，来自《联合国人权宣言》或自定义规则）。
2. **AI批评和修订（监督阶段）：** 模型被提示生成回应。然后，模型的另一个实例（充当批评者）会根据宪法原则被提示，以批评回应并重写它，使其更好地符合宪法。这会生成用于监督微调 (fine-tuning)的数据，直接提升模型遵循原则的能力。
3. **AI偏好生成（强化学习 (reinforcement learning)阶段）：** 与一般的RLAIF工作流程类似，会生成回应对。AI标注模型根据其对宪法的遵循程度进行比较，生成偏好数据$(x, y_w, y_l)$。
4. **奖励模型训练和RLHF：** 奖励模型根据这些AI生成的偏好进行训练，目标模型再针对这个奖励模型使用强化学习进行进一步微调。

宪法AI展示了RLAIF如何将复杂的行为准则植入到大型语言模型中，而无需人类针对数百万个示例逐一明确评估输出是否符合每项原则。

### RLAIF的优点

- **可扩展性：** AI标注员生成偏好数据的速度比人类标注员快得多，成本也可能更低，从而能够获得更大的偏好数据集。
- **一致性：** 如果宪法定义良好且AI标注员能力合格，生成的偏好可能比来自不同人类标注员（具有不同解释）的偏好更一致。
- **定向对齐 (alignment)：** RLAIF可以有效地将模型对齐到非常具体或复杂的原则，而这些原则对人类来说，在大规模评估时可能既乏味又困难。

### 挑战与考量

- **AI反馈的质量：** 整个过程严重依赖AI标注员的能力和对齐 (alignment)程度。如果AI标注员误解了宪法、表现出偏见，或未能理解原则背后的意图，这些缺陷将直接被编码到奖励模型中，进而影响最终的策略模型。AI反馈的质量是一个重要的影响因素。
- **宪法工程：** 制定一部有效的宪法并非易事。原则必须全面、明确、一致，并且能够抵御正在训练的模型进行的对抗性解释或“规避”。定义不清的原则可能导致模型出现意料之外或不理想的行为。
- **对齐成本：** RLAIF是否真的解决了对齐问题，还是仅仅转移了问题？我们现在将目标模型对齐到AI标注员，而AI标注员本身又对齐到一部书面宪法。确保AI标注员*忠实地*代表预定的人类价值观，而不仅仅是宪法的字面文本，仍然是一项挑战。存在将“AI对价值观的解释”编码进去而非价值观本身的风险。
- **模型要求：** RLAIF通常需要使用功能强大的前沿模型作为AI标注员，这对于所有团队或应用来说可能不切实际。RLAIF的性能受到标注模型能力的限制。

RLAIF为扩展对齐工作提供了一个有趣的方向，尤其是在执行复杂行为规则方面。然而，它也带来了新的依赖和挑战，主要围绕AI反馈的质量和指导原则的设计。它通常不被视为人类反馈的完全替代，而可能是一种互补方法，或许用于初步的广泛对齐或特定原则的执行，之后可能再进行有针对性的人工微调 (fine-tuning)或评估。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  本开创性论文介绍了宪法AI，这是一种RLAIF方法，使用一系列原则指导AI评论者生成模型对齐的反馈，从而减少对大量人工标注的依赖。
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) — Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (2024)
  Journal: ICML 2024; DOI: [10.48550/arXiv.2401.10020](https://doi.org/10.48550/arXiv.2401.10020)
  本论文介绍了语言模型生成自身奖励反馈进行后续训练的，展示了AI反馈机制的进展，其中模型直接进行自我改进。
