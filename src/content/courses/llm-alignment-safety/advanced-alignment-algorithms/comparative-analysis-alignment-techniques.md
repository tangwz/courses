---
course: "llm-alignment-safety"
chapter: "advanced-alignment-algorithms"
lesson: "comparative-analysis-alignment-techniques"
sourceId: 3161
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/comparative-analysis-alignment-techniques"
title: "对齐方法对比分析"
description: "分析不同高级对齐方法的优缺点及适用性。"
order: 6
plots: []
sourceHash: "9b3ce8dcda520119adda5e6317965bd569926dc032efb6cfaee0f7b925a42272"
sourceCorrections: []
---

高级对齐 (alignment)方法种类多样，它们建立在基础的人类反馈强化学习 (reinforcement learning) (RLHF) 之上或对其进行扩展。理解这些方法的相对优缺点及其理想的应用情境非常重要。没有哪一种方法是普适最优的；选择通常取决于具体目标、可用资源以及所处理对齐问题的特点。一份对比分析有助于指导您的选择和实施决策。

我们将从几个方面比较RLHF（作为上一章建立的基线）、直接偏好优化（DPO）、宪法式AI（CAI）、AI反馈强化学习（RLAIF）和对比方法：

- **数据需求：** 训练或引导模型需要哪类数据？
- **计算成本：** 对齐阶段的计算要求是什么？
- **实施难度：** 整个流程的设置和管理难度如何？
- **可扩展性：** 该方法如何随着模型大小和数据量扩展？
- **对齐机制：** 该技术如何从根本上引导模型行为？
- **潜在不足：** 该方法有哪些已知的局限性或风险？

### RLHF 与直接偏好优化 (DPO) 的对比

RLHF 和 DPO 都使用人类（或 AI）偏好数据，通常是响应对，一个响应优于另一个。主要区别在于如何使用这些偏好数据。

- **RLHF：** 遵循三阶段过程：监督微调 (fine-tuning)（SFT）、奖励模型（RM）训练和强化学习 (reinforcement learning)优化（常使用PPO）。RM明确学习一个基于偏好的评分函数，然后指导大型语言模型（LLM）的策略更新。
  - **优点：** 方法成熟，在对齐 (alignment)大型模型方面已有成功案例。明确的奖励模型可以被查看。
  - **缺点：** 流程复杂，分多阶段；RM训练可能不稳定或校准不准；PPO优化对超参数 (parameter) (hyperparameter)敏感，且计算量可能很大。可能出现奖励作弊问题。
- **DPO：** 直接优化大型语言模型的策略以满足观察到的偏好，绕过了明确的RM训练步骤。它使用一个损失函数 (loss function)，通常源于与RLHF目标的一个理论联系，直接促使模型增加优选响应相对于劣选响应的可能性。损失函数可能如下所示：
  
  $$
  L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
  $$
  
  其中 $\pi_\theta$ 是正在优化的策略，$\pi_{ref}$ 是一个参考策略（通常是SFT模型），$(x, y_w, y_l)$ 表示一个提示 $x$ 及来自偏好数据集 $D$ 的优选响应 $y_w$ 和劣选响应 $y_l$，$\beta$ 是一个缩放因子，$\sigma$ 是Sigmoid函数。
  - **优点：** 流程更简单（无需单独的RM训练），训练通常比PPO更稳定，在偏好调优阶段计算要求低于完整RLHF。避免了RM不准确可能导致的问题。
  - **缺点：** 仍需要偏好数据，隐式假定其推导中使用的潜在偏好模型结构，如果需要非常复杂的奖励信号，灵活性可能不足。性能对$\beta$超参数敏感。

**何时选择：** 当您有偏好数据并寻求一个更简单、可能更稳定的训练过程时，DPO 通常是 RLHF 的有力替代方法。如果您需要 RM 的明确奖励分数用于其他目的（如评估或内容过滤），RLHF 可能仍然是优选。

### 基于偏好（RLHF/DPO）与基于原则（宪法式AI）的对比

宪法式AI (CAI) 提供了一种不同于直接从偏好中学习的方法。

- **宪法式AI：** 依赖于一套预定义的原则（“宪法”）来指导模型行为。它通常包括模型根据宪法批评和修订自身输出的阶段，常使用AI反馈（监督学习 (supervised learning) + AI反馈强化学习 (reinforcement learning)）。
  - **优点：** 减少了初始对齐 (alignment)对大规模*人类*偏好标注的需求，促进与明确规则一致的行为，可以通过借助大型语言模型自身进行评论来更好地扩展。
  - **缺点：** 有效性很大程度上取决于宪法的质量和全面性（这很难编写），依赖模型可靠解释和应用原则的能力（这可能失败），可能导致过于僵化或“律师般”的行为，存在AI误解或在宪法中发现漏洞的风险。
- **RLHF/DPO：** 从数据示例中隐式学习偏好。
  - **优点：** 可以捕捉难以用规则表达的精细或复杂偏好，对齐建立在观察到的期望行为之上。
  - **缺点：** 需要大量偏好数据收集，可能继承数据或标注者中存在的偏见，除非明确原则反映在偏好中，否则不保证遵守。

**何时选择：** 当您有明确定义的原则希望模型遵循，并希望减少对细粒度人类偏好数据的依赖时，CAI 具有吸引力。RLHF/DPO 通常更适合捕捉直接来源于交互数据的复杂、隐式的意涵。通常，会采用混合方法，例如，使用 CAI 原则指导初始的无害性训练，然后使用 RLHF 改进有用性。

### 人类反馈（RLHF）与AI反馈（RLAIF）的对比

虽然 RLHF 使用人类偏好，但 RLAIF 替换为 AI 生成的偏好，这些偏好通常由类似于 CAI 的原则或一个单独的、能干的“判断”大型语言模型指导。

- **RLAIF：** 使用AI模型在响应对之间生成偏好标签，通常基于指令或宪法。这些偏好数据随后进入RL（如PPO）或DPO流程。
  - **优点：** 与人类相比，反馈生成高度可扩展，迭代周期可能更快。
  - **缺点：** 对齐 (alignment)质量完全取决于生成反馈的AI的质量和偏见。存在放大现有模型偏见或创建反馈循环导致模型强化自身独特行为的风险。需要一个能干的指令遵循模型来充当判断者。
- **RLHF：** 使用人类判断。
  - **优点：** 作为人类价值观和偏好的真实依据（假定良好的标注质量），可以捕捉AI可能遗漏的细微差别。
  - **缺点：** 收集成本高且速度慢，容易受到人类不一致性和偏见的影响。

**何时选择：** RLAIF 主要受到可扩展性的驱动。当人工标注是主要瓶颈时，它非常有用，常与 CAI 原则结合使用。然而，它需要仔细验证以确保 AI 反馈与预期的人类价值观对齐，并且不引入有害偏见。当将对齐直接建立在人类判断之上是首要任务时，RLHF 仍然是标准。

### 对比方法

对比方法的运作方式不同，侧重于教导模型区分期望和非期望输出，而不是明确建模奖励或基于排序偏好直接优化策略。

- **对比方法：** 通常涉及训练模型为正面示例分配更高的可能性，为负面示例（例如，有害、有偏见或跑题的响应）分配更低的可能性。这可以在微调 (fine-tuning)期间使用专门的损失函数 (loss function)完成。
  - **优点：** 对特定对齐 (alignment)目标有效（例如，降低毒性、改进风格一致性），计算上可能比完整的强化学习 (reinforcement learning)循环更简单。可以融入SFT阶段或作为单独步骤。
  - **缺点：** 可能无法像基于偏好的方法那样捕捉整体有用性或复杂的权衡。有效性取决于正面/负面示例的质量和覆盖范围。可能需要仔细平衡以避免抑制期望的相关行为。

**何时选择：** 对比方法对于特定行为调整很有价值，例如施加特定的负面约束（例如，“不要生成有害内容”）或促进特定的正面属性（例如，“以特定风格响应”）。它们可以是对SFT的有益补充，或与基于偏好的方法一起使用。

### 对比总结

下表总结了这些高级对齐 (alignment)方法的主要特点：

| 特点 | RLHF | DPO | 宪法式AI (CAI) | RLAIF | 对比方法 |
| --- | --- | --- | --- | --- | --- |
| **主要数据** | 人类偏好 | 人类/AI偏好 | 宪法 + AI反馈 | AI偏好 | 正面/负面示例 |
| **核心机制** | 明确奖励模型 + RL策略优化 | 通过偏好损失直接策略优化 | 基于原则的自我批评/改进 | AI判断者 + RL/DPO策略优化 | 微调 (fine-tuning)期间的对比损失 |
| **计算成本** | 高（RM + RL训练） | 中等（直接优化） | 中高（迭代改进） | 中高（判断者 + RL/DPO） | 低中（微调） |
| **实施难度** | 高（多阶段，PPO调优） | 中等（比RLHF简单） | 中等（宪法设计，AI反馈） | 中等（取决于判断者，优化方法） | 低中 |
| **可扩展性** | 中等（人类数据瓶颈） | 中等（偏好数据瓶颈） | 高（AI反馈扩展性好） | 高（AI反馈扩展性好） | 高（如果示例易于生成） |
| **主要优势** | 成熟，学习复杂偏好 | 比RLHF更简单、更稳定 | 减少人类数据需求，明确规则 | 可扩展的反馈生成 | 目标行为控制 |
| **局限性** | 复杂，不稳定，需要人类数据 | 隐式假设，需要偏好数据 | 宪法质量，AI可靠性 | AI偏见放大，依赖判断者 | 可能无法很好捕捉整体偏好 |

### 选择您的方法

选择最合适的对齐 (alignment)方法需要考虑几个因素：

1. **对齐目标：** 您是优化通用有用性和无害性（RLHF/DPO）、强制执行特定原则（CAI）、实现特定行为调整（对比方法），还是需要大规模反馈（RLAIF）？
2. **数据可用性：** 您是否能获得大规模人类偏好数据（有利于RLHF/DPO），或者生成好/坏行为示例更容易（有利于对比方法）？您能否制定明确原则（有利于CAI）？
3. **计算资源：** RLHF，特别是PPO，可能要求很高。DPO 提供一个可能更轻量级的替代方案。CAI/RLAIF 的成本取决于生成和优化步骤。
4. **实施专业知识：** RLHF 流程复杂。DPO 在一定程度上简化了这一点。CAI 需要在提示工程 (prompt engineering)和可能管理迭代AI反馈循环方面的专业知识。
5. **风险承受能力：** RLAIF 带有与 AI 判断者偏见相关的风险。CAI 的风险取决于宪法的稳健性。RLHF/DPO 的风险与数据偏见和奖励作弊/隐式模型利用有关。

在实践中，这些方法并非总是独立使用。许多最先进的模型采用混合方法。例如，一个模型可能先进行 SFT，然后使用 CAI 原则结合 RLAIF 进行对齐以建立基线安全性，再通过 RLHF 或 DPO 与人类偏好一起改进有用性。该领域持续快速发展，因此需要持续评估这些方法并接受新出现的技术。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  提出了使用人类反馈强化学习（RLHF）将大型语言模型与人类偏好对齐的方法，证明了其在提高有用性和无害性方面的有效性。
- [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 36; DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  提出了直接偏好优化（DPO），一种更简单、更稳定的方法，用于将语言模型与偏好对齐，避免了显式奖励模型训练。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  提出了宪法AI，一种使用AI生成反馈将模型与一套原则对齐的方法，从而减少了对人工标注的需求。
