---
course: "llm-constitutional-ai-rlaif"
chapter: "integrating-cai-rlaif"
lesson: "cai-output-for-rlaif"
sourceId: 4535
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-6-integrating-cai-rlaif/cai-output-for-rlaif"
title: "将 CAI 产出作为 RLAIF 的输入"
description: "利用 CAI SL 阶段调整后的产出，为 RLAIF 提供初始数据或进行提升。"
order: 2
plots: []
sourceHash: "8716e84b139c7539e6025a18bcf10d32bf107bae40671af897066d1a431a7ad6"
sourceCorrections: []
---

宪法人工智能 (CAI) 过程，特别是其监督学习 (supervised learning) (SL) 阶段，不仅生成微调 (fine-tuning)模型，还产生重要成果。此阶段包括向大型语言模型 (LLM) 提供提示，让人工智能根据某个宪法来评价回复，然后让人工智能根据评价修改回复。这会得到一个数据集，包含大量符合宪法原则的行为调整示例。我们可以不将 CAI SL 阶段和 RLAIF 阶段视为独立的步骤，而是将前者的输出应用于后者，以改进对齐 (alignment)效果。这种整合旨在将 CAI 基于原则的指导融入 RLAIF 基于偏好的优化中，以期实现更有效的对齐。

### 使用 CAI 微调 (fine-tuning)模型启动 RLAIF

一种直接的整合方式是，使用在 CAI SL 阶段微调过的 LLM 作为 RLAIF 训练的起点。回顾一下，CAI SL 阶段通过评价和修改回复对的模型进行微调，从而通过自我纠正的示例，有效地让模型掌握宪法原则。

我们可以不用通用预训练 (pre-training)或指令微调模型来初始化 RLAIF，而以 CAI 微调模型作为开始，这会带来潜在好处：

1. **更好的初始化：** 模型已具备一定程度的宪法对齐 (alignment)。其在 RLAIF 在线生成阶段的初始输出不太可能严重违背原则，可能将强化学习 (reinforcement learning)的尝试重心放在已可接受范围内的帮助性和无害性调整上。
2. **更快的收敛：** 由于起始策略更接近所需对齐状态（由宪法隐含定义并由 RLAIF 的 AI 偏好调整），与从对齐程度较低的模型开始相比，强化学习过程可能更快收敛或达到更高程度的对齐。RLAIF 中使用的近端策略优化 (PPO) 算法会惩罚偏离参考策略的较大差异；从更好的参考策略开始可以简化优化过程。
3. **减少灾难性遗忘：** 尽管仍是一个问题，但从 CAI 微调模型开始，可能略微减轻强化学习过程遗忘宪法限制的风险，因为这些限制已部分融入模型的权重 (weight)中。

此实现仅涉及获取 CAI SL 微调阶段的最终检查点，并在 RLAIF PPO 训练循环开始时，将这些权重加载为初始策略 $\pi_{\theta}$ 和可能的参考策略 $\pi_{ref}$。

### 将 CAI 数据转换为偏好对

在 CAI SL 阶段生成的数据集通常包含以下元组：`(prompt, initial_response, critique, revised_response)`。这些结构化数据可以重新用于创建适合训练 RLAIF 偏好模型 (PM) 的偏好对。

核心想法是，对于给定提示，将符合宪法原则的修改回复视为比初始回复更优。评价和修改过程在宪法的引导下，作为一种隐含的偏好判断：模型*应该*生成类似修改回复的内容，而不是初始回复。

因此，对于给定提示 $p$、初始回复 $r_{initial}$ 和修改回复 $r_{revised}$，我们可以生成一个偏好对 $(r_{chosen}, r_{rejected})$，其中 $r_{chosen} = r_{revised}$，$r_{rejected} = r_{initial}$。

这会创建数据集 $D_{CAI\_prefs} = \{ (p, r_{revised}, r_{initial}) \}$。然后，此数据集可以与标准 RLAIF AI 标注器生成的偏好数据（其比较当前策略的两个输出 $r_A$ 和 $r_B$）结合起来。

**考虑要点：**

- **数据扩充：** $D_{CAI\_prefs}$ 可以扩充主要的 RLAIF 偏好数据集，提供明确基于宪法原则的示例。
- **筛选：** 并非所有 CAI 修改都必然表示与 RLAIF 目标相关的强烈偏好。次要的风格编辑可能不如与安全性或偏见相关的修正重要。您可以筛选 CAI 数据，以仅包含由特定宪法原则（例如与减少危害相关的原则）触发的修改。
- **加权：** 在组合数据集上训练偏好模型时，您可能需要根据对每个来源的置信度或强调宪法依从性的意愿，为来源于 $D_{CAI\_prefs}$ 的样本与来源于标准 AI 标注器的样本分配不同的权重 (weight)。
- **偏好强度：** 二进制偏好 $(r_{revised} \succ r_{initial})$ 可能过于简单。评价的严重程度或修改的幅度可能为偏好强度分数提供依据，尽管这会增加复杂性。

偏好模型随后通过使用以下损失函数 (loss function)进行训练，以最大化所选回复的似然并最小化被拒绝回复的似然：


$$
\mathcal{L}_{PM} = - \mathbb{E}_{(p, r_{chosen}, r_{rejected}) \sim D_{combined}} [\log(\sigma(f_{PM}(p, r_{chosen}) - f_{PM}(p, r_{rejected})))]
$$


其中 $f_{PM}$ 是偏好模型的评分函数，$\sigma$ 是 Sigmoid 函数，$D_{combined}$ 是 RLAIF 生成数据和 CAI 衍生偏好数据的并集。

### 为 RLAIF 经验缓冲区提供初始数据

另一种方法是直接将成功的 CAI 示例注入 RLAIF PPO 阶段使用的经验缓冲区。PPO 算法通过采样提示、使用当前策略 $\pi_{\theta}$ 生成回复、使用奖励模型（基于偏好训练）评估这些回复，以及更新策略来学习。

我们可以用从 CAI SL 数据集衍生的元组 $(p, r_{revised})$ 来扩充经验缓冲区。当 PPO 算法采样经验以更新策略时，它会偶尔提取这些高质量、经过宪法原则核查的示例。

**益处：**

- **指引：** 在强化学习 (reinforcement learning)过程早期提供所需行为的直接示例。
- **稳定性：** 通过确保策略更新受已知良好路径的影响，可以潜在地稳定训练，尤其是在初始策略或奖励模型有噪声的情况下。

此实现需要修改 PPO 训练器中的经验收集或采样机制，以将这些预计算示例与在线生成内容一起包含在内。必须注意为这些注入的样本分配适当的奖励分数和潜在的优势估计值，这很可能使用训练好的 RLAIF 偏好/奖励模型来为提示 $p$ 的 $r_{revised}$ 打分。

> 利用 CAI 产出提升 RLAIF 的整合要点：(1) 使用 CAI 微调 (fine-tuning)模型初始化 RLAIF 策略。(2) 使用 CAI（提示、初始、修改）数据生成偏好对，用于 PM 训练。(3) 使用 CAI 示例为 PPO 经验缓冲区提供初始数据。(4) 潜在地在 AI 偏好标注器内部使用 CAI 微调模型本身。

通过巧妙地重用 CAI SL 阶段的产出，RLAIF 训练可以用对齐 (alignment)程度更高的模型进行初始化，从符合宪法原则的偏好数据中获益，并可能更快收敛或达到更好的状态。这种整合将 CAI 过程从仅仅一个预训练 (pre-training)步骤转变为后续强化学习阶段有价值的数据和初始化的来源。然而，必须仔细考虑数据选择、加权以及 RLAIF 学习到的隐含偏好与 CAI 期间编码的明确原则之间的潜在冲突。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法级AI（CAI），包括其基于批判和修订的监督学习阶段以及随后的AI反馈强化学习（RLAIF）。它为CAI及其RLAIF应用奠定了基础。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种健壮的在线强化学习方法，广泛应用于语言模型对齐，包括RLAIF。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2022); DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  详细介绍了用于对齐遵循指令的语言模型的“基于人类反馈的强化学习”（RLHF）流程。它提供了基于偏好的语言模型对齐的例子。
