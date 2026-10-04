---
course: "llm-constitutional-ai-rlaif"
chapter: "implementing-constitutional-ai"
lesson: "fine-tuning-llm-cai-data"
sourceId: 4478
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-3-implementing-constitutional-ai/fine-tuning-llm-cai-data"
title: "使用 CAI 数据微调 LLM"
description: "使用生成的 CAI 数据集执行监督微调步骤。"
order: 6
plots: ["plots/4478-0.json"]
sourceHash: "2c808bb107fbc42585dae77dabfb616c8b5fcfe28db7b720ace8b5a18936c766"
sourceCorrections: []
---

对大型语言模型 (LLM) 进行微调 (fine-tuning)，使用专门数据作为监督学习 (supervised learning)阶段的核心训练步骤，尤其当这些数据包含符合宪法原则的修订时。目标是将修订响应中体现的宪法原则直接融入模型的参数 (parameter)中。此目的是生成一个模型 $M_{SFT}$，它能够在推理 (inference)时无需明确调用批判和修订机制，便能生成符合宪法原则的输出。

这个过程将批判和修订阶段的复杂多步骤推理提炼到 LLM 的权重 (weight)中。微调数据集由成对的 $(P, R_{revised})$ 组成，这里 $P$ 是原始提示，而 $R_{revised}$ 是 AI 生成的、符合宪法原则的修订响应，这组数据用作训练信号。

### 监督微调 (fine-tuning)机制

归根结底，这一步采用标准的监督微调 (SFT)，这是一种使预训练 (pre-training)模型适应特定下游任务或风格的常见技术。然而，CAI 生成数据的特性带来了一些特别的考虑。

1. **输入与目标：** 训练期间模型的输入是提示 $P$。模型学习预测的目标是修订后的响应 $R_{revised}$。我们正在训练模型，使其能够直接将提示映射到满足数据生成阶段所用宪法约束的响应。
2. **模型架构：** 通常，微调过程从用于生成初始响应的相同基础模型 $M_{base}$ 开始。这可确保模型保持其通用能力，同时根据 CAI 数据调整其行为。
3. **损失函数 (loss function)：** 采用标准的自回归 (autoregressive)语言模型损失，通常是交叉熵。损失是在模型对下一个词元 (token)的预测概率分布与目标序列 $R_{revised}$ 中实际的下一个词元之间计算的。

   
   $$
   \mathcal{L}_{SFT} = - \sum_{i} \log P(token_i | P, token_{<i}; \theta)
   $$
   

   这里，$token_i$ 表示目标序列 $R_{revised}$ 中的第 $i$ 个词元，$P$ 是输入提示，$token_{<i}$ 是前面的目标词元，$\theta$ 表示正在优化的模型参数 (parameter)。

### 训练数据准备

正确格式化数据对有效微调 (fine-tuning)很重要。每个训练样本通常将提示和修订后的响应连接起来，常使用特殊词元 (token)来划分不同部分。

常见的格式可能如下所示：

```
<|system|> 你是一个符合宪法原则的有用助手。 <|user|> {提示文本} <|assistant|> {修订响应文本}<|endoftext|>
```

关键是，损失仅在与 `{修订响应文本}` 部分对应的词元上计算。提示词 (prompt)元以及任何前面的上下文 (context)或特殊词元在损失计算期间被屏蔽。这确保模型学习根据提示 *生成* 所需的响应，而不是学习预测提示本身。像 Hugging Face 的 `transformers` 等库提供了工具（例如，数据整理器 Data Collators）来自动处理这种屏蔽。

### 微调 (fine-tuning)超参数 (parameter) (hyperparameter)及注意事项

虽然标准的 SFT 实践适用，但考虑到 CAI 数据集可能的人工合成和目标导向特性，使用 CAI 数据进行微调需要仔细调整超参数：

- **学习率：** 通常需要比预训练 (pre-training)小得多的学习率（例如，$1e^{-5}$ 到 $5e^{-6}$），以避免破坏模型的现有知识，同时温和地引导其行为。学习率调度器，如带有预热阶段的余弦衰减，是标准做法。
- **批次大小：** 由可用 GPU 内存决定。较大的批次大小通常会带来更稳定的梯度，但需要更多资源。梯度累积可以模拟更大的批次大小。
- **训练轮次：** 训练通常只涉及少数几个训练轮次（通常为 1-3）。过拟合 (overfitting) CAI 数据集是一个风险；目标是实现对齐 (alignment)，而不是记住特定的修订。密切监控验证损失。
- **优化器：** AdamW 仍是常见选择。
- **参数高效微调 (PEFT)：** 像低秩适应 (LoRA) 这样的技术在这里非常适合。PEFT 方法可以通过仅更新模型参数的一小部分来显著降低微调的计算成本和内存占用。这也有助于减轻模型通用能力的灾难性遗忘，因为核心预训练权重 (weight)基本未被改动。当资源受限或微调非常大的模型时，特别考虑 PEFT。
- **数据多样性：** CAI 微调的有效性在很大程度上取决于用于生成初始数据集的提示的多样性和覆盖范围。如果提示只涵盖狭窄的主题范围或宪法原则，那么产生的对齐效果可能不稳定。扩充 CAI 数据或在生成过程中确保全面的提示覆盖很重要。

### 训练期间的监控与评估

持续评估微调 (fine-tuning)进度：

- **损失曲线：** 跟踪训练损失和验证损失。验证损失增加表明存在过拟合 (overfitting)。



![SFT 损失走势](plots/4478-0.json)



> SFT 的损失曲线。训练损失降低，而验证损失在第 7 步之后开始增加，这表明可能存在过拟合。

- **定性评估：** 定期使用多样化的保留提示从检查点模型生成响应。手动检查这些响应，以评估模型是否正在采纳预期的宪法行为（例如，增加的帮助性、降低的有害性、遵守特定原则）。
- **自动化指标：** 如果有可用，使用旨在衡量对齐 (alignment)方面的自动化指标（例如，毒性评分、基于训练用于检测宪法违反的分类器的符合性评分）。这为第七章讨论的更严格评估做准备。

### 结果：$M_{SFT}$

成功完成此微调 (fine-tuning)过程将生成 CAI 对齐 (alignment)模型，表示为 $M_{SFT}$。该模型已将 AI 批判和修订过程所理解的宪法原则整合到其生成行为中。与原始 $M_{base}$ 相比，它现在应该能够以更符合指定宪法的方式响应提示。

这个 $M_{SFT}$ 模型代表了宪法 AI 监督阶段的最高成果。它可以直接部署，接受进一步的严格评估（第七章），或者作为后续对齐阶段（例如，如我们将在第六章讨论的，基于 AI 反馈的强化学习 (reinforcement learning) (RLAIF)）的一个改进起点。这一微调步骤的质量对于 CAI 方法的整体成功非常重要。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了宪法AI框架，包括生成AI反馈和修订的方法，这些反馈和修订用于监督微调。
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; Publisher: arXiv; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  详细介绍了监督微调（SFT）阶段，这是使大型语言模型与人类意图和指令对齐的初始步骤。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种通过仅更新少量额外参数来有效微调大型语言模型的方法。
- [Fine-tune a pretrained model with custom datasets](https://huggingface.co/docs/transformers/training#fine-tune-a-pretrained-model-with-custom-datasets) — Hugging Face (2024)
  Publisher: Hugging Face
  关于如何使用Hugging Face `transformers`库对语言模型进行监督微调的实践指南，涵盖数据准备和训练循环。
