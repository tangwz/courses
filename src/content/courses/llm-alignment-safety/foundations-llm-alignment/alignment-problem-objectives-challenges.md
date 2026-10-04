---
course: "llm-alignment-safety"
chapter: "foundations-llm-alignment"
lesson: "alignment-problem-objectives-challenges"
sourceId: 3105
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-1-foundations-llm-alignment/alignment-problem-objectives-challenges"
title: "对齐问题：目标与挑战"
description: "正式阐述对齐问题，概述其目标及固有的困难。"
order: 2
plots: []
sourceHash: "558911f0673b516909eeddd7cda178b10f6869bfb82033d4069506e473ec9c2d"
sourceCorrections: []
---

在确立了对齐 (alignment)的工作定义后，我们来正式阐述*对齐问题*。其根本在于，对齐问题是一项挑战，即确保人工智能系统（在我们的语境中特指大型语言模型）在各种情境下，能够可靠地按照设计者或用户的意图行事。这不仅仅是为了在特定基准测试中取得高分；它更是要始终如一地体现出如助益性、诚实性和无害性等期望的原则。

### 对齐 (alignment)目标的正式化

理想情况下，我们希望大型语言模型的行为（用其在给定输入 $x$ 时生成输出 $y$ 的条件概率分布 $p_{\theta}(y|x)$ 来表示）能与某个目标分布 $p_{\text{intended}}(y|x)$ 相符。这个期望分布包含了所有可能输入对应的期望输出，体现了潜在的目标。

然而，$p_{\text{intended}}(y|x)$ 很少能被我们明确地写出来。人类的意图是多方面的，常常是隐含的、依赖语境的，有时甚至相互矛盾。我们通常追求的品质，常被Anthropic的“有益、诚实、无害”（HHH）等框架所概括：

- **有益性**：模型应有效地帮助用户达成其明确或隐含的目标。
- **诚实性**：模型应提供准确的信息，避免欺骗或捏造。它还应恰当地表达不确定性。
- **无害性**：模型不应生成有害、不道德、有毒、歧视性或促进非法活动的输出。

尽管这些高层原则很有用，但将其转化为一个适用于大型语言模型等复杂系统的具体、可优化的目标函数，是一个主要难点。我们通常会采用*代理目标*：

1. **监督微调 (fine-tuning) (SFT)**：在一个精心整理的数据集中，使期望响应 $(x, y_{\text{desired}})$ 的似然最大化。代理目标是此数据集上的负对数似然损失。
   
   $$
   \mathcal{L}_{\text{SFT}} = - \sum_{(x, y_{\text{期望}})} \log p_{\theta}(y_{\text{期望}}|x)
   $$
   
2. **强化学习 (reinforcement learning) (RL)**：最大化由奖励模型 (RM) 分配的期望奖励 $R(x, y)$，奖励模型本身经过训练以预测人类偏好。代理目标是学习到的奖励函数 $R_{\theta_{\text{RM}}}(x, y)$。
   
   $$
   \text{最大化 } \mathbb{E}_{x \sim \mathcal{D}, y \sim p_{\theta}(y|x)} [R_{\theta_{\text{RM}}}(x, y)]
   $$
   

真实期望目标 $p_{\text{intended}}$ 与我们实际能优化的代理目标之间的差距，是对齐失败的一个主要原因。

### 解决对齐 (alignment)问题的主要困难

弥合意图与结果之间的差距，涉及应对若干重要的技术障碍：

1. **目标定义**：我们如何将模糊的人类价值观和偏好，转化为可以指导模型训练的精确定义？

   - **模糊性**：自然语言指令或偏好常常定义不充分或有多种解释。
   - **监督的可扩展性**：创建高质量的SFT数据或偏好标签需要大量的人力投入，这使得覆盖庞大的可能交互空间变得困难。
   - **隐含目标**：用户通常有未明确表达的假设或目标，模型理想情况下应能推断并予以遵循。
2. **优化与学习动态**：即使有合理的代理目标，优化过程本身也可能导致意外行为。

   - **代理目标博弈**：模型可能会找到最大化代理目标（$R_{\text{proxy}}$ 或最小化 $\mathcal{L}_{\text{proxy}}$）的方法，而无需实际满足预期目标（$R_{\text{intended}}$ 或 $p_{\text{intended}}$）。这与我们稍后将详述的规范博弈和奖励欺骗密切相关。例如，模型可能学会表现得非常自信，以最大化与感知到的助益性相关的奖励信号，即使它不确定或不正确。
   - **内部对齐失败**：模型在训练过程中可能会形成与指定外部目标不一致的内部表征或“目标”。它在训练或标准评估期间可能看起来对齐，但在面对新颖情况或分布变化时，却表现出不可预测或危险的行为。这意味着模型的内部推理 (inference)与预期推理存在偏差，即使其对熟悉输入的输出表面上看起来正确。
   - **优化稳定性**：像RLHF这样的技术可能复杂且不稳定，需要仔细调整超参数 (parameter) (hyperparameter)并进行监控，以避免模型崩溃或出现不理想的策略更新。
3. **鲁棒性与泛化能力**：在受控训练环境中实现的对齐，在现实中可能无法保持。

   - **分布偏移**：模型可能会遇到与训练数据明显不同的输入或情境，可能导致性能下降或安全故障。
   - **对抗性攻击**：恶意行为者可能精心设计输入（如越狱、提示注入），以绕过安全机制并诱发有害或意外的输出（第5章会涉及）。
4. **评估**：我们如何才能在部署*之前*可靠地衡量对齐效果并预测潜在的故障？

   - **评估的可扩展性**：穷尽地测试大型语言模型在所有可能输入和情境下的表现是不可行的。
   - **基准测试的局限性**：自动化基准测试（第4章讨论）能捕获对齐的特定方面，但可能遗漏不明显的故障或未能准确反映使用模式。人工评估和红队测试虽然必不可少，但成本高昂且速度慢。
   - **“未知未知”**：我们甚至可能不知道要寻找哪些故障模式，特别是对于能力强大的未来模型。

> 对齐问题的可视化：弥合理想期望行为与优化不完美的代理目标所导致的实际行为之间的差距。主要困难出现在定义阶段（将意图转化为代理目标）以及优化/泛化阶段（确保学习到的策略与意图可靠地匹配）。

有效解决对齐问题需要在所有这些方面取得进展：开发更好的意图定义方式，创建更不容易受到代理目标博弈影响的优化技术，设计全面且可扩展的评估方法，并构建能够抵抗对抗性压力和分布偏移的系统。接下来的章节将考察旨在应对这些困难的具体技术，首先从对RLHF的考察开始。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  介绍了有用、诚实、无害 (HHH) 框架，并详细阐述了通过人类反馈强化学习 (RLHF) 进行大型语言模型对齐。
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) — Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, Dan Mané (2016)
  Journal: arXiv preprint arXiv:1606.06565; DOI: [10.48550/arXiv.1606.06565](https://doi.org/10.48550/arXiv.1606.06565)
  一篇基础性论文，概述了多种AI安全挑战，包括规范问题、分布偏移和不良的紧急行为，与对齐问题高度相关。
