# SFT在RLHF流程中的作用

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf/sft-role-in-rlhf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

监督微调 (fine-tuning)（SFT）是标准人类反馈强化学习 (reinforcement learning)（RLHF）工作流中奠定基础的第一步。预训练 (pre-training)的大语言模型（LLM）虽具备丰富的知识，但通常缺少对齐 (alignment)任务所需的具体指令遵循能力、期望的输出格式或对话风格。SFT通过使用精选的高质量提示-响应对数据集（常称为演示数据）来调整基础LLM，从而弥补了这一不足。

可以将SFT阶段看作是为模型提供在目标情境中*如何*表现的初始训练。此时并非学习偏好，而是学习游戏的基本规则 – 如何回应提示、使用何种格式，以及可能采纳特定角色（例如，一个乐于助人的助手）。

### 为强化学习 (reinforcement learning)初始化策略

SFT的主要技术作用是生成一个初始策略，通常表示为$\pi_{SFT}$，它作为后续强化学习（RL）阶段的起点。RL算法（在此情境中通常是近端策略优化PPO）并非从原始预训练 (pre-training)模型的参数 (parameter)开始优化，而是从SFT模型的参数开始。

这种初始化提供了重要的优势：

1. **更优的起点：** $\pi_{SFT}$已比原始预训练模型更接近期望的行为空间。它理解任务格式，并且平均能生成更相关的响应。
2. **提高样本效率：** 由于$\pi_{SFT}$已经能产生相当好的输出，RL阶段需要较少的尝试来找到高奖励轨迹（提示-响应对）。模型不会浪费太多时间生成完全不相关或格式不佳的文本，从而能更快地根据学到的奖励信号专注于优化输出。
3. **稳定性：** 从适应目标数据分布的模型开始，与从通用模型开始相比，可以带来更稳定的RL训练。

下图说明了SFT模型如何融入整个流程：

> 此流程图说明了RLHF的三个阶段。在阶段1生成的SFT模型（$\pi_{SFT}$）用于初始化RL策略（$\pi_0$），并常作为阶段3中KL惩罚的参考策略。它也可用于为阶段2的人工比较数据收集生成初始响应。

### 建立行为基线与参考点

除了初始化权重 (weight)，SFT阶段还建立了一个行为基线。演示数据教导模型：

- **指令遵循：** 如何理解和遵循提示中给出的指令。
- **输出格式：** 遵守特定的格式，如Markdown、代码块、列表等。
- **风格与语气：** 采纳SFT数据中隐含的期望角色、正式程度或安全限制。

非常重要的一点是，SFT模型（$\pi_{SFT}$）通常也作为RL阶段PPO算法中使用的KL散度惩罚的**参考策略**。PPO目标函数包含一个像$\beta \cdot D_{KL}(\pi_{RL} || \pi_{SFT})$的项，其中$\pi_{RL}$是正在训练的策略。这个KL项惩罚RL策略在每一步都偏离初始SFT策略过远。

这为何重要？

1. **保持能力：** 这有助于保留预训练 (pre-training)和SFT阶段学到的通用语言能力和知识，防止RL优化在最大化奖励信号的同时，大幅降低流畅性或连贯性。
2. **正则化 (regularization)：** 它起到正则化器的作用，防止策略退化为简单、重复的输出，这类输出可能**钻奖励模型规则的空子**（奖励作弊）。
3. **受控适应：** 它确保RL阶段专注于基于偏好优化模型，而不是以不期望的方式根本改变其核心行为。

### 奖励模型训练的预计算

尽管这不是SFT模型的主要作用，但它常用于后续奖励建模阶段的数据生成过程。为了收集人类偏好数据，提示会被输入到一个或多个模型（通常包括SFT模型本身）以生成多个候选响应。然后，人工标注者会比较这些响应（例如，选择一对中更好的一个）。因此，拥有一个有能力的SFT模型有助于创建训练有效奖励模型所需的、相关且多样化的候选响应。

总之，SFT阶段不只是一个初步步骤，它更是RLHF流程的一个基本组成部分。它使LLM适应目标范围，为高效的RL训练初始化策略，建立所需的行为格式和风格，并提供（$\pi_{SFT}$）作为参考点，这对通过KL散度约束进行稳定和有效的PPO优化来说非常必要。一个执行良好的SFT阶段能显著简化后续更复杂的奖励建模和强化学习 (reinforcement learning)阶段。

## 参考资料

- [Scaling Instruction-Finetuned Transformers](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  探讨了指令微调（SFT）在不同任务和模型上的有效性，提供了SFT如何增强指令遵循能力的理解。
- [Aligning Language Models to Follow Instructions](https://openai.com/blog/aligning-language-models-to-follow-instructions) — OpenAI (2022)
  Publisher: OpenAI Blog
  OpenAI的一篇通俗博客文章，解释了InstructGPT论文，清晰概述了RLHF流程和SFT的作用。

---

[上一节](../01-RLHF%E4%B8%8E%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E7%9A%84%E5%88%9D%E6%AD%A5%E4%BB%8B%E7%BB%8D/05-%E8%AE%BE%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](02-%E9%AB%98%E8%B4%A8%E9%87%8FSFT%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E6%95%B4%E7%90%86.md)
