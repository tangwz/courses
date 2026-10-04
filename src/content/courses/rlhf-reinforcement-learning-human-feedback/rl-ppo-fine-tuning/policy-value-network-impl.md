---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rl-ppo-fine-tuning"
lesson: "policy-value-network-impl"
sourceId: 5144
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/policy-value-network-impl"
title: "策略网络与价值网络的实现"
description: "为PPO设置actor（策略）网络和critic（价值）网络。"
order: 2
plots: []
sourceHash: "aaf68e4c4c7e4f11cde1312c3a44279c70c733815290077afcd70c371127250a"
sourceCorrections: []
---

Actor-critic方法是近端策略优化（PPO）算法中的标准做法。这需要维护两个不同但彼此关联的网络：一个策略网络（actor），它决定采取哪些动作（例如，生成哪些token）；以及一个价值网络（critic），它估计给定状态下的预期回报。在RLHF中微调 (fine-tuning)大型语言模型时，通常会实现这些网络。

### 策略网络（Actor）

策略网络就是语言模型本身。其主要作用是给定输入提示后，逐个token地生成文本序列。

- **初始化**：策略网络使用在监督微调 (fine-tuning)（SFT）阶段（第2章中提到）之后获得的模型权重 (weight)进行初始化。这提供了一个稳固的起点，确保模型已经具备良好的生成能力，并遵循在SFT期间学习到的所需风格或格式。
- **架构**：架构与所用的底层大型语言模型（例如Transformer解码器）相同。与SFT模型相比，通常不需要结构上的改变。
- **作用**：在强化学习 (reinforcement learning)阶段，对于给定提示（状态 $s$），策略网络 $\pi_\theta(a|s)$ 输出下一个token（动作 $a$）在词汇表 (vocabulary)上的概率分布。文本生成涉及从该分布中依序采样。
- **优化目标**：策略网络的参数 (parameter) $\theta$ 使用PPO目标函数进行更新。目标是增加生成序列获得奖励模型高分的概率，同时KL散度项（下一节中讨论）防止策略与初始SFT模型的行为偏离过大。

### 价值网络（Critic）

价值网络估计给定状态下的预期累积未来奖励。在PPO中，此估计值作为一个基准，用于减少策略梯度更新的方差。

- **职责**：价值网络学习状态价值函数 $V_\phi(s)$。给定状态 $s$（提示加上目前已生成的序列），它预测策略从该点起预期获得的总奖励。此预测对于计算优势估计非常重要，通常使用广义优势估计（GAE）等方法，这些方法量化 (quantization)了某个动作相对于该状态下平均动作的优势程度。优势大致为 $A(s,a) \approx R(s, a) + \gamma V_\phi(s') - V_\phi(s)$，其中 $R(s,a)$ 是即时奖励（从奖励模型获得），$s'$ 是下一个状态，$\gamma$ 是折扣因子。
- **架构**：价值网络通常与策略网络共享核心架构和参数 (parameter)。它重用LLM主体（例如Transformer层）学习到的强大表示。在共享LLM主体的最终隐藏状态表示之上，会添加一个单独的“价值头”——通常是一个简单的线性层。此头输出一个表示预测状态价值 $V_\phi(s)$ 的单个标量值。
- **初始化**：虽然共享主体（像策略网络一样）继承自SFT模型的权重 (weight)，但价值头通常随机初始化或使用小权重，因为它需要在强化学习 (reinforcement learning)训练期间从头开始学习新的函数（价值估计）。
- **优化目标**：价值网络的参数 $\phi$（主要是价值头，但也可能包括共享主体参数）与策略网络同步训练。目标通常是最小化预测值 $V_\phi(s)$ 与在PPO推演阶段计算出的实际观测回报（目标）之间的均方误差（MSE）（例如，使用GAE）：
  
  $$
  L(\phi) = \mathbb{E}_{(s, V^{target})} \left[ (V_\phi(s) - V^{target})^2 \right]
  $$
  
  其中 $V^{target}$ 表示状态 $s$ 的计算回报目标。

### 参数 (parameter)共享策略

在actor-critic方法中，策略网络和价值网络之间共享参数是一种常见且有效的方法，对于大型语言模型尤其如此。

- **效率**：这大大减少了需要训练和存储的参数数量，使过程在计算上更可行。
- **表示学习**：大型语言模型主体学习到的丰富表示对于生成文本（策略）和估计未来奖励（价值）都很有用。共享机制使价值网络能够运用预训练 (pre-training)和SFT期间学习到的这些特征。

典型的实现是使用SFT模型作为基础。在强化学习 (reinforcement learning)训练循环的每次前向传播中，输入提示由共享的大型语言模型主体处理。最终的隐藏状态随后被送入两个单独的头部：

1. **策略头**：通常是原始的语言建模头，输出词汇表 (vocabulary)上的logits。
2. **价值头**：一个新的线性层，输出一个单个标量值。

> 该图说明了常见的参数共享架构。输入状态由共享的大型语言模型主体处理，其输出分别送入单独的头部进行策略预测和价值估计。

### 库的使用实现

像Hugging Face的TRL（Transformer强化学习 (reinforcement learning)）这样的库简化了这种设置。它们通常提供包装类，例如`AutoModelForCausalLMWithValueHead`。这个类接收一个标准的预训练 (pre-training)因果语言模型（如GPT-2、Llama等），并自动在现有语言模型头旁边附加一个价值头。

```python
# 使用Hugging Face TRL的示例（伪代码）
from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer

# 加载SFT模型作为基础
model_name = "path/to/your/sft_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# 现在'model'包含策略（LM）头和随机初始化的价值头。
# 在训练期间，一次前向传播会产生两个输出：

prompt = "Translate to French: Hello world"
inputs = tokenizer(prompt, return_tensors="pt")

# 前向传播返回logits、过去的键值以及价值估计
outputs = model(**inputs)
policy_logits = outputs.logits
value_estimate = outputs.value # 价值头输出的标量值

# TRL中的PPO训练器负责使用这些输出
# 进行动作生成、优势计算以及更新两个网络。
```

通过从SFT模型初始化策略网络并使用带有专用价值头的共享架构，我们为PPO算法搭建了必要的组成部分。价值网络在提供准确状态价值估计方面的作用对强化学习微调 (fine-tuning)阶段中稳定有效的策略更新来说非常重要。接下来的部分将详细说明这些组件在PPO更新循环中如何相互配合，特别是关注KL散度惩罚和优势计算。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍近端策略优化（PPO）算法的原始论文，详细说明其目标函数、演员-评论家结构和策略更新方法。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel (2015)
  Journal: arXiv preprint arXiv:1506.02438; DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  本文介绍了广义优势估计（GAE），这是一种广泛用于减少策略梯度方法中方差的技术，对PPO中价值网络的功能至关重要。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 35; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  一篇重要的论文，它使用PPO将人类反馈强化学习（RLHF）应用于大型语言模型微调，展示了策略和价值网络的实际考量。
- [TRL Documentation: PPO Trainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face TRL库PPO训练器的官方文档，提供使用AutoModelForCausalLMWithValueHead实现PPO的实践指导和代码示例。
